"""
ResNet implementation with enhanced performance and stability features.

This module provides ResNet models (18, 34, 50, 101, 152).

The implementation maintains compatibility with torchvision weights
while adding performance optimizations and monitoring capabilities.

References:
- He et al. (2016) "Deep Residual Learning for Image Recognition"
- torchvision ResNet implementation
"""

import logging
from typing import Type, Union, List, Optional, Callable, Dict, Any, Tuple

import torch
import torch.nn as nn
from torchvision import models as torch_models
from torchvision.models import resnet
from dynvision.base import BaseModel
from dynvision.utils import check_stability


logger = logging.getLogger(__name__)

__all__ = [
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
]


def validate_resnet_parameters(
    num_classes: int,
    groups: int,
    width_per_group: int,
) -> None:
    """Validate ResNet architecture parameters."""
    if num_classes <= 0:
        raise ValueError("Number of classes must be positive")
    if groups <= 0:
        raise ValueError("Number of groups must be positive")
    if width_per_group <= 0:
        raise ValueError("Width per group must be positive")
    """
    Base ResNet implementation with enhanced features.
    
    Args:
        block: ResNet block type (BasicBlock or Bottleneck)
        layers: Number of blocks per layer
        num_classes: Number of output classes
        zero_init_residual: Whether to zero-initialize residual BN
        groups: Number of groups for group convolution
        width_per_group: Width of each group
        replace_stride_with_dilation: Whether to replace stride with dilation
        norm_layer: Normalization layer
        mixed_precision: Whether to use automatic mixed precision
        stability_check: Whether to check numerical stability
    """


class ResNetx4(torch_models.ResNet, BaseModel):
    def __init__(
        self,
        block: Type[Union[resnet.BasicBlock, resnet.Bottleneck]],
        layers: List[int],
        input_dims: List[int] = (1, 3, 224, 224),
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        mixed_precision: bool = True,
        stability_check: bool = False,
        **kwargs: Any,
    ) -> None:
        BaseModel.__init__(
            self,
            block=block,
            layers=layers,
            input_dims=input_dims,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
            mixed_precision=mixed_precision,
            stability_check=stability_check,
            **kwargs,
        )

        # Validate parameters
        validate_resnet_parameters(num_classes, groups, width_per_group)

        # Store configuration
        self.mixed_precision = mixed_precision
        self.stability_check = stability_check
        self._define_architecture()
        self._init_parameters()

    def _define_architecture(self) -> None:
        """Define ResNet architecture."""
        self.layer_names = ["layer1", "layer2", "layer3", "layer4"]
        self.layer_operations = [
            "layer",  # apply (recurrent) convolutional layer
            "addskip",  # add skip connection
            "nonlin",  # apply nonlinearity
            "supralin",  # apply supralinearity
            "record",  # record activations in responses dict
            "delay",  # set and get delayed activations for next layer
            "pool",  # apply pooling
            "norm",  # apply normalization
        ]
        torch_models.ResNet.__init__(
            self,
            block=self.block,
            layers=self.layers,
            num_classes=self.num_classes,
            zero_init_residual=self.zero_init_residual,
            groups=self.groups,
            width_per_group=self.width_per_group,
            replace_stride_with_dilation=self.replace_stride_with_dilation,
            norm_layer=self.norm_layer,
        )
        self.classifier = self.fc

    def _init_parameters(self) -> None:
        weights_enum = MODEL_CONFIGS[self.model_name]["weights"]
        weights = weights_enum.DEFAULT
        self.load_state_dict(weights.get_state_dict(progress=False, check_hash=True))
        self._set_trainable_params()

    def _forward_impl(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass implementation.

        Args:
            x: Input tensor

        Returns:
            Tuple containing:
                - Model output
                - Dictionary of layer responses
        """
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.stability_check:
            check_stability(x, "initial")

        # Main layers
        x1 = self.layer1(x)
        if self.stability_check:
            check_stability(x1, "layer1")

        x2 = self.layer2(x1)
        if self.stability_check:
            check_stability(x2, "layer2")

        x3 = self.layer3(x2)
        if self.stability_check:
            check_stability(x3, "layer3")

        x4 = self.layer4(x3)
        if self.stability_check:
            self.check_stability(x4, "layer4")

        # Classification
        x = self.avgpool(x4)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        if self.stability_check:
            check_stability(x, "classifier")

        responses = {
            "layer1": x1,
            "layer2": x2,
            "layer3": x3,
            "layer4": x4,
            "classifier": x,
        }
        return x, responses

    def _forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass."""
        return self._forward_impl(x)

    def forward(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with base class handling."""
        return BaseModel.forward(self, x, **kwargs)

    def reset(self) -> None:
        """Reset model state."""
        pass

    def _determine_residual_timesteps(self, **kwargs) -> int:
        """Determine residual timesteps (none for ResNet)."""
        return 0


# ResNet model configurations
MODEL_CONFIGS = {
    "ResNet18": {
        "block": resnet.BasicBlock,
        "layers": [2, 2, 2, 2],
        "weights": torch_models.ResNet18_Weights,
    },
    "ResNet34": {
        "block": resnet.BasicBlock,
        "layers": [3, 4, 6, 3],
        "weights": torch_models.ResNet34_Weights,
    },
    "ResNet50": {
        "block": resnet.Bottleneck,
        "layers": [3, 4, 6, 3],
        "weights": torch_models.ResNet50_Weights,
    },
    "ResNet101": {
        "block": resnet.Bottleneck,
        "layers": [3, 4, 23, 3],
        "weights": torch_models.ResNet101_Weights,
    },
    "ResNet152": {
        "block": resnet.Bottleneck,
        "layers": [3, 8, 36, 3],
        "weights": torch_models.ResNet152_Weights,
    },
}


class ResNetBase(ResNetx4):
    """Base class for ResNet variants."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        """Initialize ResNet variant.

        Args:
            model_name: Name of ResNet variant
            **kwargs: Additional arguments
        """
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")

        config = MODEL_CONFIGS[model_name]
        self.model_name = model_name
        super().__init__(block=config["block"], layers=config["layers"], **kwargs)

    def _set_trainable_params(self) -> None:
        """Set trainable parameters (classifier only)."""
        try:
            self.trainable_parameter_names = [
                p for p in list(self.state_dict().keys()) if "classifier" in p
            ]
            n_params = len(self.trainable_parameter_names)
            logger.info(
                f"Number of trainable parameters in {self.model_name}: {n_params}"
            )

        except Exception as e:
            logger.error(f"Failed to set trainable parameters: {str(e)}")
            raise

    def test_stability(self) -> None:
        """Test model stability with various inputs."""
        logger.info(f"Testing {self.model_name} stability")
        device = next(self.parameters()).device
        x = torch.randn(1, 3, 224, 224, device=device)

        # Test normal input
        try:
            self(x)
            logger.info("Normal input test passed")
        except Exception as e:
            logger.error(f"Normal input test failed: {str(e)}")
            raise

        # Test extreme values
        try:
            self(torch.full_like(x, float("inf")))
            assert False, "Should raise stability error"
        except ValueError:
            logger.info("Extreme value test passed")

        # Test NaN values
        try:
            self(torch.full_like(x, float("nan")))
            assert False, "Should raise stability error"
        except ValueError:
            logger.info("NaN value test passed")


# Create ResNet variants using the base class
class ResNet18(ResNetBase):
    """ResNet-18 model."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("model_name", None)
        super().__init__("ResNet18", **kwargs)


class ResNet34(ResNetBase):
    """ResNet-34 model."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("model_name", None)
        super().__init__("ResNet34", **kwargs)


class ResNet50(ResNetBase):
    """ResNet-50 model."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("model_name", None)
        super().__init__("ResNet50", **kwargs)


class ResNet101(ResNetBase):
    """ResNet-101 model."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("model_name", None)
        super().__init__("ResNet101", **kwargs)


class ResNet152(ResNetBase):
    """ResNet-152 model."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("model_name", None)
        super().__init__("ResNet152", **kwargs)


def test_resnet_model(
    model_class: Type[ResNetBase],
    input_shape: Tuple[int, ...] = (20, 1, 224, 224),
    device: Optional[torch.device] = None,
) -> None:
    """Test ResNet model implementation.

    Args:
        model_class: ResNet model class to test
        input_shape: Input tensor shape
        device: Device to run test on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Testing {model_class.__name__} on {device}")

    try:
        # Create and setup model
        model = model_class(
            input_dims=input_shape,
            device=device,
            mixed_precision=True,
            stability_check=True,
        )
        model.setup("fit")
        logger.info("Model creation successful")

        # Test forward pass
        x = torch.randn(1, *input_shape, device=device)
        y = model(x)
        logger.info(f"Forward pass successful: {x.shape} -> {y.shape}")

        # Test stability
        model.test_stability()
        logger.info("Stability tests passed")

        # Log trainable parameters
        trainable_params = [
            f"{name} [{tuple(param.shape)}]"
            for name, param in model.named_parameters()
            if param.requires_grad
        ]
        logger.info("Trainable Parameters:\n\t%s", "\n\t".join(trainable_params))

        logger.info(f"{model_class.__name__} tests passed!")

    except Exception as e:
        logger.error(f"{model_class.__name__} tests failed: {str(e)}")
        raise


if __name__ == "__main__":

    # Test all ResNet variants
    for model_class in [ResNet18, ResNet34, ResNet50, ResNet101, ResNet152]:
        test_resnet_model(model_class)
