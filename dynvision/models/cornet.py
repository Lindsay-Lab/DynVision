"""
CorNet: Recurrent Convolutional Neural Network with Biological Features

This module implements the CorNet-RT and CorNet-Z architectures, a biologically-inspired neural network that incorporates recurrent processing and hierarchical organization similar to the primate visual cortex. The model consists of four layers: V1, V2, V4, and IT, each.

References:
- Kubilius et al. (2019) "CORnet: Modeling the Neural Mechanisms of Core Object Recognition"
- DiCarlo et al. (2012) "How Does the Brain Solve Visual Object Recognition?"
"""

import logging
from typing import Optional, Dict, Any, Tuple, Union

import torch
import torch.nn as nn

from dynvision.base import BaseModel
from dynvision.utils import check_stability

__all__ = ["CorNetZ", "CorNetRT"]


logger = logging.getLogger(__name__)


class CorBlockRT(nn.Module):
    """
    Recurrent processing block inspired by cortical area organization.

    This block implements recurrent processing with feedforward pathways. The block maintains a hidden state that evolves over time.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        mixed_precision: Whether to use mixed precision
        stability_check: Whether to check numerical stability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        stability_check: bool = False,
    ) -> None:
        super().__init__()
        self.stability_check = stability_check
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self._define_architecture()
        self.reset()

    def _define_architecture(self) -> None:
        """Define block architecture with feedforward and recurrent pathways."""
        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.kernel_size // 2,
        )
        self.norm1 = nn.GroupNorm(32, self.out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(32, self.out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def reset(self) -> None:
        """Reset hidden state."""
        self.hidden_state = None

    def get_hidden_state(self, i: Optional[int] = None) -> Optional[torch.Tensor]:
        """Get hidden state. Index is ignored since there is only one memory slot."""
        return self.hidden_state

    def set_hidden_state(self, hidden_state: Optional[torch.Tensor]) -> None:
        """Set hidden state."""
        self.hidden_state = hidden_state

    def forward(self, x_0: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        # Feedforward path
        if x_0 is None:  # at t=0, no input except V1
            x_1 = None
        else:
            x_1 = self.conv1(x_0)
            x_1 = self.norm1(x_1)
            x_1 = self.relu1(x_1)

            if self.stability_check:
                check_stability(x_1, "feedforward")

        # Combine with recurrent state
        if x_1 is None and self.hidden_state is None:
            return None
        elif x_1 is None:
            x = self.hidden_state
        elif self.hidden_state is None:  # at t=0, initialize state
            x = x_1
        else:
            x = x_1 + self.hidden_state

        # Recurrent processing
        x_2 = self.conv2(x)
        x_2 = self.norm2(x_2)
        x_2 = self.relu2(x_2)

        if self.stability_check:
            check_stability(x_2, "recurrent")

        return x_2


class CorNetRT(BaseModel):
    def __init__(
        self,
        input_dims: tuple = (20, 3, 224, 224),
        n_classes: int = 1000,
        stability_check: bool = False,
        init_with_pretrained: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            input_dims=input_dims,
            n_classes=n_classes,
            stability_check=stability_check,
            init_with_pretrained=init_with_pretrained,
            **kwargs,
        )

        self.model_letter = "rt"
        self.model_hash = "933c001c"
        self._define_architecture()
        self._init_parameters()

    def _init_parameters(self) -> None:
        if self.init_with_pretrained:
            self.load_pretrained_state_dict(check_mismatch_layer=["classifier.2"])
            self.trainable_parameter_names = [
                p for p in list(self.state_dict().keys()) if "classifier.2" in p
            ]
        else:
            self.trainable_parameter_names = [
                p for p in list(self.state_dict().keys())
            ]

    def download_pretrained_state_dict(self):
        url = f"https://s3.amazonaws.com/cornet-models/cornet_{self.model_letter.lower()}-{self.model_hash}.pth"
        ckpt_data = torch.utils.model_zoo.load_url(url, map_location=self.device)
        state_dict = ckpt_data["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        return state_dict

    def translate_pretrained_layer_names(self):
        translate_layer_names = {
            "conv1": "conv2",
            "norm1": "norm2",
            "conv_input": "conv1",
            "norm_input": "norm1",
            "decoder.linear": "classifier.2",
        }
        return translate_layer_names

    def _define_architecture(self):
        self.layer_names = ["V1", "V2", "V4", "IT"]
        # define operations order within layer
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
        # V1
        self.V1 = CorBlockRT(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=4,
        )
        # V2
        self.V2 = CorBlockRT(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,
        )
        # V4
        self.V4 = CorBlockRT(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=2,
        )
        # IT
        self.IT = CorBlockRT(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=2,
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, self.n_classes),
        )

    def reset(self):
        for layer in [self.V1, self.V2, self.V4, self.IT]:
            layer.reset()

    def _forward(
        self, x_0: torch.Tensor, t: torch.Tensor = None, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_0 = self._expand_input_channels(x_0)

        # V1 processing
        h_1 = self.V1.get_hidden_state()  # previous output
        x_1 = self.V1(x_0)
        self.V1.set_hidden_state(x_1)

        if self.stability_check and x_1 is not None:
            check_stability(x_1, "V1")

        # V2 processing
        h_2 = self.V2.get_hidden_state()
        x_2 = self.V2(h_1)
        self.V2.set_hidden_state(x_2)

        if self.stability_check and x_2 is not None:
            check_stability(x_2, "V2")

        # V4 processing
        h_3 = self.V4.get_hidden_state()
        x_3 = self.V4(h_2)
        self.V4.set_hidden_state(x_3)

        if self.stability_check and x_3 is not None:
            check_stability(x_3, "V4")

        # IT processing
        h_4 = self.IT.get_hidden_state()
        x_4 = self.IT(h_3)
        self.IT.set_hidden_state(x_4)

        if self.stability_check and x_4 is not None:
            check_stability(x_4, "IT")

        # Classification
        if x_4 is None:
            x = None
        else:
            x = self.classifier(x_4)

        if self.stability_check:
            if torch.isnan(x).any():
                logger.error("NaN values detected in classifier output")
                raise ValueError("Numerical instability in classifier")

        responses = {
            "V1": x_1,
            "V2": x_2,
            "V4": x_3,
            "IT": x_4,
            "classifier": x,
        }
        return x, responses


class CorBlockZ(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x_0=None):
        if x_0 is None:
            out = None
        else:
            x_1 = self.conv(x_0)
            x_2 = self.nonlin(x_1)
            x_3 = self.pool(x_2)
            out = x_3

        return out


class CorNetZ(BaseModel):
    def __init__(
        self,
        input_dims: tuple = (10, 3, 224, 224),
        n_classes: int = 1000,
        **kwargs,
    ) -> None:
        kwargs = {k: v for k, v in locals().items() if k != "self"}
        super(CorNetZ, self).__init__(**kwargs)

        self.model_letter = "z"
        self.model_hash = "5c427c9c"
        self._define_architecture()
        self._init_parameters()

    def _init_parameters(self):
        self.load_pretrained_state_dict(check_mismatch_layer=["classifier.2"])
        self.trainable_parameter_names = [
            p for p in list(self.state_dict().keys()) if "classifier.2" in p
        ]

    def translate_pretrained_layer_names(self):
        translate_layer_names = {
            "decoder.linear": "classifier.2",
        }
        return translate_layer_names

    def download_pretrained_state_dict(self):
        url = f"https://s3.amazonaws.com/cornet-models/cornet_{self.model_letter.lower()}-{self.model_hash}.pth"
        ckpt_data = torch.utils.model_zoo.load_url(url, map_location=self.device)
        state_dict = ckpt_data["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        return state_dict

    def _define_architecture(self):
        # V1
        self.V1 = CorBlockZ(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
        )
        # V2
        self.V2 = CorBlockZ(
            in_channels=64,
            out_channels=128,
        )
        # V4
        self.V4 = CorBlockZ(
            in_channels=128,
            out_channels=256,
        )
        # IT
        self.IT = CorBlockZ(
            in_channels=256,
            out_channels=512,
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, self.n_classes),
        )

    def reset(self):
        pass

    def _forward(
        self, x_0: torch.Tensor, t: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        x_0 = self._expand_input_channels(x_0)

        # V1
        x_1 = self.V1(x_0)

        # V2
        x_2 = self.V2(x_1)

        # V4
        x_3 = self.V4(x_2)

        # IT
        x_4 = self.IT(x_3)

        # Classifier
        if x_4 is None:
            x = None
        else:
            x = self.classifier(x_4)

        responses = {
            "V1": x_1,
            "V2": x_2,
            "V4": x_3,
            "IT": x_4,
            "classifier": x,
        }
        return x, responses


def test_cornet(
    input_shape: Tuple[int, ...] = (20, 1, 224, 224),
    device: Optional[torch.device] = None,
    model_class=CorNetRT,
) -> None:
    """Test CorNet implementation.

    Tests include:
    - Model creation and setup
    - Forward pass with random input
    - Stability checks with extreme values
    - Parameter verification

    Args:
        input_shape: Input tensor shape
        device: Device to run test on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Testing CorNet-RT on {device}")

    try:
        # Create and setup model
        model = model_class(
            input_dims=input_shape,
            device=device,
            stability_check=True,
        )
        model.setup("fit")
        logger.info("Model creation successful")

        # Test forward pass
        x = torch.randn(1, *input_shape, device=device)
        y = model(x)
        logger.info(f"Forward pass successful: {x.shape} -> {y.shape}")

        # Test stability
        try:
            model(torch.full_like(x, float("inf")))
            assert False, "Should raise stability error"
        except ValueError:
            logger.info("Stability check passed")

        # Log trainable parameters
        trainable_params = [
            f"{name} [{tuple(param.shape)}]"
            for name, param in model.named_parameters()
            if param.requires_grad
        ]
        logger.info("Trainable Parameters:\n\t%s", "\n\t".join(trainable_params))

        logger.info("All tests passed!")

    except Exception as e:
        logger.error(f"Tests failed: {str(e)}")
        raise


if __name__ == "__main__":

    test_cornet(CorNetRT)
    test_cornet(CorNetZ)
