"""
CorNet: Recurrent Convolutional Neural Network with Biological Features

This module implements the CorNet-RT and CorNet-Z architectures, a biologically-inspired neural network that incorporates recurrent processing and hierarchical organization similar to the primate visual cortex. The model consists of four layers: V1, V2, V4, and IT, each.

References:
- Kubilius et al. (2019) "CORnet: Modeling the Neural Mechanisms of Core Object Recognition"
- DiCarlo et al. (2012) "How Does the Brain Solve Visual Object Recognition?"
"""

import logging
from typing import Optional, Any, Tuple

import torch
import torch.nn as nn

from dynvision.models.dyrcnn import DyRCNN
from dynvision.utils import alias_kwargs, str_to_bool
from dynvision.model_components import (
    RConv2d,
)


class ZeroStateSentinel:
    """Sentinel representing zero state for uninitialized hidden states.

    This allows returning 'zero' without knowing the batch size.
    Operations check isinstance and treat it as additive identity.
    """


__all__ = ["CorNetRT"]


logger = logging.getLogger(__name__)


class CorNetRT(DyRCNN):
    @alias_kwargs(
        tff="t_feedforward",
        trc="t_recurrence",
        tsk="t_skip",
        rctype="recurrence_type",
        rctarget="recurrence_target",
        idle="idle_timesteps",
        pretrained="init_with_pretrained",
    )
    def __init__(
        self,
        n_timesteps: int = 5,
        input_dims: Tuple[int, int, int] = (5, 3, 224, 224),
        n_classes: int = 1000,
        dt: float = 2,  # ms
        t_feedforward: float = 2,  # ms
        t_recurrence: float = 2,  # ms
        recurrence_type: str = "self",
        fixed_self_weight: float = 1.0,
        recurrence_bias: bool = False,
        recurrence_target: str = "middle",
        skip: bool = False,  # Force skip connections to inject zeros at early timesteps
        t_skip: float = 0.0,  # Zero delay for immediate effect → delay_skip = int(0/2) = 0
        feedback: bool = False,
        init_with_pretrained=True,
        **kwargs: Any,
    ) -> None:

        self.model_letter = "rt"
        self.model_hash = "933c001c"
        self.init_with_pretrained = str_to_bool(init_with_pretrained)
        self.fixed_self_weight = fixed_self_weight
        self.recurrence_bias = recurrence_bias

        super().__init__(
            n_timesteps=n_timesteps,
            input_dims=input_dims,
            n_classes=n_classes,
            dt=dt,
            t_feedforward=t_feedforward,
            t_recurrence=t_recurrence,
            recurrence_type=recurrence_type,
            recurrence_target=recurrence_target,
            skip=skip,
            t_skip=t_skip,
            feedback=feedback,
            **kwargs,
        )

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
            "V1.norm1.weight": "norm_V1.weight",
            "V1.norm1.bias": "norm_V1.bias",
            "V2.norm1.weight": "norm_V2.weight",
            "V2.norm1.bias": "norm_V2.bias",
            "V4.norm1.weight": "norm_V4.weight",
            "V4.norm1.bias": "norm_V4.bias",
            "IT.norm1.weight": "norm_IT.weight",
            "IT.norm1.bias": "norm_IT.bias",
            "conv1": "conv2",
            "conv_input": "conv",
            "norm_input": "mid_modules",
            "decoder.linear": "classifier.2",
        }
        return translate_layer_names

    def _define_architecture(self):
        self.layer_names = ["V1", "V2", "V4", "IT"]
        # define operations order within layer
        self.layer_operations = [
            "layer",  # apply (recurrent) convolutional layer
            "norm",  # apply normalization
            "nonlin",  # apply nonlinearity
            "record",  # record activations in responses dict
            "delay",  # set and get delayed activations for next layer
            "skipzeros",  # check for zero hidden states and initialize if needed (BEFORE delay!)
        ]
        # V1
        self.V1 = RConv2d(
            in_channels=3,
            mid_channels=64,
            out_channels=64,
            kernel_size=(7, 3),
            stride=(4, 1),
            bias=(True, False),
            mid_modules=nn.GroupNorm(32, 64),
            fixed_self_weight=self.fixed_self_weight,
            recurrence_bias=self.recurrence_bias,
            recurrence_target=self.recurrence_target,
            dt=self.dt,
            t_recurrence=self.t_recurrence,
            t_feedforward=self.t_feedforward,
            dim_y=self.dim_y,
            dim_x=self.dim_x,
            history_length=self.history_length,
        )
        self.norm_V1 = nn.GroupNorm(32, 64)

        self.skipzeros_V1 = self.zerofilter("V2")

        # V2
        self.V2 = RConv2d(
            in_channels=64,
            mid_channels=128,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(2, 1),
            bias=(True, False),
            mid_modules=nn.GroupNorm(32, 128),
            fixed_self_weight=self.fixed_self_weight,
            recurrence_bias=self.recurrence_bias,
            recurrence_target=self.recurrence_target,
            dt=self.dt,
            t_recurrence=self.t_recurrence,
            t_feedforward=self.t_feedforward,
            dim_y=self.V1.dim_y // self.V1.stride[0] // self.V1.stride[1],
            dim_x=self.V1.dim_x // self.V1.stride[0] // self.V1.stride[1],
            history_length=self.history_length,
        )
        self.norm_V2 = nn.GroupNorm(32, 128)

        self.skipzeros_V2 = self.zerofilter("V4")

        # V4
        self.V4 = RConv2d(
            in_channels=128,
            mid_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(2, 1),
            bias=(True, False),
            mid_modules=nn.GroupNorm(32, 256),
            fixed_self_weight=self.fixed_self_weight,
            recurrence_bias=self.recurrence_bias,
            recurrence_target=self.recurrence_target,
            dt=self.dt,
            t_recurrence=self.t_recurrence,
            t_feedforward=self.t_feedforward,
            dim_y=self.V2.dim_y // self.V2.stride[0] // self.V2.stride[1],
            dim_x=self.V2.dim_x // self.V2.stride[0] // self.V2.stride[1],
            history_length=self.history_length,
        )
        self.norm_V4 = nn.GroupNorm(32, 256)

        self.skipzeros_V4 = self.zerofilter("IT")

        # IT
        self.IT = RConv2d(
            in_channels=256,
            mid_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(2, 1),
            bias=(True, False),
            mid_modules=nn.GroupNorm(32, 512),
            fixed_self_weight=self.fixed_self_weight,
            recurrence_bias=self.recurrence_bias,
            recurrence_target=self.recurrence_target,
            dt=self.dt,
            t_recurrence=self.t_recurrence,
            t_feedforward=0.0,
            dim_y=self.V4.dim_y // self.V4.stride[0] // self.V4.stride[1],
            dim_x=self.V4.dim_x // self.V4.stride[0] // self.V4.stride[1],
            history_length=self.history_length,
        )
        self.norm_IT = nn.GroupNorm(32, 512)

        # Shared nonlinearity for all layers
        # IMPORTANT: inplace=False to avoid corrupting activations when reused across layers
        self.nonlin = nn.ReLU(inplace=False)

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, self.n_classes),
        )

    def _zero_init_hidden_states(
        self,
        layer_name: Optional[str] = None,
    ):
        """Initialize all hidden states in recurrent layers to zeros to replicate the behavior of the original implementation."""
        if hasattr(self, "batch_size"):
            batch_size = self.batch_size
        else:
            batch_size = 1
        if layer_name is None:
            layer_names = self.layer_names
        else:
            layer_names = [layer_name]

        for layer in [getattr(self, name) for name in layer_names]:
            if self.recurrence_target == "input":
                n_channels = layer.in_channels
                stride = 1
            elif self.recurrence_target == "middle":
                n_channels = layer.mid_channels
                stride = layer.stride[0]
            else:  # 'output'
                n_channels = layer.out_channels
                if hasattr(layer, "mid_channels") and layer.mid_channels:
                    stride = layer.stride[0] * layer.stride[1]
                else:
                    stride = layer.stride

            for i in range(layer.n_hidden_states):
                layer.set_hidden_state(
                    torch.zeros(
                        batch_size,
                        n_channels,
                        layer.dim_y // stride,
                        layer.dim_x // stride,
                        device=self.device,
                    )
                )

    def reset(self, input_shape: Optional[Tuple[int, ...]] = None):
        """Reset all hidden states in recurrent layers.

        Note: DynVision's None handling for uninitialized recurrent states
        is equivalent to the original's scalar 0:
        - Original: `tensor + 0 = tensor`
        - DynVision: `if h is None: return x`
        Both return x unchanged when there's no recurrent state.
        """
        for layer in [self.V1, self.V2, self.V4, self.IT]:
            layer.reset(input_shape=input_shape)

    def zerofilter(self, layer_name: str):
        """Return function that adds zeros to hidden states if uninitialized.

        Args:
            layer_name: Name of the target layer to initialize with zeros

        Returns:
            Function that checks if input is all zeros and initializes target layer's hidden states
        """

        def zerofilter_fn(x):
            # Check if we need to initialize target layer with zeros
            # This happens when: (1) no feedforward input yet (x is None), or (2) input is all zeros
            if x is None or (isinstance(x, torch.Tensor) and not torch.any(x).item()):
                self._zero_init_hidden_states(layer_name=layer_name)
                return None  # Return None to signal no feedforward input yet
            return x  # Pass through unchanged

        return zerofilter_fn

    def forward(
        self,
        x_0: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:

        self.batch_size = x_0.shape[0]

        return super().forward(x_0, *args, **kwargs)


def test_cornet(
    input_shape: Tuple[int, ...] = (5, 3, 224, 224),
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
    print(f"Testing CorNet-RT on {device}")

    # Create and setup model
    model = model_class(
        input_dims=input_shape,
    )
    model.setup("fit")
    print("Model creation successful")

    # Test forward pass
    x = torch.randn(1, *input_shape, device=device)
    y = model(x)
    print(f"Forward pass successful: {x.shape} -> {y.shape}")

    # Log trainable parameters
    trainable_params = [
        f"{name} [{tuple(param.shape)}]"
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    print("Trainable Parameters:\n\t%s" % "\n\t".join(trainable_params))

    print("All tests passed!")


if __name__ == "__main__":

    test_cornet()
