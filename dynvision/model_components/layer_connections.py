"""
Modules to implement connections between non-successive layers in convolutional neural networks.
"""

from typing import Union, Callable, Optional, Dict, Any
import logging
from fractions import Fraction
from dynvision.utils import apply_parametrization
from dynvision.model_components.integration_strategy import setup_integration_strategy

import torch
import torch.nn as nn
from torch.amp import autocast
import torch.nn.init as init
from pytorch_lightning import LightningModule


logger = logging.getLogger(__name__)

__all__ = ["Skip", "Feedback"]


class ConnectionBase(LightningModule):
    """
    The connection module adds a hidden state (h) of a source module to the input tensor (x). There are two ways to use this module:

    a) Initialize without parameters `conn = Connection()`, and manually pass hidden state tensor to the forward call `conn(x, h)`.
    b) Initialize with a source module and delay index `conn = Connection(source=source, delay_index=0)` to internally retrieve the relevant hidden state when calling `conn(x)`. Note: the source module must have a `get_hidden_state` function that takes the delay_index as an argument.

    In either case, when h doesn't have the same shape as x, a 1x1 convolution is applied to h to match the shapes. If the scale_factor = shape h / shape x is a positive integer a corresponding stride is applied in the convolution, if scale_factor is < 1 a corresponding upsampling is applied using a given upsampling mode (default=bilinear).
    To define the convolution, you have to pass the `in_channels` (n_channels of h), `out_channels` (n_channels of x), and stride (factor between spatial dimension of h and x). If you don't pass them, the module will try to infer them from the source module.
    If the shape mismatch is not previously known or the architecture should remain more flexible, you can instead set `auto_adapt=True`, which will automatically infer the `in_channels`, `out_channels`, and `stride` upon the first forward pass that combines x and h. Note that this may cause issues with loading state_dicts or checkpoints.
    """

    def __init__(
        self,
        source: Optional[nn.Module] = None,
        delay_index: int = 0,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        scale_factor: float = 1,
        bias: bool = True,
        integration_strategy: Union[Callable, str] = "additive",
        parametrization: Callable[[torch.Tensor], torch.Tensor] = None,
        upsample_mode: str = "nearest",
        auto_adapt: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.source = source
        self.delay_index = delay_index
        self.bias = bias
        self.scale_factor = scale_factor
        self.integration_strategy = integration_strategy
        self.parametrization = parametrization
        self.setup_transform = True
        self.auto_adapt = auto_adapt

        if not auto_adapt:
            # infer in_channels, out_channels from source module
            if in_channels is None:
                in_channels = source.out_channels
            if out_channels is None:
                out_channels = in_channels

            if scale_factor is not None and scale_factor >= 1:
                if not isinstance(scale_factor, int) and not scale_factor.is_integer():
                    raise ValueError(
                        "scale_factor must be an integer when greater than 1."
                    )
                scale_factor = int(scale_factor)
                x_proxy = torch.empty(out_channels, 1, 1)
                h_proxy = torch.empty(in_channels, scale_factor, scale_factor)
            else:
                s = Fraction(scale_factor).limit_denominator()
                x_proxy = torch.empty(out_channels, s.denominator, s.denominator)
                h_proxy = torch.empty(in_channels, s.numerator, s.numerator)

            self._setup_conv(x=x_proxy, h=h_proxy)
            self._setup_upsample(x=x_proxy, h=h_proxy, mode=upsample_mode)

    def setup(self, stage: Optional[str] = None) -> None:
        """Handle precision setup from PyTorch Lightning."""
        super().setup(stage)
        if stage == "fit" and self.auto_adapt:
            # Reset transform to ensure proper initialization with correct dtype
            self.setup_transform = True

    def _setup_conv(self, x: torch.Tensor, h: torch.Tensor) -> Union[nn.Module, bool]:
        if x is None or h is None:
            self.conv = False
            return False

        in_channels = h.shape[-3]
        out_channels = x.shape[-3]

        if in_channels == out_channels:
            self.conv = False
            self.setup_transform = False
            return False

        # Get device and dtype from source or input
        device = x.device
        dtype = x.dtype
        if self.source is not None:
            source_param = next(self.source.parameters())
            device = source_param.device
            dtype = source_param.dtype
            logger.debug(
                f"Feedback Connection: Using source module device={device}, dtype={dtype}"
            )
        else:
            logger.debug(
                f"Feedback Connection: Using input tensor device={device}, dtype={dtype}"
            )

        # Create convolution with explicit device and dtype
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.bias,
            dtype=dtype,
            device=device,
        )
        if self.parametrization is not None:
            self.conv = apply_parametrization(self.conv, self.parametrization)

        self._init_parameters(in_channels, out_channels)
        self.integrate_signal = setup_integration_strategy(self.integration_strategy)
        self.setup_transform = False

    def _setup_upsample(
        self, x: torch.Tensor, h: torch.Tensor, mode: str = "nearest"
    ) -> Union[nn.Module, bool]:
        if x is None or h is None:
            self.upsample = False
            return False

        scale_factor = x.shape[-1] / h.shape[-1]
        if scale_factor == 1:
            self.upsample = False
            return False

        self.upsample = nn.Upsample(size=x.shape[-2:], mode=mode)

    def reset_transform(self) -> None:
        self.setup_transform = True

    def _init_parameters(self, in_channels: int = 1, out_channels: int = 1) -> None:
        if hasattr(self, "conv") and isinstance(self.conv, nn.Conv2d):
            # Scale-aware initialization
            std = min(0.001, 1.0 / (in_channels * out_channels) ** 0.5)
            init.trunc_normal_(self.conv.weight, mean=0.0, std=std)
            if self.conv.bias is not None:
                init.zeros_(self.conv.bias)

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x is None:
            return None

        # Get hidden state if not provided
        if h is None:
            try:
                h = self.source.get_hidden_state(self.delay_index)
                logger.debug(
                    f"Retrieved hidden state h: {'None' if h is None else h.shape}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to get hidden state: {str(e)} \n Source must have get_hidden_state function!"
                )
                return x

        if h is None:
            return x

        if self.setup_transform:
            self._setup_conv(x, h)
            self._setup_upsample(x, h)

        if self.conv:
            h = self.conv(h)

        if self.upsample:
            h = self.upsample(h)

        # Normalize feedback with gradient preservation
        # h_norm = torch.norm(h, p=2, dim=(1, 2, 3), keepdim=True)
        # scale = torch.where(h_norm > 1e-6, h_norm, torch.ones_like(h_norm))
        # h = h / scale
        output = self.integrate_signal(x, h)
        return output


class Skip(ConnectionBase):
    """
    The skip connection module adds the input tensor (x) of an earlier layer to the output tensor (h) of a deeper layer.
    """

    pass


class Feedback(ConnectionBase):
    """
    The feedback connection module adds a hidden state (h) of a deeper layer to the input tensor (x) of an earlier layer.
    """

    pass


if __name__ == "__main__":
    input_shape = (3, 32, 32)
    batch_size = 1
    print("Input shape:", (batch_size, *input_shape))

    for class_name in __all__:
        print(f"Testing {class_name}:")

        # Create an instance of the class
        module = globals()[class_name]

        # Create a dummy source module with a get_hidden_state method
        class DummySource(nn.Module):
            def __init__(self, out_channels, scale=64):
                super(DummySource, self).__init__()
                self.out_channels = out_channels
                self.scale = scale

            def get_hidden_state(self, delay_index):
                return torch.randn(
                    batch_size, self.out_channels, self.scale, self.scale
                )

        if class_name == "Skip":
            source = DummySource(out_channels=10, scale=64)
        elif class_name == "Feedback":
            source = DummySource(out_channels=1, scale=24)
        else:
            source = None

        model = module(source=source, delay_index=0, auto_adapt=True)

        # Generate random input
        random_input = torch.randn(batch_size, *input_shape)

        # Perform forward pass
        output = model(random_input)

        # Print number of parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {n_params}")

        # Define an arbitrary loss function
        loss_fn = nn.MSELoss()

        # Generate random target
        target = torch.randn_like(output)

        # Compute loss
        loss = loss_fn(output, target)

        # Perform backward pass
        loss.backward()

        # Print results
        print(f"Model Output: {output.shape if output is not None else 'None'}\n")

    print("All tests passed!")
