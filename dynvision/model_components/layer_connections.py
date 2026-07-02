"""
Modules to implement connections between non-successive layers in convolutional neural networks.
"""

from typing import Union, Callable, Optional, Any, Tuple
import logging
from fractions import Fraction
from dynvision.utils import apply_parametrization
from dynvision.model_components.integration_strategy import setup_integration_strategy

import torch
import torch.nn as nn
import torch.nn.init as init
from pytorch_lightning import LightningModule

logger = logging.getLogger(__name__)

__all__ = ["Skip", "Feedback"]


class ConnectionBase(LightningModule):
    """
    The connection module adds a hidden state (h) of a source module to the input tensor (x). There are two ways to use this module:

    a) Initialize without parameters `conn = Connection()`, and manually pass hidden state tensor to the forward call `conn(x, h)`.
    b) Initialize with a source module and delay index `conn = Connection(source=source, delay_index=0)` to internally retrieve the relevant hidden state when calling `conn(x)`. Note: the source module must have a `get_hidden_state` function that takes the delay_index as an argument.

    Shape Adaptation:
    When h doesn't have the same shape as x, two transformations are applied to h:
    1. Channel adaptation: A 1x1 convolution adapts h from `in_channels` to `out_channels`
    2. Spatial adaptation: nn.Upsample resizes h's spatial dimensions to match x's spatial dimensions

    The `scale_factor` parameter represents the scaling needed to transform h to match x (x_size / h_size):
    - scale_factor > 1: x is larger than h, so h will be upsampled (e.g., scale_factor=2 means upsample h by 2×)
    - scale_factor < 1: x is smaller than h, so h will be downsampled (e.g., scale_factor=0.5 means downsample h by 2×)
    - scale_factor = 1: x and h have same spatial dimensions, no scaling needed

    Usage:
    - Explicit setup: Pass `in_channels`, `out_channels`, and `scale_factor` to define transformations upfront
    - Auto-adapt: Set `auto_adapt=True` to infer shapes on first forward pass (may cause checkpoint issues)
    """

    def __init__(
        self,
        source: Optional[nn.Module] = None,
        delay_index: Optional[int] = None,
        t_connection: Optional[int] = None,
        dt: Optional[float] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        kernel_size: int = 1,
        stride: int = 1,
        scale_factor: float = 1,
        bias: bool = True,
        integration_strategy: Union[Callable, str] = "additive",
        parametrization: Callable[[torch.Tensor], torch.Tensor] = None,
        upsample_mode: str = "nearest",
        auto_adapt: bool = False,
        force_conv: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.source = source
        self.delay_index = delay_index
        self.t_connection = t_connection
        self.dt = dt
        self.bias = bias
        self.kernel_size = kernel_size
        self.stride = stride
        self.scale_factor = scale_factor
        self.integration_strategy = integration_strategy
        self.parametrization = parametrization
        self.setup_transform = True
        self.auto_adapt = auto_adapt
        self.force_conv = force_conv
        self.scale_factor = scale_factor
        self.upsample_mode = upsample_mode

        self._set_delay_index()

        if not auto_adapt:
            # infer in_channels, out_channels from source module
            if in_channels is None:
                in_channels = source.out_channels
            if out_channels is None:
                out_channels = in_channels

            if scale_factor is not None and scale_factor >= 1:
                # scale_factor >= 1: x is larger, upsample h by scale_factor
                if not isinstance(scale_factor, int) and not scale_factor.is_integer():
                    raise ValueError(
                        "scale_factor must be an integer when greater than 1."
                    )
                scale_factor = int(scale_factor)
                x_proxy = torch.empty(out_channels, scale_factor, scale_factor)
                h_proxy = torch.empty(in_channels, 1, 1)
            else:
                # scale_factor < 1: x is smaller, downsample h
                # e.g., scale_factor=0.5 means x is half the size, so h needs 1/0.5 = 2× downsampling
                s = Fraction(scale_factor).limit_denominator()
                x_proxy = torch.empty(out_channels, s.numerator, s.numerator)
                h_proxy = torch.empty(in_channels, s.denominator, s.denominator)

            self._setup_conv(x=x_proxy, h=h_proxy)
            self._setup_upsample(scale_factor=scale_factor, mode=upsample_mode)

    def _set_delay_index(self) -> None:
        if self.delay_index is not None:
            return
        elif self.t_connection is not None and self.dt is not None:
            # for skip connections, +1 accounts that previous layers already had their hidden state updated this timestep
            self.delay_index = int(self.t_connection / self.dt) + 1
        else:
            raise ValueError(
                "Either delay_index or both t_connection and dt must be provided."
            )

    def setup(self, stage: Optional[str] = None) -> None:
        """Handle precision setup from PyTorch Lightning."""
        super().setup(stage)
        if stage == "fit" and self.auto_adapt:
            # Reset transform to ensure proper initialization with correct dtype
            self.setup_transform = True

    def _setup_conv(self, x: torch.Tensor, h: torch.Tensor) -> None:
        # Always setup integration signal first
        self.integrate_signal = setup_integration_strategy(self.integration_strategy)

        if x is None or h is None:
            self.conv = False
            return

        in_channels = h.shape[-3]
        out_channels = x.shape[-3]

        # Create conv if channels differ OR if force_conv is True
        if in_channels == out_channels and not self.force_conv:
            self.conv = False
            self.setup_transform = False
            return

        # Get device and dtype from source or input
        device = x.device
        dtype = x.dtype
        if self.source is not None:
            source_param = next(self.source.parameters())
            device = source_param.device
            dtype = source_param.dtype
            logger.debug(
                f"Connection: Creating Conv2d with device={device}, dtype={dtype}, "
                f"in_channels={in_channels}, out_channels={out_channels}"
            )

        # Create convolution
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.kernel_size // 2,
            bias=self.bias,
        ).to(device=device, dtype=dtype)
        self.add_module("conv", conv)

        if self.parametrization is not None:
            self.conv = apply_parametrization(self.conv, self.parametrization)

        self._init_parameters(in_channels, out_channels)
        self.setup_transform = False

    def _setup_upsample(
        self,
        size: Tuple[int, int] = None,
        scale_factor: float = None,
        mode: str = "nearest",
    ) -> None:

        if size is None and scale_factor is None:
            self.upsample = False
        elif size is not None:
            self.upsample = nn.Upsample(size=size, mode=mode)
        elif scale_factor == 1:
            self.upsample = False
        else:
            # adjust scaling factor if stride is > 1
            scale_factor = scale_factor / self.stride
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

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
        # Get hidden state if not provided
        if h is None:
            try:
                h = self.source.get_hidden_state(self.delay_index)
                logger.debug(
                    f"Retrieved hidden state h: {'None' if h is None else h.shape}"
                )
            except Exception as e:
                logger.error(f"Failed to get state from source: {str(e)}")
                return x

        # Handle cases where x or h is None before transformation
        if x is None and h is None:
            return None
        elif h is None:
            return x

        # Setup transform if needed (but avoid during training)
        if self.setup_transform:
            if self.training:
                logger.debug(
                    "Setting up connection transform during training - this may break gradients"
                )
            self._setup_conv(x, h)
            self._setup_upsample(size=x.shape[-2:], mode=self.upsample_mode)

        # Apply transformations to h to match x's expected shape
        if self.conv:
            h = self.conv(h)

        if self.upsample:
            h = self.upsample(h)

        # Now handle the case where x is None (early timesteps)
        if x is None:
            return h  # h is now properly transformed

        # Both x and h exist, integrate them
        logger.debug(
            f"Layer Connection forward: x shape: {x.shape}, h shape: {h.shape}"
        )
        output = self.integrate_signal(x, h)
        return output


class Skip(ConnectionBase):
    """
    The skip connection module adds the input tensor (x) of an earlier layer to the output tensor (h) of a deeper layer.
    """


class Feedback(ConnectionBase):
    """
    The feedback connection module adds a hidden state (h) of a deeper layer to the input tensor (x) of an earlier layer.
    """


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
