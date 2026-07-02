"""
Model of the retina and LGN as two convolutional layers with biological inspiration.
"""

import logging
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from pytorch_lightning import LightningModule


logger = logging.getLogger(__name__)

__all__ = ["Retina"]


def validate_input(x: torch.Tensor, expected_channels: int) -> None:
    """Validate input tensor dimensions and values."""
    if x.dim() != 4:
        raise ValueError(f"Expected 4D input tensor, got {x.dim()}D")
    if x.size(1) != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} input channels, got {x.size(1)}"
        )
    if torch.isnan(x).any():
        raise ValueError("Input contains NaN values")
    if torch.isinf(x).any():
        raise ValueError("Input contains Inf values")


class Retina(LightningModule):
    """
    Model of the retina and LGN as two convolutional layers with ReLU activation.

    The model implements a biologically-inspired visual processing pipeline:
    1. First conv layer models retinal ganglion cells
    2. ReLU activation models neural firing thresholds
    3. Second conv layer models LGN processing

    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        mid_channels: Number of channels in first layer (default: 36)
        out_channels: Number of output channels (default: 18)
        kernel_size: Convolution kernel size (default: 9)
        bias: Whether to use bias in convolutions (default: True)
        mixed_precision: Whether to use automatic mixed precision (default: True)

    Reference:
        Lindsey et al. (2019) doi:10.48550/arXiv.1901.00945
    """

    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 36,
        out_channels: int = 18,
        kernel_size: int = 9,
        bias: bool = True,
        mixed_precision: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.mixed_precision = mixed_precision
        self.device = device

        self._define_architecture()

        # Move to specified device
        if device is not None:
            self.to(device)

    def _define_architecture(self) -> None:
        """Define the neural network architecture."""
        # First layer - retinal ganglion cells
        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.mid_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            bias=self.bias,
            device=self.device,
        )

        # Second layer - LGN processing
        self.conv2 = nn.Conv2d(
            in_channels=self.mid_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            bias=self.bias,
            device=self.device,
        )

        # Neural firing threshold
        self.nonlin = nn.ReLU(inplace=True)

    def _init_parameters(self) -> None:
        """Initialize network parameters using Kaiming initialization."""
        for layer in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the retina model.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Processed tensor of shape (batch_size, out_channels, height, width)
        """
        # First layer - retinal ganglion cells
        x = self.conv1(x)

        # Neural firing threshold
        x = self.nonlin(x)

        # Second layer - LGN processing
        x = self.conv2(x)

        return x


if __name__ == "__main__":
    input_shape = (3, 32, 32)
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Testing on device: {device}")
    logger.info(f"Input shape: {(batch_size, *input_shape)}")

    # Test model
    model = Retina(device=device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info("\nModel Summary:")
    print(summary(model, input_shape, device=device))

    # Test forward pass
    x = torch.randn(batch_size, *input_shape, device=device, requires_grad=True)
    y = model(x)
    logger.info(f"\nOutput shape: {y.shape}")

    # Test backward pass
    loss = F.mse_loss(y, torch.randn_like(y))
    loss.backward()
    logger.info("Backward pass successful")

    # Test stability checks
    try:
        model(torch.full((1, *input_shape), float("inf"), device=device))
        assert False, "Should raise stability error"
    except ValueError:
        logger.info("Stability check passed")

    logger.info("All tests passed!")
