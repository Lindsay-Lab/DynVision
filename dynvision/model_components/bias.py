"""
Bias modules for neural networks with different spatial resolutions.

This module provides standardized bias implementations at different granularities:
- SpatialBias: Per-unit bias (channels, height, width)
- FeatureBias: Per-channel bias (channels, 1, 1)
- ScalarBias: Single bias value per layer (1,)

These are useful for models like CordsNet that use spatially-varying biases.

All bias modules support None input handling: when input is None and allow_null_input=True,
the module returns just the bias tensor. This is useful in temporal models where certain
operations (like feedback) may need to generate output without external input.
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "SpatialBias",
    "FeatureBias",
    "ScalarBias",
]


class SpatialBias(nn.Module):
    """
    Spatial bias with independent values for each spatial location.

    This creates a bias tensor of shape (channels, height, width) where
    each spatial location has its own bias value. This is the most flexible
    but also most parameter-intensive bias type.

    Used in models like CordsNet where each unit has its own bias.

    Args:
        channels: Number of feature channels
        height: Spatial height dimension
        width: Spatial width dimension
        init_value: Initial bias value (default: 0.0)
        requires_grad: Whether bias is trainable (default: True)
        allow_null_input: Whether to allow None input (returns bias only) (default: True)
        dtype: Data type for bias tensor (default: torch.float32)

    Example:
        >>> bias = SpatialBias(channels=64, height=56, width=56)
        >>> x = torch.randn(2, 64, 56, 56)
        >>> out = bias(x)
        >>> out.shape
        torch.Size([2, 64, 56, 56])
        >>> # With None input
        >>> out = bias(None)
        >>> out.shape
        torch.Size([64, 56, 56])
    """

    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        init_value: float = 0.0,
        requires_grad: bool = True,
        allow_null_input: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.channels = channels
        self.height = height
        self.width = width
        self.allow_null_input = allow_null_input

        # Create bias parameter
        bias_tensor = torch.full(
            (channels, height, width),
            init_value,
            dtype=dtype,
        )
        self.bias = nn.Parameter(bias_tensor, requires_grad=requires_grad)

    def forward(
        self, x: Optional[torch.Tensor] = None, batch_size: int = 1
    ) -> torch.Tensor:
        """
        Add spatial bias to input tensor.

        Args:
            x: Input tensor of shape (batch, channels, height, width), or None
            batch_size: Batch size to use when x is None (default: 1)

        Returns:
            Output tensor with bias added. If x is None and allow_null_input=True,
            returns the bias tensor with batch dimension added.

        Raises:
            ValueError: If x is None and allow_null_input=False, or if input shape
                       doesn't match bias shape.
        """
        # Handle None input
        if x is None:
            if not self.allow_null_input:
                raise ValueError(
                    "Input is None but allow_null_input=False. "
                    "Set allow_null_input=True to enable bias-only output."
                )
            # Add batch dimension: (C, H, W) -> (batch_size, C, H, W)
            return self.bias.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Verify input shape matches bias shape
        if x.shape[1:] != self.bias.shape:
            raise ValueError(
                f"Input shape {x.shape[1:]} does not match bias shape {self.bias.shape}"
            )

        # Add bias (broadcast over batch dimension)
        return x + self.bias

    def __repr__(self):
        return (
            f"SpatialBias(channels={self.channels}, "
            f"height={self.height}, width={self.width}, "
            f"requires_grad={self.bias.requires_grad}, "
            f"allow_null_input={self.allow_null_input})"
        )


class FeatureBias(nn.Module):
    """
    Feature bias with one value per channel.

    This creates a bias tensor of shape (channels, 1, 1) that is broadcast
    across spatial dimensions. This is the standard bias type in CNNs.

    Args:
        channels: Number of feature channels
        init_value: Initial bias value (default: 0.0)
        requires_grad: Whether bias is trainable (default: True)
        allow_null_input: Whether to allow None input (returns bias only) (default: True)
        dtype: Data type for bias tensor (default: torch.float32)

    Example:
        >>> bias = FeatureBias(channels=64)
        >>> x = torch.randn(2, 64, 56, 56)
        >>> out = bias(x)
        >>> out.shape
        torch.Size([2, 64, 56, 56])
        >>> # With None input
        >>> out = bias(None)
        >>> out.shape
        torch.Size([64, 1, 1])
    """

    def __init__(
        self,
        channels: int,
        init_value: float = 0.0,
        requires_grad: bool = True,
        allow_null_input: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.channels = channels
        self.allow_null_input = allow_null_input

        # Create bias parameter with shape (channels, 1, 1) for broadcasting
        bias_tensor = torch.full(
            (channels, 1, 1),
            init_value,
            dtype=dtype,
        )
        self.bias = nn.Parameter(bias_tensor, requires_grad=requires_grad)

    def forward(
        self, x: Optional[torch.Tensor] = None, batch_size: int = 1
    ) -> torch.Tensor:
        """
        Add feature bias to input tensor.

        Args:
            x: Input tensor of shape (batch, channels, height, width), or None
            batch_size: Batch size to use when x is None (default: 1)

        Returns:
            Output tensor with bias added (broadcast across spatial dims).
            If x is None and allow_null_input=True, returns the bias tensor with batch dimension.

        Raises:
            ValueError: If x is None and allow_null_input=False, or if input channel
                       count doesn't match bias channels.
        """
        # Handle None input
        if x is None:
            if not self.allow_null_input:
                raise ValueError(
                    "Input is None but allow_null_input=False. "
                    "Set allow_null_input=True to enable bias-only output."
                )
            # Add batch dimension: (C, 1, 1) -> (batch_size, C, 1, 1)
            return self.bias.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Verify input channels match bias channels
        if x.shape[1] != self.channels:
            raise ValueError(
                f"Input has {x.shape[1]} channels but bias expects {self.channels}"
            )

        return x + self.bias

    def __repr__(self):
        return (
            f"FeatureBias(channels={self.channels}, "
            f"requires_grad={self.bias.requires_grad}, "
            f"allow_null_input={self.allow_null_input})"
        )


class ScalarBias(nn.Module):
    """
    Scalar bias with a single value for the entire layer.

    This creates a bias scalar that is broadcast across all dimensions.
    Rarely used but included for completeness.

    Args:
        init_value: Initial bias value (default: 0.0)
        requires_grad: Whether bias is trainable (default: True)
        allow_null_input: Whether to allow None input (returns bias only) (default: True)
        dtype: Data type for bias tensor (default: torch.float32)

    Example:
        >>> bias = ScalarBias()
        >>> x = torch.randn(2, 64, 56, 56)
        >>> out = bias(x)
        >>> out.shape
        torch.Size([2, 64, 56, 56])
        >>> # With None input
        >>> out = bias(None)
        >>> out.shape
        torch.Size([])
    """

    def __init__(
        self,
        init_value: float = 0.0,
        requires_grad: bool = True,
        allow_null_input: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.allow_null_input = allow_null_input

        # Create scalar bias parameter
        bias_tensor = torch.tensor(init_value, dtype=dtype)
        self.bias = nn.Parameter(bias_tensor, requires_grad=requires_grad)

    def forward(
        self, x: Optional[torch.Tensor] = None, batch_size: int = 1
    ) -> torch.Tensor:
        """
        Add scalar bias to input tensor.

        Args:
            x: Input tensor of any shape, or None
            batch_size: Batch size to use when x is None (default: 1)

        Returns:
            Output tensor with bias added (broadcast across all dims).
            If x is None and allow_null_input=True, returns the bias tensor with batch dimension added.

        Raises:
            ValueError: If x is None and allow_null_input=False.
        """
        # Handle None input
        if x is None:
            if not self.allow_null_input:
                raise ValueError(
                    "Input is None but allow_null_input=False. "
                    "Set allow_null_input=True to enable bias-only output."
                )
            # Add batch dimension: () -> (batch_size,)
            return self.bias.unsqueeze(0).expand(batch_size)

        return x + self.bias

    def __repr__(self):
        return (
            f"ScalarBias(requires_grad={self.bias.requires_grad}, "
            f"allow_null_input={self.allow_null_input})"
        )


def create_bias(
    bias_type: str,
    channels: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    init_value: float = 0.0,
    requires_grad: bool = True,
    allow_null_input: bool = True,
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """
    Factory function to create bias modules.

    Args:
        bias_type: Type of bias ("spatial", "feature", "scalar", or "none")
        channels: Number of feature channels (required for spatial/feature)
        height: Spatial height (required for spatial)
        width: Spatial width (required for spatial)
        init_value: Initial bias value
        requires_grad: Whether bias is trainable
        allow_null_input: Whether to allow None input (returns bias only)
        dtype: Data type for bias tensor (default: torch.float32)

    Returns:
        Bias module of requested type, or Identity if bias_type="none"

    Example:
        >>> bias = create_bias("spatial", channels=64, height=56, width=56)
        >>> bias = create_bias("feature", channels=64)
        >>> bias = create_bias("scalar")
        >>> bias = create_bias("none")  # Returns Identity
    """
    bias_type = bias_type.lower()

    if bias_type in ["none", "null", None]:
        return nn.Identity()

    elif bias_type == "spatial":
        if channels is None or height is None or width is None:
            raise ValueError(
                "spatial bias requires channels, height, and width parameters"
            )
        return SpatialBias(
            channels=channels,
            height=height,
            width=width,
            init_value=init_value,
            requires_grad=requires_grad,
            allow_null_input=allow_null_input,
            dtype=dtype,
        )

    elif bias_type == "feature":
        if channels is None:
            raise ValueError("feature bias requires channels parameter")
        return FeatureBias(
            channels=channels,
            init_value=init_value,
            requires_grad=requires_grad,
            allow_null_input=allow_null_input,
            dtype=dtype,
        )

    elif bias_type == "scalar":
        return ScalarBias(
            init_value=init_value,
            requires_grad=requires_grad,
            allow_null_input=allow_null_input,
            dtype=dtype,
        )

    else:
        raise ValueError(
            f"Unknown bias type: {bias_type}. "
            f"Choose from: 'spatial', 'feature', 'scalar', 'none'"
        )


if __name__ == "__main__":
    # Test different bias types
    batch_size = 2
    channels = 64
    height, width = 56, 56

    x = torch.randn(batch_size, channels, height, width)

    print(f"Input shape: {x.shape}")
    print()

    # Test SpatialBias
    spatial_bias = SpatialBias(channels, height, width)
    out = spatial_bias(x)
    print(f"SpatialBias: {spatial_bias}")
    print(f"  Parameters: {sum(p.numel() for p in spatial_bias.parameters()):,}")
    print(f"  Output shape: {out.shape}")
    print(f"  None input shape: {spatial_bias(None).shape}")
    print()

    # Test FeatureBias
    feature_bias = FeatureBias(channels)
    out = feature_bias(x)
    print(f"FeatureBias: {feature_bias}")
    print(f"  Parameters: {sum(p.numel() for p in feature_bias.parameters()):,}")
    print(f"  Output shape: {out.shape}")
    print(f"  None input shape: {feature_bias(None).shape}")
    print()

    # Test ScalarBias
    scalar_bias = ScalarBias()
    out = scalar_bias(x)
    print(f"ScalarBias: {scalar_bias}")
    print(f"  Parameters: {sum(p.numel() for p in scalar_bias.parameters()):,}")
    print(f"  Output shape: {out.shape}")
    print(f"  None input shape: {scalar_bias(None).shape}")
    print()

    # Test factory function
    bias = create_bias("spatial", channels, height, width)
    print(f"Factory (spatial): {bias}")

    bias = create_bias("feature", channels=channels)
    print(f"Factory (feature): {bias}")

    bias = create_bias("none")
    print(f"Factory (none): {bias}")

    # Test allow_null_input=False
    print("\nTesting allow_null_input=False:")
    strict_bias = ScalarBias(allow_null_input=False)
    try:
        strict_bias(None)
        print("  ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"  ✓ Correctly raised error: {e}")

    print("\n✓ All bias types tested successfully!")
