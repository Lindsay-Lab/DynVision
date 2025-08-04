"""Topographic recurrent connections for biologically-inspired neural networks.

This module implements topographically organized lateral and feedback connections inspired by the organization of the visual cortex. The connections maintain spatial relationships between neurons, similar to the retinotopic organization found in biological visual systems.


Biological Background:
The visual cortex maintains a topographic organization where nearby neurons process information from simialr features and nearby regions of visual space. This organization is preserved through various types of connections:
- Local lateral connections between neurons with similar response properties
- Patchy connections that link nearby neurons within a layer

References:
[1] Gilbert & Wiesel (1989). Columnar specificity of intrinsic horizontal and
    corticocortical connections in cat visual cortex.
[2] Bosking et al. (1997). Orientation selectivity and the arrangement of
    horizontal connections in tree shrew striate cortex.
"""

import logging
from typing import Optional, Any

import torch
import torch.nn as nn
import math
from pytorch_lightning import LightningModule


logger = logging.getLogger(__name__)

__all__ = ["LocalLateralConnection", "LocalSeparableConnection"]


def is_square_number(n: int) -> bool:
    if n < 0:
        return False
    sqrt_n = int(math.sqrt(n))
    return sqrt_n * sqrt_n == n


def next_square_number(c: int) -> int:
    i = int(math.sqrt(c)) + 1
    return i**2


def extend_to_square_channel_number(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    n_channels = x.size(dim)
    if is_square_number(n_channels):
        return x
    add_n = next_square_number(n_channels) - n_channels
    repeated_slices = x.narrow(dim, n_channels - add_n, add_n).flip(dims=[dim])
    return torch.cat((x, repeated_slices), dim=dim)


def get_coordinates(shape: tuple) -> torch.Tensor:
    """
    Generates a tensor of given shape with each element containing its coordinates (along an additional dimension).
    """
    *n_channels, dim_y, dim_x = shape
    x = torch.arange(dim_x, dtype=torch.int64)
    y = torch.arange(dim_y, dtype=torch.int64)
    if len(n_channels) > 1:
        raise ValueError("Invalid shape!")
    elif len(n_channels) == 1:
        z = torch.arange(n_channels[0], dtype=torch.int64)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        coordinates = torch.stack((zz, yy, xx), dim=-1)
    else:
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((yy, xx), dim=-1)
    return coordinates


def _layer_to_plane(layer: torch.Tensor) -> torch.Tensor:
    """
    Maps a 3D (+coord dim) layer tensor to a 2D plane tensor by arranging the feature maps in a grid.
    """
    n_channels, dim_y, dim_x, dim_coords = layer.shape
    feature_dim = int(math.sqrt(n_channels))
    plane_dim_x = feature_dim * dim_x
    plane_dim_y = feature_dim * dim_y
    plane = torch.zeros(
        (plane_dim_y, plane_dim_x, dim_coords), dtype=torch.int64, device=layer.device
    )
    flip_indices = torch.arange(feature_dim - 1, -1, -1, device=layer.device)
    for y in range(dim_y):
        for x in range(dim_x):
            feature = layer[:, y, x].reshape(feature_dim, feature_dim, dim_coords)
            if x % 2 == 1:
                feature = feature[:, flip_indices]
            if y % 2 == 1:
                feature = feature[flip_indices, :]
            y0, y1 = y * feature_dim, (y + 1) * feature_dim
            x0, x1 = x * feature_dim, (x + 1) * feature_dim
            plane[y0:y1, x0:x1] = feature
    return plane


def _plane_to_layer(
    plane: torch.Tensor, n_channels: int, dim_y: int, dim_x: int
) -> torch.Tensor:
    """
    Maps a 2D (+coord dim) plane tensor to a 3D layer tensor by extracting feature maps from a grid.
    """
    plane = plane.squeeze()
    plane_dim_y, plane_dim_x, dim_coords = plane.shape
    layer = torch.zeros(
        (n_channels, dim_y, dim_x, dim_coords), dtype=torch.int64, device=plane.device
    )
    feature_dim = int(math.sqrt(n_channels))
    flip_indices = torch.arange(feature_dim - 1, -1, -1, device=plane.device)
    for y in range(dim_y):
        for x in range(dim_x):
            y0, y1 = y * feature_dim, (y + 1) * feature_dim
            x0, x1 = x * feature_dim, (x + 1) * feature_dim
            feature = plane[y0:y1, x0:x1]
            if x % 2 == 1:
                feature = feature[:, flip_indices]
            if y % 2 == 1:
                feature = feature[flip_indices, :]
            layer[:, y, x] = feature.reshape(-1, dim_coords)
    return layer


def create_layer_to_plane_mapping(
    n_channels: int, dim_y: int, dim_x: int
) -> torch.Tensor:
    """
    Create a lookup table that maps a 3D layer tensor to a 2D plane tensor.
    """
    feature_dim = int(math.sqrt(n_channels))
    plane_dim_y = feature_dim * dim_y
    plane_dim_x = feature_dim * dim_x
    plane_coordinates = get_coordinates((plane_dim_y, plane_dim_x))
    return _plane_to_layer(plane_coordinates, n_channels, dim_y, dim_x)


def create_plane_to_layer_mapping(
    n_channels: int, dim_y: int, dim_x: int
) -> torch.Tensor:
    """
    Create a lookup table that maps a 2D plane tensor to a 3D layer tensor.
    """
    layer_coordinates = get_coordinates((n_channels, dim_y, dim_x))
    return _layer_to_plane(layer_coordinates)


class LocalLateralConnection(LightningModule):
    """Local lateral connections for biologically-inspired neural networks.

    Biological Features:
    - Local lateral connections for short-range interactions between neurons with similar response properties

    Args:
        in_channels (int): Number of input channels
        dim_y (int): Height of input feature maps
        dim_x (int): Width of input feature maps
        kernel_size (int, optional): Size of convolutional kernel
        bias (bool): Whether to include bias (default: True)
        device: Computation device
        max_weight_init (float): Maximum initial weight value (default: 0.05)
        stability_threshold (float): Maximum allowed magnitude (default: 1e6)
        mixed_precision (bool): Whether to use mixed precision (default: True)
    """

    def __init__(
        self,
        in_channels: int,
        dim_y: int,
        dim_x: int,
        kernel_size: Optional[int] = None,
        bias: bool = True,
        device: Optional[torch.device] = None,
        max_weight_init: float = 0.05,
        mixed_precision: bool = True,
        **kwargs: Any,
    ):
        super(LocalLateralConnection, self).__init__()
        # Validate parameters
        if dim_y <= 0 or dim_x <= 0:
            raise ValueError(f"Invalid dimensions: dim_y={dim_y}, dim_x={dim_x}")

        self.n_channels = in_channels
        self.dim_y = dim_y
        self.dim_x = dim_x
        self.feature_dim = int(math.sqrt(in_channels))
        self.plane_dim_y = self.feature_dim * dim_y
        self.plane_dim_x = self.feature_dim * dim_x
        self.mixed_precision = mixed_precision
        
        if is_square_number(in_channels):
            self.square_n_channels = in_channels
        else:
            logger.warning(f"in_channels={in_channels} is not a perfect square, internally extending to next square number")
            self.square_n_channels = next_square_number(in_channels)

        # Create lookup mappings
        # Note: The naming may be a bit counter-intuitive:
        # - plane_to_layer_mapping is used in layer_to_plane (maps plane coordinates to layer indices).
        # - layer_to_plane_mapping is used in plane_to_layer (maps layer coordinates to plane indices).
        self.plane_to_layer_mapping = create_plane_to_layer_mapping(
            self.square_n_channels, dim_y, dim_x
        )
        self.layer_to_plane_mapping = create_layer_to_plane_mapping(
            self.square_n_channels, dim_y, dim_x
        )

        # Precompute flat index mappings for efficient gather operations.
        # For optimized_layer_to_plane: using self.plane_to_layer_mapping of shape [plane_dim_y, plane_dim_x, 3]
        z_index = self.plane_to_layer_mapping[..., 0]  # channel indices in layer
        y_index = self.plane_to_layer_mapping[..., 1]  # y indices in layer
        x_index = self.plane_to_layer_mapping[..., 2]  # x indices in layer
        flat_plane_mapping = (
            (z_index * (self.dim_y * self.dim_x) + y_index * self.dim_x + x_index)
            .view(-1)
            .long()
            .to(device)
        )
        self.register_buffer("flat_plane_mapping", flat_plane_mapping)

        # For optimized_plane_to_layer: using self.layer_to_plane_mapping of shape [n_channels, dim_y, dim_x, 3]
        # Here, we use the y and x coordinates in the plane.
        y_index2 = self.layer_to_plane_mapping[..., 0]
        x_index2 = self.layer_to_plane_mapping[..., 1]
        flat_layer_mapping = (
            (y_index2 * self.plane_dim_x + x_index2).view(-1).long().to(device)
        )
        self.register_buffer("flat_layer_mapping", flat_layer_mapping)

        # Determine kernel size for the plane mapping convolution.
        plane_kernel_size = self.feature_dim // 4
        if plane_kernel_size % 2 == 0:
            plane_kernel_size += 1
        plane_kernel_size = max(3, plane_kernel_size)

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=plane_kernel_size,
            stride=1,
            padding=plane_kernel_size // 2,
            padding_mode="reflect",
            bias=bias,
            device=device,
        )
        self.max_weight_init = max_weight_init

    def _init_parameters(self):
        nn.init.uniform_(
            self.conv.weight, a=-self.max_weight_init, b=self.max_weight_init
        )

    def optimized_layer_to_plane(self, layer: torch.Tensor) -> torch.Tensor:
        """
        Efficiently map a 3D layer tensor [B, n_channels, dim_y, dim_x]
        to a 2D plane tensor [B, 1, plane_dim_y, plane_dim_x] using precomputed flat indices.
        """
        batch_size = layer.size(0)
        # Flatten the layer tensor along the channel and spatial dimensions.
        layer_flat = layer.view(
            batch_size, -1
        )  # shape: [B, n_channels * dim_y * dim_x]
        # Gather values using the precomputed flat mapping.
        plane_flat = torch.gather(
            layer_flat, 1, self.flat_plane_mapping.expand(batch_size, -1)
        )
        plane = plane_flat.view(batch_size, 1, self.plane_dim_y, self.plane_dim_x)
        return plane

    def optimized_plane_to_layer(self, plane: torch.Tensor) -> torch.Tensor:
        """
        Efficiently map a 2D plane tensor [B, 1, plane_dim_y, plane_dim_x]
        back to a 3D layer tensor [B, n_channels, dim_y, dim_x] using precomputed flat indices.
        """
        batch_size = plane.size(0)
        plane_flat = plane.view(
            batch_size, -1
        )  # shape: [B, plane_dim_y * plane_dim_x]
        layer_flat = torch.gather(
            plane_flat, 1, self.flat_layer_mapping.expand(batch_size, -1)
        )
        layer = layer_flat.view(batch_size, self.square_n_channels, self.dim_y, self.dim_x)
        return layer

    def forward(self, x0: torch.Tensor) -> torch.Tensor:

        batch_size = x0.size(0)
        
        # Extend input to square channel number if necessary
        x0 = extend_to_square_channel_number(x0, dim=1)

        # Map the layer to a plane efficiently
        p0 = self.optimized_layer_to_plane(x0)

        # Apply convolution on the plane
        p1 = self.conv(p0)

        # Map back from plane to layer
        x1 = self.optimized_plane_to_layer(p1)

        # Trim any extra channels
        x1 = x1[:, : self.n_channels]

        return x1

    def trace_forward(self, x0: torch.Tensor) -> torch.jit.ScriptModule:
        return torch.jit.trace(self.forward, x0)


class LocalSeparableConnection(LightningModule):
    """Separable local connections for biologically-inspired neural networks.

    Biological Features:
    - Local lateral connections for short-range interactions
    - Patchy connections for longer-range interactions

    Args:
        in_channels (int): Number of input channels
        dim_y (int): Height of input feature maps
        dim_x (int): Width of input feature maps
        kernel_size (int): Size of convolutional kernel
        bias (bool): Whether to include bias (default: True)
        max_weight_init (float): Maximum initial weight value (default: 0.05)
        stability_threshold (float): Maximum allowed magnitude (default: 1e6)
        mixed_precision (bool): Whether to use mixed precision (default: True)
    """

    def __init__(
        self,
        in_channels: int,
        dim_y: int,
        dim_x: int,
        kernel_size: int,
        bias: bool = True,
        max_weight_init: float = 0.05,
        stability_threshold: float = 1e6,
        mixed_precision: bool = True,
        **kwargs: Any,
    ):
        super(LocalSeparableConnection, self).__init__()

        self.stability_threshold = stability_threshold
        self.mixed_precision = mixed_precision

        # Initialize components
        self.local_conv = LocalLateralConnection(
            in_channels=in_channels,
            dim_y=dim_y,
            dim_x=dim_x,
            kernel_size=kernel_size,
            bias=bias,
            max_weight_init=max_weight_init,
            mixed_precision=mixed_precision,
        )

        self.patchy_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=bias,
        )

        # Initialize weights
        nn.init.uniform_(
            self.patchy_conv.weight, a=-max_weight_init, b=max_weight_init
        )

        logger.info(
            f"Initialized LocalSeparableConnection with {in_channels} channels"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Efficient computation with mixed precision
        with torch.amp.autocast("cuda", enabled=self.mixed_precision):
            # Local connections (faster, shorter range)
            h = self.local_conv(x)

            # Patchy connections (longer range)
            h = self.patchy_conv(h)

            return h
