"""Supra-linear activation for biologically-inspired neural networks.

This module implements supra-linear activation functions inspired by biological neurons.
The supralinear transformation models the enhanced response properties observed in
cortical neurons, where the output increases more rapidly than linear with input strength.

References:
[1] Rubin, D. B., Van Hooser, S. D., & Miller, K. D. (2015). The stabilized
    supralinear network: A unifying circuit motif underlying multi-input
    integration in sensory cortex. Neuron
[2] Lindsay et al. (2019). Do Biologically-Realistic Recurrent Architectures Produce Biologically-Realistic Models?
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from pytorch_lightning import LightningModule


logger = logging.getLogger(__name__)

__all__ = ["SupraLinearity"]


class SupraLinearity(LightningModule):
    """Supra-linear activation function for biologically-inspired neural networks.

    This module implements a biologically-inspired supra-linear transformation:
    f(x) = k * sign(x) * |x|^n

    Parameters:
        n (float): Exponent controlling supralinearity (default: 1.8)
            Typical cortical values range from 1.5-2.0
        k (float): Scaling factor for response gain (default: 0.8)
            Controls the overall response magnitude
        requires_grad (bool): Whether parameters are trainable (default: False)
        stability_threshold (float): Maximum allowed input magnitude (default: 1e6)
            Prevents numerical instability
        mixed_precision (bool): Whether to use mixed precision (default: True)
            Improves performance on modern GPUs
    """

    def __init__(
        self,
        n: float = 1.8,
        k: float = 0.8,
        requires_grad: bool = False,
        stability_threshold: float = 1e6,
        mixed_precision: bool = True,
    ) -> None:
        super(SupraLinearity, self).__init__()

        # Validate parameters
        if n <= 1.0:
            logger.warning(f"n={n} <= 1.0 may not provide supra-linear behavior")
        if k <= 0.0:
            raise ValueError(f"k={k} must be positive")

        # Initialize parameters
        self.n = nn.Parameter(
            torch.tensor(n, dtype=torch.float32), requires_grad=requires_grad
        )
        self.k = nn.Parameter(
            torch.tensor(k, dtype=torch.float32), requires_grad=requires_grad
        )

        self.stability_threshold = stability_threshold
        self.mixed_precision = mixed_precision

    def forward(self, x: Optional[Tensor] = None) -> Optional[Tensor]:
        if x is None:
            return None

        # Stability check
        if torch.abs(x).max() > self.stability_threshold:
            logger.warning(
                f"Input magnitude exceeds stability threshold: {torch.abs(x).max()}"
            )
            x = torch.clamp(x, -self.stability_threshold, self.stability_threshold)

        # Efficient computation with mixed precision
        with torch.amp.autocast("cuda", enabled=self.mixed_precision):
            # Handle negative values efficiently
            abs_x = torch.abs(x)
            # Clamp small values for numerical stability
            abs_x = abs_x.clamp(min=1e-6)
            # Compute power with sign preservation
            signs = torch.sign(x)
            return self.k * signs * torch.pow(abs_x, self.n)
