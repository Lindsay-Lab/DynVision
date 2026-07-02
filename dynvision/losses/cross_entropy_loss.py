"""Cross Entropy Loss implementation with support for model responses.

This version returns element-wise losses (no reduction) and relies on BaseLoss
for normalization by the number of valid timesteps.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .base_loss import BaseLoss


class CrossEntropyLoss(BaseLoss):
    def __init__(
        self,
        reduction: str = "mean",
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -1,
    ) -> None:
        super().__init__(reduction=reduction)
        self.weight = weight
        self.ignore_index = ignore_index
        self.allow_broadcasting = False  # Cross entropy requires exact shape match

    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        responses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Compute element-wise cross-entropy (no reduction)
        # outputs: (N, C), targets: (N,)
        device = outputs.device
        dtype = outputs.dtype

        # Ensure targets on same device
        targets = targets.to(device)

        element_loss = F.cross_entropy(
            outputs,
            targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        # Mask ignored targets
        valid_mask = (
            (targets != self.ignore_index)
            & (targets >= 0)
            & (targets < outputs.size(1))
        )
        valid_mask = valid_mask.to(device=device)

        # Zero-out invalid entries
        element_loss = element_loss * valid_mask.to(dtype)

        # Return element-wise losses; BaseLoss.apply_reduction will handle normalization
        return element_loss
