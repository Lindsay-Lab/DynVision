"""Base class for implementing custom loss functions.

This module provides a base class for implementing custom loss functions with
consistent patterns for input handling, validation, and error reporting.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn


class BaseLoss(nn.Module):
    """Base class for custom loss functions.

    This class provides common functionality for loss functions including:
    - Consistent output handling (flat outputs and responses)
    - Input validation
    - Error reporting
    - Reduction modes (mean, sum, none)
    """

    valid_reductions = ["none", "mean", "sum"]

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in self.valid_reductions:
            raise ValueError(
                f"Invalid reduction '{reduction}'. Must be one of {self.valid_reductions}"
            )
        self.reduction = reduction

    def extract_outputs(
        self,
        outputs: Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
        require_responses: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if isinstance(outputs, tuple):
            flat_outputs, *extra = outputs
            responses = extra[0] if extra else None
        else:
            flat_outputs = outputs
            responses = None

        if require_responses and responses is None:
            raise ValueError(
                f"{self.__class__.__name__} requires model responses but none were provided"
            )

        return flat_outputs, responses

    def validate_shapes(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        allow_broadcasting: bool = True,
    ) -> None:
        if not allow_broadcasting and outputs.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Output shape {outputs.shape} does not match target shape {targets.shape}"
            )

        if allow_broadcasting:
            try:
                _ = torch.broadcast_tensors(outputs, targets)
            except RuntimeError:
                raise ValueError(
                    f"Output shape {outputs.shape} cannot be broadcast to target shape {targets.shape}"
                )

    def apply_reduction(
        self, loss: torch.Tensor, num_valid_timesteps: Optional[int] = None
    ) -> torch.Tensor:
        """Apply the specified reduction; if num_valid_timesteps is provided and
        reduction is 'mean', normalize by that instead of taking the tensor mean.
        """
        if self.reduction == "mean":
            if num_valid_timesteps is not None:
                denom = int(num_valid_timesteps)
                if denom <= 0:
                    return torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
                return loss.sum() / float(denom)
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss  # 'none'

    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        responses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            f"Loss computation not implemented for {self.__class__.__name__}"
        )

    def forward(
        self,
        outputs: Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
        targets: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        # Extract outputs and responses
        flat_outputs, responses = self.extract_outputs(
            outputs, require_responses=getattr(self, "requires_responses", False)
        )

        # Validate shapes
        self.validate_shapes(
            flat_outputs,
            targets,
            allow_broadcasting=getattr(self, "allow_broadcasting", True),
        )

        num_valid = kwargs.get("num_valid_timesteps", None)
        # If not provided and targets exist and subclass exposes ignore_index, try to infer
        if num_valid is None and targets is not None and hasattr(self, "ignore_index"):
            try:
                mask = targets != getattr(self, "ignore_index")
                num_valid = int(mask.sum().item())
            except Exception:
                num_valid = None

        loss = self.compute_loss(
            outputs=flat_outputs, targets=targets, responses=responses
        )
        return self.apply_reduction(loss, num_valid_timesteps=num_valid)
