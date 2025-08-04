"""Loss functions for neural network training and regularization.

This package provides various loss functions for training neural networks,
including both standard losses and specialized implementations for specific
training objectives.

Available Losses:
    - BaseLoss: Base class for implementing custom losses
    - CrossEntropyLoss: Standard cross entropy loss with response handling
    - EnergyLoss: Loss based on network activation energies
    - MeanSquaredActivationLoss: MSE loss for activation targeting

Example:
    ```python
    from dynvision.losses import CrossEntropyLoss, EnergyLoss

    # Standard classification loss
    ce_loss = CrossEntropyLoss()
    
    # Energy-based regularization
    energy_loss = EnergyLoss()
    
    # Combined loss
    def compute_loss(outputs, targets):
        classification_loss = ce_loss(outputs, targets)
        activation_loss = energy_loss(outputs, targets)
        return classification_loss + 0.1 * activation_loss
    ```
"""

from torch.nn.modules.loss import _Loss
import torch
from .base_loss import BaseLoss
from .cross_entropy_loss import CrossEntropyLoss
from .energy_loss import EnergyLoss

__all__ = [
    "BaseLoss",
    "CrossEntropyLoss",
    "EnergyLoss",
]


def mask_non_labels(outputs, label_indices, non_label_index=-1, selector="!="):
    # make sure outputs and label_indices are flattened
    if selector == "!=":
        mask = torch.where(label_indices != non_label_index)[0]
    elif selector == "==":
        mask = torch.where(label_indices == non_label_index)[0]
    else:
        raise ValueError(f"Invalid selector: {selector}. Must be '!=' or '=='.")

    label_indices = label_indices[mask].to(outputs.device)
    mask = mask.to(outputs.device)
    outputs = outputs[mask]

    return outputs, label_indices
