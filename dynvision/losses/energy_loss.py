import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Tuple
from .base_loss import BaseLoss


class EnergyLoss(BaseLoss):
    """Energy loss that computes statistics during forward pass using hooks."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        self.requires_responses = False  # We don't need stored responses!
        self.allow_broadcasting = True
        self.energy_stats = {}
        self.hooks = []
        self.norm_factors = {}

    def register_hooks(self, model: nn.Module) -> None:
        """Register forward hooks on model layers to capture energy statistics."""
        self.remove_hooks()  # Clean up any existing hooks

        for name, module in model.named_modules():
            if self._should_monitor_layer(module):
                hook = module.register_forward_hook(
                    lambda module, input, output, name=name: self._accumulate_energy(
                        name, output
                    )
                )
                self.hooks.append(hook)

    def _should_monitor_layer(self, module: nn.Module) -> bool:
        """Determine which layers to monitor for energy calculation."""
        # Monitor conv layers, linear layers, but skip activations, pooling, etc.
        return isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d))

    def _accumulate_energy(self, layer_name: str, activation: torch.Tensor) -> None:
        """Accumulate energy statistics for a layer during forward pass."""
        if activation is None:
            return

        # Calculate energy (L2 norm) for this batch
        batch_energy = torch.norm(
            activation, p=2, dim=tuple(range(1, activation.ndim))
        )

        # Store or accumulate energy statistics
        if layer_name not in self.energy_stats:
            self.energy_stats[layer_name] = {
                "total_energy": 0.0,
                "count": 0,
                "current_batch_energy": batch_energy,
            }
            # Calculate normalization factor once
            n_units = activation.shape[1:].numel()  # All dims except batch
            self.norm_factors[layer_name] = n_units
        else:
            self.energy_stats[layer_name]["current_batch_energy"] = batch_energy
            self.energy_stats[layer_name]["total_energy"] += batch_energy
            self.energy_stats[layer_name]["count"] += 1

    def forward(
        self,
        outputs: Union[
            torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]
        ] = None,
        targets: torch.Tensor = None,
    ) -> torch.Tensor:

        loss = self.compute_loss()

        return self.apply_reduction(loss)

    def compute_loss(
        self,
        outputs: torch.Tensor = None,
        targets: torch.Tensor = None,
        responses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute energy loss from accumulated statistics."""
        if not self.energy_stats:
            return torch.tensor(0.0, requires_grad=True)

        total_energy = torch.tensor(0.0, requires_grad=True)
        layer_count = 0

        for layer_name, stats in self.energy_stats.items():
            if "current_batch_energy" in stats:
                # Get the energy for current batch and normalize
                batch_energy = stats["current_batch_energy"]
                norm_factor = self.norm_factors[layer_name]

                # Ensure gradients flow through
                if batch_energy.requires_grad:
                    normalized_energy = batch_energy.mean() / norm_factor
                    total_energy = total_energy + normalized_energy
                    layer_count += 1

        if layer_count > 0:
            return total_energy / layer_count
        else:
            return torch.tensor(0.0, requires_grad=True)

    def remove_hooks(self) -> None:
        """Clean up registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __del__(self):
        """Ensure hooks are cleaned up."""
        self.remove_hooks()
