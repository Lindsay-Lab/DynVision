import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Tuple
from .base_loss import BaseLoss


class EnergyLoss(BaseLoss):
    """Energy loss that computes statistics during forward pass using hooks."""

    def __init__(self, reduction: str = "mean", p: int = 1) -> None:
        super().__init__(reduction=reduction)
        self.requires_responses = False  # We don't need stored responses!
        self.allow_broadcasting = True
        self.batch_energy = {}
        self.hooks = []
        self.norm_factors = {}
        self.p = p

    def register_hooks(self, model: nn.Module) -> None:
        """Register forward hooks on model modules to capture energy statistics."""
        self.remove_hooks()  # Clean up any existing hooks

        for name, module in model.named_modules():
            if self._should_monitor_module(module):
                hook = module.register_forward_hook(
                    lambda module, input, output, name=name: self._accumulate_energy(
                        name, output
                    )
                )
                self.hooks.append(hook)

    def _should_monitor_module(self, module: nn.Module) -> bool:
        """Determine which modules to monitor for energy calculation."""
        # Monitor conv modules, linear modules, but skip activations, pooling, etc.
        return isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d))

    def _accumulate_energy(self, module_name: str, activation: torch.Tensor) -> None:
        """Store current batch energy for a module during forward pass."""
        if activation is None:
            return

        # Calculate energy for this batch
        batch_energy = torch.norm(
            activation, p=self.p, dim=tuple(range(1, activation.ndim))
        )

        if module_name not in self.norm_factors:
            # Calculate normalization factor once
            n_units = activation.shape[1:].numel()  # All dims except batch
            self.norm_factors[module_name] = n_units ** (1 / self.p)

        self.batch_energy[module_name] = batch_energy

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
        """Compute energy loss from current batch statistics only."""
        if not self.batch_energy:
            return torch.tensor(0.0, requires_grad=True)

        total_energy = torch.tensor(0.0, requires_grad=True)
        module_count = 0

        for module_name, batch_energy in self.batch_energy.items():
            # Get the energy for current batch and normalize
            norm_factor = self.norm_factors[module_name]

            # Ensure gradients flow through
            if batch_energy.requires_grad:
                with torch.no_grad():  # todo: double check why this is working
                    normalized_energy = batch_energy / norm_factor

                total_energy = total_energy + normalized_energy
                module_count += 1

        # Clear current batch energy immediately after use to free memory
        del self.batch_energy
        self.batch_energy = {}

        if module_count > 0:
            return total_energy / module_count
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
