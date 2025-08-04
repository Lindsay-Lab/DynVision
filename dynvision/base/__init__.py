"""
DynVision Base Classes

Modular base classes for building biologically-inspired neural networks.
"""

import os
import logging
import torch
import numpy as np
from types import FunctionType, MethodType
from .coordination import DtypeDeviceCoordinator, DtypeDeviceCoordinatorMixin
from .storage import StorageBuffer, StorageBufferMixin
from .monitoring import Monitoring, MonitoringMixin
from .temporal import TemporalBase
from .lightning import LightningBase

logger = logging.getLogger(__name__)

# Export all components
__all__ = [
    # Main classes
    "BaseModel",  # Complete framework - most users want this
    "CoreModel",  # Just core neural network + coordination
    "MonitoredModel",  # Core + monitoring, no Lightning
    # Individual components
    "TemporalBase",  # Core neural network functionality
    "LightningBase",  # Lightning training framework
    # Storage components
    "StorageBuffer",  # Storage without Lightning hooks
    "StorageBufferMixin",  # Storage with Lightning hooks
    # Monitoring components
    "Monitoring",  # Monitoring without Lightning hooks
    "MonitoringMixin",  # Monitoring with Lightning hooks
    # Coordination components
    "DtypeDeviceCoordinator",  # Device coordination
    "DtypeDeviceCoordinatorMixin",  # Device coordination with Lightning hooks
]


class BaseModel(
    TemporalBase,  # Core neural network functionality
    LightningBase,  # PyTorch Lightning training framework
    StorageBufferMixin,  # Response storage with Lightning hooks
    MonitoringMixin,  # Debugging/logging with Lightning hooks
    DtypeDeviceCoordinatorMixin,  # Device coordination with Lightning hooks
):
    """
    Complete DynVision model with all functionality.

    Inheritance order ensures proper MRO:
    1. TemporalBase: Core methods (_forward, forward)
    2. LightningBase: Training framework (calls TemporalBase methods)
    3. StorageBufferMixin: Storage with Lightning hooks
    4. MonitoringMixin: Monitoring with Lightning hooks
    5. DtypeDeviceCoordinatorMixin: Coordination with Lightning hooks

    Provides:
    - Core neural network computation (TemporalBase)
    - PyTorch Lightning integration (LightningBase)
    - Device/dtype coordination (DtypeDeviceCoordinatorMixin)
    - Response storage and management (StorageBufferMixin)
    - Comprehensive logging and debugging (MonitoringMixin)
    """

    def __init__(self, **kwargs):
        # Initialize coordination as root node
        DtypeDeviceCoordinatorMixin.__init__(self)
        if os.environ.get("WORLD_SIZE", "1") == "1":
            self.is_root_node = True
        else:
            logger.info(
                "Distributed training detected - disabling custom dtype coordination"
            )
            self.is_root_node = False

        super().__init__(**kwargs)

        # Automatically capture all hyperparameters after all initialization
        self._save_all_hyperparameters()

    def _save_all_hyperparameters(self):
        """
        Automatically save all serializable attributes as hyperparameters.
        Only serializable attributes are included, filtering out functions,
        modules, and runtime state automatically.
        """

        # Define serializable types for hyperparameters
        serializable_types = (int, float, str, bool, list, tuple, dict, type(None))

        hparams = {}

        for name, value in vars(self).items():
            # Skip private/internal attributes
            if name.startswith("_"):
                continue

            # Skip runtime state and Lightning-specific attributes
            skip_attributes = {
                # Lightning runtime
                "training",
                "device",
                "trainer",
                "logger",
                "global_step",
                "current_epoch",
                # Runtime state
                "storage",
                # Model components (not hyperparameters)
                "criterion",
                "child_nodes",
                "parent_node",
            }
            if name in skip_attributes:
                continue

            # Include serializable values
            if isinstance(value, serializable_types):
                hparams[name] = value
            elif isinstance(value, (torch.Tensor, np.ndarray)):
                # Include scalar tensors/arrays as their scalar value
                if hasattr(value, "numel") and value.numel() == 1:
                    hparams[name] = value.item()
                elif hasattr(value, "size") and np.prod(value.shape) == 1:
                    hparams[name] = float(value)
            # Automatically skip non-serializable objects (modules, functions, etc.)

        # Save all captured hyperparameters
        self.save_hyperparameters(hparams)

        # Log what was captured for debugging
        logger.info(f"Captured {len(hparams)} hyperparameters: {list(hparams.keys())}")

    def sync_persistent_state(self) -> None:
        """Override to sync responses and other persistent state."""
        super().sync_persistent_state()

        # Sync responses dictionary
        if hasattr(self, "responses"):
            target_dtype = self.get_target_dtype()
            target_device = self.get_target_device()

            for layer_name, response in self.responses.items():
                if isinstance(response, torch.Tensor):
                    self.responses[layer_name] = response.to(
                        dtype=target_dtype, device=target_device
                    )
                elif isinstance(response, list):
                    self.responses[layer_name] = [
                        (
                            r.to(dtype=target_dtype, device=target_device)
                            if isinstance(r, torch.Tensor)
                            else r
                        )
                        for r in response
                    ]

    def setup(self, stage=None) -> None:
        """Lightning setup hook - build coordination network."""
        try:
            super().setup(stage)
        except AttributeError:
            pass

        if hasattr(self, "set_residual_timesteps"):
            self.set_residual_timesteps()

        if hasattr(self, "reset"):
            self.reset()

        # Build coordination network and sync everything
        if self.is_root_node:
            self.build_coordination_network()
            self.propagate_dtype_sync()


# Flexible building blocks for advanced usage
class CoreModel(TemporalBase, DtypeDeviceCoordinatorMixin):
    """Core neural network functionality with device coordination only."""

    pass


class MonitoredModel(TemporalBase, MonitoringMixin, DtypeDeviceCoordinatorMixin):
    """Core neural network with monitoring, but no Lightning integration."""

    pass
