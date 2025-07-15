"""
DynVision Base Classes

Modular base classes for building biologically-inspired neural networks.
"""

from .coordination import DtypeDeviceCoordinator, DtypeDeviceCoordinatorMixin
from .data_buffer import StorageBuffer, StorageBufferMixin
from .monitoring import Monitoring, MonitoringMixin
from .dynvision import DynVision
from .lightning import LightningBase


class BaseModel(
    DynVision,  # Core neural network functionality
    LightningBase,  # PyTorch Lightning training framework
    StorageBufferMixin,  # Response storage with Lightning hooks
    MonitoringMixin,  # Debugging/logging with Lightning hooks
    DtypeDeviceCoordinatorMixin,  # Device coordination with Lightning hooks
):
    """
    Complete DynVision model with all functionality.

    Inheritance order ensures proper MRO:
    1. DynVision: Core methods (_forward, forward, model_step dependencies)
    2. LightningBase: Training framework (calls DynVision methods)
    3. StorageBufferMixin: Storage with Lightning hooks
    4. MonitoringMixin: Monitoring with Lightning hooks
    5. DtypeDeviceCoordinatorMixin: Coordination with Lightning hooks

    Provides:
    - Core neural network computation (DynVision)
    - PyTorch Lightning integration (LightningBase)
    - Device/dtype coordination (DtypeDeviceCoordinatorMixin)
    - Response storage and management (StorageBufferMixin)
    - Comprehensive logging and debugging (MonitoringMixin)
    """

    def __init__(self, **kwargs):
        # Initialize coordination as root node
        DtypeDeviceCoordinatorMixin.__init__(self)
        self.is_root_node = True

        # Initialize other components
        super().__init__(**kwargs)

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
class CoreModel(DynVision, DtypeDeviceCoordinator):
    """Core neural network functionality with device coordination only."""

    pass


class MonitoredModel(DynVision, Monitoring, DtypeDeviceCoordinator):
    """Core neural network with monitoring, but no Lightning integration."""

    pass


# Export all components
__all__ = [
    # Main classes
    "BaseModel",  # Complete framework - most users want this
    "CoreModel",  # Just core neural network + coordination
    "MonitoredModel",  # Core + monitoring, no Lightning
    # Individual components
    "DynVision",  # Core neural network functionality
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
