from typing import Any, List, Optional
import os
import torch
import torch.nn as nn
import logging
from dynvision.project_paths import project_paths
from pytorch_lightning import LightningModule

logger = logging.getLogger(__name__)


class DtypeDeviceCoordinator:
    """
    Coordinates dtype and device consistency across Lightning module networks.
    Uses auto-discovery to build coordination networks for modules with persistent state.
    """

    def __init__(self, target_dtype: Optional[torch.dtype] = None):
        self.is_root_node = False
        self.child_nodes: List["DtypeDeviceCoordinator"] = []
        self.parent_node: Optional["DtypeDeviceCoordinator"] = None
        self._target_dtype: Optional[torch.dtype] = target_dtype

        # Let lightning handle coordination in distributed setups
        if os.environ.get("WORLD_SIZE", "1") == "1":
            self._coordination_built = False
        else:
            self._coordination_built = True

    def connect_child_node(self, child: "DtypeDeviceCoordinator") -> None:
        """Connect a child node to this coordinator."""
        if child not in self.child_nodes:
            self.child_nodes.append(child)
            child.connect_to_parent(self)

    def connect_to_parent(self, parent: "DtypeDeviceCoordinator") -> None:
        """Connect this coordinator to a parent node."""
        self.parent_node = parent

    def build_coordination_network(self) -> None:
        """
        Recursively discover and register all modules that need coordination.
        Only called by root node.
        """
        if not self.is_root_node or self._coordination_built:
            return

        def register_recursively(
            module: nn.Module, parent_node: Optional["DtypeDeviceCoordinator"] = None
        ):
            # Check if module has coordination capabilities
            if isinstance(module, DtypeDeviceCoordinator) and self.needs_coordination(
                module
            ):
                if parent_node and module != self:  # Don't connect root to itself
                    parent_node.connect_child_node(module)

                # Recursively check children
                current_node = (
                    module
                    if isinstance(module, DtypeDeviceCoordinator)
                    else parent_node
                )
                for child_module in module.children():
                    register_recursively(child_module, current_node)

        logger.info("Building dtype/device coordination network...")
        register_recursively(self)
        self._coordination_built = True

        # Log network structure
        self._log_network_structure()

    def _log_network_structure(self, level: int = 0) -> None:
        """Log the coordination network structure for debugging."""
        indent = "  " * level
        module_name = getattr(self, "__class__", "Unknown").__name__
        logger.info(
            f"{indent}{module_name} ({'Root' if self.is_root_node else 'Child'})"
        )

        for child in self.child_nodes:
            child._log_network_structure(level + 1)

    def get_target_dtype(self) -> torch.dtype:
        """Get the target dtype from root node or determine from Lightning trainer."""
        if hasattr(self, "_target_dtype") and self._target_dtype is not None:
            return self._target_dtype
        elif self.is_root_node:
            self._target_dtype = self._determine_dtype_from_lightning()
            return self._target_dtype
        elif self.parent_node:
            return self.parent_node.get_target_dtype()
        else:
            return self._determine_dtype_from_parameters()

    def get_target_device(self) -> torch.device:
        """Get the target device from Lightning model."""
        if hasattr(self, "device"):
            return self.device
        elif self.parent_node:
            return self.parent_node.get_target_device()
        else:
            return torch.device("cpu")

    def _determine_dtype_from_lightning(self) -> torch.dtype:
        """Determine target dtype from Lightning trainer precision."""
        try:
            precision = str(self.trainer.precision)

            dtype_map = {
                "bf16": torch.bfloat16,
                "bf16-mixed": torch.bfloat16,
                "16": torch.float16,
                "16-mixed": torch.float16,
                "32": torch.float32,
                "32-true": torch.float32,
                "64": torch.float64,
                "64-true": torch.float64,
            }

            target_dtype = dtype_map.get(precision, torch.float32)
            logger.info(
                f"Determined target dtype from Lightning trainer: {target_dtype} (precision: {precision})"
            )
            return target_dtype

        except RuntimeError:
            return self._determine_dtype_from_parameters()

    def _determine_dtype_from_parameters(self) -> torch.dtype:
        """Fallback: determine dtype from model parameters."""
        if hasattr(self, "parameters"):
            try:
                return list(self.parameters())[-1].dtype
            except Exception as e:
                logger.error(
                    f"Error determining dtype from parameters: {e}. Defaulting to torch.float32."
                )
                pass
        return torch.float32

    def create_aligned_tensor(self, *args, **kwargs) -> torch.Tensor:
        """Create tensor with correct dtype and device for this coordination network."""

        # Extract size argument
        if args:
            size = args[0]
            args = args[1:]
        else:
            size = kwargs.pop("size", (1,))

        # Extract creation method BEFORE setting other defaults
        creation_method = kwargs.pop("creation_method", "randn")

        # Set correct dtype and device (only if not already specified)
        kwargs.setdefault("dtype", self.get_target_dtype())
        kwargs.setdefault("device", self.get_target_device())

        # Create tensor based on method
        if creation_method == "randn":
            return torch.randn(size, **kwargs)  # Only pass size as positional arg
        elif creation_method == "zeros":
            return torch.zeros(size, **kwargs)
        elif creation_method == "ones":
            return torch.ones(size, **kwargs)
        else:
            # Default fallback
            return torch.randn(size, **kwargs)

    def needs_coordination(self, module: nn.Module) -> bool:
        """Enhanced detection for recurrence modules."""
        coordination_indicators = [
            "hidden_states",
            "responses",
            "records",
            "storage",
        ]
        has_coordination_capability = isinstance(module, DtypeDeviceCoordinatorMixin)
        has_persistent_state = any(
            hasattr(module, attr) for attr in coordination_indicators
        )

        return has_coordination_capability and has_persistent_state

    def propagate_dtype_sync(self) -> None:
        """Propagate dtype synchronization from root to all child nodes."""
        if not self.is_root_node:
            return

        target_dtype = self.get_target_dtype()
        target_device = self.get_target_device()

        self._sync_all_parameters(target_dtype)

        self.sync_persistent_state()

        self._sync_children_recursive(target_dtype, target_device)

    def _sync_all_parameters(self, target_dtype: torch.dtype) -> None:
        """Sync all model parameters to target dtype."""
        params_synced = 0
        for name, param in self.named_parameters():
            if param.dtype != target_dtype:
                param.data = param.data.to(dtype=target_dtype)
                params_synced += 1

        if params_synced > 0:
            logger.info(f"Synced {params_synced} parameters to {target_dtype}")

    def _sync_children_recursive(
        self, target_dtype: torch.dtype, target_device: torch.device
    ) -> None:
        """Recursively sync all child nodes."""
        for child in self.child_nodes:
            child.sync_persistent_state()
            child._sync_children_recursive(target_dtype, target_device)

    def sync_persistent_state(self) -> None:
        """
        Sync persistent state (hidden states, cached activations, etc.) to target dtype/device.
        Override this method in subclasses with specific persistent state.
        """
        target_dtype = self.get_target_dtype()
        target_device = self.get_target_device()

        # Common persistent state attributes to sync
        state_attrs = [
            "hidden_states",
            "stored_activations",
            "responses",
            "cached_outputs",
        ]

        for attr_name in state_attrs:
            if hasattr(self, attr_name):
                attr_value = getattr(self, attr_name)
                synced_value = self._sync_attribute(
                    attr_value, target_dtype, target_device
                )
                setattr(self, attr_name, synced_value)

    def _sync_attribute(
        self, value: Any, target_dtype: torch.dtype, target_device: torch.device
    ) -> Any:
        """Sync various types of attributes to target dtype/device."""
        if isinstance(value, torch.Tensor):
            return value.to(dtype=target_dtype, device=target_device)
        elif isinstance(value, list):
            return [
                self._sync_attribute(item, target_dtype, target_device)
                for item in value
            ]
        elif isinstance(value, dict):
            return {
                k: self._sync_attribute(v, target_dtype, target_device)
                for k, v in value.items()
            }
        else:
            return value

    def validate_datamodule_precision(self) -> None:
        """Validate that FFCV dataloader precision matches model precision."""
        if not self.is_root_node:
            return

        try:
            datamodule = getattr(self.trainer, "datamodule", None)
            if datamodule is None:
                return

            model_dtype = self.get_target_dtype()

            # Check FFCV precision parameter
            ffcv_precision = getattr(datamodule, "precision", None)

            if ffcv_precision is not None:
                # Map FFCV precision to torch dtypes
                ffcv_dtype_map = {
                    "fp16": torch.float16,
                    "bf16": torch.bfloat16,
                    "fp32": torch.float32,
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32,
                }

                expected_dtype = ffcv_dtype_map.get(str(ffcv_precision), None)

                if expected_dtype and expected_dtype != model_dtype:
                    logger.warning(
                        f"FFCV dataloader precision ({ffcv_precision} -> {expected_dtype}) "
                        f"doesn't match model precision ({model_dtype}). "
                        f"This may cause dtype mismatches during training."
                    )
                else:
                    logger.info(
                        f"FFCV dataloader precision aligned with model: {model_dtype}"
                    )
        except RuntimeError:
            return


# Mixin class for Lightning modules
class DtypeDeviceCoordinatorMixin(DtypeDeviceCoordinator, LightningModule):
    """
    Mixin for PyTorch Lightning modules that need dtype/device coordination.
    """

    def debug_coordination_status(self):
        """Enhanced debugging to understand Lightning's precision behavior."""
        print(f"\n=== COORDINATION DEBUG ===")
        print(f"Is root node: {self.is_root_node}")
        print(f"Coordinator target dtype: {self.get_target_dtype()}")

        # Check trainer precision
        try:
            trainer = self.trainer
            print(f"Trainer precision: {trainer.precision}")
            print(
                f"Trainer precision plugin: {type(trainer.precision_plugin).__name__}"
            )
        except:
            print("No trainer available")

        # Check actual parameter dtypes
        print(f"\nFirst 5 parameters:")
        for name, param in list(self.named_parameters())[:5]:
            print(f"  {name}: {param.dtype}")

        # Check child nodes
        if hasattr(self, "child_nodes"):
            print(f"\nChild nodes: {len(self.child_nodes)}")
            for i, child in enumerate(self.child_nodes):
                print(f"  Child {i}: {type(child).__name__}")
                # Check child parameters too
                for name, param in list(child.named_parameters())[:2]:
                    print(f"    {name}: {param.dtype}")

        print(f"=== END DEBUG ===\n")

    def setup(self, stage: Optional[str] = None) -> None:
        """Lightning hook: setup coordination network."""
        try:
            super().setup(stage)
        except AttributeError:
            pass

        if self.is_root_node:
            self.build_coordination_network()
            self.propagate_dtype_sync()

    def on_fit_start(self) -> None:
        """Lightning hook: validate datamodule and final sync."""
        try:
            super().on_fit_start()
        except AttributeError:
            pass

        self.debug_coordination_status()  # Add this temporarily

        if self.is_root_node:
            self.validate_datamodule_precision()
            self.propagate_dtype_sync()

    def on_train_start(self) -> None:
        """Lightning hook: ensure sync at epoch start."""
        try:
            super().on_train_start()
        except AttributeError:
            pass

        if self.is_root_node:
            self.propagate_dtype_sync()

    def on_validation_start(self) -> None:
        """Lightning hook: ensure sync at epoch start."""
        try:
            super().on_validation_start()
        except AttributeError:
            pass

        if self.is_root_node:
            self.propagate_dtype_sync()
