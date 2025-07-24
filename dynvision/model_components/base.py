"""
Provides base classes for constructing biologically-inspired neural network models.

Classes:
    - UtilityBase (nn.Module): Core utilities for standardizing input/label dimensions,
      state dictionary management, and parameter logging.
    - LightningBase (UtilityBase, pl.LightningModule): Extends UtilityBase with PyTorch Lightning
      support for training, validation, and testing, including methods for forward propagation,
      optimizer/scheduler configuration, and evaluation.

Usage:
    Subclass LightningBase to implement custom models with advanced recurrent and feedback features.
"""

import io
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import gc

# Debuging
import psutil
import threading
import time
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb

from dynvision import losses
from dynvision.data.operations import _adjust_data_dimensions, _adjust_label_dimensions
from dynvision.utils import (
    alias_kwargs,
    path_to_index,
    load_config,
    on_same_device,
    determine_target_dtype,
)
from dynvision.project_paths import project_paths

__all__ = ["UtilityBase", "LightningBase"]

defaults = SimpleNamespace(
    **load_config(project_paths.scripts.configs / "config_defaults.yaml")
)

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
        self._coordination_built = False

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
                return next(self.parameters()).dtype
            except StopIteration:
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
            "stored_activations",
            "responses",
            "cached_outputs",
            "state_buffer",
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
class DtypeDeviceCoordinatorMixin(DtypeDeviceCoordinator):
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


class UtilityBase(nn.Module):
    """
    A base class providing utility functions for neural network models.
    Includes methods for adjusting input dimensions, managing state dictionaries,
    and handling tensor operations.
    """

    def _expand_input_channels(
        self, x: Optional[torch.Tensor], n_target_channels: int = 3
    ) -> Optional[torch.Tensor]:
        """
        Expand the input tensor's channels to match the target number of channels.

        Args:
            x (Optional[torch.Tensor]): Input tensor with shape (batch_size, n_channels, ...).
            n_target_channels (int): Target number of channels.

        Returns:
            Optional[torch.Tensor]: Tensor with expanded channels or None if input is None.
        """
        if x is None:
            return None

        batch_size, n_channels, *shape = x.shape

        if n_channels == 1:
            x = x.expand(batch_size, n_target_channels, *shape)
        elif n_channels == n_target_channels:
            pass
        else:
            raise ValueError(f"Invalid number of input channels: {n_channels}")

        return x

    def _calculate_layer_output_shape(
        self,
        input_shape: Tuple[int, ...],
        kernel_size: int,
        stride: int,
        padding: int = 0,
    ) -> Tuple[int, ...]:
        """
        Calculate the output shape of a layer given its input shape and parameters.

        Args:
            input_shape (Tuple[int, ...]): Shape of the input tensor.
            kernel_size (int): Size of the kernel.
            stride (int): Stride of the operation.
            padding (int, optional): Padding applied to the input. Defaults to 0.

        Returns:
            Tuple[int, ...]: Shape of the output tensor.
        """
        return tuple(
            (dim - kernel_size + 2 * padding) // stride + 1 for dim in input_shape
        )

    def _determine_residual_timesteps(
        self, max_timesteps: int = 100, dtype: Optional[torch.dtype] = None
    ) -> int:
        """
        Determine the number of residual timesteps required for an input to be processed through the unrolled model.

        This method uses a random input tensor to forward propagate through the model
        and checks for non-empty outputs. The process stops when a non-empty output
        is detected or a maximum of max_timesteps iterations is reached.

        Args:
            max_timesteps (int): Maximum number of timesteps to check. Defaults to 100.
            dtype (Optional[torch.dtype]): Data type for the random input tensor. Defaults to None.
            force_rerun (bool): Whether to force rerunning the determination process. Defaults to False.

        Returns:
            int: Number of residual timesteps required.

        Raises:
            ValueError: If the number of residual timesteps exceeds max_timesteps.
        """
        random_input = self.create_aligned_tensor(
            size=(1, self.n_channels, self.dim_y, self.dim_x), creation_method="randn"
        )

        if hasattr(self, "reset"):
            self.reset()

        x = None
        t = -1

        def is_empty_output(x):
            return (x is None) or (
                torch.all(x.eq(0)) or (torch.isnan(x).all()) or x.grad_fn is None
            )

        logger.info("Determining residual timesteps...")

        while is_empty_output(x):
            t += 1
            x, _ = self._forward(random_input, t=t, feedforward_only=True)
            if t > max_timesteps:
                raise ValueError(
                    f"Unable to determine residual timesteps (> {max_timesteps})!"
                )

        logger.info(f"Residual timesteps: {t}")

        if hasattr(self, "reset"):
            self.reset()

        return t

    def set_residual_timesteps(
        self,
        n_timesteps: Optional[int] = None,
        max_timesteps: int = 100,
        dtype: Optional[torch.dtype] = None,
        force_rerun: bool = False,
    ) -> None:
        """
        Set the number of residual timesteps for the model.

        Args:
            n_timesteps (Optional[int]): Number of residual timesteps to set. If None, it will be determined automatically.
            max_timesteps (int): Maximum number of timesteps to check when determining residual timesteps. Defaults to 100.
            dtype (Optional[torch.dtype]): Data type for the random input tensor used in determining residual timesteps. Defaults to None.
            force_rerun (bool): Whether to force rerunning the determination process. Defaults to False.
        """
        if n_timesteps is not None:
            if hasattr(self, "n_residual_timesteps"):
                logger.debug(
                    f"Overwriting existing n_residual_timesteps: {self.n_residual_timesteps} with {n_timesteps}"
                )
        elif hasattr(self, "n_residual_timesteps") and not force_rerun:
            return
        else:
            n_timesteps = self._determine_residual_timesteps(
                max_timesteps=max_timesteps, dtype=dtype
            )

        self.register_buffer(
            "n_residual_timesteps", torch.tensor(n_timesteps, dtype=torch.int)
        )

    def _add_missing_parameters_to_state_dict(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Add missing parameters to the model's state dictonary to the input's state dictionary.

        Args:
            state_dict (Dict[str, torch.Tensor]): State dictionary to update.

        Returns:
            Dict[str, torch.Tensor]: Updated state dictionary with missing parameters added.
        """
        missing_parameter_names = []
        for key in self.state_dict().keys():
            if key not in state_dict.keys():
                missing_parameter_names.append(key)
                state_dict[key] = self.state_dict()[key]
        if missing_parameter_names:
            logger.info(
                f"Adding missing parameters to loaded state dict: {missing_parameter_names}"
            )
        return state_dict

    def _remove_unexpected_parameters_from_state_dict(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Remove unexpected parameters from the input's state dictionary that are not found in the model's state dictionary.

        Args:
            state_dict (Dict[str, torch.Tensor]): State dictionary to update.

        Returns:
            Dict[str, torch.Tensor]: Updated state dictionary with unexpected parameters removed.
        """
        unexpected_parameter_names = []
        for key in state_dict.keys():
            if key not in self.state_dict().keys():
                unexpected_parameter_names.append(key)
        if unexpected_parameter_names:
            logger.warning(
                f"Removing unexpected parameters from loaded state dict: {unexpected_parameter_names}"
            )
            for key in unexpected_parameter_names:
                del state_dict[key]
        return state_dict

    def load_pretrained_state_dict(
        self, check_mismatch_layer: List[str] = [], strict: bool = True
    ) -> None:
        """
        Load a pretrained state dictionary into the model. This function requires the model to implement a method to download the pretrained weights 'download_pretrained_state_dict()'. If modules in the model have different names in the pretrained weights, a method 'translate_pretrained_layer_names()' should be implemented to return a mapping between the pretrained and model layer names in the form of a dictionary {"pretrained_name": "new_name"}', where pretrained_name can be any substring of a named parameter.

        Args:
            check_mismatch_layer (List[str], optional): List of layer names to check for mismatched shapes. Defaults to [].
            strict (bool, optional): Whether to strictly enforce that the keys in `state_dict` match the model's keys. Defaults to True.
        """
        # Load the pretrained weights
        if hasattr(self, "download_pretrained_state_dict"):
            state_dict = self.download_pretrained_state_dict()
        else:
            raise NotImplementedError("No method to download pretrained weights")

        # translate keys in loaded state dict
        if hasattr(self, "translate_pretrained_layer_names"):
            translate_layer_names = self.translate_pretrained_layer_names()
            external_keys = list(state_dict.keys())
            new_state_dict = copy(state_dict)
            for key in external_keys:
                for old_key, new_key in translate_layer_names.items():
                    if old_key in key:
                        new_key = key.replace(old_key, new_key)
                        new_state_dict[new_key] = state_dict[key]
                        if old_key not in translate_layer_names.values():
                            del new_state_dict[key]
                        continue
            state_dict = new_state_dict

        pretrained_keys = list(state_dict.keys())

        # adapt to different number of output classes
        for layer_name in check_mismatch_layer:
            layer_keys = [key for key in pretrained_keys if key.startswith(layer_name)]

            for key in layer_keys:
                pretrained_shape = state_dict[key].shape
                initialized_shape = self.state_dict()[key].shape

                if pretrained_shape != initialized_shape:
                    state_dict[key] = self.state_dict()[key]
                    pretrained_keys.remove(key)

        self.load_state_dict(state_dict, strict=strict)

        # select trainable parameters
        parameter_names = list(self.state_dict().keys())
        self.trainable_parameter_names = [
            p for p in parameter_names if p not in pretrained_keys
        ]

    def load_state_dict(
        self, state_dict: Dict[str, torch.Tensor], **kwargs: Any
    ) -> None:
        state_dict = self._add_missing_parameters_to_state_dict(state_dict)
        state_dict = self._remove_unexpected_parameters_from_state_dict(state_dict)
        super().load_state_dict(state_dict, **kwargs)

    def hasattr(self, *args: Any) -> bool:
        if len(args) == 1:
            obj = self
            attribute_name = args[0]
        elif len(args) == 2:
            obj, attribute_name = args
        else:
            raise ValueError(
                "getattr accepts either a single attribute name (str) or an object and an attribute name."
            )
        attributes = attribute_name.split(".")
        for attr_name in attributes:
            if not hasattr(obj, attr_name):
                return False
            obj = getattr(obj, attr_name)
        return True

    def getattr(self, *args: Any):
        if len(args) == 1:
            obj = self
            attribute_name = args[0]
        elif len(args) == 2:
            obj, attribute_name = args
        else:
            raise ValueError(
                "getattr accepts either a single attribute name (str) or an object and an attribute name."
            )

        attributes = attribute_name.split(".")
        for attr_name in attributes:
            obj = getattr(obj, attr_name)

        return obj

    def set_trainable_parameters(
        self, parameter_names: List[str] = [], force_train_all: bool = False
    ) -> None:
        if hasattr(self, "trainable_parameter_names") and len(
            self.trainable_parameter_names
        ):
            logger.warning(
                f"self.trainable_parameter_names is not empty: {self.trainable_parameter_names}! This will be overwritten!"
            )

        if not len(parameter_names):
            parameter_names = list(self.state_dict().keys())

        self.trainable_parameter_names = []
        for parameter_name in parameter_names:
            if self.hasattr(parameter_name):
                parameter = self.getattr(parameter_name)
            else:
                logger.warning(
                    f"Parameter {parameter_name} not found in model, can't be set to trainable!"
                )
                continue
            if force_train_all:
                try:
                    parameter.requires_grad = True
                except Exception as e:
                    logger.warning(
                        f"Parameter {parameter} can't be set to requires_grad=True! \n\t{e}"
                    )
            if hasattr(parameter, "requires_grad") and parameter.requires_grad:
                self.trainable_parameter_names.append(parameter_name)

    def set_trainable_parameter(self, parameter_name: str) -> None:
        if hasattr(self, parameter_name):
            if hasattr(self, "trainable_parameter_names"):
                if parameter_name not in self.trainable_parameter_names:
                    self.trainable_parameter_names.append(parameter_name)
            else:
                self.trainable_parameter_names = [parameter_name]
        else:
            logger.warning(
                f"Parameter {parameter_name} not found in model, can't be set to trainable!"
            )

    def get_trainable_parameter_names(self) -> List[str]:
        """
        Get the names of trainable parameters in the model.

        Returns:
            List[str]: List of trainable parameter names.
        """
        if not hasattr(self, "trainable_parameter_names"):
            self.set_trainable_parameters()
        return self.trainable_parameter_names

    def get_trainable_parameters(self) -> List[torch.Tensor]:
        """
        Get the trainable parameters of the model.

        Returns:
            List[torch.Tensor]: List of trainable parameters.
        """
        if not hasattr(self, "trainable_parameter_names"):
            self.set_trainable_parameters()
        return list(self.trainable_parameters())

    def trainable_parameters(self) -> torch.Generator:
        """
        Yield trainable parameters of the model. Refers to 'trainable_parameter_names' attribute if defined, else `parameters`.

        Returns:
            torch.Generator: Generator yielding trainable parameters.
        """
        if hasattr(self, "trainable_parameter_names"):
            for name, param in self.named_parameters():
                if name in self.trainable_parameter_names:
                    yield param
        else:
            yield from self.parameters()

    def named_trainable_parameters(self) -> torch.Generator:
        """
        Yield trainable parameters of the model. Refers to 'trainable_parameter_names' attribute if defined, else `named_parameters`.

        Returns:
            torch.Generator: Generator yielding tuples of parameter names and trainable parameters.
        """
        if hasattr(self, "trainable_parameter_names"):
            for name, param in self.named_parameters():
                if name in self.trainable_parameter_names:
                    yield name, param
        else:
            yield from self.named_parameters()

    def print_trainable_parameter_names(self) -> None:
        """
        Print the names of trainable and fixed parameters in the model.
        """
        if not hasattr(self, "trainable_parameter_names"):
            self.set_trainable_parameters()

        trainable, fixed = [], []
        for name, param in self.named_parameters():
            if name in self.trainable_parameter_names:
                trainable += [name]
            else:
                fixed += [name]

        logger.info(f"Trainable Parameters:\n\t{trainable}")
        logger.info(f"Fixed Parameters:\n\t{fixed}")

    def log_param_stats(
        self,
        section: str = "params",
        metrics: List[str] = ["min", "max", "norm"],
        log_only_trainable: bool = False,
    ) -> None:
        """
        Log statistics of model parameters.

        Args:
            section (str, optional): Section name for logging. Defaults to "params".
            metrics (List[str], optional): List of metrics to log. Defaults to ["min", "max", "norm"].
            log_only_trainable (bool, optional): Whether to log only trainable parameters. Defaults to False.
        """
        for name, param in self.named_parameters():
            if log_only_trainable and not param.requires_grad:
                continue

            if len(param.data.size()):
                for metric in metrics:
                    if hasattr(param.data, metric):
                        self.log(
                            f"{section}/{name}_{metric}",
                            getattr(param.data, metric)(),
                            sync_dist=True,
                        )
                    elif metric == "full":
                        self.log(
                            f"{section}/{name}_{metric}",
                            param,
                            sync_dist=True,
                        )
                    else:
                        logger.debug(f"Metric {metric} not available!")
            else:
                self.log(f"{section}/{name}", param.data, sync_dist=True)

    def get_dataframe(
        self, layer_name: str = defaults.classifier_name
    ) -> pd.DataFrame:
        """
        Generate a DataFrame containing classifier responses and associated metadata.

        Args:
            layer_name (str, optional): Name of the classifier layer. Defaults to `defaults.classifier_name`.

        Returns:
            pd.DataFrame: DataFrame containing classifier responses and metadata.
        """
        if hasattr(self, "responses"):
            response = self.responses[layer_name]
        else:
            logger.warning("No responses stored!")
            return pd.DataFrame()

        if isinstance(self.label_indices, list):
            label_indices = torch.cat(self.label_indices)
            guess_indices = torch.cat(self.guess_indices)
            image_indices = torch.cat(self.image_indices)
        else:
            label_indices = self.label_indices
            guess_indices = self.guess_indices
            image_indices = self.image_indices

        # cast to cpu and numpy
        valid_data_length = min(len(response), len(label_indices))
        response = response[-valid_data_length:].cpu().float().numpy()
        label_indices = label_indices[-valid_data_length:].cpu().numpy()
        guess_indices = guess_indices[-valid_data_length:].cpu().numpy()
        image_indices = image_indices[-valid_data_length:].cpu().numpy()

        n_samples, n_timesteps, n_classes = response.shape
        sample_indices, times_indices, class_indices = np.meshgrid(
            np.arange(n_samples),
            np.arange(n_timesteps),
            np.arange(n_classes),
            indexing="ij",
        )

        label_sets = np.array(["".join(row.astype(str)) for row in label_indices])
        df = pd.DataFrame(
            {
                "sample_index": sample_indices.ravel(),
                "times_index": times_indices.ravel(),
                "class_index": class_indices.ravel(),
                "response": response.ravel(),
                "label_index": label_indices.ravel().repeat(n_classes),
                "guess_index": guess_indices.ravel().repeat(n_classes),
                "image_index": image_indices.ravel().repeat(n_classes),
                "label_set": label_sets.repeat(n_classes * n_timesteps),
            }
        )
        del (
            response,
            label_indices,
            guess_indices,
            image_indices,
            sample_indices,
            times_indices,
            class_indices,
        )
        return df

    def _concatenate_responses(self) -> None:
        """
        Concatenate stored responses and results, using the int value of the model's attribute 'store_responses' to limit the maximum number of stored responses.
        """
        # Set the maximum number of responses to store (due to memory constraints)
        key = next(iter(self.responses))
        batch_size = self.responses[key][0].shape[0]

        if int(self.store_responses) > 1:
            max_index = int(self.store_responses) // batch_size
            max_index = max(1, max_index)
        else:
            max_index = len(self.responses[key])

        result_attributes = [
            "guess_indices",
            "label_indices",
            "image_indices",
            "times_indices",
        ]

        # Store task results
        for attr_name in result_attributes:
            attribute = getattr(self, attr_name)
            attribute = attribute if isinstance(attribute, list) else [attribute]
            for i, a in enumerate(attribute):
                if a is not None:
                    attribute[i] = a.cpu()

            attribute = self._concatenate_tensors(attribute[-max_index:], dim=0)
            setattr(self, attr_name, attribute)

        # Store task responses
        for layer_name, response in self.responses.items():
            response = response if isinstance(response, list) else [response]
            for i, r in enumerate(response):
                if r is not None:
                    response[i] = r.cpu()

            response = self._concatenate_tensors(response[-max_index:], dim=0)
            self.responses[layer_name] = response

    def _concatenate_tensors(
        self, tensors: List[Optional[torch.Tensor]], dim: int = 1
    ) -> Optional[torch.Tensor]:
        """
        Concatenate a list of tensors along a given dimension.

        Args:
            tensors (List[Optional[torch.Tensor]]): List of tensors to concatenate.
            dim (int, optional): Dimension along which to concatenate. Defaults to 1.

        Returns:
            Optional[torch.Tensor]: Concatenated tensor or None if input is None.
        """
        if tensors is None:
            return None

        # determine common tensor shape
        shape = None
        device = self.device
        for tensor in tensors:
            if tensor is not None:
                shape = tensor.shape
                device = tensor.device
                break

        if shape is None:
            logger.warning("No tensors to concatenate!")
            return None

        for i, tensor in enumerate(tensors):
            if tensor is None:
                tensors[i] = torch.zeros(shape, device=device)
            elif tensor.device != device:
                tensors[i] = tensor.to(device)

        concatenated_tensor = torch.cat(tensors, dim=dim)
        return concatenated_tensor

    def _unsqueeze_tensor(
        self, tensor: Optional[torch.Tensor], dim: int = 1
    ) -> Optional[torch.Tensor]:
        """
        Add a singleton dimension to a tensor.

        Args:
            tensor (Optional[torch.Tensor]): Input tensor.
            dim (int, optional): Dimension to add. Defaults to 1.

        Returns:
            Optional[torch.Tensor]: Tensor with added dimension or None if input is None.
        """
        if tensor is not None:
            tensor = tensor.unsqueeze(dim)
        return tensor

    def _check_tensors(
        self,
        generator_name: str = "named_parameters",
        data_attr: str = "data",
        raise_error: bool = False,
    ) -> None:
        """
        Check for NaN/Inf values and dtype consistency in model parameters.
        """
        iterator = getattr(self, generator_name)
        if isinstance(iterator, dict):
            iterator = iterator.items()
        elif callable(iterator):
            iterator = iterator()
        else:
            raise ValueError(
                f"The attribute {generator_name} is neither a dict nor a generator."
            )

        nonfinite_detected = False
        model_dtype = next(self.parameters()).dtype
        dtype_mismatches = []

        logger.info(f"\nChecking {generator_name} {data_attr}:")
        logger.info("-" * 100)
        logger.info(
            f"{'Module Name':<30} {'Shape':<20} {'Type':<16} {'Device':<8} {'Min':>8} {'Max':>8} {'Norm':>8}"
        )
        logger.info("-" * 100)

        for name, tensor in iterator:
            if data_attr and self.hasattr(tensor, data_attr):
                tensor = self.getattr(tensor, data_attr)
            else:
                logger.debug(f"Attribute {data_attr} not found in {name}!")

            if isinstance(tensor, list):
                tensor = self._concatenate_tensors(tensor, dim=0)

            # Check dtype consistency
            if tensor.dtype != model_dtype:
                dtype_mismatches.append((name, tensor.dtype, model_dtype))

            if len(tensor.size()):
                valid_data = tensor[torch.isfinite(tensor)]
                if valid_data.numel() > 0:
                    shape_str = str(tensor.size()).replace("torch.Size", "")
                    logger.info(
                        f"{name:<30} {shape_str:<20} {str(tensor.dtype):<16} {str(tensor.device):<8} "
                        f"{valid_data.min().item():>8.3f} {valid_data.max().item():>8.3f} {valid_data.norm().item():>8.3f}"
                    )
                else:
                    logger.warning(
                        f"{name:<30} {'[NaN/Inf]':<20} {str(tensor.dtype):<16} {str(tensor.device):<8} {'---':>8} {'---':>8} {'---':>8}"
                    )
            else:
                logger.info(
                    f"{name:<30} {'[scalar]':<20} {str(tensor.dtype):<16} {str(tensor.device):<8} {tensor:>8.3f} {tensor:>8.3f} {tensor:>8.3f}"
                )

            if (torch.isnan(tensor)).any():
                logger.warning(f"\t NaN detected in {name}: ")
                logger.warning(
                    f"\t {(torch.isnan(tensor)).sum().item()} / {tensor.numel()}"
                )
            if torch.isinf(tensor).any():
                logger.warning(f"\t Inf detected in {name}: ")
                logger.warning(
                    f"\t {(torch.isinf(tensor)).sum().item()} / {tensor.numel()}"
                )
            if (~torch.isfinite(tensor)).any():
                nonfinite_detected = True

        # Report dtype mismatches
        if dtype_mismatches:
            logger.warning("Detected dtype mismatches:")
            for name, param_dtype, expected_dtype in dtype_mismatches:
                logger.warning(f"\t {name}: {param_dtype} (expected {expected_dtype})")

        if raise_error and (nonfinite_detected or dtype_mismatches):
            raise ValueError("NaN/Inf values or dtype mismatches detected!")

        return None

    def _check_gradients(self, raise_error: bool = False) -> None:
        self._check_tensors(
            generator_name="named_trainable_parameters",
            data_attr="grad.data",
            raise_error=raise_error,
        )

    def _check_weights(self, raise_error: bool = False) -> None:
        self._check_tensors(
            generator_name="named_parameters",
            data_attr="data",
            raise_error=raise_error,
        )

    def _check_responses(self, raise_error: bool = False) -> None:
        self._check_tensors(
            generator_name="get_responses",
            data_attr="",
            raise_error=raise_error,
        )

    def _clear_gpu_memory(self) -> None:
        """
        Clear GPU memory by emptying the cache and synchronizing.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        gc.collect()
        return None

    def safely_named_parameters(self, module=None, replace={".": "_"}):
        if module is None:
            module = self

        if isinstance(module, nn.Sequential):
            generator = (
                (f"Sequential.{child_name}.{param_name}", param)
                for child_name, child in module.named_children()
                if hasattr(child, "named_parameters")
                for param_name, param in child.named_parameters()
            )

        elif hasattr(module, "named_parameters"):
            generator = module.named_parameters()

        else:
            logger.warning(
                f"module {module} has no named parameters to safely rename!"
            )
            generator = iter([])

        for name, param in generator:
            for k, v in replace.items():
                name = name.replace(k, v)
            yield name, param

    def get_safely_named_parameters_dict(self, module=None, replace={".": "_"}):
        if module is None:
            module = self
        if not hasattr(module, "safely_named_parameters_dict"):
            setattr(
                module,
                "safely_named_parameters_dict",
                {
                    k: v
                    for k, v in self.safely_named_parameters(
                        module=module, replace=replace
                    )
                },
            )
        return module.safely_named_parameters_dict

    def _log_system_info(self) -> None:
        """Log essential system information at training start."""
        logger.info("=" * 60)
        logger.info("🚀 TRAINING STARTED")
        logger.info("=" * 60)

        # Model basics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"Model: {self.__class__.__name__} | Params: {total_params:,} ({trainable_params:,} trainable)"
        )
        logger.info(
            f"Config: {self.n_classes} classes | {self.n_timesteps} timesteps | non_label_idx: {self.non_label_index}"
        )

        # System info
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name()
            logger.info(f"Device: {device_name}")

    def _log_training_summary(self) -> None:
        """Log training completion summary."""
        try:
            logger.info(
                f"✅ Training completed: {self.trainer.current_epoch} epochs | {self.trainer.global_step} steps"
            )
        except RuntimeError:
            return

    def _validate_batch_data(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, stage: str
    ) -> None:
        """Validate batch data for critical issues."""
        inputs, labels = batch[:2]

        # Adjust dimensions for validation
        inputs = _adjust_data_dimensions(inputs)
        labels = _adjust_label_dimensions(labels)

        # Check for NaN/Inf in inputs
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            logger.warning(
                f"⚠️  [{stage.upper()}] Batch {batch_idx}: NaN/Inf detected in inputs"
            )

        # Validate label range
        label_min, label_max = labels.min().item(), labels.max().item()
        if label_min < 0 or label_max >= self.n_classes:
            invalid_labels = labels[(labels < 0) | (labels >= self.n_classes)]
            unique_invalid = torch.unique(invalid_labels).tolist()
            logger.warning(
                f"⚠️  [{stage.upper()}] Batch {batch_idx}: Invalid labels {unique_invalid} (expect 0-{self.n_classes-1})"
            )

        # Log first batch info
        if batch_idx == 0:
            logger.info(
                f"📦 [{stage.upper()}] First batch: {inputs.shape} | Labels: [{label_min}, {label_max}] | Device: {inputs.device}"
            )

    def _check_training_health(self, loss: torch.Tensor, batch_idx: int) -> None:
        """Check for training health issues."""
        if loss is None:
            return

        # Check for NaN/Inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(
                f"⚠️  Batch {batch_idx}: Loss is {'NaN' if torch.isnan(loss) else 'Inf'}"
            )

        # Check for extremely high loss
        loss_val = loss.item() if hasattr(loss, "item") else float(loss)
        if loss_val > 100:
            logger.warning(f"⚠️  Batch {batch_idx}: Very high loss {loss_val:.4f}")

    def _should_log_detailed(self, batch_idx: int, stage: str) -> bool:
        """Determine if detailed logging should occur."""
        try:
            epoch = self.trainer.current_epoch
            # Log more frequently early in training
            if epoch < 2:
                return batch_idx % 20 == 0
            elif epoch < 5:
                return batch_idx % 50 == 0
            else:
                return batch_idx % 100 == 0
        except RuntimeError:
            return batch_idx < 5


class LightningBase(UtilityBase, pl.LightningModule, DtypeDeviceCoordinatorMixin):
    """
    A base class for PyTorch Lightning models, extending UtilityBase.
    Provides additional functionality for training, validation, and testing.
    """

    @alias_kwargs(
        trc="t_recurrence",
        tff="t_feedforward",
        rctype="recurrence_type",
        lr="learning_rate",
        solver="dynamics_solver",
        taugrad="train_tau",
        lossrt="loss_reaction_time",
    )
    def __init__(
        self,
        input_dims: Tuple[int] = (20, 3, 224, 224),
        retain_graph: bool = defaults.retain_graph,
        store_responses: int = defaults.store_train_responses,
        criterion_params: List[Tuple[str, Dict[str, Any]]] = [
            (loss, defaults.loss_configs[loss]) for loss in defaults.loss
        ],
        loss_reaction_time: float = defaults.loss_reaction_time,
        n_timesteps: int = 1,
        dt: float = defaults.dt,
        tau: float = defaults.tau,
        t_feedforward: float = defaults.t_feedforward,
        t_recurrence: float = defaults.t_recurrence,
        classifier_name: str = defaults.classifier_name,
        store_responses_on_cpu: bool = defaults.store_responses_on_cpu,
        non_label_index: int = defaults.non_label_index,
        optimizer: str = defaults.optimizer,
        optimizer_kwargs: Dict[str, Any] = defaults.optimizer_kwargs,
        optimizer_configs: Dict[str, Dict[str, Any]] = defaults.optimizer_configs,
        learning_rate: float = defaults.learning_rate,
        lr_parameter_groups: Dict[str, Dict[str, Any]] = defaults.lr_parameter_groups,
        scheduler: str = defaults.scheduler,
        scheduler_kwargs: Dict[str, Any] = defaults.scheduler_kwargs,
        scheduler_configs: Dict[str, Dict[str, Any]] = defaults.scheduler_configs,
        log_level: str = defaults.log_level,
        log_every_n_steps: int = defaults.log_every_n_steps,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # Store the arguments as attributes
        for name, value in kwargs.items():
            if name != "self":
                setattr(self, name, value)
        self.retain_graph = retain_graph
        self.lr_parameter_groups = lr_parameter_groups
        self.log_level = log_level
        self.criterion_params = criterion_params
        self.store_responses = store_responses
        self.learning_rate = float(learning_rate)
        self.optimizer = optimizer
        self.dt = float(dt)
        self.tau = float(tau)
        self.t_feedforward = float(t_feedforward)
        self.t_recurrence = float(t_recurrence)
        self.loss_reaction_time = float(loss_reaction_time)
        self.classifier_name = classifier_name
        self.store_responses_on_cpu = store_responses_on_cpu
        self.non_label_index = non_label_index
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_configs = optimizer_configs
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs
        self.scheduler_configs = scheduler_configs
        self.log_every_n_steps = int(log_every_n_steps)

        self.is_root_node = True  # Main model is always root

        # Process the input dims and timesteps
        x = self.create_aligned_tensor(size=(1, *input_dims), creation_method="randn")
        x = _adjust_data_dimensions(x)
        _, data_timesteps, self.n_channels, self.dim_y, self.dim_x = x.shape

        if data_timesteps == 1:
            self.n_timesteps = n_timesteps
        elif n_timesteps == 1:
            self.n_timesteps = data_timesteps
        else:
            self.n_timesteps = max(n_timesteps, data_timesteps)
            logger.warning(
                f"The model is initialized with {n_timesteps} timesteps. "
                f"The provided data dimensions have {data_timesteps} timesteps. "
                f"Choosing the larger number of timesteps!"
            )
        self.input_dims = (self.n_timesteps, self.n_channels, self.dim_y, self.dim_x)

        self._init_loss(self.criterion_params)

        self.save_hyperparameters()  # redundant with logger?

    def setup(self, stage: Optional[str]) -> None:
        """
        Set up the model for training or evaluation.

        Args:
            stage (Optional[str]): Stage of the setup process (e.g., "fit", "test").
        """
        if not hasattr(self, "n_residual_timesteps"):
            self.set_residual_timesteps()
        self.reset()

    def _define_architecture(self) -> None:
        raise NotImplementedError("Define the model architecture!")

    def _init_parameters(self) -> None:
        logger.info("Initializing model parameters...")
        for module in self.children():
            if hasattr(module, "_init_parameters"):
                module._init_parameters()

    def verify_initialization(self) -> None:
        pass

    def _forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        feedforward_only: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the model for a single time step. This is a general formulation of the time step operation that uses the model's attributes layer_names and layer_operations to handle the execution of the operations in each layer when they are named according to the following convention: {operation}_{layer_name}, or {operation} if the operation is not layer-specific.
        The main layer computation (e.g. a convolution) can just named {layer_name}.
        Reserved layer operations that are available are 'record' and 'delay' (given that the layer has a set_hidden_state and get_hidden_state method).

        Args:
            x (torch.Tensor): Input tensor for time step t.
            t (Optional[torch.Tensor], optional): Time step index. Defaults to None.
            feedforward_only (bool, optional): Whether to perform only feedforward operations. Defaults to False.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Model output and dictionary of responses.
        """
        batch_size, n_channels, y_dim, x_dim = x.shape

        if hasattr(self, "input_adaption"):
            x = self.input_adaption(x)

        responses = {}
        if not hasattr(self, "layer_operations"):
            # define default operations order within layer
            self.layer_operations = [
                "layer",  # apply (recurrent) convolutional layer
                "ext",  # add external input
                "skip",  # add skip connection
                "feedback",  # add feedback connection
                "tstep",  # apply dynamical systems ode solver step
                "nonlin",  # apply nonlinearity
                "supralin",  # apply supralinearity
                "record",  # record activations in responses dict
                "delay",  # set and get delayed activations for next layer
                "pool",  # apply pooling
                "norm",  # apply normalization
            ]

        for layer_name in self.layer_names:

            layer = getattr(self, layer_name)

            for operation in self.layer_operations:

                if feedforward_only and operation in [
                    "addskip",
                    "addext",
                    "addfeedback",
                ]:
                    continue

                module_name = f"{operation}_{layer_name}"

                if operation == "layer":
                    if hasattr(layer, "_get_name"):
                        module_name = layer._get_name()
                    else:
                        module_name = "layer"
                    x = layer(x)

                elif operation == "record":
                    responses[layer_name] = x

                elif operation == "delay" and hasattr(layer, "set_hidden_state"):
                    layer.set_hidden_state(x)
                    x = layer.get_hidden_state(0)

                elif operation == "tstep" and hasattr(self, module_name):
                    module = getattr(self, module_name)
                    h = layer.get_hidden_state(-1)
                    x = module(x, h)

                # apply layer operations (if defined)
                elif hasattr(self, module_name) and x is not None:
                    module = getattr(self, module_name)
                    x = module(x)

                elif hasattr(self, operation) and x is not None:
                    module = getattr(self, operation)
                    x = module(x)

                else:
                    pass

                if x is not None and (~torch.isfinite(x)).any():
                    logger.warning(
                        f"NaN/inf detected in {module_name} output! \n\t {(~torch.isfinite(x)).sum()}/{x.numel()} NaNs"
                    )

        if x is None:
            x = torch.zeros((batch_size, self.n_classes), device=self.device)
        else:
            classifier = getattr(self, self.classifier_name)
            x = classifier(x)

        responses[self.classifier_name] = x

        return x, responses

    def forward(
        self,
        x_0: torch.Tensor,
        store_responses: bool = False,
        feedforward_only: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x_0 (torch.Tensor): Input tensor.
            store_responses (bool, optional): Whether to store responses. Defaults to False.
            feedforward_only (bool, optional): Whether to perform only feedforward operations. Defaults to False.

        Returns:
            torch.Tensor: Model outputs.
        """
        store_responses = store_responses if store_responses else self.store_responses

        x_0 = _adjust_data_dimensions(x_0)
        batch_size, n_timesteps, dim_channels, dim_y, dim_x = x_0.shape

        outputs = torch.zeros(
            (batch_size, n_timesteps, self.n_classes), device=x_0.device
        )

        if hasattr(self, "reset"):
            self.reset()

        # Forward the model over all timesteps
        for t in torch.arange(n_timesteps, device=x_0.device):
            x = x_0[:, t, ...]

            x, hidden_state = self._forward(x, t, feedforward_only=feedforward_only)

            if x is not None:
                outputs[:, t, :] = x

            if store_responses:
                self._store_responses(
                    hidden_state, store_n_responses=int(store_responses), t=t
                )

        del x, hidden_state
        return outputs

    def _init_outputs(self) -> None:
        self.guess_indices = []
        self.label_indices = []
        self.image_indices = []
        self.times_indices = []

    def _init_responses(
        self,
        response_dict: Dict[str, torch.Tensor] = {},
        store_n_responses: Optional[int] = None,
    ) -> None:
        logger.info("Initializing responses")

        if store_n_responses is None:
            store_n_responses = int(self.store_responses)

        if not store_n_responses or hasattr(self, "responses"):
            return

        if self.store_responses_on_cpu:
            device = "cpu"
        else:
            device = self.device

        self.responses = {}
        self.reset()

        if not response_dict or any(v is None for v in response_dict.values()):
            random_input = self.create_aligned_tensor(
                size=(1, self.n_channels, self.dim_y, self.dim_x),
                creation_method="randn",
            )
            while any(v is None for v in response_dict.values()):
                _, response_dict = self._forward(random_input, feedforward_only=True)

        if hasattr(self, "n_residual_timesteps"):
            n_timesteps = self.n_timesteps + self.n_residual_timesteps
        else:
            n_timesteps = self.n_timesteps

        for layer_name in response_dict.keys():
            self.responses[layer_name] = torch.zeros(
                (
                    store_n_responses,
                    n_timesteps,
                    *response_dict[layer_name].shape[1:],
                ),
                device=device,
            )
        self.reset()
        return None

    def _store_responses(
        self,
        response_dict: Dict[str, torch.Tensor] = {},
        store_n_responses: Optional[int] = None,
        t: Optional[int] = None,
    ) -> None:
        if not hasattr(self, "responses"):
            self._init_responses(response_dict, store_n_responses=store_n_responses)

        if not response_dict:
            logger.warning("No responses to update!")
            return None

        for layer_name, response in response_dict.items():
            if response is None:
                continue
            elif response.device != self.responses[layer_name].device:
                response = response.to(self.responses[layer_name].device)
            else:
                pass

            response = response.detach()
            batch_size = response.shape[0]
            layer_response_size = self.responses[layer_name].shape[0]

            if t is None:
                if layer_response_size > batch_size:
                    # shift the responses to the left (mimic deque behavior)
                    response_history = self.responses[layer_name][batch_size:, ...]
                    self.responses[layer_name][:batch_size, ...] = response_history
                    self.responses[layer_name][-batch_size:, ...] = response
                else:
                    self.responses[layer_name][:, ...] = response[
                        -layer_response_size:
                    ]
            else:
                if layer_response_size > batch_size:
                    # shift the responses to the left (mimic deque behavior)
                    response_history = self.responses[layer_name][batch_size:, t, ...]
                    self.responses[layer_name][:-batch_size, t, ...] = response_history
                    self.responses[layer_name][-batch_size:, t, ...] = response
                else:
                    self.responses[layer_name][:, t, ...] = response[
                        -layer_response_size:
                    ]
        del response
        return None

    def get_responses(self) -> Dict[str, torch.Tensor]:
        """
        Get the stored responses of the model.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of stored responses.
        """
        if hasattr(self, "responses"):
            return self.responses
        else:
            logger.warning("No responses stored!")
            return {}

    def set_responses(self, responses: Dict[str, torch.Tensor]) -> None:
        """
        Set the stored responses of the model.

        Args:
            responses (Dict[str, torch.Tensor]): Dictionary of responses to set.
        """
        if hasattr(self, "responses"):
            logger.warning("Overwriting stored responses!")
        self.responses = responses
        return None

    def predictor(self, outputs: torch.Tensor) -> torch.Tensor:
        return torch.argmax(outputs, dim=-1)

    def _expand_timesteps(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, label_indices, *extra = batch
        inputs = _adjust_data_dimensions(inputs)
        label_indices = _adjust_label_dimensions(label_indices)

        if inputs.size(1) == 1 and self.n_timesteps > 1:
            # input data is not yet extended
            inputs = inputs.expand(-1, self.n_timesteps, -1, -1, -1)
            label_indices = label_indices.expand(-1, self.n_timesteps)

        return (inputs, label_indices, *extra)

    def _extend_residual_timesteps(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, label_indices, *extra = batch

        inputs = _adjust_data_dimensions(inputs)
        label_indices = _adjust_label_dimensions(label_indices)

        if self.n_residual_timesteps > 0:
            # add 0s at the end as inputs for residual timesteps
            new_shape = (
                inputs.size(0),
                self.n_timesteps + self.n_residual_timesteps,
                *inputs.shape[2:],
            )
            new_inputs = torch.zeros(
                new_shape, device=inputs.device, dtype=inputs.dtype
            )
            new_inputs[:, : self.n_timesteps, ...] = inputs

            # add voidid buffer labels at the beginning for residual timesteps

            new_shape = (inputs.size(0), self.n_timesteps + self.n_residual_timesteps)
            new_label_indices = torch.full(
                new_shape,
                self.non_label_index,
                device=label_indices.device,
                dtype=label_indices.dtype,
            )
            new_label_indices[:, -self.n_timesteps :] = label_indices

            batch = (new_inputs, new_label_indices, *extra)

        else:

            batch = (inputs, label_indices, *extra)

        return batch

    def model_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        store_responses: bool = False,
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:

        batch = self._expand_timesteps(batch)
        inputs, label_indices, *paths = self._extend_residual_timesteps(batch)

        # forward
        # (you may override the model's store_response setting here)
        outputs = self.forward(inputs, store_responses=store_responses)

        # calculate loss
        loss = self.compute_loss(
            outputs,
            responses=self.responses if hasattr(self, "responses") else None,
            label_indices=label_indices,
            reaction_time=self.loss_reaction_time,
        )

        # calculate accuracy
        guess_indices = self.predictor(outputs)
        accuracy = self.calc_accuracy(label_indices, guess_indices)

        del outputs
        return loss, accuracy, guess_indices

    def _init_loss(self, criterion_params: List[Tuple[str, Dict[str, Any]]]) -> None:
        if hasattr(self, "criterion"):
            if isinstance(self.criterion, list) and (
                isinstance(self.criterion[0], nn.Module)
                or isinstance(self.criterion[0], tuple)
            ):
                # already initialized
                return None
            elif self.criterion is None:
                pass
            else:
                raise ValueError(
                    "self.criterion should be a list of loss functions or tuples of loss functions and weights"
                )

        self.criterion = []

        if not isinstance(criterion_params, list):
            criterion_params = [criterion_params]

        for criterion_name, criterion_config in criterion_params:
            if "weight" in criterion_config.keys():
                criterion_weight = criterion_config.pop("weight")
            else:
                criterion_weight = 1

            # Add ignore_index for cross entropy loss if not specified
            if (
                criterion_name.lower() in ["crossentropyloss", "cross_entropy_loss"]
                and "ignore_index" not in criterion_config
            ):
                criterion_config["ignore_index"] = self.non_label_index
                logger.info(
                    f"Setting ignore_index={self.non_label_index} for {criterion_name}"
                )

            logger.info(
                f"Criterion: {criterion_name} with weight: {criterion_weight} and config: {criterion_config}"
            )

            if hasattr(losses, criterion_name):
                criterion_fn = getattr(losses, criterion_name)(**criterion_config)
                self.criterion += [(criterion_fn, criterion_weight)]
            else:
                raise ValueError(f"Invalid loss function: {criterion_name}")

        return None

    def compute_loss(
        self,
        outputs: torch.Tensor,
        label_indices: torch.Tensor,
        responses: Optional[Dict[str, torch.Tensor]] = None,
        reaction_time: float = 0,
    ) -> torch.Tensor:

        *_, n_classes = outputs.shape

        if reaction_time:
            reaction_timesteps = self.n_residual_timesteps + int(
                reaction_time / self.dt
            )
            outputs = (
                outputs[:, reaction_timesteps:, :].contiguous().view(-1, n_classes)
            )
            label_indices = label_indices[:, reaction_timesteps:].contiguous().view(-1)
        else:
            outputs = outputs.view(-1, n_classes)
            label_indices = label_indices.view(-1)

        # Quick validation (minimal overhead)
        invalid_mask = (label_indices < 0) | (label_indices >= n_classes)
        if invalid_mask.any():
            valid_mask = ~invalid_mask
            if valid_mask.any():
                outputs = outputs[valid_mask]
                label_indices = label_indices[valid_mask]
            else:
                logger.warning("All labels invalid, returning zero loss")
                return torch.tensor(0.0, device=outputs.device, requires_grad=True)

        loss_values = torch.zeros(len(self.criterion), device=outputs.device)
        for i, criterion_fn in enumerate(self.criterion):
            if isinstance(criterion_fn, tuple):
                criterion_fn, weight = criterion_fn
            else:
                weight = 1

            loss_value = weight * criterion_fn((outputs, responses), label_indices)
            loss_values[i] = loss_value

        loss = loss_values.sum()

        # Quick NaN check (minimal overhead)
        if torch.isnan(loss):
            logger.warning(f"⚠️  NaN loss detected")

        return loss

    def calc_accuracy(
        self,
        label_indices: torch.Tensor,
        guess_indices: torch.Tensor,
    ) -> float:
        # Create mask for valid labels (excluding non_label_index)
        valid_mask = (
            (label_indices >= 0)
            & (label_indices != self.non_label_index)
            & (label_indices < self.n_classes)
        )

        if valid_mask.sum() == 0:
            # No valid labels, return 0 accuracy
            logger.warning("No valid labels found for accuracy calculation")
            return 0.0

        accuracy = (
            (guess_indices[valid_mask] == label_indices[valid_mask])
            .float()
            .mean()
            .item()
        )
        return accuracy

    def backward(self, loss: torch.Tensor) -> None:
        if self.retain_graph:
            loss.backward(retain_graph=True)
        else:
            try:
                loss.backward()
            except RuntimeError as e:
                if "retain_graph" in str(e) or "computation graph" in str(e):
                    logger.warning(f"Retrying with retain_graph=True due to: {e}")
                    loss.backward(retain_graph=True)
                    self.retain_graph = True
                else:
                    raise e

    def _shared_eval_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        store_responses: bool = False,
    ) -> Tuple[torch.Tensor, float]:

        batch = self._expand_timesteps(batch)
        inputs, label_indices, *paths = self._extend_residual_timesteps(batch)

        loss, accuracy, guess_indices = self.model_step(
            batch, batch_idx, store_responses
        )

        self._store_outputs(batch, batch_idx, guess_indices, label_indices, *paths)

        return loss, accuracy

    def _store_outputs(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        guess_indices: torch.Tensor,
        label_indices: torch.Tensor,
        *extra: Any,
    ) -> None:

        batch_size, n_timesteps = guess_indices.shape

        self.guess_indices.append(guess_indices.detach().cpu())
        self.label_indices.append(label_indices.detach().cpu())

        if extra and (isinstance(extra[0], str) or isinstance(extra[0], Path)):
            image_indices = torch.tensor(
                [path_to_index(path) for path in extra], device="cpu"
            )
        else:
            image_indices = (
                torch.arange(batch_size, device="cpu") + batch_idx * batch_size
            )

        image_indices = image_indices.unsqueeze(1).expand(batch_size, n_timesteps)
        self.image_indices.append(image_indices)

        times_indices = torch.arange(n_timesteps, device="cpu")
        times_indices = times_indices.unsqueeze(0).expand(batch_size, n_timesteps)
        self.times_indices.append(times_indices)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Perform a single training step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Batch of input data and labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Training loss.
        """
        batch_size = batch[0].size(0)
        loss, accuracy, *_ = self.model_step(batch, batch_idx, store_responses=False)

        metrics = {"train_loss": loss, "train_accuracy": accuracy}
        self.log_dict(metrics, prog_bar=True, batch_size=batch_size, sync_dist=True)
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        store_responses=None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Perform a single validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Batch of input data and labels.
            batch_idx (int): Index of the batch.

        Returns:
            Tuple[torch.Tensor, float]: Validation loss and accuracy.
        """
        # Memory before processing
        ###################33
        import psutil

        process = psutil.Process(os.getpid())

        mem_before = process.memory_info().rss / 1024**2
        gpu_before = (
            torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        )

        logger.warning(
            f"VAL STEP START - Batch {batch_idx}: RAM: {mem_before:.0f}MB, GPU: {gpu_before:.0f}MB"
        )
        ###################33

        if store_responses is None:
            store_responses = self.store_responses

        batch_size = batch[0].size(0)
        loss, accuracy = self._shared_eval_step(
            batch, batch_idx, store_responses=store_responses
        )

        # Memory after processing
        ############33333
        mem_after = process.memory_info().rss / 1024**2
        gpu_after = (
            torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        )
        mem_increase = mem_after - mem_before
        gpu_increase = gpu_after - gpu_before

        logger.warning(
            f"VAL STEP END - Batch {batch_idx}: RAM: +{mem_increase:.0f}MB, GPU: +{gpu_increase:.0f}MB"
        )

        # Emergency cleanup if memory usage is too high
        if mem_increase > 1000:  # More than 1GB increase
            logger.error(
                f"🚨 EMERGENCY: {mem_increase:.0f}MB increase in validation step!"
            )
            self._clear_gpu_memory()
            import gc

            gc.collect()
        ###################33

        metrics = {"val_loss": loss, "val_accuracy": accuracy}
        self.log_dict(metrics, prog_bar=True, batch_size=batch_size, sync_dist=True)

        self._clear_gpu_memory()
        return loss, accuracy

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, float]:
        """
        Perform a single test step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Batch of input data and labels.
            batch_idx (int): Index of the batch.

        Returns:
            Tuple[torch.Tensor, float]: Test loss and accuracy.
        """
        batch_size = batch[0].size(0)
        loss, accuracy = self._shared_eval_step(
            batch, batch_idx, store_responses=self.store_responses
        )

        metrics = {"test_loss": loss, "test_accuracy": accuracy}
        self.log_dict(
            metrics, prog_bar=True, on_step=True, batch_size=batch_size, sync_dist=True
        )
        return loss, accuracy

    def optimizer_step(self, *args: Any, **kwargs: Any) -> None:
        super().optimizer_step(*args, **kwargs)
        if self.log_level.upper() == "DEBUG":
            with torch.no_grad():
                self._check_gradients()
                self._check_weights(raise_error=True)

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

    ### HOOKS

    def on_train_start(self) -> None:
        """Initialize training with weight check and basic system info."""
        try:
            super().on_train_start()
        except AttributeError:
            pass

        self._check_weights()
        self._log_system_info()

    def on_train_end(self) -> None:
        """Final diagnostics and cleanup."""
        self._check_responses()
        self._log_training_summary()

    def on_train_batch_start(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Check for critical data issues early in training."""
        if batch_idx < 3:  # Only check first few batches
            self._validate_batch_data(batch, batch_idx, "train")

    def on_validation_batch_start(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Check for critical data issues in validation."""
        if batch_idx == 0:  # Only check first validation batch
            self._validate_batch_data(batch, batch_idx, "val")

    def on_train_batch_end(
        self, outputs: Any, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Monitor training progress and check for issues."""
        loss = outputs if isinstance(outputs, torch.Tensor) else outputs.get("loss")
        self._check_training_health(loss, batch_idx)

        if batch_idx % self.log_every_n_steps == 0:
            self.log_param_stats()

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        """Check gradients before optimizer step."""
        if self.log_level.upper() == "DEBUG":
            self._check_gradients()

    def on_test_start(self) -> None:
        self._init_outputs()

    def on_validation_start(self) -> None:
        self._init_outputs()

    def on_validation_epoch_end(self) -> None:
        self._init_outputs()

    def _group_lr_parameters(self, base_lr: float) -> List[Dict[str, Any]]:
        """Group parameters with appropriate learning rates and gradient clipping.

        This helper method organizes parameters into groups based on their type
        (regular, recurrent, or feedback) and assigns appropriate learning rates
        and gradient clipping settings to each group.

        Args:
            base_lr: The base learning rate after batch size scaling
            recurrence_factor: Factor to scale recurrent weights' learning rate
            feedback_factor: Factor to scale feedback weights' learning rate

        Returns:
            List of parameter group dictionaries for the optimizer
        """
        params = {key: [] for key in self.lr_parameter_groups.keys()}
        params["regular"] = []

        # Categorize parameters
        for name, param in self.named_trainable_parameters():
            for key in params.keys():
                if key != "regular" and key in name:
                    params[key].append(param)
                    break
            else:
                params["regular"].append(param)

        param_groups = []

        for group_name, group_params in params.items():

            if group_params:  # Only create groups with parameters
                group_config = self.lr_parameter_groups.get(group_name, {})
                group_config["name"] = group_name
                group_config["params"] = group_params

                # Set learning rate based on group configuration
                lr_factor = group_config.pop("lr_factor", 1.0)
                if not "lr" in group_config:
                    group_config["lr"] = base_lr * lr_factor

                param_groups.append(group_config)

        return param_groups

    def _get_base_learning_rate(self) -> float:
        """Get the base learning rate from model attributes.

        Returns:
            float: Base learning rate value
        """
        base_lr = getattr(self, "lr", getattr(self, "learning_rate", None))
        if base_lr is None:
            raise ValueError("No learning rate specified in model attributes")
        return base_lr

    def _create_optimizer(self, scaled_lr: float) -> torch.optim.Optimizer:
        """Create and configure the optimizer.

        Args:
            scaled_lr: Scaled learning rate
            recurrence_lr_factor: Factor for scaling recurrent weights' learning rate
            feedback_lr_factor: Factor for scaling feedback weights' learning rate

        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        if isinstance(self.optimizer, str):
            if not hasattr(torch.optim, self.optimizer):
                raise ValueError(f"Unknown optimizer: {self.optimizer}")

            param_groups = self._group_lr_parameters(scaled_lr)

            optimizer_class = getattr(torch.optim, self.optimizer)

            optimizer = optimizer_class(param_groups, **self.optimizer_kwargs)
            return optimizer
        else:
            return self.optimizer

    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """Create and configure the learning rate scheduler.

        Args:
            optimizer: The optimizer to schedule

        Returns:
            Dict[str, Any]: Scheduler configuration
        """
        if not hasattr(torch.optim.lr_scheduler, self.scheduler):
            raise ValueError(f"Unknown scheduler: {self.scheduler}")

        scheduler = getattr(torch.optim.lr_scheduler, self.scheduler)(
            optimizer, **self.scheduler_kwargs
        )

        return {
            "scheduler": scheduler,
            **self.scheduler_configs,
        }

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers.

        This is a standard PyTorch Lightning hook that sets up the optimizer
        with appropriate parameter groups and learning rates. Learning rate scaling
        based on batch size is handled by the trainer configuration.

        Returns:
            Dict containing optimizer and scheduler configurations

        Raises:
            ValueError: If required optimizer or scheduler configurations are invalid
        """
        self.print_trainable_parameter_names()

        base_lr = self._get_base_learning_rate()

        optimizer = self._create_optimizer(base_lr)

        lr_scheduler_config = self._create_scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
            **self.optimizer_configs,
        }

    def log_table(
        self,
        key: str,
        columns: Optional[List[str]] = None,
        data: Optional[List[List[Any]]] = None,
        dataframe: Optional[pd.DataFrame] = None,
        step: Optional[int] = None,
    ) -> None:
        if self.logger:
            self.logger.log_table(key=key, dataframe=dataframe, step=step)

    def log_figure(
        self, fig: plt.Figure, key: str, step: Optional[int] = None
    ) -> None:
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plt.close()
        self.log(
            {f"{key}", wandb.Image(buffer, caption=key)},
            step=step,
            rank_zero_only=True,
        )


if __name__ == "__main__":
    input_shape = (1, 28, 28)

    model = LightningBase(
        input_dims=input_shape, n_timesteps=8, n_classes=10, cumulative_readout=True
    )

    random_input = torch.randn(2, *input_shape)

    outputs = model(random_input)

    print(f"Random Input ({random_input.shape})")
    print(f"Model Output Timesteps: {len(outputs)}")
    for t, output in enumerate(outputs):
        print(f"\tt {t}: ({output.shape}): \t{output}")
