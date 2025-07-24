from copy import copy
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dynvision.data.operations import _adjust_data_dimensions, _adjust_label_dimensions


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
