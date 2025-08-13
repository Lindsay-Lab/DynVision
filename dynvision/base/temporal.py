"""Core neural network functionality for biologically-inspired models."""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
from copy import copy
import logging

from dynvision.data.operations import _adjust_data_dimensions, _adjust_label_dimensions
from dynvision.utils import alias_kwargs
from .storage import DataBuffer

logger = logging.getLogger(__name__)


class TemporalBase(nn.Module):
    """Core neural network functionality for temporal dynamics models."""

    @alias_kwargs(
        trc="t_recurrence",
        tff="t_feedforward",
        tfb="t_feedback",
        tsk="t_skip",
        pattern="data_presentation_pattern",
        rctype="recurrence_type",
        rctarget="recurrence_target",
        solver="dynamics_solver",
    )
    def __init__(
        self,
        # Core model architecture
        input_dims: Tuple[int] = (20, 3, 224, 224),
        n_timesteps: int = 1,
        n_classes: int = 1000,
        # Temporal dynamics
        dt: float = 1.0,
        tau: float = 4.0,
        t_feedforward: float = 0.0,
        t_recurrence: float = 3.0,
        t_feedback: Optional[float] = None,
        t_skip: Optional[float] = None,
        data_presentation_pattern: Union[List[int], str] = [1],
        # Architecture configuration
        classifier_name: str = "classifier",
        dynamics_solver: str = "euler",
        recurrence_type: str = "none",
        recurrence_target: str = "output",
        **kwargs: Any,
    ) -> None:
        nn.Module.__init__(self)
        super().__init__(**kwargs)

        # Store core model attributes
        self.dt = float(dt)
        self.tau = float(tau)
        self.t_feedforward = float(t_feedforward)
        self.t_recurrence = float(t_recurrence)
        self.t_feedback = float(t_feedforward if t_feedback is None else t_feedback)
        self.t_skip = float(t_feedback if t_skip is None else t_skip)
        self.history_length = max(
            self.t_feedforward, self.t_recurrence, self.t_feedback, self.t_skip
        )
        self.classifier_name = classifier_name
        self.dynamics_solver = str(dynamics_solver)
        self.recurrence_type = str(recurrence_type)
        self.recurrence_target = str(recurrence_target)
        self.data_presentation_pattern = data_presentation_pattern

        # Process feedforward delay
        self.delay_feedforward = int(t_feedforward / dt)

        # Process input dimensions and determine timesteps
        self._process_input_dimensions(input_dims, n_timesteps)

        # Set number of classes (may be updated later)
        self.n_classes = n_classes

        # Store any additional kwargs as attributes
        for name, value in kwargs.items():
            if name != "self":
                setattr(self, name, value)

        # Define the architecture
        self._define_architecture()

        # Initialize parameters
        self._init_parameters()

        # Verify initialization
        self.verify_initialization()

        # Set residual timesteps
        self.set_residual_timesteps()

        # Reset model state
        if hasattr(self, "reset"):
            self.reset()

    # Data processing
    #################
    def _process_batch(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int = 0
    ) -> torch.Tensor:
        """Process a batch of input tensors."""
        batch = self._expand_timesteps(batch)
        batch = self._extend_residual_timesteps(batch)
        return batch

    def _process_input_dimensions(
        self, input_dims: Tuple[int], n_timesteps: int
    ) -> None:
        """Process input dimensions and determine final timesteps."""
        # Create a sample tensor to understand dimensions
        if hasattr(self, "create_aligned_tensor"):
            x = self.create_aligned_tensor(
                size=(1, *input_dims), creation_method="randn"
            )
        else:
            x = torch.randn(1, *input_dims)

        x = _adjust_data_dimensions(x)
        _, data_timesteps, self.n_channels, self.dim_y, self.dim_x = x.shape

        # Determine final number of timesteps
        if data_timesteps == 1:
            self.n_timesteps = n_timesteps
        elif n_timesteps == 1:
            self.n_timesteps = data_timesteps
        else:
            self.n_timesteps = max(n_timesteps, data_timesteps)
            logger.warning(
                f"Model initialized with {n_timesteps} timesteps, "
                f"data has {data_timesteps} timesteps. "
                f"Using the larger number: {self.n_timesteps}"
            )

        self.input_dims = (self.n_timesteps, self.n_channels, self.dim_y, self.dim_x)

    # Architecture
    ##############
    def _define_architecture(self) -> None:
        """Override this method to define your model architecture."""
        raise NotImplementedError("Define the model architecture!")

    def _init_parameters(self) -> None:
        logger.info("Initializing model parameters...")
        for module in self.children():
            if hasattr(module, "_init_parameters"):
                module._init_parameters()

    def verify_initialization(self) -> None:
        pass

    def reset(self) -> None:
        """Reset the model state, in particular hidden states."""
        pass

    # Core forward pass
    ###################
    def _forward(
        self,
        x: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        feedforward_only: bool = False,
        store_responses: bool = True,
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

        responses = {}

        batch_size, n_channels, y_dim, x_dim = x.shape

        # if x.max() < 0 and x.min() == x.max():  # ToDO: revaluate null input
        #     x = None

        if hasattr(self, "input_adaption"):
            x = self.input_adaption(x)

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

                elif operation == "record" and store_responses:
                    responses[layer_name] = x

                elif operation == "delay" and hasattr(layer, "set_hidden_state"):
                    layer.set_hidden_state(x)
                    x = layer.get_hidden_state(self.delay_feedforward)

                elif operation == "tstep" and hasattr(self, module_name):
                    module = getattr(self, module_name)
                    h = layer.get_newest_hidden_state()
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
            x = torch.zeros(
                (batch_size, self.n_classes), device=self.device, requires_grad=True
            )
        else:
            classifier = getattr(self, self.classifier_name)
            x = classifier(x)

        if store_responses:
            responses[self.classifier_name] = x

        return x, responses

    def forward(
        self,
        x_0: torch.Tensor,
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
        x_0 = _adjust_data_dimensions(x_0)
        batch_size, n_timesteps, dim_channels, dim_y, dim_x = x_0.shape

        if hasattr(self, "reset"):
            self.reset()

        store_responses = (
            hasattr(self, "storage") and self.storage.responses.should_store()
        )
        if store_responses:
            responses = DataBuffer(
                max_size=n_timesteps,
                strategy="fixed",
                cpu_offload=self.storage.responses.cpu_offload,
                detach_tensors=self.storage.responses.detach_tensors,
                thread_safe=self.storage.responses.thread_safe,
            )

        # Collect outputs in a list to preserve gradients
        output_list = []

        # Forward the model over all timesteps
        for t in torch.arange(n_timesteps, device=x_0.device):
            x = x_0[:, t, ...]

            x, responses_t = self._forward(
                x,
                t,
                feedforward_only=feedforward_only,
                store_responses=store_responses,
            )

            if x is not None:
                # Add time dimension and append to list
                output_list.append(x.unsqueeze(1))
            else:
                # Handle None case - create zero tensor with gradients
                zero_output = torch.zeros(
                    (batch_size, 1, self.n_classes),
                    device=x_0.device,
                    dtype=x_0.dtype,
                    requires_grad=True,
                )
                output_list.append(zero_output)

            if store_responses and len(responses_t):
                t_responses = {
                    k: v.unsqueeze(1) if v is not None else None
                    for k, v in responses_t.items()
                }
                responses.append(t_responses)

        # Concatenate all outputs along time dimension - preserves gradients
        outputs = torch.cat(output_list, dim=1)

        if store_responses:
            response_dict = responses.to_dict(dim=1)  # Concatenate time dimension
            self.storage.store_responses(response_dict)
            del responses, response_dict

        del output_list
        return outputs

    def predictor(self, outputs: torch.Tensor) -> torch.Tensor:
        return torch.argmax(outputs, dim=-1)

    # Input processing utilities
    ############################
    def _expand_input_channels(
        self, x: Optional[torch.Tensor], n_target_channels: int = 3
    ) -> Optional[torch.Tensor]:
        """
        Expands or repeats the input tensor's channel dimension to match n_target_channels.

        Args:
            x (Optional[torch.Tensor]): Input tensor of shape (..., C, H, W).
            n_target_channels (int): Desired number of channels.

        Returns:
            Optional[torch.Tensor]: Tensor with n_target_channels in the channel dimension.
        """
        if x is None:
            return None
        if x.shape[-3] == n_target_channels:
            return x
        if x.shape[-3] == 1:
            # Repeat single channel to match target channels
            return x.repeat_interleave(n_target_channels, dim=-3)
        elif x.shape[-3] < n_target_channels:
            # Pad with zeros to match target channels
            pad_shape = list(x.shape)
            pad_shape[-3] = n_target_channels - x.shape[-3]
            pad = torch.zeros(*pad_shape, dtype=x.dtype, device=x.device)
            return torch.cat([x, pad], dim=-3)
        else:
            # Truncate channels if more than needed
            return x[..., :n_target_channels, :, :]

    # Temporal dynamics
    ###################
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

    def _expand_timesteps(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, label_indices, *extra = batch
        inputs = _adjust_data_dimensions(inputs)
        label_indices = _adjust_label_dimensions(label_indices)

        if inputs.size(1) == 1 and self.n_timesteps > 1:

            inputs = inputs.expand(-1, self.n_timesteps, -1, -1, -1)
            label_indices = label_indices.expand(-1, self.n_timesteps)

            # optionally modify based on data_presentation pattern
            if (
                hasattr(self, "data_presentation_pattern")
                and len(self.data_presentation_pattern) > 1
            ):
                # Cache the processed presentation pattern
                if not hasattr(self, "_cached_presentation_pattern"):
                    self._cache_presentation_pattern()

                presentation_pattern = self._cached_presentation_pattern
                zero_mask = ~presentation_pattern

                # Only modify if there are timesteps to zero out
                if zero_mask.any():
                    inputs[:, zero_mask] = 0
                    label_indices[:, zero_mask] = self.non_label_index

        return (inputs, label_indices, *extra)

    def _cache_presentation_pattern(self) -> None:
        """Cache the processed presentation pattern to avoid recomputation."""
        if isinstance(self.data_presentation_pattern, (str, list)):
            pattern = [int(i) for i in self.data_presentation_pattern]
        elif isinstance(self.data_presentation_pattern, int):
            logger.warning(
                "presentation pattern given as int, this obscures leading 0s!"
            )
            pattern = [int(i) for i in str(self.data_presentation_pattern)]
        else:
            raise ValueError(
                f"type of pattern is not str or list:", self.data_presentation_pattern
            )

        pattern = torch.tensor(pattern, dtype=torch.bool)

        # Resize pattern to match n_timesteps if needed
        if len(pattern) != self.n_timesteps:
            if len(pattern) == 1:
                # Special case: single value repeated
                pattern = pattern.expand(self.n_timesteps)
            else:
                # Efficient nearest neighbor resampling
                old_len = len(pattern)
                indices = torch.arange(self.n_timesteps) * old_len // self.n_timesteps
                indices = torch.clamp(indices, 0, old_len - 1)
                pattern = pattern[indices]

        self._cached_presentation_pattern = pattern

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

    # Parameter management
    ######################
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

    # Trainable parameter management
    ################################
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

    # Utility methods
    #################
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
