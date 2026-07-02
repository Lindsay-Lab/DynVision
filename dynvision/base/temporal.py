"""Core neural network functionality for biologically-inspired models."""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from dynvision import losses
from dynvision.data.operations import _adjust_data_dimensions, _adjust_label_dimensions
from dynvision.utils import alias_kwargs, str_to_bool
from .storage import DataBuffer

logger = logging.getLogger(__name__)
# logging.getLogger("dynvision.base.temporal").setLevel(logging.DEBUG)


class TemporalBase(nn.Module):
    """Core neural network functionality for temporal dynamics models."""

    # Class-level defaults for layer operation behavior
    # These can be overridden per instance or subclass
    DEFAULT_NON_FEEDFORWARD_OPERATIONS = [
        "addfeedback",
        "addskip",
    ]
    DEFAULT_OPERATIONS_SKIPPED_ON_NULL_INPUT = [
        "nonlin",
        "supralin",
        "pool",
        "norm",
    ]

    @alias_kwargs(
        trc="t_recurrence",
        tff="t_feedforward",
        tfb="t_feedback",
        tsk="t_skip",
        lossrt="loss_reaction_time",
        pattern="data_presentation_pattern",
        shufflepattern="shuffle_presentation_pattern",
        rctype="recurrence_type",
        rctarget="recurrence_target",
        solver="dynamics_solver",
        idle="idle_timesteps",
        ffonly="feedforward_only",
        tbptt="truncated_bptt_timesteps",
    )
    def __init__(
        self,
        # Core model architecture
        input_dims: Tuple[int] = (20, 3, 224, 224),
        n_timesteps: int = 1,
        n_classes: int = 1000,
        # Temporal dynamics
        dt: float = 2.0,
        tau: float = 5.0,
        t_feedforward: float = 0.0,
        t_recurrence: float = 6.0,
        t_feedback: Optional[float] = None,
        t_skip: Optional[float] = None,
        skip: bool = False,
        feedback: bool = False,
        data_presentation_pattern: Union[List[int], str] = [1],
        shuffle_presentation_pattern: bool = False,
        non_label_index: int = -1,
        non_input_value: float = 0.0,
        idle_timesteps: int = 0,
        loss_reaction_time: float = 0.0,
        feedforward_only: bool = False,
        truncated_bptt_timesteps: int = 0,
        # Architecture configuration
        classifier_name: str = "classifier",
        dynamics_solver: str = "euler",
        recurrence_type: str = "none",
        recurrence_target: str = "output",
        # Operation selection configuration
        non_feedforward_operations: Optional[List[str]] = None,
        operations_skipped_on_null_input: Optional[List[str]] = None,
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
        self.t_skip = float(self.t_feedback if t_skip is None else t_skip)
        self.skip = str_to_bool(skip)
        self.feedback = str_to_bool(feedback)
        self.history_length = max(
            self.t_feedforward,
            self.t_recurrence,
            self.t_feedback if self.feedback else 0,
            # +dt because skip source's delay stores current timestep before
            # addskip runs, requiring one extra buffer slot (see delay_index_skip +1)
            self.t_skip + self.dt if self.skip else 0,
        )
        self.classifier_name = classifier_name
        self.dynamics_solver = str(dynamics_solver)
        self.recurrence_type = str(recurrence_type)
        self.recurrence_target = str(recurrence_target)
        self.data_presentation_pattern = data_presentation_pattern
        self.shuffle_presentation_pattern = str_to_bool(shuffle_presentation_pattern)
        self.loss_reaction_time = float(loss_reaction_time)
        self.non_label_index = int(non_label_index)
        self.non_input_value = float(non_input_value)
        self.idle_timesteps = int(idle_timesteps)
        self.feedforward_only = str_to_bool(feedforward_only)
        self.truncated_bptt_timesteps = int(truncated_bptt_timesteps)

        # Set operation selections (use defaults if not provided)
        self.non_feedforward_operations = (
            non_feedforward_operations
            if non_feedforward_operations is not None
            else list(self.DEFAULT_NON_FEEDFORWARD_OPERATIONS)
        )
        self.operations_skipped_on_null_input = (
            operations_skipped_on_null_input
            if operations_skipped_on_null_input is not None
            else list(self.DEFAULT_OPERATIONS_SKIPPED_ON_NULL_INPUT)
        )

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
            input_shape = (
                1,
                self.n_timesteps,
                self.n_channels,
                self.dim_y,
                self.dim_x,
            )
            self.reset(input_shape)

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

    def reset(self, input_shape: Optional[Tuple[int, ...]] = None) -> None:
        """Reset the model state, in particular hidden states."""

    def compute_idle_initial_states(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Dict[str, List[Optional[torch.Tensor]]]:
        """Compute initial hidden states through idle timesteps.

        .. deprecated::
            This method is deprecated and no longer used by forward().
            The forward() method now handles idle timesteps inline, running them
            in a separate loop before the real timesteps. Hidden states flow
            naturally through the layers' buffers without caching/restoring.
            This approach maintains proper buffer state and allows gradients to
            flow correctly (with detachment at the transition to limit memory).

        Runs the model for idle_timesteps with null input to allow spontaneous
        activity to converge. Returns the converged hidden state values that
        can be used to initialize the model for training timesteps.

        This is a pure state initialization step - it runs in torch.no_grad()
        context and does not contribute to loss or gradients. The returned
        values are detached from any computation graph.

        Args:
            batch_size: Batch size for idle timestep computation
            device: Device to run computation on
            dtype: Data type for tensors

        Returns:
            Dictionary mapping layer names to lists of hidden state tensors.
            These can be used with layer.initialize_hidden_states(values)
        """
        import warnings

        warnings.warn(
            "compute_idle_initial_states is deprecated. The forward() method now "
            "handles idle timesteps inline without cache/restore.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not hasattr(self, "idle_timesteps") or self.idle_timesteps <= 0:
            return {}

        # Create null input for idle timesteps
        null_input = torch.full(
            (batch_size, self.n_channels, self.dim_y, self.dim_x),
            self.non_input_value,
            device=device,
            dtype=dtype,
            requires_grad=False,
        )

        # Run idle timesteps without building computation graph
        with torch.no_grad():
            for t in range(self.idle_timesteps):
                x, _ = self._forward(
                    null_input,
                    t=t,
                    feedforward_only=False,
                    store_responses=False,
                )
                del x  # Don't accumulate outputs

        # Cache hidden states from all layers
        cached_states = {}
        for name, layer in self.named_modules():
            if hasattr(layer, "cache_hidden_states"):
                cached = layer.cache_hidden_states()
                if cached:  # Only store if layer has hidden states
                    cached_states[name] = cached

        return cached_states

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
                "addext",  # add external input
                "addskip",  # add skip connection
                "addfeedback",  # add feedback connection
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

        if hasattr(self, "input_adaption"):
            x = self.input_adaption(x)

        for layer_name in self.layer_names:

            layer = getattr(self, layer_name)

            for operation in self.layer_operations:
                module_name = f"{operation}_{layer_name}"

                # Skip operations that require temporal dynamics during feedforward-only pass
                if feedforward_only and operation in self.non_feedforward_operations:
                    continue

                # Skip operations that require valid input when input is None
                if x is None and operation in self.operations_skipped_on_null_input:
                    continue

                # RConv2d layer operation
                if operation == "layer":
                    if hasattr(layer, "_get_name"):
                        module_name = layer._get_name()
                    else:
                        module_name = "layer"
                    x = layer(x, feedforward_only=feedforward_only)

                # Record layer responses
                elif operation == "record" and store_responses:
                    responses[layer_name] = x

                # Forward delay of layer activations
                elif (
                    operation == "delay"
                ):  # ToDo: standardize the forward delay mechanism
                    # Try layer-specific delay function first (e.g., delay_layer0)
                    if hasattr(self, f"delay_{layer_name}"):
                        delay_func = getattr(self, f"delay_{layer_name}")
                        x = delay_func(x)
                    # Then try layer's delay method (uses per-layer t_feedforward)
                    elif hasattr(layer, "delay"):
                        x = layer.delay(x)
                    # Fall back to original implementation
                    elif hasattr(layer, "set_hidden_state"):
                        layer.set_hidden_state(x)
                        # increment delay by one to account for "set then get" pattern:
                        # delay=1 gets what was just set, delay=2 gets previous timestep
                        if self.delay_feedforward > 0:
                            x = layer.get_hidden_state(self.delay_feedforward + 1)

                # Dynamical systems time step operation
                elif operation == "tstep" and hasattr(self, module_name):
                    module = getattr(self, module_name)
                    h = layer.get_newest_hidden_state()
                    if h is not None and h.device != x.device:
                        h = h.to(x.device)
                    x = module(x, h)

                # Apply any other operations (if defined)
                elif hasattr(self, module_name):
                    module = getattr(self, module_name)
                    x = module(x)
                elif hasattr(self, operation):
                    module = getattr(self, operation)
                    x = module(x)

                else:
                    logger.debug(f"No {operation} defined for {layer_name}. Skipping.")

                # if isinstance(x, torch.Tensor) and (~torch.isfinite(x)).any():
                #     logger.warning(
                #         f"NaN/inf detected in {module_name} output! \n\t {(~torch.isfinite(x)).sum()}/{x.numel()} NaNs"
                #     )

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
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x_0 (torch.Tensor): Input tensor.
            store_responses (bool, optional): Whether to store responses. Defaults to False.

        Returns:
            torch.Tensor: Model outputs.
        """
        x_0 = _adjust_data_dimensions(x_0)
        batch_size, n_timesteps, dim_channels, dim_y, dim_x = x_0.shape

        if hasattr(self, "reset"):
            self.reset(x_0.shape)

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

        # Get idle timesteps count
        idle_steps = getattr(self, "idle_timesteps", 0)

        # =====================================================================
        # IDLE TIMESTEPS LOOP
        # Run model with null input to bring network into stable state.
        # Hidden states evolve naturally through the layers' buffers.
        # Outputs, responses, and activity loss are NOT recorded during this phase.
        # =====================================================================
        if idle_steps > 0:
            # Create null input for idle timesteps
            null_input = torch.full(
                (batch_size, self.n_channels, self.dim_y, self.dim_x),
                self.non_input_value,
                device=x_0.device,
                dtype=x_0.dtype,
            )

            # Flag for activity loss hooks to skip accumulation
            self._in_idle_period = True

            for t_idle in range(idle_steps):
                # Forward pass with null input - populates hidden states
                x, _ = self._forward(
                    null_input,
                    t=t_idle,
                    feedforward_only=self.feedforward_only,
                    store_responses=False,  # Don't store responses during idle
                )
                del x  # Discard output - not used for loss

            # End of idle period - detach hidden states to truncate BPTT
            # This prevents gradient flow through idle timesteps (memory efficient)
            # while preserving the hidden state VALUES as initial conditions
            for layer in self.modules():
                if hasattr(layer, "detach_hidden_states"):
                    layer.detach_hidden_states()

            # Clear idle period flag
            self._in_idle_period = False

            # Clean up
            del null_input

        # =====================================================================
        # REAL TIMESTEPS LOOP
        # Forward pass with actual input data.
        # Outputs and responses are recorded, gradients flow normally.
        # =====================================================================
        for t in range(n_timesteps):
            x = x_0[:, t, ...]

            x, responses_t = self._forward(
                x,
                t=t,
                feedforward_only=self.feedforward_only,
                store_responses=store_responses,
            )

            # Truncated BPTT: Detach hidden states periodically to limit memory accumulation
            if (
                hasattr(self, "truncated_bptt_timesteps")
                and self.truncated_bptt_timesteps > 0
            ):
                timestep_num = t + 1  # 1-based indexing for real timesteps
                if timestep_num % self.truncated_bptt_timesteps == 0:
                    logger.info(
                        f"[TBPTT] Detaching hidden states at real timestep {timestep_num} "
                        f"(every {self.truncated_bptt_timesteps} timesteps)"
                    )
                    for layer in self.modules():
                        if hasattr(layer, "detach_hidden_states"):
                            layer.detach_hidden_states()

            if x is None:
                # Handle None case - create zero tensor with gradients
                zero_output = torch.zeros(
                    (batch_size, 1, self.n_classes),
                    device=x_0.device,
                    dtype=x_0.dtype,
                    requires_grad=True,
                )
                output_list.append(zero_output)
            elif x.size(0) == 1 and x.size(0) != batch_size:
                # Handle case where only model self-generated activity (with batch size 1) reaches the classifier
                x = x.repeat(batch_size, 1).unsqueeze(1)
            else:
                x = x.unsqueeze(1)

            output_list.append(x)

            if store_responses and len(responses_t):
                # Move responses to CPU immediately to prevent GPU memory accumulation
                t_responses = {
                    k: v.unsqueeze(1).cpu().detach() if v is not None else None
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
        input_shape = (1, self.n_channels, self.dim_y, self.dim_x)
        random_input = self.create_aligned_tensor(
            size=input_shape, creation_method="randn"
        )

        if hasattr(self, "reset"):
            self.reset(input_shape)

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
            self.reset(input_shape)

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
        """Expand single-timestep inputs to match model's n_timesteps.

        When input has a single timestep but the model expects multiple timesteps,
        this method expands the input and applies temporal masking based on
        data_presentation_pattern and loss_reaction_time.

        Args:
            batch: Tuple of (inputs, label_indices, *extra)

        Returns:
            Tuple of (expanded_inputs, expanded_labels, *extra) with masking applied
        """
        inputs, label_indices, *extra = batch
        inputs = _adjust_data_dimensions(inputs)
        label_indices = _adjust_label_dimensions(label_indices)

        if inputs.size(1) == 1 and self.n_timesteps > 1:
            # Expand single timestep to all timesteps (creates views, not copies)
            inputs = inputs.expand(-1, self.n_timesteps, -1, -1, -1)
            label_indices = label_indices.expand(-1, self.n_timesteps)

            # Apply temporal masking (pattern zeros and reaction time)
            inputs, label_indices = self._apply_temporal_masking(inputs, label_indices)

        return (inputs, label_indices, *extra)

    def _apply_temporal_masking(
        self, inputs: torch.Tensor, label_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data presentation pattern and loss reaction time masking.

        Two independent masking operations:

        1. **Pattern masking** (when pattern contains zeros):
           - Inputs: Set to non_input_value where pattern is 0 (no stimulus)
           - Labels: Set to non_label_index where pattern is 0 (ignore in loss)

        2. **Reaction time masking** (when loss_reaction_time > 0):
           - Labels only: Set to non_label_index for the first N timesteps after
             each stimulus onset, giving the model time to "react" before evaluation

        For a static image (pattern="1"), reaction time masking still applies:
        the first N timesteps after t=0 are masked for loss computation.

        Args:
            inputs: Input tensor of shape (batch, timesteps, channels, height, width)
            label_indices: Label tensor of shape (batch, timesteps)

        Returns:
            Tuple of (masked_inputs, masked_labels)
        """
        # Quick check using string inspection to avoid GPU-CPU sync
        pattern_has_zeros = "0" in str(self.data_presentation_pattern)
        has_reaction_time = self.loss_reaction_time > 0

        # Early exit if no masking needed
        if not pattern_has_zeros and not has_reaction_time:
            return inputs, label_indices

        # Get presentation pattern and ensure correct device
        presentation_pattern = self._get_presentation_pattern()
        if presentation_pattern.device != inputs.device:
            presentation_pattern = presentation_pattern.to(inputs.device)

        # Compute masks (only if needed)
        pattern_mask = ~presentation_pattern if pattern_has_zeros else None
        reaction_mask = (
            self._compute_reaction_mask(presentation_pattern)
            if has_reaction_time
            else None
        )

        # Apply input masking (only pattern zeros affect inputs)
        if pattern_mask is not None:
            inputs = inputs.clone()  # Clone since expanded tensors are views
            inputs[:, pattern_mask] = self.non_input_value

        # Apply label masking (both pattern zeros and reaction time affect labels)
        if pattern_mask is not None or reaction_mask is not None:
            label_indices = label_indices.clone()
            # Combine masks
            if pattern_mask is not None and reaction_mask is not None:
                label_mask = pattern_mask | reaction_mask
            elif pattern_mask is not None:
                label_mask = pattern_mask
            else:
                label_mask = reaction_mask
            label_indices[:, label_mask] = self.non_label_index

        return inputs, label_indices

    def _cache_presentation_pattern(self) -> None:
        """Cache the base presentation pattern for reuse and shuffling."""
        if isinstance(self.data_presentation_pattern, (str, list)):
            raw_pattern = [int(i) for i in self.data_presentation_pattern]
        elif isinstance(self.data_presentation_pattern, int):
            logger.warning(
                "presentation pattern given as int, this obscures leading 0s!"
            )
            raw_pattern = [int(i) for i in str(self.data_presentation_pattern)]
        else:
            raise ValueError(
                "type of pattern is not str or list:", self.data_presentation_pattern
            )

        if not raw_pattern:
            raise ValueError("data_presentation_pattern must not be empty")

        base_pattern = torch.tensor(raw_pattern, dtype=torch.bool)
        self._base_presentation_pattern = base_pattern
        self._cached_presentation_pattern = self._expand_pattern(base_pattern)

    def _expand_pattern(self, pattern: torch.Tensor) -> torch.Tensor:
        """Expand a base pattern tensor to match self.n_timesteps."""
        if pattern.numel() == 1:
            return pattern.expand(self.n_timesteps)

        if self.n_timesteps <= 0:
            raise ValueError(
                "n_timesteps must be positive to expand presentation pattern"
            )

        device = pattern.device
        indices = (
            torch.arange(self.n_timesteps, device=device)
            * pattern.numel()
            // self.n_timesteps
        )
        indices = torch.clamp(indices, 0, pattern.numel() - 1)
        return pattern[indices]

    def _compute_reaction_mask(self, pattern: torch.Tensor) -> torch.Tensor:
        """Return mask for timesteps ignored due to loss reaction time.

        Marks the first N timesteps after each stimulus onset (rising edge in pattern)
        for exclusion from loss computation. Uses fully vectorized operations for
        GPU efficiency with zero CPU-GPU synchronization.
        """
        if self.loss_reaction_time <= 0:
            return torch.zeros_like(pattern, dtype=torch.bool)

        pattern = pattern.to(dtype=torch.bool)
        if pattern.numel() == 0:
            return torch.zeros_like(pattern, dtype=torch.bool)

        reaction_steps = max(1, math.ceil(self.loss_reaction_time / self.dt))

        # Detect rising edges (stimulus onsets) - where pattern goes from False to True
        padded = torch.cat(
            [torch.zeros(1, dtype=torch.bool, device=pattern.device), pattern]
        )
        rising_edges = padded[1:] & ~padded[:-1]

        # Get indices of rising edges
        edge_indices = torch.where(rising_edges)[0]

        if edge_indices.numel() == 0:
            return torch.zeros_like(pattern, dtype=torch.bool)

        # Use broadcasting to create ranges: [edge, edge+1, ..., edge+reaction_steps-1]
        offsets = torch.arange(reaction_steps, device=pattern.device)
        # Shape: (num_edges, reaction_steps)
        all_indices = edge_indices.unsqueeze(1) + offsets.unsqueeze(0)

        # Flatten and filter indices that are within bounds
        all_indices = all_indices.flatten()
        valid_indices = all_indices[all_indices < len(pattern)]

        # Create mask
        mask = torch.zeros_like(pattern, dtype=torch.bool)
        mask[valid_indices] = True

        return mask

    def _get_presentation_pattern(self) -> torch.Tensor:
        """Return the presentation pattern, shuffling when requested."""
        if not hasattr(self, "_cached_presentation_pattern"):
            self._cache_presentation_pattern()

        if not self.shuffle_presentation_pattern:
            return self._cached_presentation_pattern

        base_pattern = getattr(self, "_base_presentation_pattern", None)
        if base_pattern is None or base_pattern.numel() <= 1:
            return self._cached_presentation_pattern

        perm = torch.randperm(base_pattern.size(0), device=base_pattern.device)
        shuffled_base = base_pattern[perm]
        return self._expand_pattern(shuffled_base)

    def _extend_residual_timesteps(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        inputs, label_indices, *extra = batch
        batch_size, n_timesteps, _, _, _ = inputs.shape

        inputs = _adjust_data_dimensions(inputs)
        label_indices = _adjust_label_dimensions(label_indices)

        if self.n_residual_timesteps > 0:
            # add 0s at the end as inputs for residual timesteps
            new_shape = (
                inputs.size(0),
                n_timesteps + self.n_residual_timesteps,
                *inputs.shape[2:],
            )
            new_inputs = torch.full(
                new_shape,
                self.non_input_value,
                device=inputs.device,
                dtype=inputs.dtype,
            )
            new_inputs[:, :n_timesteps, ...] = inputs

            # add voidid buffer labels at the beginning for residual timesteps

            new_shape = (inputs.size(0), n_timesteps + self.n_residual_timesteps)
            new_label_indices = torch.full(
                new_shape,
                self.non_label_index,
                device=label_indices.device,
                dtype=label_indices.dtype,
            )
            new_label_indices[:, self.n_residual_timesteps :] = label_indices

            batch = (new_inputs, new_label_indices, *extra)

        else:

            batch = (inputs, label_indices, *extra)

        return batch

    # Loss management
    ##################
    def _init_loss(self) -> None:
        if getattr(self, "_loss_initialized", False):
            return

        if not hasattr(self, "criterion_params") or self.criterion_params is None:
            self.criterion_params = [("CrossEntropyLoss", {"weight": 1.0})]

        self.criterion = []

        for criterion_name, criterion_config in self.criterion_params:

            if "weight" in criterion_config.keys():
                criterion_weight = criterion_config.pop("weight")
            else:
                criterion_weight = 1

            if (
                criterion_name.lower() in ["crossentropyloss", "cross_entropy_loss"]
                and "ignore_index" not in criterion_config
            ):
                criterion_config["ignore_index"] = self.non_label_index
                logger.info(
                    f"Setting ignore_index={self.non_label_index} for {criterion_name}"
                )

            if criterion_weight <= 0:
                logger.info(
                    f"Skipping criterion {criterion_name} because weight={criterion_weight}"
                )
                continue

            if hasattr(losses, criterion_name):
                logger.info(
                    f"Criterion: {criterion_name} with weight: {criterion_weight} and config: {criterion_config}"
                )
                criterion_fn = getattr(losses, criterion_name)(**criterion_config)
                self.criterion += [(criterion_fn, criterion_weight)]
            else:
                raise ValueError(f"Invalid loss function: {criterion_name}")

            try:
                criterion_fn.register_hooks(self)
                logger.debug(f"registered hook for {criterion_fn}")
            except Exception:
                pass

        self._loss_initialized = True

    def setup(self, stage: Optional[str] = None) -> None:
        if not getattr(self, "_loss_initialized", False):
            self._init_loss()

        super_setup = getattr(super(), "setup", None)
        if callable(super_setup):
            super_setup(stage)

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
        self, check_mismatch_layer: List[str] = [], strict: bool = False
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

        logger.debug(f"Original state_dict keys: {list(state_dict.keys())}")

        # translate keys in loaded state dict
        if hasattr(self, "translate_pretrained_layer_names"):
            translate_layer_names = self.translate_pretrained_layer_names()
            logger.debug(f"Translation mapping: {translate_layer_names}")

            new_state_dict = {}

            for key in list(state_dict.keys()):
                new_key = key
                # Try to find a matching translation
                for old_pattern, new_pattern in translate_layer_names.items():
                    if old_pattern in key:
                        new_key = key.replace(old_pattern, new_pattern)
                        logger.debug(f"Translating: {key} -> {new_key}")
                        break

                new_state_dict[new_key] = state_dict[key]

            state_dict = new_state_dict
            logger.debug(f"Translated state_dict keys: {list(state_dict.keys())}")

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
