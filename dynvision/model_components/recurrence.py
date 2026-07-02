from typing import Callable, Optional, Union, Dict, Any, Tuple, List

import torch
import torch.nn as nn
from torchsummary import summary

from dynvision.model_components.topographic_recurrence import (
    LocalLateralConnection,
    LocalSeparableConnection,
)

# from dynvision.base import DtypeDeviceCoordinatorMixin
from dynvision.model_components.integration_strategy import setup_integration_strategy
from dynvision.utils import apply_parametrization, str_to_bool, calculate_conv_out_dim
from dynvision.base.storage import DataBuffer
from pytorch_lightning import LightningModule
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "DepthwiseSeparableConnection",
    "DepthPointwiseConnection",
    "PointDepthwiseConnection",
    "FullConnection",
    "SelfConnection",
    "InputAdaption",
    "RecurrentConnectedConv2d",
    "RConv2d",  # alias
]


class RecurrenceBase(LightningModule):
    """
    Base class for recurrent connections providing common parameter initialization
    and argument validation.
    """

    def __init__(self, max_weight_init: float = 0.05, **kwargs) -> None:
        super().__init__()
        self.max_weight_init = max_weight_init
        self.is_root_node = False

    def validate_init_args(self, kwargs: Dict[str, Any]) -> None:
        """Validate required arguments for convolutional recurrence."""
        required = {"in_channels", "kernel_size"}
        missing = required - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required arguments for convolution: {missing}")

    def _init_parameters(self) -> None:
        """Common parameter initialization for recurrent connections."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if hasattr(module, "weight.original0"):
                    module.weight.original0.data.fill_(1.0)
                if hasattr(module, "weight.original1"):
                    nn.init.uniform_(
                        module.weight.original1.weight,
                        a=-self.max_weight_init,
                        b=self.max_weight_init,
                    )
                elif hasattr(module, "weight"):
                    nn.init.uniform_(
                        module.weight, a=-self.max_weight_init, b=self.max_weight_init
                    )
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def get_target_dtype(self) -> torch.dtype:
        if hasattr(self, "parameters") and len(list(self.parameters())) > 0:
            return next(self.parameters()).dtype
        return None

    def get_target_device(self) -> torch.device:
        return self.device


class ForwardRecurrenceBase(RecurrenceBase):
    """
    Base class for combined forward and recurrence operations.
    """

    def reset(self, input_shape: Optional[Tuple[int, ...]] = None) -> None:
        """Reset hidden states using DataBuffer with CyclicStrategy.

        Configuration:
        - cpu_offload=False: Keep on GPU for fast recurrent access
        - detach_tensors=False: Preserve gradients through recurrence
        - thread_safe=True: Safe for distributed training
        """
        self._hidden_states = DataBuffer(
            max_size=self.n_hidden_states,
            strategy="cyclic",
            cpu_offload=False,
            detach_tensors=False,
            thread_safe=True,
        )

    def get_hidden_state(self, delay: Optional[int] = None) -> Optional[torch.Tensor]:
        """
        Get the hidden state at a specific delay.
        The delay is measured in timesteps from the present.

        Args:
            delay (Optional[int]): Delay in timesteps. delay=0 is the most recent state,
                                   delay=1 is one timestep back, etc. If None, defaults to 0.

        Returns:
            Optional[torch.Tensor]: Hidden state tensor at the specified delay, or None if unavailable.
        """
        # Default to most recent state
        if delay is None or delay == 0:
            delay = 1

        # Check if buffer is empty
        if len(self._hidden_states) == 0:
            return None

        # Check if requested delay exceeds available history
        # Use > not >= to allow recurrence (accessed before set) to work:
        # At t=1: buffer=[state_t0] (1 entry), delay_recurrence=1 should return state_t0
        if delay > len(self._hidden_states):
            return None

        # Convert delay to buffer index: delay=1 → -1, delay=2 → -2, etc.
        try:
            return self._hidden_states.get(-delay)
        except (IndexError, ValueError):
            # Index out of range - shouldn't happen with our check, but handle gracefully
            return None

    def get_oldest_hidden_state(self) -> Optional[torch.Tensor]:
        """Get the oldest hidden state in the buffer.

        Returns:
            Optional[torch.Tensor]: Oldest hidden state, or None if buffer is empty.
        """
        if len(self._hidden_states) == 0:
            return None
        try:
            return self._hidden_states.get(0)
        except (IndexError, ValueError):
            return None

    def get_newest_hidden_state(self) -> Optional[torch.Tensor]:
        """Get the newest hidden state in the buffer.

        Returns:
            Optional[torch.Tensor]: Newest hidden state, or None if buffer is empty.
        """
        if len(self._hidden_states) == 0:
            return None
        try:
            return self._hidden_states.get(-1)
        except (IndexError, ValueError):
            return None

    def set_hidden_state(self, h: torch.Tensor, delay: Optional[int] = None) -> None:
        """Store a hidden state in the buffer.

        Args:
            h (torch.Tensor): Hidden state tensor to store.
            delay (Optional[int]): If None, appends the state as the newest entry.
                                   Setting at specific delays is not supported with DataBuffer.

        Raises:
            NotImplementedError: If delay is specified (not supported with circular buffer).
        """
        if delay is None:
            self._hidden_states.append(h)
            return
        else:
            raise NotImplementedError(
                "Setting hidden states at specific delays is not supported with DataBuffer. "
                "Only appending new states (delay=None) is allowed."
            )

    def detach_hidden_states(self) -> None:
        """Detach all hidden states from the computation graph.

        This preserves the hidden state values but breaks gradient connections,
        freeing memory from intermediate activations. Useful after idle timesteps
        to prevent backpropagation through the warmup period.
        """
        if hasattr(self, "_hidden_states") and self._hidden_states is not None:
            self._hidden_states.detach()
            # Clear GPU cache immediately to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def reenable_grad_on_hidden_states(self) -> None:
        """Re-enable gradients on hidden states after no_grad context.

        DEPRECATED: This method is kept for backward compatibility but should not
        be used. Use initialize_hidden_states() instead for proper gradient flow.

        This is used after running idle timesteps in a torch.no_grad() context.
        It detaches the tensors (removing the no_grad computation graph) and
        re-enables gradient tracking for future forward passes.

        This allows us to:
        1. Run idle timesteps without building computation graphs (saves memory)
        2. Keep the converged hidden state values
        3. Enable gradients for the actual training timesteps
        """
        if hasattr(self, "_hidden_states") and self._hidden_states is not None:
            self._hidden_states.detach_and_reenable_grad()

    def cache_hidden_states(self) -> List[Optional[torch.Tensor]]:
        """Cache current hidden state values for later restoration.

        Returns a list of cloned tensors representing the current buffer state.
        These values can be used to initialize a fresh buffer later.

        Returns:
            List of cached hidden state tensors (cloned to preserve values)

        Example:
            # After idle timesteps
            cached = layer.cache_hidden_states()

            # Reset and reinitialize
            layer.reset(input_shape)
            layer.initialize_hidden_states(cached)
        """
        if not hasattr(self, "_hidden_states") or self._hidden_states is None:
            return []

        cached = []
        for i in range(len(self._hidden_states)):
            tensor = self._hidden_states[i]
            if tensor is not None:
                # Clone to preserve values independent of original buffer
                cached.append(tensor.clone())
            else:
                cached.append(None)
        return cached

    def initialize_hidden_states(self, values: List[Optional[torch.Tensor]]) -> None:
        """Initialize hidden state buffer with pre-computed values.

        This method resets the hidden state buffer and populates it with the
        provided values. Used to set initial conditions from idle timesteps
        or other state preparation processes.

        The buffer is first reset (cleared), then initialized with the values.
        This ensures a fresh buffer that will participate in new computation
        graphs during subsequent forward passes.

        Args:
            values: List of tensors to initialize buffer with. Should match
                   the expected buffer structure (same length as n_hidden_states)

        Example:
            # Compute initial states through idle period
            with torch.no_grad():
                for t in range(idle_timesteps):
                    layer.forward(null_input)

            # Cache and reset
            cached = layer.cache_hidden_states()
            layer.reset(input_shape)
            layer.initialize_hidden_states(cached)

            # Now layer is ready for training with converged initial states
        """
        # Reset buffer (creates fresh DataBuffer)
        self.reset(input_shape=None)

        # Initialize with provided values
        if hasattr(self, "_hidden_states") and self._hidden_states is not None:
            self._hidden_states.initialize_from_values(values)

    def sync_persistent_state(self) -> None:
        """Sync hidden states to match model's device and dtype.

        Only performs synchronization if there's actually a device/dtype mismatch.
        """
        if not hasattr(self, "_hidden_states") or len(self._hidden_states) == 0:
            return

        target_dtype = self.get_target_dtype()
        target_device = self.get_target_device()

        # Check if any tensor needs syncing
        needs_sync = False
        for i in range(len(self._hidden_states)):
            try:
                h = self._hidden_states.get(i)
                if h is not None and (
                    h.dtype != target_dtype or h.device != target_device
                ):
                    needs_sync = True
                    break
            except (IndexError, ValueError):
                break

        if needs_sync:
            # Create new buffer and sync all tensors
            synced_buffer = DataBuffer(
                max_size=self.n_hidden_states,
                strategy="cyclic",
                cpu_offload=False,
                detach_tensors=False,
                thread_safe=True,
            )

            for i in range(len(self._hidden_states)):
                try:
                    hidden = self._hidden_states.get(i)
                    if hidden is not None:
                        if (
                            hidden.dtype != target_dtype
                            or hidden.device != target_device
                        ):
                            synced_buffer.append(
                                hidden.to(dtype=target_dtype, device=target_device)
                            )
                        else:
                            synced_buffer.append(hidden)  # Keep original reference
                except (IndexError, ValueError):
                    break

            self._hidden_states = synced_buffer


class DepthwiseSeparableConnection(RecurrenceBase):
    """
    Implements a depthwise separable convolution connection.

    This connection type first applies a depthwise convolution (separate convolution
    for each channel) followed by a pointwise convolution (1x1 convolution across channels).
    This decomposition reduces parameter count while maintaining expressivity.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        out_channels: Optional[int] = None,
        bias: bool = False,
        max_weight_init: float = 0.05,
        parametrization: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the DepthwiseSeparableConnection.

        Args:
            in_channels (int): Number of input channels.
            kernel_size (int): Size of the convolution kernel.
            bias (bool): Whether to include a bias term. Default is False.
            max_weight_init (float): Maximum weight initialization value. Default is 0.05.
            parametrization (Callable): Function to apply to the convolution layers. Default is identity.
        """
        super().__init__(max_weight_init=max_weight_init, **kwargs)

        # Store initialization arguments as attributes
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.parametrization = parametrization

        self._define_architecture()

    def _define_architecture(self) -> None:
        """Define the architecture of the depthwise separable connection."""
        self.depthwise_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            groups=self.in_channels,
            bias=self.bias,
        )

        self.pointwise_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.bias,
        )

        if self.parametrization is not None:
            self.depthwise_conv = apply_parametrization(
                self.depthwise_conv, self.parametrization
            )
            self.pointwise_conv = apply_parametrization(
                self.pointwise_conv, self.parametrization
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.depthwise_conv(x)
        h = self.pointwise_conv(h)
        return h


class DepthPointwiseConnection(DepthwiseSeparableConnection):
    """
    Implements a depth-pointwise convolution connection.
    """

    pass


class PointDepthwiseConnection(DepthwiseSeparableConnection):
    """
    Implements a point-depthwise convolution connection.
    """

    def _define_architecture(self) -> None:
        """Define the architecture of the depthwise separable connection."""
        self.pointwise_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.bias,
        )
        self.depthwise_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            groups=self.out_channels,
            bias=self.bias,
        )

        if self.parametrization is not None:
            self.depthwise_conv = apply_parametrization(
                self.depthwise_conv, self.parametrization
            )
            self.pointwise_conv = apply_parametrization(
                self.pointwise_conv, self.parametrization
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pointwise_conv(x)
        h = self.depthwise_conv(h)
        return h


class FullConnection(RecurrenceBase):
    """
    Implements a full convolution connection.

    This connection type applies a full convolutional operation where each unit
    receives input from all units within a nearby spatial region across all channels.
    This provides the most complete form of local connectivity but with the highest
    parameter count.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        out_channels: Optional[int] = None,
        bias: bool = False,
        max_weight_init: float = 0.05,
        parametrization: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the FullConnection.

        Args:
            in_channels (int): Number of input channels.
            kernel_size (int): Size of the convolution kernel.
            bias (bool): Whether to include a bias term. Default is False.
            max_weight_init (float): Maximum weight initialization value. Default is 0.05.
            parametrization (Callable): Function to apply to the convolution layers. Default is identity.
        """
        super().__init__(max_weight_init=max_weight_init, **kwargs)

        # Store initialization arguments as attributes
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.parametrization = parametrization

        self._define_architecture()

    def _define_architecture(self) -> None:
        """Define the architecture of the full connection."""
        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            bias=self.bias,
        )
        if self.parametrization is not None:
            self.conv = apply_parametrization(self.conv, self.parametrization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SelfConnection(RecurrenceBase):
    """
    Implements a self-connection with optional fixed weight and bias.

    This is the simplest form of recurrence where a unit connects only to itself.
    It can be used to model the persistence of neural activity over time or as a
    simplified form of neural adaptation.
    """

    def __init__(
        self,
        max_weight_init: float = 0.2,
        fixed_weight: Optional[float] = None,
        bias: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize the SelfConnection.

        Args:
            fixed_weight (Optional[float]): Fixed weight value. Default is None.
            max_weight_init (float): Maximum weight initialization value. Default is 0.2.
            bias (bool): Whether to include a bias term. Default is True.
        """
        super().__init__(max_weight_init=max_weight_init, **kwargs)

        # Store initialization arguments as attributes
        self.bias = bias
        self.fixed_weight = fixed_weight
        self._define_architecture()

    def _define_architecture(self) -> None:

        if self.fixed_weight is None:
            self.weight = nn.Parameter(torch.zeros(1))
        else:
            self.weight = nn.Parameter(
                torch.tensor(
                    [self.fixed_weight],
                    requires_grad=False,
                    device=self.get_target_device(),
                )
            )

        if self.bias:
            self.bias = nn.Parameter(
                torch.zeros(1), requires_grad=self.fixed_weight is None
            )
            # self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.weight.device:
            logger.error(
                f"Input device {x.device} does not match weight device {self.weight.device}"
            )
            self.weight = self.weight.to(device=x.device)
        out = x * self.weight

        if self.bias:
            out = out + self.bias

        return out

    def _init_parameters(self):
        """Initialize parameters."""
        with torch.no_grad():
            nn.init.uniform_(self.weight, -self.max_weight_init, self.max_weight_init)
            if self.bias is not None:
                nn.init.zeros_(self.bias)


class InputAdaption(LightningModule):
    """
    Implements input adaptation with a self-connection.
    """

    def __init__(
        self,
        fixed_weight: Optional[float] = None,
        max_weight_init: float = 0.2,
        bias: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the InputAdaption.

        Args:
            fixed_weight (Optional[float]): Fixed weight value. Default is None.
            max_weight_init (float): Maximum weight initialization value. Default is 0.2.
        """
        super(InputAdaption, self).__init__()

        self.recurrence = SelfConnection(
            fixed_weight=fixed_weight,
            max_weight_init=max_weight_init,
            bias=bias,
            **kwargs,
        )
        self.reset()

    def reset(self, input_shape: Optional[Tuple[int, ...]] = None) -> None:
        """
        Reset the hidden state.
        """
        self.hidden_state = None

    def forward(self, x: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        if self.hidden_state is None:
            self.hidden_state = x
            return x

        else:
            h = self.recurrence(self.hidden_state)
            self.hidden_state = h
            return h


class RecurrentConnectedConv2d(ForwardRecurrenceBase):
    """
    Implements a recurrently connected convolutional layer.

    This layer combines feedforward convolutional processing with various types
    of recurrent connections, allowing for dynamic temporal processing and
    feature integration over time.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Optional[int] = None,
        mid_channels: Optional[int] = None,
        mid_modules: Optional[nn.Module] = None,
        padding: Optional[int] = None,
        bias: bool = True,
        dim_y: Optional[int] = None,
        dim_x: Optional[int] = None,
        dt: float = 1,  # ms
        recurrence_target: str = "output",
        recurrence_type: str = "self",
        recurrence_kernel_size: Optional[int] = None,  # defaults to kernel_size
        recurrence_bias: Optional[bool] = None,  # defaults to bias
        t_recurrence: float = 0,  # ms
        t_feedforward: Optional[float] = 0,  # ms,
        integration_strategy: Union[Callable, str] = "additive",
        history_length: Optional[int] = None,  # ms
        fixed_self_weight: Optional[float] = None,
        max_weight_init: float = 0.001,
        parametrization: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        feedforward_only: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        # Store core convolution parameters
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.stride = stride
        self.parametrization = parametrization
        self.mid_modules = mid_modules

        # Handle list/tuple arguments for two-stage convolution
        if mid_channels is not None:
            # Convert single values to tuples for two-stage convolution
            self.stride = (
                stride if isinstance(stride, (list, tuple)) else (stride, stride)
            )
            self.kernel_size = (
                kernel_size
                if isinstance(kernel_size, (list, tuple))
                else (kernel_size, kernel_size)
            )

            if isinstance(padding, (list, tuple)):
                self.padding = padding
            elif padding is None:
                self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
            else:
                self.padding = (padding, padding)

            self.bias = bias if isinstance(bias, (list, tuple)) else (bias, bias)
        else:
            self.padding = kernel_size // 2 if padding is None else padding

        # Store recurrence convolution parameters
        self.recurrence_kernel_size = recurrence_kernel_size or (
            self.kernel_size[0] if mid_channels else kernel_size
        )
        self.recurrence_bias = (
            recurrence_bias
            if recurrence_bias is not None
            else (self.bias[0] if mid_channels else bias)
        )

        # Store spatial dimensions
        self.dim_y = dim_y
        self.dim_x = dim_x

        # Store recurrence parameters
        self.dt = dt
        self.recurrence_type = recurrence_type
        self.t_recurrence = t_recurrence
        self.t_feedforward = t_feedforward  # Optional per-layer feedforward delay
        self.integration_strategy = integration_strategy
        self.recurrence_target = recurrence_target
        self.fixed_self_weight = fixed_self_weight
        self.max_weight_init = max_weight_init
        self.feedforward_only = str_to_bool(feedforward_only)

        # Configure hidden state memory
        self._setup_hidden_state_memory(history_length)

        # Only define architecture, initialization happens in setup
        self._define_architecture()
        self.reset()

    def _calculate_conv_out_dim(
        self, in_dim=None, kernel_size=None, padding=None, stride=None
    ) -> int:
        in_dim = in_dim or self.dim_y
        kernel_size = kernel_size or self.kernel_size
        padding = padding or self.padding
        stride = stride or self.stride
        try:
            return calculate_conv_out_dim(in_dim, kernel_size, padding, stride)
        except Exception as e:
            logger.warning(
                f"Error calculating convolution output dimensions: {str(e)}"
            )
            return None

    def _calculate_feedforward_output_dims(self) -> tuple[int, int]:
        """
        Calculate the output dimensions after convolution.
        """
        if self.mid_channels is None:
            dim_y = self._calculate_conv_out_dim(self.dim_y)
            dim_x = self._calculate_conv_out_dim(self.dim_x)
        else:
            dim_y = self._calculate_conv_out_dim(
                self.dim_y,
                kernel_size=self.kernel_size[0],
                stride=self.stride[0],
                padding=self.padding[0],
            )
            dim_y = self._calculate_conv_out_dim(
                dim_y,
                kernel_size=self.kernel_size[1],
                stride=self.stride[1],
                padding=self.padding[1],
            )
            dim_x = self._calculate_conv_out_dim(
                self.dim_x,
                kernel_size=self.kernel_size[0],
                stride=self.stride[0],
                padding=self.padding[0],
            )
            dim_x = self._calculate_conv_out_dim(
                dim_x,
                kernel_size=self.kernel_size[1],
                stride=self.stride[1],
                padding=self.padding[1],
            )
        return dim_y, dim_x

    def _calculate_recurrence_output_dims(self) -> tuple[int, int]:
        """
        Calculate the output dimensions after recurrence.
        """
        if self.recurrence_target == "input":
            dim_y = self.dim_y
            dim_x = self.dim_x
        elif self.recurrence_target == "middle":
            dim_y = self._calculate_conv_out_dim(
                self.dim_y,
                kernel_size=self.kernel_size[0],
                stride=self.stride[0],
                padding=self.padding[0],
            )
            dim_x = self._calculate_conv_out_dim(
                self.dim_x,
                kernel_size=self.kernel_size[0],
                stride=self.stride[0],
                padding=self.padding[0],
            )
        elif self.recurrence_target == "output":
            dim_y, dim_x = self._calculate_feedforward_output_dims()
        else:
            raise ValueError(f"Invalid recurrence target: {self.recurrence_target}")
        return dim_y, dim_x

    def _setup_hidden_state_memory(
        self, history_length: Optional[float] = None
    ) -> None:
        if history_length is None:
            self.history_length = max(self.t_recurrence, self.t_feedforward)
        else:
            self.history_length = history_length

        self.n_hidden_states = int(self.history_length / self.dt)  # + 1
        self.delay_recurrence = int(self.t_recurrence / self.dt)

    def _define_architecture(self) -> None:
        # Define feedforward convolution
        self._setup_feedforward_conv()

        # Define recurrent connection if needed
        if self.feedforward_only:
            self.recurrence = None
        else:
            self._setup_recurrence()

    def _setup_feedforward_conv(self) -> None:
        if self.mid_channels is None:
            self.conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.bias,
            )
            if self.parametrization is not None:
                self.conv = apply_parametrization(self.conv, self.parametrization)
        else:
            self.conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.mid_channels,
                kernel_size=self.kernel_size[0],
                stride=self.stride[0],
                padding=self.padding[0],
                bias=self.bias[0],
            )
            self.nonlin = nn.ReLU(inplace=False)
            self.conv2 = nn.Conv2d(
                in_channels=self.mid_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size[1],
                stride=self.stride[1],
                padding=self.padding[1],
                bias=self.bias[1],
            )
            if self.parametrization is not None:
                self.conv = apply_parametrization(self.conv, self.parametrization)
                self.conv2 = apply_parametrization(self.conv2, self.parametrization)

    def _setup_recurrence(self) -> None:
        """Set up the recurrent connection based on specified type."""
        if self.recurrence_target == "input":
            out_channels = self.in_channels
        elif self.recurrence_target == "middle":
            out_channels = self.mid_channels
        elif self.recurrence_target == "output":
            out_channels = self.out_channels
        else:
            raise ValueError(f"Invalid recurrence target: {self.recurrence_target}")

        # Setup up upsampling if needed
        in_dim_y, in_dim_x = self._calculate_feedforward_output_dims()
        out_dim_y, out_dim_x = self._calculate_recurrence_output_dims()

        if None in (in_dim_y, in_dim_x, out_dim_y, out_dim_x):
            self.upsample = False
        elif in_dim_y == out_dim_y and in_dim_x == out_dim_x:
            self.upsample = False
        else:
            self.upsample = nn.Upsample(size=(out_dim_y, out_dim_x))

        recurrence_params = dict(
            kernel_size=self.recurrence_kernel_size,
            bias=self.recurrence_bias,
            max_weight_init=self.max_weight_init,
            fixed_weight=self.fixed_self_weight,
            in_channels=self.out_channels,
            out_channels=out_channels,
            dim_y=in_dim_y,
            dim_x=in_dim_x,
        )

        # Map recurrence types to their implementations
        recurrence_types = {
            "self": SelfConnection,
            "full": FullConnection,
            "depthpointwise": DepthPointwiseConnection,
            "pointdepthwise": PointDepthwiseConnection,
            "local": LocalLateralConnection,
            "localdepthwise": LocalSeparableConnection,
            None: None,
            "none": None,
        }
        if self.recurrence_type.lower() not in recurrence_types:
            raise ValueError(f"Invalid recurrence type: {self.recurrence_type}")

        if self.recurrence_type.lower() in [None, "none"]:
            self.recurrence = None
            return
        else:
            recurrence_class = recurrence_types[self.recurrence_type.lower()]
            self.recurrence = recurrence_class(**recurrence_params)
            if self.parametrization is not None:
                if hasattr(self.recurrence, "conv"):
                    self.recurrence.conv = apply_parametrization(
                        self.recurrence.conv, self.parametrization
                    )
                elif hasattr(self.recurrence, "depthwise_conv"):
                    self.recurrence.depthwise_conv = apply_parametrization(
                        self.recurrence.depthwise_conv, self.parametrization
                    )
                elif hasattr(self.recurrence, "pointwise_conv"):
                    self.recurrence.pointwise_conv = apply_parametrization(
                        self.recurrence.pointwise_conv, self.parametrization
                    )
                elif hasattr(self.recurrence, "weight"):
                    self.recurrence.weight = apply_parametrization(
                        self.recurrence.weight, self.parametrization
                    )

        # Configure recurrence influence
        self.integrate_signal = setup_integration_strategy(self.integration_strategy)

    def _init_parameters(self) -> None:
        def init_conv_layer(conv_layer):
            if hasattr(conv_layer, "weight.original0"):
                conv_layer.weight.original0.data.fill_(1.0)
            if hasattr(conv_layer, "weight.original1"):
                nn.init.kaiming_normal_(
                    conv_layer.weight.original1, nonlinearity="relu"
                )
            elif hasattr(conv_layer, "weight"):
                nn.init.kaiming_normal_(conv_layer.weight, nonlinearity="relu")
            if hasattr(conv_layer, "bias") and conv_layer.bias is not None:
                nn.init.constant_(conv_layer.bias, 0)

        init_conv_layer(self.conv)

        if self.mid_channels is not None:
            init_conv_layer(self.conv2)

        try:
            self.recurrence._init_parameters()
        except Exception:
            pass

    def forward_recurrence(
        self,
        x: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
    ):
        # Get previous activation for recurrent input
        if h is None:  # passed hidden state takes precedence
            h = self.get_hidden_state(self.delay_recurrence)

        # Response to recurrent input
        if h is None or self.recurrence is None:
            h = None
        else:
            h = self.recurrence(h)

        # Mixing-in recurrent influence
        if h is None:
            return x
        elif bool(self.upsample):
            h = self.upsample(h)

        if x is None:
            x = h
        else:
            x = self.integrate_signal(x, h)

        return x

    def forward_feedforward(
        self,
        x: Optional[torch.Tensor] = None,
    ):
        if x is None:
            return None

        x = self.conv(x)

        if self.mid_modules is not None:
            x = self.mid_modules(x)
        if self.mid_channels is not None:
            x = self.nonlin(x)

        return x

    def forward_feedforward2(
        self,
        x: Optional[torch.Tensor] = None,
    ):
        if x is None:
            return None
        elif self.mid_channels is not None:
            x = self.conv2(x)
        return x

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
        feedforward_only: bool = False,
        **kwargs,
    ) -> Optional[torch.Tensor]:

        if self.feedforward_only or self.recurrence is None:
            x = self.forward_feedforward(x)
            if self.mid_channels:
                x = self.forward_feedforward2(x)
            return x

        elif self.recurrence_target == "input":
            # Adding recurrence to layer input
            x = self.forward_recurrence(x, h)
            # Feedforward combined activity
            x = self.forward_feedforward(x)
            # Feedforward2 combined activity
            x = self.forward_feedforward2(x)

        elif self.recurrence_target == "output":
            # Feedforward input activity
            x = self.forward_feedforward(x)
            # Feedforward2 combined activity
            x = self.forward_feedforward2(x)
            # Adding recurrence to layer output
            x = self.forward_recurrence(x, h)

        elif self.recurrence_target == "middle":
            # Feedforward input activity
            x = self.forward_feedforward(x)
            # Adding recurrence to layer output
            x = self.forward_recurrence(x, h)
            # Feedforward2 combined activity
            x = self.forward_feedforward2(x)

        else:
            raise ValueError(f"Invalid recurrence target: {self.recurrence_target}")

        return x

    def delay(
        self, x: torch.Tensor, delay_feedforward: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply delay operation: store current state and retrieve delayed state.

        This method handles the feedforward delay by storing the current activation
        in the hidden state buffer and retrieving a delayed version for feedforward
        to the next layer.

        Args:
            x: Current activation to store
            delay_feedforward: Optional override for feedforward delay in timesteps.
                             If None, uses layer's t_feedforward parameter.

        Returns:
            Delayed activation for feedforward to next layer
        """
        # Determine delay amount
        if delay_feedforward is not None:
            # Use provided delay
            delay = delay_feedforward
        elif hasattr(self, "t_feedforward"):
            # Use per-layer t_feedforward if defined
            delay = int(self.t_feedforward / self.dt)
        else:
            # Default to no delay if not specified
            delay = 0

        # Retrieve delayed state before updating with current state
        if delay > 0:
            out = self.get_hidden_state(delay)
        else:
            out = x

        # Store current state
        self.set_hidden_state(x)

        return out


# Create alias for shorter name
RConv2d = RecurrentConnectedConv2d


if __name__ == "__main__":
    input_shape = (3, 32, 32)
    batch_size = 1
    logger.info("Input shape: %s", (batch_size, *input_shape))

    for class_name in __all__:
        logger.info("Testing %s:", class_name)

        # Create an instance of the class
        module = globals()[class_name]
        if class_name == "RecurrentConnectedConv2d":
            model = module(
                in_channels=input_shape[0],
                out_channels=6,
                kernel_size=3,
                dim_y=input_shape[1],
                dim_x=input_shape[2],
            )
        elif class_name == "InputAdaption":
            model = module()
        elif class_name == "SelfConnection":
            model = module()
        else:
            model = module(in_channels=input_shape[0], kernel_size=3)

        # Print number of parameters
        n_params = sum(p.numel() for p in model.parameters())
        logger.info("Number of parameters: %d", n_params)

        # Print model summary
        if class_name not in ["SelfConnection", "InputAdaption"]:
            print(summary(model, input_shape))

        # Generate random input
        if class_name == "InputAdaption":
            require_grad = True
        else:
            require_grad = False
        random_input = torch.randn(
            batch_size, *input_shape, requires_grad=require_grad
        )

        # Perform forward pass
        output = model(random_input)

        # Define an arbitrary loss function
        loss_fn = nn.MSELoss()

        # Generate random target
        target = torch.randn_like(output)

        # Compute loss
        loss = loss_fn(output, target)

        # Perform backward pass
        loss.backward()

        # Print results
        logger.info(
            "Model Output: %s\n", output.shape if output is not None else "None"
        )

    logger.info("All tests passed!")
