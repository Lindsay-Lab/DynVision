from collections import deque
from typing import Callable, Optional, Union, Deque, Dict, Any

import torch
import torch.nn as nn
from torchsummary import summary

from dynvision.model_components.topographic_recurrence import (
    LocalLateralConnection,
    LocalSeparableConnection,
)
from dynvision.model_components.base import DtypeDeviceCoordinatorMixin
from dynvision.model_components.integration_strategy import setup_integration_strategy
from dynvision.utils import apply_parametrization, str_to_bool, calculate_conv_out_dim
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
]


class RecurrenceBase(LightningModule, DtypeDeviceCoordinatorMixin):
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

    def reset(self) -> None:
        self.hidden_states: Deque[torch.Tensor] = deque(maxlen=self.n_hidden_states)

    def get_hidden_state(self, delay: Optional[int] = None) -> Optional[torch.Tensor]:
        """
        Get the hidden state at a specific index.
        The index is the delay in time steps.
        Index 0 is the newest entry, index 1 is the previous entry.

        Args:
            i (Optional[int]): Index of the hidden state. Default is None.

        Returns:
            Optional[torch.Tensor]: Hidden state tensor.
        """
        if delay is None:
            return self.hidden_states
        elif delay >= 0:
            i = -1 * (delay + 1)
        else:
            i = delay

        if abs(i) > len(self.hidden_states):
            return None
        else:
            return self.hidden_states[i]

    def get_oldest_hidden_state(self) -> Optional[torch.Tensor]:
        if len(self.hidden_states):
            return self.hidden_states[0]
        else:
            return None

    def get_newest_hidden_state(self) -> Optional[torch.Tensor]:
        if len(self.hidden_states):
            return self.hidden_states[-1]
        else:
            return None

    def set_hidden_state(self, h: torch.Tensor, delay: Optional[int] = None) -> None:
        """Fast reference storage - only coordinate if actually needed."""
        if delay is None:
            self.hidden_states.append(h)
            return
        elif delay >= 0:
            i = -1 * (delay + 1)
        else:
            i = delay

        self.hidden_states[i] = h

    def sync_persistent_state(self) -> None:
        """Only sync if there's actually a mismatch."""
        if not hasattr(self, "hidden_states") or not self.hidden_states:
            return

        target_dtype = self.get_target_dtype()
        target_device = self.get_target_device()

        # Check if any tensor needs syncing
        needs_sync = any(
            h.dtype != target_dtype or h.device != target_device
            for h in self.hidden_states
        )

        if needs_sync:
            # Only then do the coordination (preserves references when possible)
            synced_states = deque(maxlen=self.n_hidden_states)
            for hidden in self.hidden_states:
                if hidden.dtype != target_dtype or hidden.device != target_device:
                    synced_states.append(
                        hidden.to(dtype=target_dtype, device=target_device)
                    )
                else:
                    synced_states.append(hidden)  # Keep original reference

            self.hidden_states = synced_states


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
        fixed_weight: Optional[float] = None,
        max_weight_init: float = 0.2,
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
        self.fixed_weight = fixed_weight
        self.bias_enabled = bias
        self.requires_grad = fixed_weight is None

        self._define_architecture()

    def _define_architecture(self) -> None:
        """Define the architecture of the self connection."""
        self.weight = nn.Parameter(torch.Tensor([1]), requires_grad=self.requires_grad)
        if self.bias_enabled:
            self.bias = nn.Parameter(torch.Tensor([0]), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "bias"):
            return x * self.weight + self.bias
        return x * self.weight

    def _init_parameters(self) -> None:
        """
        Initialize parameters for the self-connection.
        """
        if self.requires_grad:
            nn.init.uniform_(self.weight, a=-self.max_weight_init, b=0)
        else:
            self.weight.data.fill_(self.fixed_weight)
        if hasattr(self, "bias"):
            nn.init.constant_(self.bias, 0)


class InputAdaption(LightningModule):
    """
    Implements input adaptation with a self-connection.
    """

    def __init__(
        self,
        fixed_weight: Optional[float] = None,
        max_weight_init: float = 0.2,
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
            bias=False,
            **kwargs,
        )
        self.reset()

    def reset(self) -> None:
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
        padding: Optional[int] = None,
        bias: bool = True,
        dim_y: Optional[int] = None,
        dim_x: Optional[int] = None,
        dt: float = 1,  # ms
        tau: float = 1,  # ms
        recurrence_target: str = "output",
        recurrence_type: str = "self",
        t_recurrence: float = 0,  # ms
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
        self.parametrization = parametrization

        if padding is None:
            padding = kernel_size // 2
        self.padding = padding

        if mid_channels is not None and not isinstance(stride, (list, tuple)):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        # Store spatial dimensions
        self.dim_y = dim_y
        self.dim_x = dim_x

        # Store recurrence parameters
        self.dt = dt
        self.tau = tau
        self.recurrence_type = recurrence_type
        self.t_recurrence = t_recurrence
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
            dim_y = self._calculate_conv_out_dim(self.dim_y, stride=self.stride[0])
            dim_y = self._calculate_conv_out_dim(dim_y, stride=self.stride[1])
            dim_x = self._calculate_conv_out_dim(self.dim_x, stride=self.stride[0])
            dim_x = self._calculate_conv_out_dim(dim_x, stride=self.stride[1])
        return dim_y, dim_x

    def _calculate_recurrence_output_dims(self) -> tuple[int, int]:
        """
        Calculate the output dimensions after recurrence.
        """
        if self.recurrence_target == "input":
            dim_y = self.dim_y
            dim_x = self.dim_x
        elif self.recurrence_target == "middle":
            dim_y = self._calculate_conv_out_dim(self.dim_y, stride=self.stride[0])
            dim_x = self._calculate_conv_out_dim(self.dim_x, stride=self.stride[0])
        elif self.recurrence_target == "output":
            dim_y, dim_x = self._calculate_feedforward_output_dims()
        else:
            raise ValueError(f"Invalid recurrence target: {self.recurrence_target}")
        return dim_y, dim_x

    def _setup_hidden_state_memory(
        self, history_length: Optional[float] = None
    ) -> None:
        self.history_length = (
            self.t_recurrence if history_length is None else history_length
        )
        self.n_hidden_states = int(self.history_length / self.dt) + 1
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
        conv_kwargs = dict(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
        )

        if self.mid_channels is None:
            self.conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                **conv_kwargs,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.mid_channels,
                **conv_kwargs | {"stride": self.stride[0]},
            )
            self.nonlin = nn.ReLU(inplace=False)
            self.conv2 = nn.Conv2d(
                in_channels=self.mid_channels,
                out_channels=self.out_channels,
                **conv_kwargs | {"stride": self.stride[1]},
            )

        if self.parametrization is not None:
            self.conv = apply_parametrization(self.conv, self.parametrization)

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
            kernel_size=self.kernel_size,
            bias=self.bias,
            parametrization=self.parametrization,
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
        except:
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
        else:
            x = self.conv(x)
        return x

    def forward_feedforward2(
        self,
        x: Optional[torch.Tensor] = None,
    ):
        if x is None:
            return None
        else:
            x = self.conv2(x)
        return x

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Optional[torch.Tensor]:

        if self.recurrence_target == "input":
            # Adding recurrence to layer input
            x = self.forward_recurrence(x, h)
            # Feedforward combined activity
            x = self.forward_feedforward(x)
            # Feedforward2 combined activity
            if self.mid_channels:
                x = self.forward_feedforward2(x)

        elif self.recurrence_target == "output":
            # Feedforward input activity
            x = self.forward_feedforward(x)
            # Feedforward2 combined activity
            if self.mid_channels:
                x = self.forward_feedforward2(x)
            # Adding recurrence to layer output
            x = self.forward_recurrence(x, h)

        elif self.recurrence_target == "middle":
            # Feedforward input activity
            x = self.forward_feedforward(x)
            # Adding recurrence to layer output
            x = self.forward_recurrence(x, h)
            # Feedforward2 combined activity
            if self.mid_channels:
                x = self.forward_feedforward2(x)

        else:
            raise ValueError(f"Invalid recurrence target: {self.recurrence_target}")

        return x


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
        if not class_name in ["SelfConnection", "InputAdaption"]:
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
