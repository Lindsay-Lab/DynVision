"""
DyRCNN: A biologically-inspired recurrent convolutional neural network.

This model implements key features of the visual cortex:
- Hierarchical processing (V1 -> V2 -> V4 -> IT)
- Recurrent connections within areas
- Skip and feedback connections between areas
- Retinal preprocessing
- Traveling wave patterns
- Neural dynamics with time constants

References:
- Kar et al. (2019) "Evidence that recurrent circuits are critical to the ventral stream's execution of core object recognition behavior"
- Muller et al. (2018) "Cortical travelling waves: mechanisms and computational principles"
"""

import logging
from typing import Optional, Union, Dict, Any, List

import torch
import torch.nn as nn

from dynvision.model_components import (
    EulerStep,
    SupraLinearity,
    Retina,
    InputAdaption,
    RecurrentConnectedConv2d,
    Skip,
    Feedback,
)
from dynvision.base import BaseModel
from dynvision.utils import alias_kwargs, str_to_bool

__all__ = ["DyRCNNx2", "DyRCNNx4", "DyRCNNx8", "FourLayerCNN", "TwoLayerCNN"]


logger = logging.getLogger(__name__)


class DyRCNN(BaseModel):
    """
    Base class for DyRCNN models implementing core biological features.

    The model implements:
    - Recurrent processing within areas
    - Dynamical systems evolution
    - Biological time constants
    - Optional retinal preprocessing

    Args:
        n_classes: Number of output classes
        input_dims: Input dimensions (timesteps, channels, height, width)
        dt: Integration time step (ms)
        tau: Neural time constant (ms)
        t_feedforward: Feedforward delay (ms)
        t_recurrence: Recurrent delay (ms)
        recurrence_type: Type of recurrent connections
        train_tau: Whether to learn time constants
        bias: Whether to use biases in convolutions
        max_weight_init: Maximum initial weight value
        supralinearity: Degree of supralinear activation
        input_adaption_weight: Weight for input adaptation
        skip: Whether to use skip connections
        feedback: Whether to use feedback connections
        feedforward_only: Whether to disable recurrence
    """

    @alias_kwargs(
        taugrad="train_tau",
        supralin="supralinearity",
        winit="max_weight_init",
        inadapt="input_adaption_weight",
        ffonly="feedforward_only",
        trc="t_recurrence",
        tff="t_feedforward",
        rctype="recurrence_type",
    )
    def __init__(
        self,
        # Core neural network parameters (passed to TemporalBase)
        dt: float = 2,  # ms
        tau: float = 8,  # ms
        t_feedforward: float = 10,  # ms
        t_recurrence: float = 6,  # ms
        t_feedback: float = 14,
        t_skip: float = 14,
        recurrence_type: str = "full",
        recurrence_target: str = "output",  # Target for recurrent connections
        # DyRCNN-specific biological parameters
        train_tau: bool = False,
        bias: bool = True,
        max_weight_init: float = 0.001,
        supralinearity: float = 1,
        input_adaption_weight: float = 0,
        use_retina: bool = False,
        skip: bool = False,
        feedback: bool = False,
        feedforward_only: bool = False,
        **kwargs: Any,
    ) -> None:

        # Store DyRCNN-specific biological attributes
        self.train_tau = str_to_bool(train_tau)
        self.bias = str_to_bool(bias)
        self.max_weight_init = float(max_weight_init)
        self.supralinearity = float(supralinearity)
        self.input_adaption_weight = float(input_adaption_weight)
        self.use_retina = str_to_bool(use_retina)
        self.skip = str_to_bool(skip)
        self.feedback = str_to_bool(feedback)
        self.feedforward_only = str_to_bool(feedforward_only)

        # Pass core neural network parameters to parent classes
        # BaseModel will distribute these properly to TemporalBase and LightningBase
        super().__init__(
            # Core parameters for TemporalBase
            dt=float(dt),
            tau=float(tau),
            t_feedforward=float(t_feedforward),
            t_recurrence=float(t_recurrence),
            t_feedback=float(t_feedback),
            t_skip=float(t_skip),
            recurrence_type=recurrence_type,
            recurrence_target=recurrence_target,
            # All other Lightning/training parameters pass through kwargs
            **kwargs,
        )

    def _define_architecture(self) -> None:
        raise NotImplementedError("Define the model architecture!")


class DyRCNNx4(DyRCNN):
    """
    Four-layer DyRCNN implementing a biologically-inspired visual hierarchy.
    """

    def reset(self) -> None:
        """Reset model state."""
        self.V1.reset()
        self.tstep_V1.reset()
        self.V2.reset()
        self.tstep_V2.reset()
        self.V4.reset()
        self.tstep_V4.reset()
        self.IT.reset()
        self.tstep_IT.reset()
        if hasattr(self, "input_adaption") and hasattr(self.input_adaption, "reset"):
            self.input_adaption.reset()
        if hasattr(self, "retina") and hasattr(self.retina, "reset"):
            self.retina.reset()

    # DyRCNNx4 specific architecture definition
    def _define_architecture(self) -> None:
        """Define the four-layer visual hierarchy."""
        self.layer_names = ["V1", "V2", "V4", "IT"]
        # define operations order within layer
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

        # Activation functions
        self.nonlin = nn.ReLU(inplace=False)
        if hasattr(self, "supralinearity") and float(self.supralinearity) != 1:
            self.supralin = SupraLinearity(
                n=float(self.supralinearity), requires_grad=False
            )

        # Initialize retina
        if self.use_retina:
            self.bottleneck_channels = 18
            self.retina = Retina(
                in_channels=self.n_channels,
                out_channels=self.bottleneck_channels,
                mid_channels=36,
                kernel_size=9,
                bias=self.bias,
            )

        # Common layer parameters
        layer_params = dict(
            bias=self.bias,
            recurrence_type=self.recurrence_type,
            dt=self.dt,
            tau=self.tau,
            history_length=self.history_length,
            t_recurrence=self.t_recurrence,
            max_weight_init=self.max_weight_init,
            feedforward_only=self.feedforward_only,
            recurrence_target=self.recurrence_target,
        )

        # Input adaptation
        if self.input_adaption_weight:
            self.input_adaption = InputAdaption(
                fixed_weight=self.input_adaption_weight
            )

        # V1 layer
        in_channels = self.retina.out_channels if self.use_retina else self.n_channels
        self.V1 = RecurrentConnectedConv2d(
            in_channels=in_channels,
            out_channels=36,
            kernel_size=3,
            stride=1,
            dim_y=self.dim_y,
            dim_x=self.dim_x,
            **layer_params,
        )
        self.tau_V1 = torch.nn.Parameter(
            torch.tensor(self.tau),
            requires_grad=self.train_tau,
        )
        self.tstep_V1 = EulerStep(dt=self.dt, tau=self.tau_V1)
        self.pool_V1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # V2
        self.V2 = RecurrentConnectedConv2d(
            in_channels=self.V1.out_channels,
            out_channels=64,
            kernel_size=3,
            stride=1,
            dim_y=self.V1.dim_y // self.V1.stride // self.pool_V1.stride,
            dim_x=self.V1.dim_x // self.V1.stride // self.pool_V1.stride,
            **layer_params,
        )
        self.tau_V2 = torch.nn.Parameter(
            torch.tensor(self.tau),
            requires_grad=self.train_tau,
        )
        self.tstep_V2 = EulerStep(dt=self.dt, tau=self.tau_V2)
        self.pool_V2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # V4
        self.V4 = RecurrentConnectedConv2d(
            in_channels=self.V2.out_channels,
            out_channels=121,
            kernel_size=3,
            stride=1,
            dim_y=self.V2.dim_y // self.V2.stride // self.pool_V2.stride,
            dim_x=self.V2.dim_x // self.V2.stride // self.pool_V2.stride,
            **layer_params,
        )
        if self.skip:
            self.addskip_V4 = Skip(
                source=self.V1,
                auto_adapt=True,
            )
        if self.feedback:
            self.addfeedback_V1 = Feedback(
                source=self.V4,
                auto_adapt=True,
            )
        self.tau_V4 = torch.nn.Parameter(
            torch.tensor(self.tau),
            requires_grad=self.train_tau,
        )
        self.tstep_V4 = EulerStep(dt=self.dt, tau=self.tau_V4)
        self.pool_V4 = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)

        # IT
        self.IT = RecurrentConnectedConv2d(
            in_channels=self.V4.out_channels,
            out_channels=256,
            kernel_size=3,
            stride=2,
            dim_y=self.V4.dim_y // self.V4.stride // self.pool_V4.stride,
            dim_x=self.V4.dim_x // self.V4.stride // self.pool_V4.stride,
            **layer_params,
        )
        if self.skip:
            self.addskip_IT = Skip(
                source=self.V2,
                auto_adapt=True,
            )
        if self.feedback:
            self.addfeedback_V2 = Feedback(
                source=self.IT,
                auto_adapt=True,
            )
        self.tau_IT = torch.nn.Parameter(
            torch.tensor(self.tau),
            requires_grad=self.train_tau,
        )
        self.tstep_IT = EulerStep(dt=self.dt, tau=self.tau_IT)

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.IT.out_channels, self.n_classes),
        )

    def setup(self, stage: Optional[str]) -> None:
        """Set up model for training or evaluation."""
        # First let PyTorch Lightning set up the model (including precision)
        super().setup(stage)

        # # Then initialize feedback connections
        # if stage == "fit" and self.feedback:
        #     for layer_name in self.layer_names:
        #         if hasattr(self, f"addfeedback_{layer_name}"):
        #             feedback_module = getattr(self, f"addfeedback_{layer_name}")
        #             feedback_module.setup(stage)

        #     # Do a forward pass to initialize transforms
        #     dtype = next(self.parameters()).dtype
        #     device = next(self.parameters()).device
        #     x = torch.randn((1, *self.input_dims), device=device, dtype=dtype)
        #     y = self.forward(x, store_responses=False)

        #     self.reset()

    def _init_parameters(self) -> None:
        super()._init_parameters()

        trunc_normal_params = dict(mean=0.0, std=0.004)

        for layer_name in self.layer_names:
            layer = getattr(self, layer_name)
            if hasattr(layer, "conv.weight") and not hasattr(
                layer, "conv.weight.original1"
            ):
                nn.init.trunc_normal_(layer.conv.weight, **trunc_normal_params)

        nn.init.trunc_normal_(self.classifier[-1].weight, **trunc_normal_params)
        nn.init.constant_(self.classifier[-1].bias, 0)


class DyRCNNx8(DyRCNNx4):
    """
    Four-layer DyRCNN (with double conv per layer) implementing a biologically-inspired visual hierarchy.
    """

    def _define_architecture(self) -> None:
        """Define the four-layer visual hierarchy."""
        self.layer_names = ["V1", "V2", "V4", "IT"]
        # define operations order within layer
        self.layer_operations = [
            "layer",  # apply (recurrent) convolutional layer
            "addext",  # add external input
            "addskip",  # add skip connection
            "addfeedback",  # add feedback connection
            "tstep",  # apply dynamical systems ode solver step
            "nonlin",  # apply nonlinearity
            "supralin",  # apply supralinearity
            "norm",  # apply normalization
            "delay",  # set and get delayed activations for next layer
            "record",  # record activations in responses dict
            "pool",  # apply pooling
        ]

        # Activation functions
        self.nonlin = nn.ReLU(inplace=False)
        if hasattr(self, "supralinearity") and float(self.supralinearity) != 1:
            self.supralin = SupraLinearity(
                n=float(self.supralinearity), requires_grad=False
            )

        # Initialize retina
        if self.use_retina:
            self.bottleneck_channels = 18
            self.retina = Retina(
                in_channels=self.n_channels,
                out_channels=self.bottleneck_channels,
                mid_channels=36,
                kernel_size=9,
                bias=self.bias,
            )

        # Common layer parameters
        layer_params = dict(
            bias=self.bias,
            recurrence_type=self.recurrence_type,
            dt=self.dt,
            tau=self.tau,
            history_length=self.history_length,
            t_recurrence=self.t_recurrence,
            max_weight_init=self.max_weight_init,
            feedforward_only=self.feedforward_only,
            recurrence_target=self.recurrence_target,
        )

        # Input adaptation
        if self.input_adaption_weight:
            self.input_adaption = InputAdaption(
                fixed_weight=self.input_adaption_weight
            )

        # V1 layer
        in_channels = self.retina.out_channels if self.use_retina else self.n_channels
        self.V1 = RecurrentConnectedConv2d(
            in_channels=in_channels,
            mid_channels=64,
            out_channels=64,
            kernel_size=5,
            stride=(2, 1),
            dim_y=self.dim_y,
            dim_x=self.dim_x,
            **layer_params,
        )
        self.tau_V1 = torch.nn.Parameter(
            torch.tensor(self.tau),
            requires_grad=self.train_tau,
        )
        self.tstep_V1 = EulerStep(dt=self.dt, tau=self.tau_V1)

        self.pool_V1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # V2
        self.V2 = RecurrentConnectedConv2d(
            in_channels=self.V1.out_channels,
            mid_channels=144,
            out_channels=144,
            kernel_size=3,
            stride=(2, 1),
            dim_y=self.V1.dim_y
            // self.V1.stride[0]
            // self.V1.stride[1]
            // self.pool_V1.stride,
            dim_x=self.V1.dim_x
            // self.V1.stride[0]
            // self.V1.stride[1]
            // self.pool_V1.stride,
            **layer_params,
        )
        self.tau_V2 = torch.nn.Parameter(
            torch.tensor(self.tau),
            requires_grad=self.train_tau,
        )
        self.tstep_V2 = EulerStep(dt=self.dt, tau=self.tau_V2)

        self.pool_V2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # V4
        self.V4 = RecurrentConnectedConv2d(
            in_channels=self.V2.out_channels,
            mid_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=(2, 1),
            dim_y=self.V2.dim_y
            // self.V2.stride[0]
            // self.V2.stride[1]
            // self.pool_V2.stride,
            dim_x=self.V2.dim_x
            // self.V2.stride[0]
            // self.V2.stride[1]
            // self.pool_V2.stride,
            **layer_params,
        )
        if self.skip:
            self.addskip_V4 = Skip(
                source=self.V1,
                auto_adapt=True,
            )
        if self.feedback:
            self.addfeedback_V1 = Feedback(
                source=self.V4,
                auto_adapt=True,
            )
        self.tau_V4 = torch.nn.Parameter(
            torch.tensor(self.tau),
            requires_grad=self.train_tau,
        )
        self.tstep_V4 = EulerStep(dt=self.dt, tau=self.tau_V4)

        # IT
        self.IT = RecurrentConnectedConv2d(
            in_channels=self.V4.out_channels,
            mid_channels=529,
            out_channels=529,
            kernel_size=3,
            stride=(2, 1),
            dim_y=self.V4.dim_y // self.V4.stride[0] // self.V4.stride[1],
            dim_x=self.V4.dim_x // self.V4.stride[0] // self.V4.stride[1],
            **layer_params,
        )
        if self.skip:
            self.addskip_IT = Skip(
                source=self.V2,
                auto_adapt=True,
            )
        if self.feedback:
            self.addfeedback_V2 = Feedback(
                source=self.IT,
                auto_adapt=True,
            )
        self.tau_IT = torch.nn.Parameter(
            torch.tensor(self.tau),
            requires_grad=self.train_tau,
        )
        self.tstep_IT = EulerStep(dt=self.dt, tau=self.tau_IT)

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.IT.out_channels, self.n_classes),
        )


class DyRCNNx2(DyRCNN):

    def reset(self):
        self.layer1.reset()
        self.tstep_layer1.reset()
        self.layer2.reset()
        self.tstep_layer2.reset()
        if hasattr(self, "input_adaption"):
            self.input_adaption.reset()

    def _define_architecture(self):
        self.layer_names = ["layer1", "layer2"]
        # define operations order within layer
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

        # Input adaption
        if self.input_adaption_weight:
            self.input_adaption = InputAdaption(
                fixed_weight=self.input_adaption_weight
            )

        if hasattr(self, "supralinearity") and float(self.supralinearity) != 1:
            self.supralin = SupraLinearity(
                n=float(self.supralinearity), requires_grad=False
            )

        # Define the convolutional layers
        layer_params = dict(
            bias=self.bias,
            recurrence_type=self.recurrence_type,
            dt=self.dt,
            tau=self.tau,
            history_length=self.history_length,
            t_recurrence=self.t_recurrence,
            recurrence_target=self.recurrence_target,
        )

        # Define the convolutional layers
        self.layer1 = RecurrentConnectedConv2d(
            in_channels=self.n_channels,
            out_channels=36,
            kernel_size=3,
            stride=1,
            dim_y=self.dim_y,
            dim_x=self.dim_x,
            **layer_params,
        )
        self.tstep_layer1 = EulerStep(dt=self.dt, tau=self.tau)
        self.nonlin_layer1 = nn.ReLU(inplace=False)
        self.pool_layer1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer2 = RecurrentConnectedConv2d(
            in_channels=self.layer1.out_channels,
            out_channels=64,
            kernel_size=3,
            stride=1,
            dim_y=self.layer1.dim_y // self.layer1.stride // self.pool_layer1.stride,
            dim_x=self.layer1.dim_x // self.layer1.stride // self.pool_layer1.stride,
            **layer_params,
        )
        self.tstep_layer2 = EulerStep(dt=self.dt, tau=self.tau)
        self.nonlin_layer2 = nn.ReLU(inplace=False)
        if self.feedback:
            self.addfeedback_layer1 = Feedback(source=self.layer2, autoadapt=True)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.layer2.out_channels, self.n_classes, device=self.device),
        )

    def setup(self, stage: Optional[str]) -> None:
        # First let PyTorch Lightning set up the model (including precision)
        super().setup(stage)

        # Then initialize feedback connections with model's dtype
        if stage == "fit" and self.feedback:
            # Get model dtype after Lightning setup
            dtype = self.get_target_dtype()
            device = self.get_target_device()

            # Reset feedback transforms to force reinitialization
            for layer_name in self.layer_names:
                if hasattr(self, f"addfeedback_{layer_name}"):
                    feedback_module = getattr(self, f"addfeedback_{layer_name}")
                    if feedback_module.auto_adapt:
                        feedback_module.reset_transform()

            # Initialize with correct dtype
            x = torch.randn((1, *self.input_dims), device=device, dtype=dtype)
            y = self.forward(x, store_responses=False)

            self.reset()

    def _init_parameters(self):
        super()._init_parameters()

        trunc_normal_params = dict(mean=0.0, std=0.004)

        for layer_name in self.layer_names:
            layer = getattr(self, layer_name)
            if hasattr(layer, "conv.weight") and not hasattr(
                layer, "conv.weight.original1"
            ):
                nn.init.trunc_normal_(layer.conv.weight, **trunc_normal_params)

        nn.init.trunc_normal_(self.classifier[-1].weight, **trunc_normal_params)
        nn.init.constant_(self.classifier[-1].bias, 0)


# Aliases for backwards compatibility
FourLayerCNN = DyRCNNx4
TwoLayerCNN = DyRCNNx2

if __name__ == "__main__":
    # Test configuration
    input_shape = (20, 3, 64, 64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Testing on device: {device}")

    # Create model
    model = DyRCNNx4(
        input_dims=input_shape,
        n_classes=200,
        store_responses=True,
        tff=13,
        device=device,
    )
    model.setup("fit")

    # Test forward pass
    x = torch.randn(1, *input_shape, device=device)
    y = model(x)

    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Output shape: {y.shape}")
    logger.info(f"Residual timesteps: {model.n_residual_timesteps}")

    # Test stability
    try:
        model(torch.full_like(x, float("inf")))
        assert False, "Should raise stability error"
    except ValueError:
        logger.info("Stability check passed")

    logger.info("All tests passed!")
