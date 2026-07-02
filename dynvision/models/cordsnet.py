"""
CordsNet: Contextual Recurrent Deep Structured Network

This module implements CordsNet, a hierarchical recurrent neural network
with complex skip connections and spatial biases.

Reference:
- Wang et al. (2020) "CordsNet: Contextual Recurrent Deep Structured Network"
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Tuple, Any

from dynvision.base import BaseModel
from dynvision.model_components import (
    RecurrentConnectedConv2d as RConv2d,
    SpatialBias,
    Skip,
    EulerStep,
)
from dynvision.project_paths import project_paths
from dynvision.utils import alias_kwargs, str_to_bool

__all__ = ["CordsNet"]

logger = logging.getLogger(__name__)


class Identity(nn.Module):
    """Identity module that ignores additional arguments."""

    def forward(self, x, *args, **kwargs):
        return x


class CordsNet(BaseModel):
    """
    CordsNet: Contextual Recurrent Deep Structured Network

    Reimplementation of CordsNet using DynVision's layer_operations pattern.
    Uses standard TemporalBase._forward() with proper delay coordination.

    Architecture:
        - 8 recurrent layers with complex skip connections
        - Channels: [64, 64, 64, 128, 128, 256, 256, 512, 512]
        - Strides: [1, 1, 2, 1, 2, 1, 2, 1]
        - Skip pattern: Each layer receives from previous + 2-layers-back
        - Spatial bias per layer (per-unit)

    Dynamics:
        Original update rule:
            r_new = (1 - alpha) * r_old + alpha * relu(recurrent(r) + feedforward(input) + bias)

        Equivalent to Euler integration with dt/tau = alpha:
            alpha = dt / tau

        Default: dt=1, tau=10 → alpha=0.1

    Args:
        n_timesteps: Number of timesteps to process (default: 100)
        input_dims: Input dimensions (t, c, y, x)
        n_classes: Number of output classes
        dt: Integration time step (controls temporal resolution)
        tau: Time constant (controls integration rate, alpha = dt/tau)
        t_feedforward: Delay for feedforward connections between layers
        t_recurrence: Delay for recurrent connections within each layer
        t_skip: Delay for skip connections
        idle_timesteps: Timesteps of spontaneous activity before input (default: 100)
        recurrence_type: Type of recurrent connection (default: "full" for 3x3 conv)
    """

    @alias_kwargs(
        tff="t_feedforward",
        trc="t_recurrence",
        tsk="t_skip",
        rctype="recurrence_type",
        rctarget="recurrence_target",
        idle="idle_timesteps",
        pretrained="init_with_pretrained",
    )
    def __init__(
        self,
        n_timesteps: int = 100,
        input_dims: Tuple[int, int, int, int] = (100, 3, 224, 224),
        n_classes: int = 1000,
        dt: float = 2.0,
        tau: float = 10.0,  # alpha = dt/tau = 0.1 by default
        t_feedforward: float = 2.0,
        t_recurrence: float = 2.0,
        t_skip: float = 2.0,
        t_feedback: float = 0.0,  # Not used in CordsNet
        recurrence_type: str = "full",  # 3x3 conv for recurrence
        recurrence_target: str = "output",
        skip: bool = True,
        feedback: bool = False,
        idle_timesteps: int = 100,
        init_with_pretrained: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize CordsNet model.

        Note: The original CordsNet uses alpha=0.2, which corresponds to dt=2, tau=10.
        """
        self.init_with_pretrained = str_to_bool(init_with_pretrained)

        super().__init__(
            n_timesteps=n_timesteps,
            input_dims=input_dims,
            n_classes=n_classes,
            dt=float(dt),
            tau=float(tau),
            t_feedforward=float(t_feedforward),
            t_recurrence=float(t_recurrence),
            t_feedback=float(t_feedback),
            t_skip=float(t_skip),
            recurrence_type=recurrence_type,
            recurrence_target=recurrence_target,
            skip=skip,
            feedback=feedback,
            idle_timesteps=int(idle_timesteps),
            **kwargs,
        )

    def _init_parameters(self) -> None:
        """Initialize model parameters, optionally loading pretrained weights."""
        if self.init_with_pretrained:
            self.load_pretrained_state_dict(check_mismatch_layer=["classifier.3"])
            # Make only the classifier trainable
            self.trainable_parameter_names = [
                p for p in list(self.state_dict().keys()) if "classifier.3" in p
            ]

        else:
            self.trainable_parameter_names = list(self.state_dict().keys())

    def download_pretrained_state_dict(self) -> dict:
        """
        Download pretrained CordsNet weights.

        Note: GitHub blob URLs don't work for direct download. Need to use raw.githubusercontent.com
        or use the actual release/download link.
        """
        # TODO: Fix URL to actual pretrained weights location
        url = "https://raw.githubusercontent.com/wmws2/cordsnet/main/cordsnetr8.pth"
        save_path = project_paths.models / "CordsNet" / "cordsnet_pretrained.pth"

        if not save_path.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading pretrained CordsNet from {url}")

            import requests

            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Downloaded pretrained model to {save_path}")
            else:
                raise RuntimeError(
                    f"Failed to download pretrained weights from {url} "
                    f"(status code: {response.status_code})"
                )
        else:
            logger.info(f"Using cached pretrained model from {save_path}")

        state_dict = torch.load(save_path, map_location=self.device, weights_only=True)
        return state_dict

    def translate_pretrained_layer_names(self) -> dict:
        """
        Map original CordsNet pretrained weight names to DynVision structure.

        Original structure:
        - inp_conv: initial 7x7 convolution on input
        - inp_avgpool: average pooling after initial conv
        - inp_skip: learned skip connection from input to layer2
        - area_conv[i]: recurrent connection for layer i (3x3 conv)
        - area_area[i]: feedforward connection for layer i (3x3 conv with stride)
        - conv_bias[i]: spatial bias for layer i [channels, height, width]
        - skip_area[0,1,2]: learned skip connections (1x1 conv with stride=2)
        - out_fc: final linear classifier

        DynVision structure:
        - layer0: RConv2d for input processing
        - layer{1-8}: RConv2d modules
          - layer.conv: feedforward (area_area)
          - layer.recurrence.conv: recurrent (area_conv)
        - addbias_layer{1-8}.bias: spatial biases (conv_bias)
        - addskip_layer{N}: Skip modules with learned convs
        - classifier: final classifier
        """
        translate_layer_names = {
            "inp_conv": "layer0.conv",
            # Note: inp_avgpool has no parameters, not in state_dict
            "out_fc": "classifier.3",
        }

        # Map layers 1-8
        for i in range(8):
            layer_name = f"layer{i+1}"
            # area_conv → recurrence conv
            translate_layer_names[f"area_conv.{i}"] = f"{layer_name}.recurrence.conv"
            # area_area → feedforward conv
            translate_layer_names[f"area_area.{i}"] = f"{layer_name}.conv"
            # conv_bias → spatial bias
            translate_layer_names[f"conv_bias.{i}"] = f"addbias_{layer_name}.bias"

        # Map skip connections
        translate_layer_names["inp_skip"] = "addskip_layer2.conv"
        translate_layer_names["skip_area.0"] = "addskip_layer4.conv"
        translate_layer_names["skip_area.1"] = "addskip_layer6.conv"
        translate_layer_names["skip_area.2"] = "addskip_layer8.conv"

        return translate_layer_names

    def reset(self, input_shape: Optional[Tuple[int, ...]] = None) -> None:
        """Reset all layer hidden states and dynamics solvers."""
        for layer_name in self.layer_names:
            # Reset layer hidden states
            layer = getattr(self, layer_name)
            if hasattr(layer, "reset"):
                layer.reset()
            # Reset dynamics solver hidden states
            tstep = getattr(self, f"tstep_{layer_name}", None)
            if tstep is not None and hasattr(tstep, "reset"):
                tstep.reset()

    def _define_architecture(self) -> None:
        """
        Define CordsNet architecture using DynVision's layer_operations pattern.

        Architecture:
        - layer0: Input processing (7x7 conv + pool, no recurrence)
        - layer1-8: Recurrent layers with 3x3 feedforward + 3x3 recurrence
        - Skip connections between non-consecutive layers
        - Spatial biases per layer

        Layer structure:
        - layer0: 3→64 (conv stride 2, pool stride 2 → 56x56)
        - layer1: 64→64 (stride 1, size 56x56)
        - layer2: 64→64 (stride 1, size 56x56) + skip from layer0
        - layer3: 64→128 (stride 2, size 28x28) + skip from layer1
        - layer4: 128→128 (stride 1, size 28x28) + skip from layer2
        - layer5: 128→256 (stride 2, size 14x14) + skip from layer3
        - layer6: 256→256 (stride 1, size 14x14) + skip from layer4
        - layer7: 256→512 (stride 2, size 7x7) + skip from layer5
        - layer8: 512→512 (stride 1, size 7x7) + skip from layer6
        """

        # Include input processing as layer0
        self.layer_names = [
            "layer0",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "layer5",
            "layer6",
            "layer7",
            "layer8",
        ]

        # Define operation sequence (matches DyRCNN pattern)
        self.layer_operations = [
            "addskip",  # Skip connection (if defined)
            "layer",  # RConv2d (feedforward + recurrence)
            "addbias",  # Spatial bias (layers 1-8 only)
            "pool",  # Pooling (layer0 only)
            "nonlin",  # Shared ReLU
            "tstep",  # Euler integration (continuous-time dynamics)
            "nonlin",
            "record",  # Record response
            "delay",  # Store and retrieve delayed activation
        ]

        # Shared nonlinearity
        self.nonlin = nn.ReLU(inplace=False)

        # Configuration arrays
        channels = [64, 64, 64, 128, 128, 256, 256, 512, 512]
        strides = [1, 1, 2, 1, 2, 1, 2, 1]

        # Layer 0: Input processing (no recurrence, no bias)
        self.layer0 = RConv2d(
            in_channels=self.n_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            recurrence_type="none",  # No recurrence for input layer
            t_feedforward=0.0,  # Immediate skip to layer2, no delay (matches original inp_skip)
            history_length=self.dt,
            dim_y=self.dim_y,  # Input spatial dimensions
            dim_x=self.dim_x,
            parametrization=nn.utils.parametrizations.weight_norm,
        )
        self.pool_layer0 = nn.AvgPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False
        )
        self.nonlin_layer0 = Identity()  # No nonlinearity for layer0!

        # Common parameters for recurrent layers
        common_params = {
            "bias": False,  # Using spatial bias modules instead
            "recurrence_type": self.recurrence_type,  # "full" = 3x3 conv
            "recurrence_kernel_size": 3,  # 3x3 recurrent convolution
            "recurrence_target": self.recurrence_target,  # Recurrence on layer output
            "recurrence_bias": False,  # No bias in recurrent conv
            "dt": self.dt,
            "t_recurrence": self.t_recurrence,
            "t_feedforward": self.t_feedforward,
            "history_length": max(
                self.t_feedforward, self.t_skip + self.dt, self.t_recurrence
            ),
            "parametrization": nn.utils.parametrizations.weight_norm,
        }

        # Calculate spatial dimensions for each layer
        # After layer0: 224 -> 112 (stride 2) -> 56 (pool stride 2)
        current_dim_y = self.dim_y // self.layer0.stride // self.pool_layer0.stride
        current_dim_x = self.dim_x // self.layer0.stride // self.pool_layer0.stride

        # Layers 1-8: Recurrent layers with spatial biases
        for i in range(8):
            layer_name = f"layer{i+1}"

            layer_params = common_params.copy()
            if i == 7:
                layer_params.update(
                    {"t_feedforward": 0.0}
                )  # No delay for last layer feedforward

            # RConv2d layer: feedforward (area_area) + recurrence (area_conv)
            layer = RConv2d(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=3,  # Feedforward kernel
                stride=strides[i],  # Feedforward stride
                padding=1,
                dim_y=current_dim_y,  # Input dimensions for this layer
                dim_x=current_dim_x,
                **layer_params,
            )
            setattr(self, layer_name, layer)

            # Update dimensions for next layer based on stride
            current_dim_y = current_dim_y // strides[i]
            current_dim_x = current_dim_x // strides[i]

            # Spatial bias module (per-unit bias)
            bias = SpatialBias(
                channels=channels[i + 1],
                height=current_dim_y,
                width=current_dim_x,
                init_value=0.0,
            )
            setattr(self, f"addbias_{layer_name}", bias)

            # Dynamics solver (Euler integration)
            # Original: r = (1-alpha)*r + alpha*relu(...)  where alpha = dt/tau
            tau = torch.nn.Parameter(
                torch.tensor(self.tau, dtype=torch.float32),
                requires_grad=False,
            )
            setattr(self, f"tau_{layer_name}", tau)

            solver = EulerStep(dt=self.dt, tau=tau)
            setattr(self, f"tstep_{layer_name}", solver)

        # Skip connections (matching original pattern)
        if self.skip:
            self._define_skip_connections()

        # sum the output of the last two layers (layer7 and layer8) to pass to the classifier
        combine_layer7_layer8 = Skip(
            source=self.layer7,
            in_channels=512,
            out_channels=512,
            scale_factor=1,
            delay_index=0,
            bias=False,
            force_conv=False,
            auto_adapt=False,
        )

        # Output classifier
        self.classifier = nn.Sequential(
            combine_layer7_layer8,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.utils.parametrizations.weight_norm(
                nn.Linear(512, self.n_classes, bias=True)
            ),
        )

    def _define_skip_connections(self) -> None:
        """
        Define skip connections matching original CordsNet pattern.

        Skip pattern from original CordsNet (lines 162-177):
        - layer2 (area 1): relu(r[0]) + inp_skip(inp)  - 3x3 conv, 64→64, same size
        - layer3 (area 2): relu(r[1]) + relu(r[0])  - direct, 64→64, same size
        - layer4 (area 3): relu(r[2]) + skip_area[0](relu(r[1]))  - 1x1 conv stride 2, 64→128
        - layer5 (area 4): relu(r[3]) + relu(r[2])  - direct, 128→128, same size
        - layer6 (area 5): relu(r[4]) + skip_area[1](relu(r[3]))  - 1x1 conv stride 2, 128→256
        - layer7 (area 6): relu(r[5]) + relu(r[4])  - direct, 256→256, same size
        - layer8 (area 7): relu(r[6]) + skip_area[2](relu(r[5]))  - 1x1 conv stride 2, 256→512

        Note: scale_factor = target_size / source_size (x_size / h_size)
        This matches nn.Upsample convention where scale_factor>1 means upsampling.

        Dimension tracking (output dimensions after each layer):
        - layer0: 56×56, layer1: 56×56, layer2: 56×56
        - layer3: 28×28 (stride 2), layer4: 28×28, layer5: 28×28
        - layer6: 14×14 (stride 2), layer7: 14×14, layer8: 7×7 (stride 2)
        """

        # layer2 ← layer0: 56×56 ← 56×56 (inp_skip, immediate/no delay)
        self.addskip_layer2 = Skip(
            source=self.layer0,
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            delay_index=0,  # Immediate skip (no delay), matching original inp_skip
            bias=False,
            force_conv=True,  # Force conv to match original 3x3 skip
            parametrization=nn.utils.parametrizations.weight_norm,
            auto_adapt=False,
        )

        # layer3 ← layer1: 28×28 ← 56×56 (direct, needs downsample)
        self.addskip_layer3 = Skip(
            source=self.layer1,
            in_channels=64,
            out_channels=64,
            t_connection=self.t_skip,
            dt=self.dt,
            force_conv=False,
            auto_adapt=False,
        )

        # layer4 ← layer2: 28×28 ← 56×56 (skip_area[0])
        self.addskip_layer4 = Skip(
            source=self.layer2,
            in_channels=64,
            out_channels=128,
            kernel_size=1,
            stride=2,
            t_connection=self.t_skip,
            dt=self.dt,
            bias=False,
            parametrization=nn.utils.parametrizations.weight_norm,
            auto_adapt=False,
        )

        # layer5 ← layer3: 14×14 ← 28×28 (direct, needs downsample)
        self.addskip_layer5 = Skip(
            source=self.layer3,
            in_channels=128,
            out_channels=128,
            t_connection=self.t_skip,
            dt=self.dt,
            force_conv=False,
            auto_adapt=False,
        )

        # layer6 ← layer4: 14×14 ← 28×28 (skip_area[1])
        self.addskip_layer6 = Skip(
            source=self.layer4,
            in_channels=128,
            out_channels=256,
            kernel_size=1,
            stride=2,
            t_connection=self.t_skip,
            dt=self.dt,
            bias=False,
            parametrization=nn.utils.parametrizations.weight_norm,
            auto_adapt=False,
        )

        # layer7 ← layer5: 7×7 ← 14×14 (direct, needs downsample)
        self.addskip_layer7 = Skip(
            source=self.layer5,
            in_channels=256,
            out_channels=256,
            t_connection=self.t_skip,
            dt=self.dt,
            force_conv=False,
            auto_adapt=False,
        )

        # layer8 ← layer6: 7×7 ← 14×14 (skip_area[2])
        self.addskip_layer8 = Skip(
            source=self.layer6,
            in_channels=256,
            out_channels=512,
            kernel_size=1,
            stride=2,
            t_connection=self.t_skip,
            dt=self.dt,
            bias=False,
            parametrization=nn.utils.parametrizations.weight_norm,
            auto_adapt=False,
        )


#  SPECIAL PREDICTOR THAT WAS APPLIED DURING TRAINING
# def predictor(self, outputs: torch.Tensor) -> torch.Tensor:
#     """Replicate CordsNet's original accuracy computation.

#     The reference implementation only evaluated predictions over the
#     concluding portion of the trial (timesteps >= 70). To mirror that
#     behaviour we fill guesses with the model's ``non_label_index`` for
#     the earlier timesteps and use per-timestep argmax predictions on the
#     last 70 steps. This ensures the upstream accuracy metric matches the
#     legacy code path, which summed accuracies across that evaluation
#     window.
#     """

#     if outputs.ndim != 3:
#         raise ValueError(
#             f"Expected classifier outputs with shape (batch, timesteps, classes), got {outputs.shape}"
#         )

#     batch_size, n_timesteps, _ = outputs.shape
#     eval_window = min(70, n_timesteps)

#     # Prepare default predictions ignored by accuracy for earlier steps.
#     guess_index = torch.full(
#         (batch_size, n_timesteps),
#         fill_value=int(getattr(self, "non_label_index", -1)),
#         dtype=torch.long,
#         device=outputs.device,
#     )

#     if eval_window == 0:
#         return guess_index

#     guess_index[:, -eval_window:] = torch.argmax(
#         outputs[:, -eval_window:, :], dim=-1
#     )

#     return guess_index


if __name__ == "__main__":
    # Test model creation
    input_shape = (100, 3, 224, 224)

    model = CordsNet(
        input_dims=input_shape,
        n_classes=1000,
        n_timesteps=100,
        dt=1,
        tau=10,
        t_feedforward=1,
        t_recurrence=1,
        t_skip=1,
        recurrence_type="full",
        idle_timesteps=100,
    )
    model.setup("fit")

    print(f"Model created successfully!")
    print(f"Layer names: {model.layer_names}")
    print(f"Layer operations: {model.layer_operations}")

    # Test forward pass
    x = torch.randn(1, *input_shape)
    y = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Residual timesteps: {model.n_residual_timesteps}")
