import requests
import torch
import torch.nn as nn

from dynvision.base import BaseModel

from dynvision.model_components import (
    EulerStep,
    RecurrentConnectedConv2d,
    Skip,
)
from dynvision.project_paths import project_paths
from dynvision.utils import alias_kwargs, str_to_bool

__all__ = ["CordsNet"]


class CordsNet(BaseModel):
    @alias_kwargs(
        tff="t_feedforward",
        trc="t_recurrence",
        rctype="recurrence_type",
        solver="dynamics_solver",
    )
    def __init__(
        self,
        n_classes=1000,
        input_dims=(20, 3, 224, 224),  # (t, c, y, x)
        dt=1,  # ms
        tau=10,  # ms
        t_feedforward=1,  # ms
        t_recurrence=1,  # ms
        recurrence_type="full",
        dynamics_solver="euler",
        bias=True,
        **kwargs,
    ) -> None:

        super().__init__(
            n_classes=n_classes,
            input_dims=input_dims,
            t_recurrence=float(t_recurrence),
            t_feedforward=float(t_feedforward),
            tau=float(tau),
            dt=float(dt),
            bias=str_to_bool(bias),
            recurrence_type=recurrence_type,
            dynamics_solver=dynamics_solver,
            **kwargs,
        )
        self.delay_ff = int(t_feedforward / dt)
        self.delay_rc = int(t_recurrence / dt)
        self._define_architecture()
        self._init_parameters()

    def setup(self, stage):
        super().setup(stage)

    def _init_parameters(self):
        # Load pretrained weights
        self.load_pretrained_state_dict(check_mismatch_layer=["classifier.2"])

        # make only the classifier trainable
        self.trainable_parameter_names = [
            p for p in list(self.state_dict().keys()) if "classifier.2" in p
        ]

    def download_pretrained_state_dict(self, version=0):
        url = "https://github.com/wmws2/cordsnet/blob/main/cordsnetr8.pth"
        save_path = project_paths.models / "CordsNet" / "CordsNet_pretrained.pt"

        if not save_path.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded pretrained model to {save_path}")
            else:
                raise Exception(f"Failed to download file from {url}")
        else:
            print(f"Pretrained model already exists at {save_path}")

        state_dict = torch.load(save_path, map_location=self.device)

        # average unit-wise bias values
        for key in state_dict.keys():
            if "conv_bias" in key:
                state_dict[key] = state_dict[key].mean(dim=(1, 2))

        return state_dict

    def translate_pretrained_layer_names(self):
        translate_layer_names = {
            "inp_conv": "layer_inp",
            "area_conv.0": "layer1.recurrence.conv",
            "area_conv.1": "layer2.recurrence.conv",
            "area_conv.2": "layer3.recurrence.conv",
            "area_conv.3": "layer4.recurrence.conv",
            "area_conv.4": "layer5.recurrence.conv",
            "area_conv.5": "layer6.recurrence.conv",
            "area_conv.6": "layer7.recurrence.conv",
            "area_conv.7": "layer8.recurrence.conv",
            "area_area.0": "layer1.conv",
            "area_area.1": "layer2.conv",
            "area_area.2": "layer3.conv",
            "area_area.3": "layer4.conv",
            "area_area.4": "layer5.conv",
            "area_area.5": "layer6.conv",
            "area_area.6": "layer7.conv",
            "area_area.7": "layer8.conv",
            "inp_skip": "skip_layer2",
            "skip_area.0": "skip_layer4.conv",
            "skip_area.1": "skip_layer6.conv",
            "skip_area.2": "skip_layer8.conv",
            "conv_bias.0": "layer1.conv.bias",
            "conv_bias.1": "layer2.conv.bias",
            "conv_bias.2": "layer3.conv.bias",
            "conv_bias.3": "layer4.conv.bias",
            "conv_bias.4": "layer5.conv.bias",
            "conv_bias.5": "layer6.conv.bias",
            "conv_bias.6": "layer7.conv.bias",
            "conv_bias.7": "layer8.conv.bias",
            "out_fc": "classifier.2",
        }
        return translate_layer_names

    def reset(self):
        for layer_name in self.layer_names[1:]:
            getattr(self, layer_name).reset()
            getattr(self, f"tstep_{layer_name}").reset()

    def _define_architecture(self):
        channels = [64, 64, 64, 128, 128, 256, 256, 512, 512]
        sizes = [56, 56, 28, 28, 14, 14, 7, 7]
        strides = [1, 1, 2, 1, 2, 1, 2, 1]
        self.depth = len(sizes)
        self.blockdepth = int(self.depth / 2 - 1)

        self.layer_names = ["layer_inp"] + [
            f"layer{i}" for i in range(1, self.depth + 1)
        ]
        self.layer_operations = [
            "addskip",  # apply skip connection
            "layer",  # apply (recurrent) convolutional layer
            "nonlin",  # apply nonlinearity
            "pool",  # apply pooling
            "tstep",  # apply dynamical systems ode solver step
            "nonlin",  # apply nonlinearity
            "record",  # record activations in responses dict
            "delay",  # set and get delayed activations for next layer
        ]

        # Input layer
        self.layer_inp = nn.utils.parametrizations.weight_norm(
            nn.Conv2d(
                in_channels=self.n_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        )
        self.pool_layer_inp = nn.AvgPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False
        )

        # Area layers
        common_layer_params = dict(
            bias=self.bias,
            recurrence_type=self.recurrence_type,
            dt=self.dt,
            tau=self.tau,
            history_length=self.t_feedforward,
            delay_recurrence=self.t_recurrence,
            dynamics_solver=self.dynamics_solver,
            parametrization=nn.utils.parametrizations.weight_norm,
            device=self.device,
        )

        for i, layer_name in enumerate(self.layer_names[1:]):
            setattr(self, f"nonlin_{layer_name}", nn.ReLU(inplace=False))
            setattr(
                self,
                f"tstep_{layer_name}",
                EulerStep(dt=self.dt, tau=self.tau),
            )

            setattr(
                self,
                layer_name,
                RecurrentConnectedConv2d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=3,
                    stride=strides[i],
                    **common_layer_params,
                ),
            )
            if i > 1:
                setattr(
                    self,
                    f"skip_{layer_name}",
                    Skip(
                        in_channels=channels[i - 1],
                        out_channels=channels[i],
                        scale_factor=strides[i - 1],
                        parametrization=nn.utils.parametrizations.weight_norm,
                        bias=False,
                        device=self.device,
                    ),
                )

        self.skip_layer2 = nn.utils.parametrizations.weight_norm(
            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                device=self.device,
            )
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.utils.parametrizations.weight_norm(
                nn.Linear(channels[self.depth], self.n_classes, bias=True),
            ),
        )
        return None


if __name__ == "__main__":
    input_shape = (20, 3, 224, 224)

    model = CordsNet(
        input_dims=input_shape,
        n_classes=10,
        store_responses=True,
        tff=1,
        trc=1,
        tau=10,
        dt=1,
    )
    model.setup("fit")

    # print(torchsummary.summary(model, input_size=input_shape[1:]))
    # print(model.state_dict().keys())

    random_input = torch.randn(1, *input_shape)

    t = 0
    output = model.forward(random_input)

    outputs = model(random_input).squeeze()  # remove batch dimension

    print(f"Random Input ({random_input.shape})")
    print(f"Residual Timesteps: {model.n_residual_timesteps}")
    print(f"Model Output: {outputs.shape}")
