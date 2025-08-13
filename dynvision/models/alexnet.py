import torch
import torch.nn as nn
from torchvision import models as torch_models

from dynvision.base import BaseModel
from dynvision.model_components import (
    InputAdaption,
    RecurrentConnectedConv2d,
)

__all__ = ["AlexNet"]


class AlexNet(BaseModel):
    def __init__(
        self,
        input_dims: tuple = (20, 3, 224, 224),
        n_classes: int = 1000,
        dropout: float = 0.5,
        recurrence_type: str = "none",
        fixed_self_weight: float = None,
        t_feedforward: float = 0,
        t_recurrence: float = 0,
        dt: float = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            input_dims=input_dims,
            n_classes=n_classes,
            dropout=dropout,
            recurrence_type=recurrence_type,
            fixed_self_weight=fixed_self_weight,
            t_feedforward=t_feedforward,
            t_recurrence=t_recurrence,
            dt=dt,
            **kwargs,
        )
        self._define_architecture()
        self._init_parameters()

    def setup(self, stage):
        super().setup(stage)

    def _init_parameters(self):
        self.load_pretrained_state_dict(check_mismatch_layer=["classifier.7"])

        # make only the classifier trainable
        self.trainable_parameter_names = [
            p for p in list(self.state_dict().keys()) if "classifier.7" in p
        ]
        return None

    def download_pretrained_state_dict(self):
        pretrained_weights = torch_models.AlexNet_Weights.DEFAULT
        pretrained_weights = torch_models.AlexNet_Weights.verify(pretrained_weights)
        return pretrained_weights.get_state_dict()

    def translate_pretrained_layer_names(self):
        translate_layer_names = {
            "features.0": "layer1.conv",
            "features.3": "layer2.conv",
            "features.6": "layer3.conv",
            "features.8": "layer4.conv",
            "features.10": "layer5.conv",
            "classifier.1": "classifier.2",
            "classifier.4": "classifier.5",
            "classifier.6": "classifier.7",
        }
        return translate_layer_names

    def _define_architecture(self):
        self.layer_names = ["layer1", "layer2", "layer3", "layer4", "layer5"]
        self.layer_operations = [
            "layer",  # apply (recurrent) convolutional layer
            "nonlin",  # apply nonlinearity
            "record",  # record activations in responses dict
            "delay",  # set and get delayed activations for next layer
            "pool",  # apply pooling
            "norm",  # apply normalization
        ]
        self.relu = nn.ReLU(inplace=True)

        # Layer 1
        self.layer1 = RecurrentConnectedConv2d(
            in_channels=3,  # self.in_channels,
            out_channels=64,
            kernel_size=11,
            stride=4,
            padding=2,
            fixed_self_weight=self.fixed_self_weight,
            recurrence_type=self.recurrence_type,
        )
        self.pool_layer1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Layer 2
        self.layer2 = RecurrentConnectedConv2d(
            in_channels=64,
            out_channels=192,
            kernel_size=5,
            padding=2,
            fixed_self_weight=self.fixed_self_weight,
            recurrence_type=self.recurrence_type,
        )
        self.pool_layer2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Layer 3
        self.layer3 = RecurrentConnectedConv2d(
            in_channels=192,
            out_channels=384,
            kernel_size=3,
            padding=1,
            fixed_self_weight=self.fixed_self_weight,
            recurrence_type=self.recurrence_type,
        )

        # Layer 4
        self.layer4 = RecurrentConnectedConv2d(
            in_channels=384,
            out_channels=256,
            kernel_size=3,
            padding=1,
            fixed_self_weight=self.fixed_self_weight,
            recurrence_type=self.recurrence_type,
        )

        # Layer 5
        self.layer5 = RecurrentConnectedConv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            padding=1,
            fixed_self_weight=self.fixed_self_weight,
            recurrence_type=self.recurrence_type,
        )
        self.pool_layer5 = nn.MaxPool2d(kernel_size=3, stride=2)

        # AvgPool
        self.norm_layer5 = nn.AdaptiveAvgPool2d((6, 6))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(p=self.dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.n_classes),
        )

    def reset(self):
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]:
            layer.reset()


if __name__ == "__main__":

    input_dims = (20, 1, 224, 224)

    random_input = torch.randn(1, *input_dims)

    model = AlexNet(input_dims=input_dims)
    model.setup("fit")

    output = model(random_input)

    trainable_params = [
        f"{name} [{tuple(param.shape)}]"
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    print("Trainable Parameters:\n\t", "\n\t".join(trainable_params))
    print()
    print(f"Random Input ({tuple(random_input.shape)})")
    print(f"Model Output ({tuple(output.shape)})")
