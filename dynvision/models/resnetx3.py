import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from dynvision.base import BaseModel

__all__ = ["ResNet20", "ResNet32", "ResNet44", "ResNet56", "ResNet110", "ResNet1202"]


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.expansion * channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        out = F.relu(out)
        return out


class ResNetx3(BaseModel):
    def __init__(
        self,
        block,
        num_blocks=[3, 3, 3],
        input_dims: tuple = (20, 3, 32, 32),
        n_classes: int = 10,
        store_responses: bool = False,
        **kwargs,
    ):
        self.in_channels = 16

        super().__init__(
            block=block,
            num_blocks=num_blocks,
            input_dims=input_dims,
            n_classes=n_classes,
            store_responses=store_responses**kwargs,
        )

        # make all layers trainable
        self.trainable_parameter_names = [p for p in list(self.state_dict().keys())]

        self._define_architecture()

    def _define_architecture(self):
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(self.block, 16, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.block, 32, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.block, 64, self.num_blocks[2], stride=2)
        self.classifier = nn.Linear(64, self.n_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels * block.expansion

        return nn.Sequential(*layers)

    def _forward(
        self, x_0: torch.Tensor, t: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        x_0 = F.relu(self.bn1(self.conv1(x_0)))
        x_1 = self.layer1(x_0)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        out = F.avg_pool2d(x_3, x_3.size()[3])
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        responses = {
            "layer1": x_1,
            "layer2": x_2,
            "layer3": x_3,
            "classifier": out,
        }
        return out, responses


class ResNet20(ResNetx3):
    def __init__(self, **kwargs):
        super().__init__(BasicBlock, num_blocks=[3, 3, 3], **kwargs)


class ResNet32(ResNetx3):
    def __init__(self, **kwargs):
        super().__init__(BasicBlock, num_blocks=[5, 5, 5], **kwargs)


class ResNet44(ResNetx3):
    def __init__(self, **kwargs):
        super().__init__(BasicBlock, num_blocks=[7, 7, 7], **kwargs)


class ResNet56(ResNetx3):
    def __init__(self, **kwargs):
        super().__init__(BasicBlock, num_blocks=[9, 9, 9], **kwargs)


class ResNet110(ResNetx3):
    def __init__(self, **kwargs):
        super().__init__(BasicBlock, num_blocks=[18, 18, 18], **kwargs)


class ResNet1202(ResNetx3):
    def __init__(self, **kwargs):
        super().__init__(BasicBlock, num_blocks=[200, 200, 200], **kwargs)


if __name__ == "__main__":

    input_dims = (20, 3, 32, 32)

    random_input = torch.randn(1, *input_dims)

    model = ResNet20(input_dims=input_dims)

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
