"""ResNet34-style encoder shared by all 3 models."""
from typing import List

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ResNet34Encoder(nn.Module):
    """
    5-stage ResNet34 encoder.
    Output feature channels: [64, 64, 128, 256, 512]
    """

    def __init__(self, in_channels: int = 1, pretrained: bool = False):
        super().__init__()
        weights = tv_models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = tv_models.resnet34(weights=weights)

        # Adapt first conv for single-channel MRI input
        if in_channels != 3:
            old_conv = resnet.conv1
            self.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            if pretrained:
                # Average pretrained weights across input channels
                with torch.no_grad():
                    self.conv1.weight.copy_(
                        old_conv.weight.mean(dim=1, keepdim=True).repeat(
                            1, in_channels, 1, 1
                        )
                    )
        else:
            self.conv1 = resnet.conv1

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64
        self.layer2 = resnet.layer2  # 128
        self.layer3 = resnet.layer3  # 256
        self.layer4 = resnet.layer4  # 512

    @property
    def out_channels(self) -> List[int]:
        return [64, 64, 128, 256, 512]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return features at 5 stages for skip connections."""
        x0 = self.relu(self.bn1(self.conv1(x)))  # stride 2, 64ch
        x1 = self.layer1(self.maxpool(x0))        # stride 4, 64ch
        x2 = self.layer2(x1)                      # stride 8, 128ch
        x3 = self.layer3(x2)                      # stride 16, 256ch
        x4 = self.layer4(x3)                      # stride 32, 512ch
        return [x0, x1, x2, x3, x4]
