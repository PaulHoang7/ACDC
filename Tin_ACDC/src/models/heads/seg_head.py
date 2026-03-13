"""Segmentation head: projects decoder features to class logits."""
import torch
import torch.nn as nn


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
