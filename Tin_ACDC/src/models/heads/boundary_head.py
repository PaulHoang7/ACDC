"""Boundary prediction head for M3."""
import torch
import torch.nn as nn


class BoundaryHead(nn.Module):
    """Predicts single-channel boundary probability map."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
