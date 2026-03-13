"""Deep supervision auxiliary heads for M3."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSupervisionHead(nn.Module):
    """Auxiliary segmentation head at an intermediate decoder stage."""

    def __init__(self, in_channels: int, num_classes: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(
        self, x: torch.Tensor, target_size: tuple
    ) -> torch.Tensor:
        """Predict and upsample to target spatial size."""
        logits = self.conv(x)
        if logits.shape[2:] != target_size:
            logits = F.interpolate(
                logits, size=target_size, mode="bilinear",
                align_corners=False
            )
        return logits
