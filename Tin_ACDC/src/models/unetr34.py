"""M1: UNetR34 — plain U-Net with ResNet34 encoder. Baseline model."""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import ResNet34Encoder
from .decoders import UNetDecoder
from .heads import SegmentationHead


class UNetR34(nn.Module):
    """
    Standard U-Net with ResNet34 encoder.
    No attention, no boundary head, no deep supervision.

    Returns: {"seg_logits": (B, C, H, W)}
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        pretrained: bool = False,
        decoder_channels=(256, 128, 64, 32),
    ):
        super().__init__()
        self.encoder = ResNet34Encoder(in_channels, pretrained=pretrained)
        self.decoder = UNetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=list(decoder_channels),
            use_attention=False,
        )
        self.seg_head = SegmentationHead(decoder_channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_size = x.shape[2:]
        features = self.encoder(x)
        dec_features = self.decoder(features)
        logits = self.seg_head(dec_features[-1])
        logits = F.interpolate(logits, size=input_size, mode="bilinear",
                               align_corners=False)
        return {"seg_logits": logits}
