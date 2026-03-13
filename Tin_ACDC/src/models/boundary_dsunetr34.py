"""M3: BoundaryDSUNetR34 — boundary-aware U-Net with deep supervision."""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import ResNet34Encoder
from .decoders import UNetDecoder
from .heads import SegmentationHead, BoundaryHead, DeepSupervisionHead


class BoundaryDSUNetR34(nn.Module):
    """
    Boundary-aware U-Net with deep supervision. Proposed thesis model.

    Returns: {
        "seg_logits": (B, C, H, W),
        "boundary_logits": (B, 1, H, W),
        "aux_seg_logits": [(B, C, H, W), ...],  # from deep supervision
    }
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        pretrained: bool = False,
        decoder_channels: Tuple[int, ...] = (256, 128, 64, 32),
        ds_stages: Tuple[int, ...] = (0, 1),
    ):
        super().__init__()
        self.ds_stages = ds_stages

        self.encoder = ResNet34Encoder(in_channels, pretrained=pretrained)
        self.decoder = UNetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=list(decoder_channels),
            use_attention=False,
        )
        self.seg_head = SegmentationHead(decoder_channels[-1], num_classes)
        self.boundary_head = BoundaryHead(decoder_channels[-1])

        # Deep supervision heads at specified decoder stages
        self.ds_heads = nn.ModuleList()
        for stage_idx in ds_stages:
            self.ds_heads.append(
                DeepSupervisionHead(decoder_channels[stage_idx], num_classes)
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_size = x.shape[2:]
        features = self.encoder(x)
        dec_features = self.decoder(features)

        # Main segmentation output
        seg_logits = self.seg_head(dec_features[-1])
        seg_logits = F.interpolate(seg_logits, size=input_size,
                                   mode="bilinear", align_corners=False)

        # Boundary output
        boundary_logits = self.boundary_head(dec_features[-1])
        boundary_logits = F.interpolate(boundary_logits, size=input_size,
                                        mode="bilinear", align_corners=False)

        # Deep supervision auxiliary outputs
        aux_seg_logits = []
        for i, stage_idx in enumerate(self.ds_stages):
            aux = self.ds_heads[i](dec_features[stage_idx], input_size)
            aux_seg_logits.append(aux)

        return {
            "seg_logits": seg_logits,
            "boundary_logits": boundary_logits,
            "aux_seg_logits": aux_seg_logits,
        }
