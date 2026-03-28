"""U-Net decoder with configurable skip fusion."""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU repeated twice."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """Attention gate for skip connections (used in AttUNetR34)."""

    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
        super().__init__()
        self.W_gate = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.W_skip = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self, gate: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        # Align spatial dims
        if g.shape[2:] != s.shape[2:]:
            g = F.interpolate(g, size=s.shape[2:], mode="bilinear",
                              align_corners=False)
        att = self.psi(self.relu(g + s))
        return skip * att


class DecoderBlock(nn.Module):
    """Single decoder stage: upsample + skip fusion + conv block."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        use_attention: bool = False,
    ):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.att_gate = AttentionGate(in_ch, skip_ch, skip_ch // 2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                          align_corners=False)
        if self.use_attention:
            skip = self.att_gate(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetDecoder(nn.Module):
    """
    4-stage U-Net decoder.

    Args:
        encoder_channels: [64, 64, 128, 256, 512] from ResNet34
        decoder_channels: [256, 128, 64, 32]
        use_attention: whether to use attention gates on skip connections
    """

    def __init__(
        self,
        encoder_channels: List[int] = [64, 64, 128, 256, 512],
        decoder_channels: List[int] = [256, 128, 64, 32],
        use_attention: bool = False,
    ):
        super().__init__()
        # Decoder goes from deepest to shallowest
        # Stage 0: 512 + 256 -> 256
        # Stage 1: 256 + 128 -> 128
        # Stage 2: 128 + 64  -> 64
        # Stage 3: 64  + 64  -> 32
        enc = list(reversed(encoder_channels))  # [512, 256, 128, 64, 64]
        self._decoder_channels = list(decoder_channels)
        self.blocks = nn.ModuleList()
        in_ch = enc[0]
        for i, out_ch in enumerate(decoder_channels):
            skip_ch = enc[i + 1]
            self.blocks.append(
                DecoderBlock(in_ch, skip_ch, out_ch,
                             use_attention=use_attention)
            )
            in_ch = out_ch

    @property
    def out_channels(self) -> int:
        return self._decoder_channels[-1]

    def forward(
        self, features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Args:
            features: encoder outputs [x0, x1, x2, x3, x4] shallow->deep
        Returns:
            decoder features at each stage [d0, d1, d2, d3] deep->shallow
        """
        skips = list(reversed(features))  # [x4, x3, x2, x1, x0]
        x = skips[0]
        decoder_features = []
        for i, block in enumerate(self.blocks):
            x = block(x, skips[i + 1])
            decoder_features.append(x)
        return decoder_features
