"""Boundary and deep supervision losses for M3."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


def generate_boundary_target(mask: np.ndarray,
                             kernel_size: int = 3) -> np.ndarray:
    """
    Generate boundary map from a segmentation mask via morphological erosion.
    mask: (H, W) integer array with class ids.
    Returns: (H, W) float32 boundary map (0 or 1).
    """
    struct = ndimage.generate_binary_structure(2, 1)
    boundary = np.zeros_like(mask, dtype=np.float32)
    for cls_id in np.unique(mask):
        if cls_id == 0:
            continue
        cls_mask = (mask == cls_id).astype(np.float32)
        eroded = ndimage.binary_erosion(cls_mask, structure=struct,
                                        iterations=1).astype(np.float32)
        boundary += cls_mask - eroded
    return (boundary > 0).astype(np.float32)


class BoundaryBCELoss(nn.Module):
    """BCE loss on boundary prediction."""

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, 1, H, W) raw boundary logits
        targets: (B, 1, H, W) binary boundary map
        """
        return F.binary_cross_entropy_with_logits(logits, targets)


class DeepSupervisionLoss(nn.Module):
    """Weighted sum of auxiliary segmentation losses."""

    def __init__(self, base_loss: nn.Module, weights: list = None):
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights

    def forward(self, aux_logits_list: list,
                targets: torch.Tensor) -> torch.Tensor:
        n = len(aux_logits_list)
        if self.weights is None:
            w = [1.0 / n] * n
        else:
            w = self.weights
        loss = 0.0
        for logits, weight in zip(aux_logits_list, w):
            loss += weight * self.base_loss(logits, targets)
        return loss
