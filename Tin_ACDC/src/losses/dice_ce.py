"""Combined Dice + CrossEntropy loss for segmentation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation."""

    def __init__(self, num_classes: int = 4, smooth: float = 1.0,
                 ignore_bg: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_bg = ignore_bg

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W)
            targets: (B, H, W) long
        """
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, self.num_classes)  # (B, H, W, C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        start_cls = 1 if self.ignore_bg else 0
        dice_sum = 0.0
        count = 0
        for c in range(start_cls, self.num_classes):
            p = probs[:, c]
            g = targets_oh[:, c]
            inter = (p * g).sum(dim=(1, 2))
            union = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2))
            dice = (2.0 * inter + self.smooth) / (union + self.smooth)
            dice_sum += dice.mean()
            count += 1
        return 1.0 - dice_sum / max(count, 1)


class DiceCELoss(nn.Module):
    """Combined Dice + CrossEntropy loss. Default for M1 and M2."""

    def __init__(self, num_classes: int = 4,
                 dice_weight: float = 1.0, ce_weight: float = 1.0):
        super().__init__()
        self.dice = SoftDiceLoss(num_classes)
        self.ce = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        return (self.dice_weight * self.dice(logits, targets)
                + self.ce_weight * self.ce(logits, targets))
