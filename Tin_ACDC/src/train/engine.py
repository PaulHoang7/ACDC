"""Training and validation engine — single epoch loops."""
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses.boundary_losses import generate_boundary_target
import numpy as np


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    seg_loss_fn: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    boundary_loss_fn: Optional[nn.Module] = None,
    ds_loss_fn: Optional[nn.Module] = None,
    lambda_b: float = 0.2,
    lambda_ds: float = 0.2,
    grad_clip: float = 0.0,
) -> Dict[str, float]:
    """Run one training epoch. Returns dict of averaged metrics."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        with autocast(enabled=scaler is not None):
            outputs = model(images)
            loss = seg_loss_fn(outputs["seg_logits"], masks)

            # Boundary loss (M3 only)
            if boundary_loss_fn is not None and "boundary_logits" in outputs:
                # Generate boundary targets on the fly
                masks_np = masks.cpu().numpy()
                boundary_targets = np.stack(
                    [generate_boundary_target(m) for m in masks_np]
                )
                boundary_targets = torch.from_numpy(boundary_targets).float()
                boundary_targets = boundary_targets.unsqueeze(1).to(device)
                loss = loss + lambda_b * boundary_loss_fn(
                    outputs["boundary_logits"], boundary_targets
                )

            # Deep supervision loss (M3 only)
            if ds_loss_fn is not None and "aux_seg_logits" in outputs:
                loss = loss + lambda_ds * ds_loss_fn(
                    outputs["aux_seg_logits"], masks
                )

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"train_loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    seg_loss_fn: nn.Module,
    device: torch.device,
    num_classes: int = 4,
) -> Dict[str, float]:
    """Run validation. Returns loss and per-class Dice."""
    model.eval()
    total_loss = 0.0
    # Per-class intersection and union accumulators
    inter = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    n_batches = 0

    for batch in tqdm(loader, desc="Val", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        outputs = model(images)
        loss = seg_loss_fn(outputs["seg_logits"], masks)
        total_loss += loss.item()
        n_batches += 1

        preds = outputs["seg_logits"].argmax(dim=1)
        for c in range(num_classes):
            p = (preds == c)
            g = (masks == c)
            inter[c] += (p & g).sum().float()
            union[c] += p.sum().float() + g.sum().float()

    dice_per_class = (2.0 * inter + 1.0) / (union + 1.0)
    # Foreground classes: RV=1, MYO=2, LV=3
    metrics = {
        "val_loss": total_loss / max(n_batches, 1),
        "val_dice_rv": dice_per_class[1].item(),
        "val_dice_myo": dice_per_class[2].item(),
        "val_dice_lv": dice_per_class[3].item(),
        "val_mean_dice": dice_per_class[1:].mean().item(),
    }
    return metrics
