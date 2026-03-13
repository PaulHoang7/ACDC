"""Evaluation metrics: Dice, IoU, HD95 at patient-volume level."""
from typing import Dict

import numpy as np
from medpy.metric.binary import hd95 as compute_hd95


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient for a single binary pair."""
    inter = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0 if inter == 0 else 0.0
    return 2.0 * inter / total


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """IoU / Jaccard for a single binary pair."""
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union


def hausdorff95(pred: np.ndarray, gt: np.ndarray,
                voxel_spacing: tuple = (1.0, 1.0, 1.0)) -> float:
    """HD95 for a single binary pair. Returns inf if either is empty."""
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf")
    return compute_hd95(pred, gt, voxelspacing=voxel_spacing)


def compute_volume_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int = 4,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """
    Compute per-class and mean metrics on a full 3D volume.

    Args:
        pred: (D, H, W) integer prediction
        gt:   (D, H, W) integer ground truth
        num_classes: number of classes including background
        voxel_spacing: voxel dimensions for HD95

    Returns:
        dict with dice_rv, dice_myo, dice_lv, mean_dice,
        iou_rv, iou_myo, iou_lv, mean_iou,
        hd95_rv, hd95_myo, hd95_lv, mean_hd95
    """
    class_names = {1: "rv", 2: "myo", 3: "lv"}
    metrics = {}
    dices, ious, hd95s = [], [], []

    for cls_id, cls_name in class_names.items():
        p = (pred == cls_id).astype(np.uint8)
        g = (gt == cls_id).astype(np.uint8)

        d = dice_score(p, g)
        j = iou_score(p, g)
        h = hausdorff95(p, g, voxel_spacing)

        metrics[f"dice_{cls_name}"] = d
        metrics[f"iou_{cls_name}"] = j
        metrics[f"hd95_{cls_name}"] = h

        dices.append(d)
        ious.append(j)
        if h != float("inf"):
            hd95s.append(h)

    metrics["mean_dice"] = np.mean(dices)
    metrics["mean_iou"] = np.mean(ious)
    metrics["mean_hd95"] = np.mean(hd95s) if hd95s else float("inf")

    return metrics
