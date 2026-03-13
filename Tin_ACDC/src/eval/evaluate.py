"""Full evaluation pipeline: predict -> reconstruct -> compute metrics."""
from pathlib import Path
from typing import Any, Dict, List

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_volume_metrics
from .reconstruct_volume import collect_slice_predictions, reconstruct_volume


@torch.no_grad()
def predict_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> List[Dict]:
    """Run inference on a full dataset, return slice-level predictions."""
    model.eval()
    predictions = []
    for batch in tqdm(loader, desc="Predict"):
        images = batch["image"].to(device)
        outputs = model(images)
        preds = outputs["seg_logits"].argmax(dim=1).cpu().numpy()

        for i in range(len(preds)):
            meta = {k: v[i] if isinstance(v, list) else v
                    for k, v in batch["meta"].items()}
            # Convert tensor meta values
            for k, v in meta.items():
                if isinstance(v, torch.Tensor):
                    meta[k] = v.item()
            predictions.append({"pred": preds[i], "meta": meta})
    return predictions


def evaluate_patient_volumes(
    predictions: List[Dict],
    metadata_df: pd.DataFrame,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
) -> pd.DataFrame:
    """
    Evaluate at patient-volume level.

    Returns a DataFrame with per-patient, per-frame metrics.
    """
    volumes = collect_slice_predictions(predictions)
    results = []

    for pid, frames in volumes.items():
        for fid, slices in frames.items():
            pred_vol = reconstruct_volume(slices)

            # Load GT volume
            row = metadata_df[
                (metadata_df["patient_id"] == pid)
                & (metadata_df["frame_id"] == fid)
            ].iloc[0]
            gt_vol = nib.load(row["mask_path"]).get_fdata().astype(np.int64)

            # Resize pred if needed (evaluation should match GT shape)
            if pred_vol.shape != gt_vol.shape:
                from skimage.transform import resize
                pred_vol = resize(
                    pred_vol.astype(float), gt_vol.shape,
                    order=0, preserve_range=True, anti_aliasing=False
                ).astype(np.int64)

            metrics = compute_volume_metrics(
                pred_vol, gt_vol, voxel_spacing=voxel_spacing
            )
            metrics["patient_id"] = pid
            metrics["frame_id"] = fid
            metrics["phase"] = row["phase"]
            results.append(metrics)

    return pd.DataFrame(results)
