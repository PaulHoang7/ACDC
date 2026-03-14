"""
PyTorch Dataset for 2D slice-wise ACDC segmentation.

Supports two loading backends:
    1. raw:    load from NIfTI volumes on the fly (slower, no disk overhead)
    2. cached: load from preprocessed .npz files (faster, requires preprocess.py)

Supports two metadata sources:
    - patient_metadata.csv (volume-level) → expands to slices at runtime (raw)
    - slice_metadata.csv   (slice-level)  → direct index (raw or cached)
"""
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .transforms import get_train_transforms, get_val_transforms


class ACDCSliceDataset(Dataset):
    """
    2D slice dataset for ACDC cardiac MRI.

    Each item returns:
        image: (1, H, W) float32 tensor
        mask:  (H, W) long tensor  {0=BG, 1=RV, 2=MYO, 3=LV}
        meta:  dict with patient_id, phase, frame_id, slice_idx
    """

    def __init__(
        self,
        metadata_csv: str,
        split_file: str,
        split_key: str = "train",
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (256, 256),
        clip_percentiles: Tuple[float, float] = (0.5, 99.5),
        keep_empty: bool = False,
        backend: str = "auto",
    ):
        """
        Args:
            metadata_csv:     path to slice_metadata.csv or patient_metadata.csv
            split_file:       path to split JSON (dev_split.json or fold_X.json)
            split_key:        "train" or "val"
            transform:        albumentations Compose (applied after loading)
            target_size:      (H, W) for on-the-fly resize in raw mode
            clip_percentiles: for on-the-fly z-score normalization (raw mode)
            keep_empty:       include all-background slices
            backend:          "raw"   = load from NIfTI every time
                              "cached"= load from .npz (requires preprocess.py --cache)
                              "auto"  = cached if .npz paths exist, else raw
        """
        self.transform = transform
        self.target_size = target_size
        self.clip_percentiles = clip_percentiles
        self.keep_empty = keep_empty

        df = pd.read_csv(metadata_csv)
        with open(split_file, "r") as f:
            split = json.load(f)
        patient_ids = set(split[split_key])

        # Filter to this split's patients
        df = df[df["patient_id"].isin(patient_ids)].reset_index(drop=True)

        # Detect CSV type: slice-level vs volume-level
        self.is_slice_csv = "slice_index" in df.columns

        # Detect backend
        has_cache = (
            "cached_image_path" in df.columns
            and df["cached_image_path"].notna().any()
            and (df["cached_image_path"].str.len() > 0).any()
        )

        if backend == "auto":
            self.backend = "cached" if has_cache else "raw"
        else:
            self.backend = backend

        # Build slice index
        if self.is_slice_csv:
            self.slices = self._build_from_slice_csv(df)
        else:
            self.slices = self._build_from_volume_csv(df)

    # ── Index builders ──────────────────────────────────────────────

    def _build_from_slice_csv(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Build index from slice_metadata.csv (already slice-level)."""
        slices = []
        for _, row in df.iterrows():
            if not self.keep_empty and not row.get("has_foreground", True):
                continue

            entry: Dict[str, Any] = {
                "patient_id": row["patient_id"],
                "phase": row["phase"],
                "frame_id": int(row["frame_id"]),
                "slice_idx": int(row["slice_index"]),
                # Raw NIfTI paths (for raw backend fallback)
                "image_path": row.get("image_path_raw", row.get("image_path", "")),
                "mask_path": row.get("mask_path_raw", row.get("mask_path", "")),
            }

            if self.backend == "cached":
                entry["cached_path"] = row.get("cached_image_path", "")
            slices.append(entry)
        return slices

    def _build_from_volume_csv(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Build index from patient_metadata.csv (volume-level → expand)."""
        slices = []
        for _, row in df.iterrows():
            mask_vol = nib.load(row["mask_path"]).get_fdata()
            n_slices = mask_vol.shape[2]
            for s in range(n_slices):
                if not self.keep_empty and mask_vol[:, :, s].max() == 0:
                    continue
                slices.append({
                    "image_path": row["image_path"],
                    "mask_path": row["mask_path"],
                    "slice_idx": s,
                    "patient_id": row["patient_id"],
                    "phase": row["phase"],
                    "frame_id": int(row["frame_id"]),
                })
        return slices

    # ── Loading ─────────────────────────────────────────────────────

    def _load_cached(self, entry: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Load from .npz file (already normalized + resized)."""
        data = np.load(entry["cached_path"])
        return data["image"].astype(np.float32), data["mask"].astype(np.int64)

    def _load_raw(self, entry: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Load from NIfTI, normalize per-volume, resize slice."""
        img_vol = nib.load(entry["image_path"]).get_fdata().astype(np.float32)
        mask_vol = nib.load(entry["mask_path"]).get_fdata().astype(np.int64)

        # Per-volume z-score normalization
        img_vol = self._normalize_volume(img_vol)

        image = img_vol[:, :, entry["slice_idx"]]
        mask = mask_vol[:, :, entry["slice_idx"]]

        # Resize if needed
        h, w = image.shape
        th, tw = self.target_size
        if h != th or w != tw:
            from skimage.transform import resize as sk_resize
            image = sk_resize(image, (th, tw), order=1,
                              preserve_range=True, anti_aliasing=False
                              ).astype(np.float32)
            mask = sk_resize(mask.astype(np.float64), (th, tw), order=0,
                             preserve_range=True, anti_aliasing=False
                             ).astype(np.int64)

        return image, mask

    def _normalize_volume(self, vol: np.ndarray) -> np.ndarray:
        """Per-volume z-score normalization with clipping."""
        lo, hi = np.percentile(vol, self.clip_percentiles)
        vol = np.clip(vol, lo, hi)
        mean = vol.mean()
        std = vol.std()
        if std < 1e-8:
            return np.zeros_like(vol, dtype=np.float32)
        return ((vol - mean) / std).astype(np.float32)

    # ── Main interface ──────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.slices[idx]

        if self.backend == "cached":
            image, mask = self._load_cached(entry)
        else:
            image, mask = self._load_raw(entry)

        # Apply augmentation (albumentations expects HW numpy arrays)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert to tensors if not already (ToTensorV2 handles this)
        if isinstance(image, torch.Tensor):
            if image.ndim == 2:
                image = image.unsqueeze(0)  # (H,W) → (1,H,W)
        else:
            image = torch.from_numpy(image).unsqueeze(0)

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()

        return {
            "image": image.float(),     # (1, H, W)
            "mask": mask.long(),         # (H, W)
            "meta": {
                "patient_id": entry["patient_id"],
                "phase": entry["phase"],
                "frame_id": entry["frame_id"],
                "slice_idx": entry["slice_idx"],
            },
        }


# ──────────────────────────────────────────────────────────────────────
# Convenience: build train/val dataloaders from configs
# ──────────────────────────────────────────────────────────────────────

def build_dataloaders(
    metadata_csv: str,
    split_file: str,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    target_size: Tuple[int, int] = (256, 256),
    aug_cfg: Optional[Dict[str, Any]] = None,
    keep_empty: bool = False,
    backend: str = "auto",
) -> Dict[str, DataLoader]:
    """
    Build train + val DataLoaders from a single split file.

    Args:
        metadata_csv: path to slice_metadata.csv or patient_metadata.csv
        split_file:   path to dev_split.json or fold_X.json
        batch_size:   training batch size (val uses same)
        num_workers:  dataloader workers
        pin_memory:   pin memory for GPU transfer
        target_size:  (H, W) image size
        aug_cfg:      augmentation config dict (from data.yaml)
        keep_empty:   include background-only slices
        backend:      "auto", "raw", or "cached"

    Returns:
        {"train": DataLoader, "val": DataLoader}
    """
    train_tf = get_train_transforms(target_size, aug_cfg)
    val_tf = get_val_transforms(target_size)

    train_ds = ACDCSliceDataset(
        metadata_csv=metadata_csv,
        split_file=split_file,
        split_key="train",
        transform=train_tf,
        target_size=target_size,
        keep_empty=keep_empty,
        backend=backend,
    )

    val_ds = ACDCSliceDataset(
        metadata_csv=metadata_csv,
        split_file=split_file,
        split_key="val",
        transform=val_tf,
        target_size=target_size,
        keep_empty=keep_empty,
        backend=backend,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return {"train": train_loader, "val": val_loader}
