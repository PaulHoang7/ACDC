"""
PyTorch Dataset for 2D slice-wise ACDC segmentation.

Supports two loading backends:
    1. raw:    load from NIfTI volumes on the fly (slower, no disk overhead)
    2. cached: load from preprocessed .npz files (faster, requires preprocess.py)
"""
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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
        clip_percentiles: Tuple[float, float] = (0.5, 99.5),
        keep_empty: bool = False,
        backend: str = "auto",
    ):
        """
        Args:
            metadata_csv: path to patient_metadata.csv OR slice_metadata.csv
            split_file:   path to split JSON (dev_split.json or fold_X.json)
            split_key:    "train" or "val"
            transform:    albumentations Compose (applied after loading)
            clip_percentiles: for on-the-fly z-score normalization (raw mode)
            keep_empty:   include all-background slices
            backend:      "raw"   = load from NIfTI every time
                          "cached"= load from .npz (requires preprocess.py)
                          "auto"  = use cached if cached_image_path col exists
        """
        self.transform = transform
        self.clip_percentiles = clip_percentiles
        self.keep_empty = keep_empty

        df = pd.read_csv(metadata_csv)
        with open(split_file, "r") as f:
            split = json.load(f)
        patient_ids = set(split[split_key])

        # Filter to this split
        df = df[df["patient_id"].isin(patient_ids)].reset_index(drop=True)

        # Detect backend
        has_cache = ("cached_image_path" in df.columns
                     and df["cached_image_path"].notna().any()
                     and df["cached_image_path"].str.len().max() > 0)

        if backend == "auto":
            self.backend = "cached" if has_cache else "raw"
        else:
            self.backend = backend

        # Build slice index
        if self.backend == "cached":
            self.slices = self._build_cached_index(df)
        else:
            self.slices = self._build_raw_index(df)

    # ── Index builders ──────────────────────────────────────────────

    def _build_cached_index(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Build index from slice_metadata.csv (already slice-level)."""
        slices = []
        for _, row in df.iterrows():
            if not self.keep_empty and not row.get("has_foreground", True):
                continue
            slices.append({
                "cached_path": row["cached_image_path"],
                "patient_id": row["patient_id"],
                "phase": row["phase"],
                "frame_id": int(row["frame_id"]),
                "slice_idx": int(row["slice_index"]),
            })
        return slices

    def _build_raw_index(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Build index from patient_metadata.csv (volume-level → expand to slices)."""
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
        """Load from NIfTI and normalize on the fly."""
        img_vol = nib.load(entry["image_path"]).get_fdata()
        mask_vol = nib.load(entry["mask_path"]).get_fdata()
        image = img_vol[:, :, entry["slice_idx"]].astype(np.float32)
        mask = mask_vol[:, :, entry["slice_idx"]].astype(np.int64)
        image = self._normalize(image)
        return image, mask

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Per-slice z-score normalization with clipping."""
        lo, hi = np.percentile(img, self.clip_percentiles)
        img = np.clip(img, lo, hi)
        mean = img.mean()
        std = img.std()
        if std < 1e-8:
            return np.zeros_like(img, dtype=np.float32)
        return ((img - mean) / std).astype(np.float32)

    # ── Main interface ──────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.slices[idx]

        if self.backend == "cached":
            image, mask = self._load_cached(entry)
        else:
            image, mask = self._load_raw(entry)

        # Apply augmentation (albumentations expects HW arrays)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Ensure correct tensor format
        if isinstance(image, torch.Tensor):
            if image.ndim == 2:
                image = image.unsqueeze(0)
        else:
            image = torch.from_numpy(image).unsqueeze(0)

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()

        return {
            "image": image.float(),
            "mask": mask.long(),
            "meta": {
                "patient_id": entry["patient_id"],
                "phase": entry["phase"],
                "frame_id": entry["frame_id"],
                "slice_idx": entry["slice_idx"],
            },
        }
