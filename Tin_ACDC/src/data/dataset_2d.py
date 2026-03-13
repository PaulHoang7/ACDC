"""PyTorch Dataset for 2D slice-wise ACDC segmentation."""
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class ACDCSliceDataset(Dataset):
    """
    Loads 2D slices from ACDC labeled 3D volumes.

    Each item returns:
        image: (1, H, W) float32 tensor
        mask:  (H, W) long tensor with class ids {0,1,2,3}
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
    ):
        import pandas as pd

        self.transform = transform
        self.clip_percentiles = clip_percentiles
        self.keep_empty = keep_empty

        # Load metadata and split
        df = pd.read_csv(metadata_csv)
        with open(split_file, "r") as f:
            split = json.load(f)
        patient_ids = set(split[split_key])

        # Filter metadata to this split
        df = df[df["patient_id"].isin(patient_ids)]

        # Build slice index: (row_idx, slice_idx)
        self.slices: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            img_vol = nib.load(row["image_path"]).get_fdata()
            mask_vol = nib.load(row["mask_path"]).get_fdata()
            n_slices = img_vol.shape[2]
            for s in range(n_slices):
                mask_slice = mask_vol[:, :, s].astype(np.int64)
                if not self.keep_empty and mask_slice.max() == 0:
                    continue
                self.slices.append({
                    "image_path": row["image_path"],
                    "mask_path": row["mask_path"],
                    "slice_idx": s,
                    "patient_id": row["patient_id"],
                    "phase": row["phase"],
                    "frame_id": row["frame_id"],
                })

    def __len__(self) -> int:
        return len(self.slices)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Per-volume z-score normalization with optional clipping."""
        lo, hi = np.percentile(img, self.clip_percentiles)
        img = np.clip(img, lo, hi)
        mean = img.mean()
        std = img.std() + 1e-8
        return ((img - mean) / std).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.slices[idx]

        # Load full volume and extract slice
        img_vol = nib.load(entry["image_path"]).get_fdata()
        mask_vol = nib.load(entry["mask_path"]).get_fdata()

        image = img_vol[:, :, entry["slice_idx"]].astype(np.float32)
        mask = mask_vol[:, :, entry["slice_idx"]].astype(np.int64)

        # Normalize
        image = self._normalize(image)

        # Apply augmentation
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]  # (1, H, W) or (H, W) tensor
            mask = augmented["mask"]

        # Ensure image has channel dim
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
