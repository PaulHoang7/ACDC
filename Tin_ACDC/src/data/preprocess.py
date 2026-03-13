"""
Preprocessing pipeline for ACDC 2D slice-wise segmentation.

Pipeline steps (per volume):
    1. Load NIfTI image + mask
    2. Verify shape match
    3. Per-volume z-score normalization (clip → normalize)
    4. Resize each slice to target_size (image=bilinear, mask=nearest)
    5. Extract 2D slices with full metadata tracing

Two output modes:
    - metadata_only: build slice-level CSV without saving arrays
    - cache:         save each slice as .npz + build slice-level CSV
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from skimage.transform import resize
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────────────────────────────────

def load_nifti_pair(image_path: str, mask_path: str
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an image-mask NIfTI pair. Returns (image, mask, affine)."""
    img_nii = nib.load(image_path)
    mask_nii = nib.load(mask_path)
    image = img_nii.get_fdata().astype(np.float32)
    mask = mask_nii.get_fdata().astype(np.int64)
    affine = img_nii.affine
    return image, mask, affine


def check_shape(image: np.ndarray, mask: np.ndarray,
                patient_id: str, frame_id: int) -> None:
    """Raise if image and mask shapes disagree."""
    if image.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch for {patient_id} frame {frame_id}: "
            f"image {image.shape} vs mask {mask.shape}"
        )


def normalize_volume(volume: np.ndarray,
                     clip_percentiles: Tuple[float, float] = (0.5, 99.5),
                     ) -> np.ndarray:
    """Per-volume z-score normalization with intensity clipping."""
    lo, hi = np.percentile(volume, clip_percentiles)
    volume = np.clip(volume, lo, hi)
    mean = volume.mean()
    std = volume.std()
    if std < 1e-8:
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume - mean) / std).astype(np.float32)


def resize_slice(
    image_2d: np.ndarray,
    mask_2d: np.ndarray,
    target_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize a single 2D slice.
      image → bilinear (order=1)
      mask  → nearest  (order=0)
    """
    img_out = resize(
        image_2d, target_size, order=1,
        preserve_range=True, anti_aliasing=False,
    ).astype(np.float32)

    mask_out = resize(
        mask_2d.astype(np.float64), target_size, order=0,
        preserve_range=True, anti_aliasing=False,
    ).astype(np.int64)

    return img_out, mask_out


def verify_class_ids(mask_2d: np.ndarray,
                     expected: Tuple[int, ...] = (0, 1, 2, 3)) -> None:
    """Warn if unexpected class ids appear in a mask slice."""
    unique = set(np.unique(mask_2d).astype(int))
    unexpected = unique - set(expected)
    if unexpected:
        raise ValueError(f"Unexpected mask classes: {unexpected}")


# ──────────────────────────────────────────────────────────────────────
# Core pipeline
# ──────────────────────────────────────────────────────────────────────

def preprocess_volume(
    image_path: str,
    mask_path: str,
    patient_id: str,
    frame_id: int,
    phase: str,
    source_split: str,
    target_size: Tuple[int, int] = (256, 256),
    clip_percentiles: Tuple[float, float] = (0.5, 99.5),
    keep_empty: bool = False,
) -> List[Dict[str, Any]]:
    """
    Preprocess one 3D volume → list of 2D slice records.

    Each record:
        image_array:  (H, W) float32 normalized + resized
        mask_array:   (H, W) int64 resized
        + full metadata
    """
    image, mask, affine = load_nifti_pair(image_path, mask_path)
    check_shape(image, mask, patient_id, frame_id)

    # Step 1: per-volume normalization (on full 3D)
    image = normalize_volume(image, clip_percentiles)

    # Step 2: extract & resize each slice
    n_slices = image.shape[2]
    voxel_spacing = tuple(np.abs(np.diag(affine)[:3]).tolist())
    records = []

    for s in range(n_slices):
        img_slice = image[:, :, s]
        mask_slice = mask[:, :, s]

        # Skip empty slices if requested
        if not keep_empty and mask_slice.max() == 0:
            continue

        verify_class_ids(mask_slice)

        # Resize
        img_resized, mask_resized = resize_slice(
            img_slice, mask_slice, target_size
        )

        records.append({
            "patient_id": patient_id,
            "source_split": source_split,
            "frame_id": frame_id,
            "phase": phase,
            "slice_index": s,
            "n_slices_in_volume": n_slices,
            "original_height": image.shape[0],
            "original_width": image.shape[1],
            "target_height": target_size[0],
            "target_width": target_size[1],
            "voxel_spacing_h": voxel_spacing[0],
            "voxel_spacing_w": voxel_spacing[1],
            "voxel_spacing_d": voxel_spacing[2],
            "has_foreground": bool(mask_resized.max() > 0),
            "unique_classes": sorted(np.unique(mask_resized).tolist()),
            "image_path_raw": image_path,
            "mask_path_raw": mask_path,
            # Arrays carried in-memory; persisted only in cache mode
            "image_array": img_resized,
            "mask_array": mask_resized,
        })

    return records


# ──────────────────────────────────────────────────────────────────────
# Full-dataset pipeline
# ──────────────────────────────────────────────────────────────────────

def run_preprocessing(
    metadata_df: pd.DataFrame,
    processed_dir: str,
    reports_dir: str,
    target_size: Tuple[int, int] = (256, 256),
    clip_percentiles: Tuple[float, float] = (0.5, 99.5),
    keep_empty: bool = False,
    cache_arrays: bool = True,
) -> pd.DataFrame:
    """
    Preprocess the full dataset.

    Args:
        metadata_df:     output of parse_dataset() — volume-level metadata
        processed_dir:   /mnt/nfs-data/tin_dataset/ACDC/processed
        reports_dir:     /mnt/nfs-data/tin_dataset/ACDC/reports
        target_size:     (H, W) after resize
        clip_percentiles: for z-score clipping
        keep_empty:      keep all-background slices?
        cache_arrays:    if True, save .npz per slice under processed_dir/2d_slices/

    Returns:
        slice-level DataFrame (also saved to reports_dir)
    """
    slices_dir = Path(processed_dir) / "2d_slices"
    reports_path = Path(reports_dir)
    slices_dir.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    all_records = []
    stats = {"volumes": 0, "slices_total": 0, "slices_with_fg": 0,
             "slices_empty_skipped": 0}

    for _, row in tqdm(metadata_df.iterrows(),
                       total=len(metadata_df), desc="Preprocessing"):
        slice_records = preprocess_volume(
            image_path=row["image_path"],
            mask_path=row["mask_path"],
            patient_id=row["patient_id"],
            frame_id=row["frame_id"],
            phase=row["phase"],
            source_split=row["split"],
            target_size=target_size,
            clip_percentiles=clip_percentiles,
            keep_empty=keep_empty,
        )

        stats["volumes"] += 1

        for rec in slice_records:
            pid = rec["patient_id"]
            fid = rec["frame_id"]
            sid = rec["slice_index"]
            fname = f"{pid}_frame{fid:02d}_slice{sid:03d}"

            if cache_arrays:
                # Save processed arrays
                npz_path = slices_dir / f"{fname}.npz"
                np.savez_compressed(
                    npz_path,
                    image=rec["image_array"],
                    mask=rec["mask_array"],
                )
                rec["cached_image_path"] = str(npz_path)
                rec["cached_mask_path"] = str(npz_path)
            else:
                rec["cached_image_path"] = ""
                rec["cached_mask_path"] = ""

            # Remove in-memory arrays from the metadata record
            rec.pop("image_array", None)
            rec.pop("mask_array", None)

            # Convert unique_classes list to string for CSV
            rec["unique_classes"] = str(rec["unique_classes"])

            all_records.append(rec)
            stats["slices_total"] += 1
            if rec["has_foreground"]:
                stats["slices_with_fg"] += 1
            else:
                stats["slices_empty_skipped"] += 0 if keep_empty else 1

    # Build slice-level DataFrame
    slice_df = pd.DataFrame(all_records)

    # Save outputs
    slice_csv_path = reports_path / "slice_metadata.csv"
    slice_df.to_csv(slice_csv_path, index=False)

    stats_path = reports_path / "preprocessing_stats.json"
    import json
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return slice_df
