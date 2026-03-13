"""Parse ACDC dataset: enumerate patients, extract metadata, build inventory."""
import re
from pathlib import Path
from typing import Any, Dict, List

import nibabel as nib
import pandas as pd


def parse_info_cfg(info_path: Path) -> Dict[str, Any]:
    """Parse a patient's Info.cfg file."""
    info = {}
    with open(info_path, "r") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                key, val = line.split(":", 1)
                val = val.strip()
                try:
                    val = int(val)
                except ValueError:
                    pass
                info[key.strip()] = val
    return info


def find_labeled_frames(patient_dir: Path) -> List[Dict[str, Any]]:
    """Find all labeled frame pairs (image + GT) in a patient folder."""
    frames = []
    for gt_path in sorted(patient_dir.glob("*_gt.nii.gz")):
        # Extract frame number from filename like patient001_frame01_gt.nii.gz
        match = re.match(r"(.+)_frame(\d+)_gt\.nii\.gz", gt_path.name)
        if not match:
            continue
        prefix = match.group(1)
        frame_id = int(match.group(2))
        img_path = patient_dir / f"{prefix}_frame{frame_id:02d}.nii.gz"
        if not img_path.exists():
            continue
        # Load header to get shape info
        img_nii = nib.load(str(img_path))
        shape = img_nii.shape  # (H, W, D)
        frames.append({
            "image_path": str(img_path),
            "mask_path": str(gt_path),
            "frame_id": frame_id,
            "num_slices": shape[2] if len(shape) >= 3 else 1,
            "height": shape[0],
            "width": shape[1],
        })
    return frames


def parse_dataset(raw_data_dir: str, split_name: str = "training") -> pd.DataFrame:
    """
    Parse the full ACDC dataset directory into a metadata DataFrame.

    Returns columns:
        patient_id, split, pathology, ed_frame, es_frame,
        frame_id, phase, image_path, mask_path, num_slices, height, width
    """
    raw_dir = Path(raw_data_dir)
    records = []

    for patient_dir in sorted(raw_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        patient_id = patient_dir.name  # e.g. "patient001"

        # Parse Info.cfg
        info_path = patient_dir / "Info.cfg"
        info = parse_info_cfg(info_path) if info_path.exists() else {}
        ed_frame = info.get("ED", None)
        es_frame = info.get("ES", None)
        pathology = info.get("Group", "unknown")

        # Find labeled frames
        frames = find_labeled_frames(patient_dir)
        for frame in frames:
            # Determine phase (ED or ES)
            phase = "unknown"
            if ed_frame is not None and frame["frame_id"] == ed_frame:
                phase = "ED"
            elif es_frame is not None and frame["frame_id"] == es_frame:
                phase = "ES"

            records.append({
                "patient_id": patient_id,
                "split": split_name,
                "pathology": pathology,
                "ed_frame": ed_frame,
                "es_frame": es_frame,
                "frame_id": frame["frame_id"],
                "phase": phase,
                "image_path": frame["image_path"],
                "mask_path": frame["mask_path"],
                "num_slices": frame["num_slices"],
                "height": frame["height"],
                "width": frame["width"],
            })

    return pd.DataFrame(records)


def verify_masks(metadata_df: pd.DataFrame) -> Dict[str, Any]:
    """Verify mask label values across the dataset."""
    all_labels = set()
    mismatches = []
    for _, row in metadata_df.iterrows():
        mask = nib.load(row["mask_path"]).get_fdata()
        labels = set(mask.astype(int).flatten())
        all_labels.update(labels)
        # Verify shape match
        img = nib.load(row["image_path"]).get_fdata()
        if img.shape != mask.shape:
            mismatches.append({
                "patient_id": row["patient_id"],
                "frame_id": row["frame_id"],
                "img_shape": img.shape,
                "mask_shape": mask.shape,
            })
    return {
        "unique_labels": sorted(all_labels),
        "shape_mismatches": mismatches,
    }
