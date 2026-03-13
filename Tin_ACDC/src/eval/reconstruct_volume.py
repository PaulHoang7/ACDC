"""Reconstruct 3D patient volumes from 2D slice predictions."""
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch


def collect_slice_predictions(
    predictions: List[Dict],
) -> Dict[str, Dict[int, Dict[int, np.ndarray]]]:
    """
    Group slice predictions by patient_id -> frame_id -> slice_idx.

    Args:
        predictions: list of dicts with keys:
            pred: (H, W) np array
            meta: {patient_id, frame_id, slice_idx, phase}
    """
    volumes = defaultdict(lambda: defaultdict(dict))
    for item in predictions:
        meta = item["meta"]
        pid = meta["patient_id"]
        fid = meta["frame_id"]
        sid = meta["slice_idx"]
        volumes[pid][fid][sid] = item["pred"]
    return dict(volumes)


def reconstruct_volume(
    slice_dict: Dict[int, np.ndarray],
) -> np.ndarray:
    """
    Stack 2D slice predictions into a 3D volume.

    Args:
        slice_dict: {slice_idx: (H, W) prediction}

    Returns:
        (H, W, D) volume
    """
    max_idx = max(slice_dict.keys())
    h, w = next(iter(slice_dict.values())).shape
    volume = np.zeros((h, w, max_idx + 1), dtype=np.int64)
    for idx, pred in slice_dict.items():
        volume[:, :, idx] = pred
    return volume
