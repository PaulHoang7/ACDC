"""Patient-level splitting: fixed dev split and 5-fold CV."""
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def fixed_split(
    patient_ids: List[str],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Create a fixed patient-level train/val split."""
    rng = np.random.RandomState(seed)
    ids = sorted(patient_ids)
    rng.shuffle(ids)
    n_train = int(len(ids) * train_ratio)
    return {
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train:]),
    }


def kfold_splits(
    patient_ids: List[str],
    n_folds: int = 5,
    seed: int = 42,
) -> List[Dict[str, List[str]]]:
    """Create n-fold patient-level CV splits."""
    ids = np.array(sorted(patient_ids))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in kf.split(ids):
        folds.append({
            "train": sorted(ids[train_idx].tolist()),
            "val": sorted(ids[val_idx].tolist()),
        })
    return folds


def save_splits(
    splits_dir: str,
    patient_ids: List[str],
    train_ratio: float = 0.8,
    n_folds: int = 5,
    seed: int = 42,
) -> None:
    """Generate and save both fixed split and k-fold splits to disk."""
    out = Path(splits_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Fixed dev split
    dev = fixed_split(patient_ids, train_ratio, seed)
    with open(out / "dev_split.json", "w") as f:
        json.dump(dev, f, indent=2)

    # K-fold CV splits
    folds = kfold_splits(patient_ids, n_folds, seed)
    for i, fold in enumerate(folds):
        with open(out / f"fold_{i}.json", "w") as f:
            json.dump(fold, f, indent=2)

    # Summary
    summary = {
        "seed": seed,
        "n_patients": len(patient_ids),
        "dev_split": {"train": len(dev["train"]), "val": len(dev["val"])},
        "n_folds": n_folds,
    }
    with open(out / "splits_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
