"""
Patient-level splitting for ACDC.

Rules (from AGENTS.md):
    - splits MUST be at patient level, NEVER slice level
    - only official training set (100 patients) is used for dev splits
    - official testing set (50 patients) is final holdout — never split
    - no patient may appear in both train and val within a split
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


# ──────────────────────────────────────────────────────────────────────
# Leakage guard
# ──────────────────────────────────────────────────────────────────────

class LeakageError(Exception):
    """Raised when train/val sets share patients."""
    pass


def check_no_leakage(train_ids: List[str], val_ids: List[str],
                     label: str = "") -> None:
    """Assert zero overlap between train and val patient sets."""
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise LeakageError(
            f"LEAKAGE DETECTED{f' in {label}' if label else ''}: "
            f"{len(overlap)} patients in both train and val: "
            f"{sorted(overlap)[:5]}..."
        )


def check_completeness(train_ids: List[str], val_ids: List[str],
                        all_ids: List[str], label: str = "") -> None:
    """Assert every patient is assigned to exactly one set."""
    combined = set(train_ids) | set(val_ids)
    missing = set(all_ids) - combined
    extra = combined - set(all_ids)
    errors = []
    if missing:
        errors.append(f"Missing patients: {sorted(missing)[:5]}")
    if extra:
        errors.append(f"Extra patients: {sorted(extra)[:5]}")
    if errors:
        raise ValueError(
            f"Completeness check failed{f' in {label}' if label else ''}: "
            + "; ".join(errors)
        )


# ──────────────────────────────────────────────────────────────────────
# Split generators
# ──────────────────────────────────────────────────────────────────────

def fixed_split(
    patient_ids: List[str],
    train_ratio: float = 0.8,
    seed: int = 42,
    stratify_labels: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """
    Fixed patient-level train/val split.

    Args:
        patient_ids:     list of patient IDs (training set only)
        train_ratio:     fraction for training
        seed:            random seed
        stratify_labels: optional per-patient label (e.g. pathology group)
                         for stratified splitting
    """
    rng = np.random.RandomState(seed)
    ids = np.array(sorted(patient_ids))
    n = len(ids)
    n_train = int(n * train_ratio)

    if stratify_labels is not None:
        # Stratified shuffle split
        labels = np.array(stratify_labels)
        # Shuffle indices preserving label groups
        indices = np.arange(n)
        rng.shuffle(indices)
        ids = ids[indices]
        labels = labels[indices]
        # Assign by group proportionally
        train_set, val_set = [], []
        for group in np.unique(labels):
            group_mask = labels == group
            group_ids = ids[group_mask].tolist()
            n_g_train = max(1, int(len(group_ids) * train_ratio))
            train_set.extend(group_ids[:n_g_train])
            val_set.extend(group_ids[n_g_train:])
    else:
        indices = np.arange(n)
        rng.shuffle(indices)
        train_set = ids[indices[:n_train]].tolist()
        val_set = ids[indices[n_train:]].tolist()

    train_set = sorted(train_set)
    val_set = sorted(val_set)

    check_no_leakage(train_set, val_set, "dev_split")
    check_completeness(train_set, val_set, sorted(patient_ids), "dev_split")

    return {"train": train_set, "val": val_set}


def kfold_splits(
    patient_ids: List[str],
    n_folds: int = 5,
    seed: int = 42,
    stratify_labels: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    K-fold patient-level CV splits.

    Returns list of dicts with keys: fold, train, val.
    """
    ids = np.array(sorted(patient_ids))

    if stratify_labels is not None:
        labels = np.array(stratify_labels)
        # Reorder labels to match sorted ids
        sort_idx = np.argsort(patient_ids)
        labels = np.array(stratify_labels)[sort_idx]
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splitter = kf.split(ids, labels)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splitter = kf.split(ids)

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter):
        train_list = sorted(ids[train_idx].tolist())
        val_list = sorted(ids[val_idx].tolist())

        check_no_leakage(train_list, val_list, f"fold_{fold_idx}")
        check_completeness(train_list, val_list, sorted(patient_ids),
                           f"fold_{fold_idx}")

        folds.append({
            "fold": fold_idx,
            "train": train_list,
            "val": val_list,
        })

    return folds


# ──────────────────────────────────────────────────────────────────────
# Save to disk
# ──────────────────────────────────────────────────────────────────────

def save_splits(
    splits_dir: str,
    patient_ids: List[str],
    train_ratio: float = 0.8,
    n_folds: int = 5,
    seed: int = 42,
    stratify_labels: Optional[List[str]] = None,
    metadata_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Generate all splits and save to disk.

    Saves:
        splits_dir/dev_split.json       — fixed 80/20
        splits_dir/fold_0.json … fold_4.json — 5-fold CV
        splits_dir/splits_summary.json  — summary with counts

    Returns summary dict.
    """
    out = Path(splits_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Fixed dev split ─────────────────────────────────────────────
    dev = fixed_split(patient_ids, train_ratio, seed, stratify_labels)
    dev_out = {
        "seed": seed,
        "train_ratio": train_ratio,
        "split_level": "patient",
        "source": "official_training_only",
        "train": dev["train"],
        "val": dev["val"],
    }
    with open(out / "dev_split.json", "w") as f:
        json.dump(dev_out, f, indent=2)

    # ── K-fold CV ───────────────────────────────────────────────────
    folds = kfold_splits(patient_ids, n_folds, seed, stratify_labels)
    for fold in folds:
        fold_out = {
            "seed": seed,
            "fold": fold["fold"],
            "n_folds": n_folds,
            "split_level": "patient",
            "source": "official_training_only",
            "train": fold["train"],
            "val": fold["val"],
        }
        with open(out / f"fold_{fold['fold']}.json", "w") as f:
            json.dump(fold_out, f, indent=2)

    # ── Summary ─────────────────────────────────────────────────────
    summary = {
        "seed": seed,
        "split_level": "patient",
        "source": "official_training_only",
        "n_patients_total": len(patient_ids),
        "dev_split": {
            "train_ratio": train_ratio,
            "n_train": len(dev["train"]),
            "n_val": len(dev["val"]),
            "leakage": False,
        },
        "cv": {
            "n_folds": n_folds,
            "folds": [],
        },
    }
    for fold in folds:
        summary["cv"]["folds"].append({
            "fold": fold["fold"],
            "n_train": len(fold["train"]),
            "n_val": len(fold["val"]),
            "leakage": False,
        })

    with open(out / "splits_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ──────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────

def generate_split_report(
    splits_dir: str,
    reports_dir: str,
    metadata_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Generate a detailed CSV report of all splits.

    Output: reports_dir/split_report.csv with columns:
        patient_id, pathology, dev_split_role, fold_0_role, ..., fold_4_role

    Also saves a human-readable text summary.
    """
    splits_path = Path(splits_dir)
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    # Load dev split
    with open(splits_path / "dev_split.json") as f:
        dev = json.load(f)

    # Load folds
    folds = []
    fold_idx = 0
    while (splits_path / f"fold_{fold_idx}.json").exists():
        with open(splits_path / f"fold_{fold_idx}.json") as f:
            folds.append(json.load(f))
        fold_idx += 1

    # Collect all patient IDs
    all_ids = sorted(set(dev["train"]) | set(dev["val"]))

    # Build report rows
    rows = []
    for pid in all_ids:
        row = {"patient_id": pid}

        # Pathology from metadata if available
        if metadata_df is not None and "pathology" in metadata_df.columns:
            match = metadata_df[metadata_df["patient_id"] == pid]
            row["pathology"] = match["pathology"].iloc[0] if len(match) > 0 else "unknown"
        else:
            row["pathology"] = "unknown"

        # Dev split role
        if pid in dev["train"]:
            row["dev_split"] = "train"
        elif pid in dev["val"]:
            row["dev_split"] = "val"
        else:
            row["dev_split"] = "MISSING"

        # Fold roles
        for fold in folds:
            fi = fold["fold"]
            if pid in fold["train"]:
                row[f"fold_{fi}"] = "train"
            elif pid in fold["val"]:
                row[f"fold_{fi}"] = "val"
            else:
                row[f"fold_{fi}"] = "MISSING"

        rows.append(row)

    report_df = pd.DataFrame(rows)
    report_df.to_csv(reports_path / "split_report.csv", index=False)

    # ── Text summary ────────────────────────────────────────────────
    lines = []
    lines.append("=" * 60)
    lines.append("ACDC SPLIT REPORT")
    lines.append("=" * 60)
    lines.append(f"Total patients: {len(all_ids)}")
    lines.append(f"Source: official training set only")
    lines.append(f"Split level: PATIENT (not slice)")
    lines.append("")

    # Dev split
    lines.append("── Dev Split (fixed) ──")
    lines.append(f"  Train: {len(dev['train'])} patients")
    lines.append(f"  Val:   {len(dev['val'])} patients")
    overlap = set(dev["train"]) & set(dev["val"])
    lines.append(f"  Leakage check: {'PASS' if not overlap else f'FAIL ({len(overlap)} shared)'}")
    lines.append("")

    # Pathology distribution in dev split
    if metadata_df is not None and "pathology" in metadata_df.columns:
        lines.append("  Pathology distribution:")
        patient_path = metadata_df.drop_duplicates("patient_id")[["patient_id", "pathology"]]
        for subset_name, subset_ids in [("train", dev["train"]), ("val", dev["val"])]:
            sub = patient_path[patient_path["patient_id"].isin(subset_ids)]
            dist = sub["pathology"].value_counts().to_dict()
            lines.append(f"    {subset_name}: {dist}")
        lines.append("")

    # Fold summaries
    lines.append(f"── {len(folds)}-Fold CV ──")
    for fold in folds:
        fi = fold["fold"]
        n_tr = len(fold["train"])
        n_va = len(fold["val"])
        ol = set(fold["train"]) & set(fold["val"])
        status = "PASS" if not ol else f"FAIL ({len(ol)} shared)"
        lines.append(f"  Fold {fi}: train={n_tr}, val={n_va}, leakage={status}")
    lines.append("")

    # Cross-fold val coverage
    all_val_ids = set()
    for fold in folds:
        all_val_ids.update(fold["val"])
    lines.append(f"  Val coverage across folds: {len(all_val_ids)}/{len(all_ids)} patients")
    missing = set(all_ids) - all_val_ids
    if missing:
        lines.append(f"  WARNING: {len(missing)} patients never in val: {sorted(missing)[:5]}")
    else:
        lines.append(f"  Every patient appears in val exactly once: OK")
    lines.append("")
    lines.append("=" * 60)

    report_text = "\n".join(lines)

    with open(reports_path / "split_report.txt", "w") as f:
        f.write(report_text)

    return report_df
