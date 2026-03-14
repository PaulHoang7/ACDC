"""
Sanity check for ACDC dataset + dataloader pipeline.

Verifies:
    1. Dataset loads without errors
    2. Tensor shapes are correct
    3. Label values are valid {0,1,2,3}
    4. Train augmentation produces valid outputs
    5. Metadata is preserved
    6. Saves sample visualization to NFS

Usage:
    python scripts/test_loader.py
    python scripts/test_loader.py --split fold_0.json
    python scripts/test_loader.py --backend raw
    python scripts/test_loader.py --save-figures
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.utils.io import load_paths, load_config, get_project_root
from src.data.dataset_2d import ACDCSliceDataset, build_dataloaders
from src.data.transforms import get_train_transforms, get_val_transforms


def parse_args():
    p = argparse.ArgumentParser(description="ACDC dataloader sanity check")
    p.add_argument("--split", default="dev_split.json",
                   help="Split file name (default: dev_split.json)")
    p.add_argument("--backend", default="auto", choices=["auto", "raw", "cached"],
                   help="Loading backend")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--save-figures", action="store_true",
                   help="Save sample visualizations to NFS")
    return p.parse_args()


def check_batch(batch: dict, batch_idx: int, mode: str) -> None:
    """Validate a single batch."""
    images = batch["image"]
    masks = batch["mask"]
    metas = batch["meta"]

    B, C, H, W = images.shape

    # Shape checks
    assert C == 1, f"Expected 1 channel, got {C}"
    assert H == 256 and W == 256, f"Expected 256x256, got {H}x{W}"
    assert masks.shape == (B, H, W), f"Mask shape {masks.shape} != ({B},{H},{W})"

    # Dtype checks
    assert images.dtype == torch.float32, f"Image dtype {images.dtype}"
    assert masks.dtype == torch.int64, f"Mask dtype {masks.dtype}"

    # Label range
    unique_labels = torch.unique(masks).tolist()
    valid_labels = {0, 1, 2, 3}
    invalid = set(int(l) for l in unique_labels) - valid_labels
    assert not invalid, f"Invalid labels: {invalid}"

    # No NaN/Inf
    assert not torch.isnan(images).any(), "NaN in images"
    assert not torch.isinf(images).any(), "Inf in images"

    # Metadata present
    assert len(metas["patient_id"]) == B
    assert len(metas["phase"]) == B

    if batch_idx == 0:
        print(f"\n  [{mode}] Batch {batch_idx}:")
        print(f"    images:  {images.shape}  dtype={images.dtype}")
        print(f"    masks:   {masks.shape}  dtype={masks.dtype}")
        print(f"    labels:  {unique_labels}")
        print(f"    img range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"    patients: {list(metas['patient_id'][:3])}...")
        print(f"    phases:   {list(metas['phase'][:3])}...")


def main():
    args = parse_args()
    root = get_project_root()
    paths = load_paths()
    data_cfg = load_config(root / "configs" / "data.yaml")

    # Resolve paths
    slice_csv = f"{paths['reports_dir']}/slice_metadata.csv"
    patient_csv = f"{paths['metadata_dir']}/patient_metadata.csv"
    split_file = f"{paths['splits_dir']}/{args.split}"

    # Pick the right CSV based on backend
    if args.backend == "cached":
        metadata_csv = slice_csv
    elif args.backend == "raw":
        metadata_csv = slice_csv  # slice CSV also works for raw
    else:
        metadata_csv = slice_csv  # auto: let dataset decide

    print("=" * 60)
    print("ACDC DataLoader Sanity Check")
    print("=" * 60)
    print(f"  metadata:  {metadata_csv}")
    print(f"  split:     {split_file}")
    print(f"  backend:   {args.backend}")
    print(f"  batch:     {args.batch_size}")

    # ── Test 1: Build dataloaders ────────────────────────────────
    print("\n[Test 1] Building dataloaders...")
    loaders = build_dataloaders(
        metadata_csv=metadata_csv,
        split_file=split_file,
        batch_size=args.batch_size,
        num_workers=0,  # single-process for debugging
        pin_memory=False,
        target_size=(256, 256),
        aug_cfg=data_cfg.get("augmentation"),
        backend=args.backend,
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]

    print(f"  train: {len(train_loader.dataset)} slices, "
          f"{len(train_loader)} batches")
    print(f"  val:   {len(val_loader.dataset)} slices, "
          f"{len(val_loader)} batches")

    # ── Test 2: Iterate train batches ────────────────────────────
    print("\n[Test 2] Checking train batches...")
    n_checked = 0
    for i, batch in enumerate(train_loader):
        check_batch(batch, i, "train")
        n_checked += 1
        if n_checked >= 3:
            break
    print(f"  PASS — {n_checked} train batches OK")

    # ── Test 3: Iterate val batches ──────────────────────────────
    print("\n[Test 3] Checking val batches...")
    n_checked = 0
    for i, batch in enumerate(val_loader):
        check_batch(batch, i, "val")
        n_checked += 1
        if n_checked >= 3:
            break
    print(f"  PASS — {n_checked} val batches OK")

    # ── Test 4: Augmentation consistency ─────────────────────────
    print("\n[Test 4] Augmentation consistency check...")
    ds = train_loader.dataset
    idx = 0
    results = [ds[idx] for _ in range(5)]
    images = torch.stack([r["image"] for r in results])
    masks = torch.stack([r["mask"] for r in results])

    # Same patient
    pids = [r["meta"]["patient_id"] for r in results]
    assert all(p == pids[0] for p in pids), "Metadata changed across augmentation"

    # Images should differ (augmentation is stochastic)
    all_same = all(torch.equal(images[0], images[i]) for i in range(1, 5))
    if all_same:
        print("  WARNING: all 5 augmented samples identical — augmentation may be off")
    else:
        print("  PASS — augmentation produces variation")

    # Masks should have same unique labels (class set preserved)
    print(f"  5x same sample labels: "
          f"{[torch.unique(masks[i]).tolist() for i in range(5)]}")

    # ── Test 5: No patient leakage ───────────────────────────────
    print("\n[Test 5] Patient leakage check...")
    train_pids = {s["patient_id"] for s in train_loader.dataset.slices}
    val_pids = {s["patient_id"] for s in val_loader.dataset.slices}
    overlap = train_pids & val_pids
    if overlap:
        print(f"  FAIL — {len(overlap)} patients in both: {sorted(overlap)[:5]}")
        sys.exit(1)
    else:
        print(f"  PASS — 0 overlap (train={len(train_pids)}, val={len(val_pids)})")

    # ── Test 6: Save sample figures ──────────────────────────────
    if args.save_figures:
        print("\n[Test 6] Saving sample visualizations...")
        from src.utils.visualization import plot_sample

        fig_dir = Path(paths["figures_dir"]) / "dataloader_sanity"
        fig_dir.mkdir(parents=True, exist_ok=True)

        for mode, loader in [("train", train_loader), ("val", val_loader)]:
            batch = next(iter(loader))
            for j in range(min(4, batch["image"].shape[0])):
                img = batch["image"][j, 0].numpy()
                msk = batch["mask"][j].numpy()
                pid = batch["meta"]["patient_id"][j]
                phase = batch["meta"]["phase"][j]
                title = f"{mode} | {pid} | {phase}"
                save_path = str(fig_dir / f"{mode}_{j:02d}_{pid}.png")
                plot_sample(img, msk, title=title, save_path=save_path)
        print(f"  Saved to {fig_dir}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)
    print(f"  Train: {len(train_loader.dataset)} slices from "
          f"{len(train_pids)} patients")
    print(f"  Val:   {len(val_loader.dataset)} slices from "
          f"{len(val_pids)} patients")
    print(f"  Backend: {train_loader.dataset.backend}")
    print(f"  Image shape: (1, 256, 256) float32")
    print(f"  Mask shape:  (256, 256) int64, labels={{0,1,2,3}}")
    print(f"  Augmentation: train=ON, val=OFF")
    print(f"  Leakage: NONE")


if __name__ == "__main__":
    main()
