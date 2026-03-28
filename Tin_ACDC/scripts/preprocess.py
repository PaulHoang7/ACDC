"""
Preprocessing CLI: parse ACDC → normalize → resize → slice → save.

All outputs land under /mnt/nfs-data/tin_dataset/ACDC/.

Usage:
    # Full pipeline: metadata + cached arrays
    python scripts/preprocess.py

    # Metadata only (no .npz cache, much faster)
    python scripts/preprocess.py --metadata-only

    # Include empty (all-background) slices
    python scripts/preprocess.py --keep-empty

    # Custom image size
    python scripts/preprocess.py --image-size 224 224
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.io import load_paths, load_config, get_project_root, ensure_dirs
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.data.parse_acdc import parse_dataset, verify_masks
from src.data.build_splits import save_splits
from src.data.preprocess import run_preprocessing
from src.utils.visualization import plot_sample

import nibabel as nib
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="ACDC preprocessing pipeline")
    p.add_argument("--metadata-only", action="store_true",
                   help="Only build slice-level CSV, skip caching .npz arrays")
    p.add_argument("--keep-empty", action="store_true",
                   help="Keep all-background slices")
    p.add_argument("--image-size", type=int, nargs=2, default=None,
                   help="Target H W (default: from data.yaml)")
    p.add_argument("--skip-parse", action="store_true",
                   help="Skip parsing step (reuse existing patient_metadata.csv)")
    p.add_argument("--skip-splits", action="store_true",
                   help="Skip split generation")
    p.add_argument("--skip-verify", action="store_true",
                   help="Skip mask verification (faster)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    root = get_project_root()
    paths = load_paths()
    ensure_dirs(paths)
    data_cfg = load_config(root / "configs" / "data.yaml")
    train_cfg = load_config(root / "configs" / "train.yaml")

    set_seed(args.seed)
    logger = get_logger("preprocess",
                        f"{paths['logs_dir']}/preprocess.log")

    # ── Step 0: resolve config ──────────────────────────────────────
    preproc = data_cfg["preprocessing"]
    target_size = tuple(args.image_size) if args.image_size else tuple(preproc["image_size"])
    clip_pct = tuple(preproc["clip_percentiles"])
    keep_empty = args.keep_empty or data_cfg["slice_extraction"].get("keep_empty_slices", False)
    cache_arrays = not args.metadata_only

    logger.info("=" * 60)
    logger.info("ACDC Preprocessing Pipeline")
    logger.info(f"  target_size   = {target_size}")
    logger.info(f"  clip_pct      = {clip_pct}")
    logger.info(f"  keep_empty    = {keep_empty}")
    logger.info(f"  cache_arrays  = {cache_arrays}")
    logger.info(f"  output        = {paths['processed']}")
    logger.info(f"  reports       = {paths['reports_dir']}")
    logger.info("=" * 60)

    # ── Step 1: parse raw dataset ───────────────────────────────────
    meta_csv = f"{paths['metadata_dir']}/patient_metadata.csv"

    if args.skip_parse and Path(meta_csv).exists():
        import pandas as pd
        logger.info(f"Reusing existing metadata: {meta_csv}")
        metadata = pd.read_csv(meta_csv)
    else:
        logger.info("Parsing ACDC training set...")
        metadata = parse_dataset(paths["raw_data"], split_name="training")
        Path(paths["metadata_dir"]).mkdir(parents=True, exist_ok=True)
        metadata.to_csv(meta_csv, index=False)
        logger.info(f"Found {len(metadata)} volumes from "
                     f"{metadata['patient_id'].nunique()} patients")
        logger.info(f"Volume-level metadata → {meta_csv}")

    # ── Step 2: verify masks ────────────────────────────────────────
    if not args.skip_verify:
        logger.info("Verifying mask labels...")
        result = verify_masks(metadata)
        logger.info(f"  Unique labels: {result['unique_labels']}")
        expected = set(preproc.get("class_ids", [0, 1, 2, 3]))
        found = set(result["unique_labels"])
        if found != expected:
            logger.warning(f"  Expected {expected}, found {found}")
        if result["shape_mismatches"]:
            for m in result["shape_mismatches"]:
                logger.warning(f"  Shape mismatch: {m}")
        else:
            logger.info("  All image-mask shapes OK")
    else:
        logger.info("Skipping mask verification (--skip-verify)")

    # ── Step 3: build splits ────────────────────────────────────────
    if not args.skip_splits:
        logger.info("Building patient-level splits...")
        patient_ids = sorted(metadata["patient_id"].unique().tolist())
        split_cfg = train_cfg.get("split", {})
        dev_cfg = split_cfg.get("dev_split", {})
        save_splits(
            splits_dir=paths["splits_dir"],
            patient_ids=patient_ids,
            train_ratio=dev_cfg.get("train_ratio", 0.8),
            n_folds=split_cfg.get("cv_folds", 5),
            seed=args.seed,
        )
        logger.info(f"  Splits → {paths['splits_dir']}")
    else:
        logger.info("Skipping split generation (--skip-splits)")

    # ── Step 4: preprocess volumes → 2D slices ─────────────────────
    logger.info("Running preprocessing pipeline...")
    slice_df = run_preprocessing(
        metadata_df=metadata,
        processed_dir=paths["processed"],
        reports_dir=paths["reports_dir"],
        target_size=target_size,
        clip_percentiles=clip_pct,
        keep_empty=keep_empty,
        cache_arrays=cache_arrays,
    )

    logger.info(f"  Total slices:    {len(slice_df)}")
    logger.info(f"  With foreground: {slice_df['has_foreground'].sum()}")
    logger.info(f"  Patients:        {slice_df['patient_id'].nunique()}")
    logger.info(f"  Phases:          {slice_df['phase'].value_counts().to_dict()}")

    if cache_arrays:
        # Report cache size
        slices_dir = Path(paths["processed"]) / "2d_slices"
        total_bytes = sum(f.stat().st_size for f in slices_dir.glob("*.npz"))
        logger.info(f"  Cache size:      {total_bytes / 1e6:.1f} MB")
        logger.info(f"  Cache dir:       {slices_dir}")
    else:
        logger.info("  Mode: metadata-only (no .npz cache)")

    logger.info(f"  Slice CSV:       {paths['reports_dir']}/slice_metadata.csv")
    logger.info(f"  Stats JSON:      {paths['reports_dir']}/preprocessing_stats.json")

    # ── Step 5: sanity visualization ────────────────────────────────
    logger.info("Generating sanity check visualizations...")
    fig_dir = Path(paths["figures_dir"]) / "preprocessing_sanity"
    fig_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = slice_df.sample(min(8, len(slice_df)), random_state=args.seed)
    for i, (_, row) in enumerate(sample_rows.iterrows()):
        if cache_arrays and row.get("cached_image_path", ""):
            data = np.load(row["cached_image_path"])
            img_slice = data["image"]
            mask_slice = data["mask"]
        else:
            # Fall back: reload from raw + re-extract slice
            img_vol = nib.load(row["image_path_raw"]).get_fdata()
            mask_vol = nib.load(row["mask_path_raw"]).get_fdata()
            img_slice = img_vol[:, :, row["slice_index"]]
            mask_slice = mask_vol[:, :, row["slice_index"]]

        title = (f"{row['patient_id']} frame{row['frame_id']:02.0f} "
                 f"slice{row['slice_index']:02.0f} ({row['phase']}) "
                 f"[{row['target_height']}x{row['target_width']}]")
        plot_sample(
            img_slice, mask_slice.astype(int),
            title=title,
            save_path=str(fig_dir / f"preproc_{i:02d}_{row['patient_id']}.png"),
        )
    logger.info(f"  Figures → {fig_dir}")

    # ── Done ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Preprocessing complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
