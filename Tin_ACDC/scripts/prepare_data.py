"""
Phase 1 script: Parse ACDC, verify masks, build splits, save metadata.
All outputs go to /mnt/nfs-data/tin_dataset/ACDC/.

Usage:
    python scripts/prepare_data.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.io import load_paths, load_config, get_project_root, ensure_dirs
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.data.parse_acdc import parse_dataset, verify_masks
from src.data.build_splits import save_splits
from src.utils.visualization import plot_sample

import nibabel as nib
import numpy as np


def main():
    set_seed(42)
    paths = load_paths()
    ensure_dirs(paths)
    train_cfg = load_config(get_project_root() / "configs" / "train.yaml")
    logger = get_logger("prepare_data",
                        f"{paths['logs_dir']}/prepare_data.log")

    # Step 1: Parse dataset
    logger.info("Parsing ACDC training set...")
    metadata = parse_dataset(paths["raw_data"], split_name="training")
    logger.info(f"Found {len(metadata)} labeled volumes from "
                f"{metadata['patient_id'].nunique()} patients")

    # Save metadata
    meta_path = f"{paths['metadata_dir']}/patient_metadata.csv"
    Path(paths["metadata_dir"]).mkdir(parents=True, exist_ok=True)
    metadata.to_csv(meta_path, index=False)
    logger.info(f"Metadata saved to {meta_path}")

    # Step 2: Verify masks
    logger.info("Verifying mask labels...")
    verify_result = verify_masks(metadata)
    logger.info(f"Unique labels found: {verify_result['unique_labels']}")
    if verify_result["shape_mismatches"]:
        logger.warning(f"Shape mismatches: {verify_result['shape_mismatches']}")
    else:
        logger.info("All image-mask shapes match.")

    # Step 3: Build splits
    logger.info("Building patient-level splits...")
    patient_ids = sorted(metadata["patient_id"].unique().tolist())
    split_cfg = train_cfg.get("split", {})
    dev_cfg = split_cfg.get("dev_split", {})
    save_splits(
        splits_dir=paths["splits_dir"],
        patient_ids=patient_ids,
        train_ratio=dev_cfg.get("train_ratio", 0.8),
        n_folds=split_cfg.get("cv_folds", 5),
        seed=train_cfg.get("seed", 42),
    )
    logger.info(f"Splits saved to {paths['splits_dir']}")

    # Step 4: Sanity visualization
    logger.info("Generating sanity check visualizations...")
    fig_dir = Path(paths["figures_dir"]) / "sanity_check"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for i, (_, row) in enumerate(metadata.sample(5, random_state=42).iterrows()):
        img = nib.load(row["image_path"]).get_fdata()
        mask = nib.load(row["mask_path"]).get_fdata()
        mid_slice = img.shape[2] // 2
        plot_sample(
            img[:, :, mid_slice],
            mask[:, :, mid_slice].astype(int),
            title=f"{row['patient_id']} frame{row['frame_id']:02d} "
                  f"slice{mid_slice} ({row['phase']})",
            save_path=str(fig_dir / f"sample_{i}_{row['patient_id']}.png"),
        )
    logger.info(f"Sanity figures saved to {fig_dir}")

    logger.info("Data preparation complete.")


if __name__ == "__main__":
    main()
