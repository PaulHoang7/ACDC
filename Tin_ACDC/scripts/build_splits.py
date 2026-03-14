"""
CLI script to generate patient-level splits for ACDC.

All split files → /mnt/nfs-data/tin_dataset/ACDC/splits/
All reports    → /mnt/nfs-data/tin_dataset/ACDC/reports/

Usage:
    # Default: 80/20 + 5-fold CV, seed=42
    python scripts/build_splits.py

    # Custom ratio and folds
    python scripts/build_splits.py --train-ratio 0.85 --n-folds 3

    # Stratify by pathology group
    python scripts/build_splits.py --stratify

    # Different seed
    python scripts/build_splits.py --seed 123

    # Skip parsing (reuse existing metadata)
    python scripts/build_splits.py --skip-parse
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.utils.io import load_paths, load_config, get_project_root, ensure_dirs
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.data.parse_acdc import parse_dataset
from src.data.build_splits import save_splits, generate_split_report


def parse_args():
    p = argparse.ArgumentParser(description="Generate ACDC patient-level splits")
    p.add_argument("--train-ratio", type=float, default=0.8,
                   help="Train fraction for fixed dev split (default: 0.8)")
    p.add_argument("--n-folds", type=int, default=5,
                   help="Number of CV folds (default: 5)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--stratify", action="store_true",
                   help="Stratify splits by pathology group from Info.cfg")
    p.add_argument("--skip-parse", action="store_true",
                   help="Reuse existing patient_metadata.csv")
    return p.parse_args()


def main():
    args = parse_args()
    root = get_project_root()
    paths = load_paths()
    ensure_dirs(paths)
    set_seed(args.seed)

    logger = get_logger("build_splits",
                        f"{paths['logs_dir']}/build_splits.log")

    # ── Step 1: get patient metadata ────────────────────────────────
    meta_csv = f"{paths['metadata_dir']}/patient_metadata.csv"

    if args.skip_parse and Path(meta_csv).exists():
        logger.info(f"Reusing existing metadata: {meta_csv}")
        metadata = pd.read_csv(meta_csv)
    else:
        logger.info("Parsing ACDC training set...")
        metadata = parse_dataset(paths["raw_data"], split_name="training")
        Path(paths["metadata_dir"]).mkdir(parents=True, exist_ok=True)
        metadata.to_csv(meta_csv, index=False)
        logger.info(f"Parsed {metadata['patient_id'].nunique()} patients → {meta_csv}")

    # ── Step 2: extract patient-level info ──────────────────────────
    # Deduplicate to patient level (each patient has 2 rows for ED/ES)
    patient_df = (metadata
                  .drop_duplicates("patient_id")[["patient_id", "pathology"]]
                  .sort_values("patient_id")
                  .reset_index(drop=True))

    patient_ids = patient_df["patient_id"].tolist()
    logger.info(f"Total patients (training set): {len(patient_ids)}")

    # Pathology distribution
    path_dist = patient_df["pathology"].value_counts().to_dict()
    logger.info(f"Pathology distribution: {path_dist}")

    # Stratification labels
    stratify_labels = None
    if args.stratify:
        stratify_labels = patient_df["pathology"].tolist()
        logger.info("Stratifying splits by pathology group")

    # ── Step 3: generate splits ─────────────────────────────────────
    logger.info(f"Generating splits: ratio={args.train_ratio}, "
                f"folds={args.n_folds}, seed={args.seed}")

    summary = save_splits(
        splits_dir=paths["splits_dir"],
        patient_ids=patient_ids,
        train_ratio=args.train_ratio,
        n_folds=args.n_folds,
        seed=args.seed,
        stratify_labels=stratify_labels,
        metadata_df=metadata,
    )

    logger.info(f"Split files saved to: {paths['splits_dir']}")
    logger.info(f"  dev_split.json: train={summary['dev_split']['n_train']}, "
                f"val={summary['dev_split']['n_val']}")
    for fold_info in summary["cv"]["folds"]:
        logger.info(f"  fold_{fold_info['fold']}.json: "
                     f"train={fold_info['n_train']}, val={fold_info['n_val']}")

    # ── Step 4: generate report ─────────────────────────────────────
    logger.info("Generating split report...")
    report_df = generate_split_report(
        splits_dir=paths["splits_dir"],
        reports_dir=paths["reports_dir"],
        metadata_df=metadata,
    )

    logger.info(f"Report CSV → {paths['reports_dir']}/split_report.csv")
    logger.info(f"Report TXT → {paths['reports_dir']}/split_report.txt")

    # Print text report to console
    report_txt = Path(paths["reports_dir"]) / "split_report.txt"
    print("\n" + report_txt.read_text())

    logger.info("Done.")


if __name__ == "__main__":
    main()
