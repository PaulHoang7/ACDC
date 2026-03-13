"""
Evaluation script: load checkpoint, predict, compute patient-level metrics.
All outputs go to /mnt/nfs-data/tin_dataset/ACDC/.

Usage:
    python scripts/evaluate.py --model m1_unetr34 --init scratch --seed 42
    python scripts/evaluate.py --model m1_unetr34 --init scratch --seed 42 --split val
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils.io import load_paths, load_config, get_project_root, ensure_dirs
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.data.dataset_2d import ACDCSliceDataset
from src.data.transforms import get_val_transforms
from src.models import build_model
from src.eval import predict_dataset, evaluate_patient_volumes


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ACDC model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--init", type=str, required=True,
                        choices=["scratch", "finetune"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val"])
    parser.add_argument("--checkpoint", type=str, default="best_model.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    root = get_project_root()
    paths = load_paths()
    ensure_dirs(paths)
    data_cfg = load_config(root / "configs" / "data.yaml")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_id = f"{args.model}_{args.init}_seed{args.seed}"
    if args.fold is not None:
        experiment_id += f"_fold{args.fold}"

    logger = get_logger("evaluate",
                        f"{paths['logs_dir']}/{experiment_id}/evaluate.log")

    # Load model
    model = build_model(args.model)
    ckpt_path = (Path(paths["checkpoints_dir"]) / experiment_id
                 / args.checkpoint)
    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # Dataset
    if args.fold is not None:
        split_file = f"{paths['splits_dir']}/fold_{args.fold}.json"
    else:
        split_file = f"{paths['splits_dir']}/dev_split.json"

    img_size = tuple(data_cfg["preprocessing"]["image_size"])
    val_tf = get_val_transforms(img_size)
    slice_csv = f"{paths['reports_dir']}/slice_metadata.csv"
    volume_csv = f"{paths['metadata_dir']}/patient_metadata.csv"
    metadata_csv = slice_csv if Path(slice_csv).exists() else volume_csv
    logger.info(f"Dataset source: {metadata_csv}")

    dataset = ACDCSliceDataset(metadata_csv, split_file, args.split,
                               transform=val_tf, keep_empty=True,
                               backend="auto")
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    logger.info(f"Evaluating {len(dataset)} slices ({args.split} set)")

    # Predict
    predictions = predict_dataset(model, loader, device)

    # Patient-level metrics (always use volume-level CSV for GT paths)
    metadata_df = pd.read_csv(volume_csv)
    results_df = evaluate_patient_volumes(predictions, metadata_df)

    # Save results
    out_dir = Path(paths["reports_dir"]) / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"metrics_{args.split}.csv"
    results_df.to_csv(results_path, index=False)

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Results for {experiment_id} ({args.split}):")
    for col in ["dice_rv", "dice_myo", "dice_lv", "mean_dice",
                "hd95_rv", "hd95_myo", "hd95_lv", "mean_hd95"]:
        if col in results_df.columns:
            logger.info(f"  {col}: {results_df[col].mean():.4f} "
                        f"± {results_df[col].std():.4f}")
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
