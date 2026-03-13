"""
Training script for ACDC segmentation models.
All outputs go to /mnt/nfs-data/tin_dataset/ACDC/.

Usage:
    python scripts/train.py --model m1_unetr34 --init scratch --seed 42
    python scripts/train.py --model m2_attunetr34 --init finetune --seed 42
    python scripts/train.py --model m3_boundarydsunetr34 --init scratch --seed 42
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.utils.io import load_paths, load_config, get_project_root, ensure_dirs
from src.utils.seed import set_seed
from src.utils.logging import ExperimentLogger
from src.utils.visualization import plot_training_curves
from src.data.dataset_2d import ACDCSliceDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models import build_model
from src.losses import DiceCELoss, BoundaryBCELoss, DeepSupervisionLoss
from src.train import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train ACDC segmentation model")
    parser.add_argument("--model", type=str, required=True,
                        choices=["m1_unetr34", "m2_attunetr34",
                                 "m3_boundarydsunetr34"])
    parser.add_argument("--init", type=str, required=True,
                        choices=["scratch", "finetune"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold index for CV. If None, uses dev_split.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count from config")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size from config")
    return parser.parse_args()


def main():
    args = parse_args()
    root = get_project_root()
    paths = load_paths()
    ensure_dirs(paths)
    data_cfg = load_config(root / "configs" / "data.yaml")
    train_cfg = load_config(root / "configs" / "train.yaml")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Experiment ID
    experiment_id = f"{args.model}_{args.init}_seed{args.seed}"
    if args.fold is not None:
        experiment_id += f"_fold{args.fold}"

    logger = ExperimentLogger(experiment_id, paths["logs_dir"])
    logger.info(f"Device: {device}")

    # Split file
    if args.fold is not None:
        split_file = f"{paths['splits_dir']}/fold_{args.fold}.json"
    else:
        split_file = f"{paths['splits_dir']}/dev_split.json"

    # Transforms
    img_size = tuple(data_cfg["preprocessing"]["image_size"])
    train_tf = get_train_transforms(img_size, data_cfg.get("augmentation"))
    val_tf = get_val_transforms(img_size)

    # Datasets
    metadata_csv = f"{paths['metadata_dir']}/patient_metadata.csv"
    train_ds = ACDCSliceDataset(metadata_csv, split_file, "train",
                                transform=train_tf)
    val_ds = ACDCSliceDataset(metadata_csv, split_file, "val",
                              transform=val_tf)

    batch_size = args.batch_size or train_cfg["training"]["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=train_cfg["training"]["num_workers"],
        pin_memory=train_cfg["training"]["pin_memory"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=train_cfg["training"]["num_workers"],
        pin_memory=train_cfg["training"]["pin_memory"],
    )

    logger.info(f"Train: {len(train_ds)} slices | Val: {len(val_ds)} slices")

    # Model
    pretrained = args.init == "finetune"
    model = build_model(args.model, pretrained=pretrained)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {args.model} | Params: {n_params:,} | Init: {args.init}")

    # Optimizer with differential LR for finetune
    opt_cfg = train_cfg["optimizer"]
    if pretrained and hasattr(model, "encoder"):
        ft_cfg = opt_cfg.get("finetune", {})
        param_groups = [
            {"params": model.encoder.parameters(),
             "lr": ft_cfg.get("encoder_lr", 1e-4)},
            {"params": [p for n, p in model.named_parameters()
                        if not n.startswith("encoder.")],
             "lr": ft_cfg.get("decoder_lr", 1e-3)},
        ]
    else:
        param_groups = model.parameters()

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
    )

    # Scheduler
    epochs = args.epochs or train_cfg["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(train_cfg["scheduler"]["eta_min"]),
    )

    # Loss
    seg_loss_fn = DiceCELoss(num_classes=4)
    boundary_loss_fn = None
    ds_loss_fn = None
    if "m3" in args.model:
        boundary_loss_fn = BoundaryBCELoss()
        ds_loss_fn = DeepSupervisionLoss(base_loss=DiceCELoss(num_classes=4))

    # Merge config for logging
    full_config = {
        "model": args.model, "init": args.init, "seed": args.seed,
        "epochs": epochs, "batch_size": batch_size,
        "paths": paths, "data": data_cfg, "train": train_cfg,
        "loss": train_cfg.get("loss", {}),
        "early_stopping": train_cfg.get("early_stopping", {}),
    }

    # Train
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler,
        seg_loss_fn=seg_loss_fn, device=device,
        experiment_logger=logger, config=full_config,
        boundary_loss_fn=boundary_loss_fn, ds_loss_fn=ds_loss_fn,
    )

    result = trainer.fit()

    # Save training curves
    curves_path = f"{paths['figures_dir']}/{experiment_id}_curves.png"
    plot_training_curves(result["history"], save_path=curves_path)

    print(f"\nDone. Best val_mean_dice: {result['best_metric']:.4f}")
    print(f"Checkpoints: {paths['checkpoints_dir']}/{experiment_id}/")
    print(f"Logs: {paths['logs_dir']}/{experiment_id}/")


if __name__ == "__main__":
    main()
