"""High-level Trainer orchestrating the full training loop."""
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from src.utils.logging import ExperimentLogger
from .callbacks import EarlyStopping, ModelCheckpoint
from .engine import train_one_epoch, validate


class Trainer:
    """Orchestrates training, validation, logging, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        seg_loss_fn: nn.Module,
        device: torch.device,
        experiment_logger: ExperimentLogger,
        config: Dict[str, Any],
        boundary_loss_fn: Optional[nn.Module] = None,
        ds_loss_fn: Optional[nn.Module] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.seg_loss_fn = seg_loss_fn
        self.device = device
        self.logger = experiment_logger
        self.config = config
        self.boundary_loss_fn = boundary_loss_fn
        self.ds_loss_fn = ds_loss_fn

        train_cfg = config.get("train", config.get("training", {}))
        loss_cfg = config.get("loss", {})

        self.epochs = train_cfg.get("epochs", 200)
        self.use_amp = train_cfg.get("mixed_precision", True)
        self.grad_clip = train_cfg.get("gradient_clip_norm", 0.0)
        self.lambda_b = loss_cfg.get("boundary_loss", {}).get("weight", 0.2)
        self.lambda_ds = loss_cfg.get("deep_supervision_loss", {}).get("weight", 0.2)

        self.scaler = GradScaler() if self.use_amp else None

        # Callbacks
        paths_cfg = config.get("paths", {})
        ckpt_dir = paths_cfg.get("checkpoints_dir", "checkpoints")
        es_cfg = config.get("early_stopping", {})

        self.checkpoint = ModelCheckpoint(
            save_dir=f"{ckpt_dir}/{experiment_logger.experiment_id}",
        )
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 30),
            min_delta=es_cfg.get("min_delta", 0.001),
        ) if es_cfg.get("enabled", True) else None

        self.history = {
            "train_loss": [], "val_loss": [],
            "val_mean_dice": [], "val_dice_rv": [],
            "val_dice_myo": [], "val_dice_lv": [],
        }

    def fit(self) -> Dict[str, Any]:
        """Run the full training loop."""
        self.logger.info(f"Starting training for {self.epochs} epochs")
        self.logger.log_config(self.config)

        for epoch in range(1, self.epochs + 1):
            # Train
            train_metrics = train_one_epoch(
                self.model, self.train_loader, self.optimizer,
                self.seg_loss_fn, self.device,
                scaler=self.scaler,
                boundary_loss_fn=self.boundary_loss_fn,
                ds_loss_fn=self.ds_loss_fn,
                lambda_b=self.lambda_b,
                lambda_ds=self.lambda_ds,
                grad_clip=self.grad_clip,
            )

            # Validate
            val_metrics = validate(
                self.model, self.val_loader,
                self.seg_loss_fn, self.device,
            )

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            # Log
            all_metrics = {**train_metrics, **val_metrics}
            for k, v in all_metrics.items():
                self.history[k].append(v) if k in self.history else None
            self.logger.log_metrics(all_metrics, epoch)
            self.logger.info(
                f"Epoch {epoch}/{self.epochs} | "
                f"loss={train_metrics['train_loss']:.4f} | "
                f"val_dice={val_metrics['val_mean_dice']:.4f}"
            )

            # Checkpoint
            improved = self.checkpoint.step(
                val_metrics, self.model, epoch,
                extra_info={"optimizer_state_dict": self.optimizer.state_dict()},
            )
            self.checkpoint.save_last(self.model, epoch)

            if improved:
                self.logger.info(
                    f"New best val_mean_dice: {val_metrics['val_mean_dice']:.4f}"
                )

            # Early stopping
            if self.early_stopping:
                if self.early_stopping.step(val_metrics["val_mean_dice"]):
                    self.logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(patience={self.early_stopping.patience})"
                    )
                    break

        self.logger.close()
        return {
            "history": self.history,
            "best_metric": self.checkpoint.best,
        }
