"""Training callbacks: early stopping and model checkpointing."""
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class EarlyStopping:
    """Stop training when monitored metric stops improving."""

    def __init__(self, patience: int = 30, min_delta: float = 0.001,
                 mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False

        if self.mode == "max":
            improved = value > self.best + self.min_delta
        else:
            improved = value < self.best - self.min_delta

        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class ModelCheckpoint:
    """Save model when monitored metric improves."""

    def __init__(self, save_dir: str, monitor: str = "val_mean_dice",
                 mode: str = "max"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best = None

    def step(self, metrics: dict, model: nn.Module,
             epoch: int, extra_info: Optional[dict] = None) -> bool:
        value = metrics[self.monitor]
        if self.best is None:
            improved = True
        elif self.mode == "max":
            improved = value > self.best
        else:
            improved = value < self.best

        if improved:
            self.best = value
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_metric": value,
            }
            if extra_info:
                state.update(extra_info)
            torch.save(state, self.save_dir / "best_model.pth")
            return True
        return False

    def save_last(self, model: nn.Module, epoch: int,
                  extra_info: Optional[dict] = None) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
        }
        if extra_info:
            state.update(extra_info)
        torch.save(state, self.save_dir / "last_model.pth")
