import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Create a logger with console and optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter(
            "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


class ExperimentLogger:
    """Logs experiment configs, metrics, and events to disk + TensorBoard."""

    def __init__(self, experiment_id: str, log_dir: str):
        self.experiment_id = experiment_id
        self.log_dir = Path(log_dir) / experiment_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(
            experiment_id, str(self.log_dir / "train.log")
        )
        self.tb_writer = SummaryWriter(
            log_dir=str(self.log_dir / "tensorboard")
        )

    def log_config(self, config: Dict[str, Any]) -> None:
        with open(self.log_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

    def log_metric(self, tag: str, value: float, step: int) -> None:
        self.tb_writer.add_scalar(tag, value, step)

    def log_metrics(
        self, metrics: Dict[str, float], step: int, prefix: str = ""
    ) -> None:
        for k, v in metrics.items():
            tag = f"{prefix}/{k}" if prefix else k
            self.tb_writer.add_scalar(tag, v, step)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def close(self) -> None:
        self.tb_writer.close()


def append_to_registry(registry_path: str, row: Dict[str, Any]) -> None:
    """Append a row to the experiment registry CSV."""
    path = Path(registry_path)
    file_exists = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
