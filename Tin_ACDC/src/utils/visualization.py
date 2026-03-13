from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

# Class colors: background=black, RV=red, MYO=green, LV=blue
CLASS_COLORS = np.array(
    [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8
)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert a label mask (H, W) to an RGB image."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(CLASS_COLORS):
        rgb[mask == cls_id] = color
    return rgb


def overlay_mask(
    image: np.ndarray, mask: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    """Overlay a colored mask on a grayscale image."""
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    img = image.astype(np.float32)
    img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(
        np.uint8
    )
    mask_rgb = mask_to_rgb(mask)
    fg = mask > 0
    blended = img.copy()
    blended[fg] = (alpha * mask_rgb[fg] + (1 - alpha) * img[fg]).astype(
        np.uint8
    )
    return blended


def plot_sample(
    image: np.ndarray,
    mask_gt: np.ndarray,
    mask_pred: Optional[np.ndarray] = None,
    title: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Plot image with GT and optional prediction overlay."""
    ncols = 3 if mask_pred is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(overlay_mask(image, mask_gt))
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    if mask_pred is not None:
        axes[2].imshow(overlay_mask(image, mask_pred))
        axes[2].set_title("Prediction")
        axes[2].axis("off")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_training_curves(
    history: Dict[str, list], save_path: Optional[str] = None
) -> None:
    """Plot loss and dice curves from training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if "train_loss" in history:
        ax1.plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Loss Curves")

    if "val_mean_dice" in history:
        ax2.plot(history["val_mean_dice"], label="Val Mean Dice")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice")
    ax2.legend()
    ax2.set_title("Validation Dice")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
