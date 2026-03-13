"""Augmentation pipeline using albumentations. Identical across all models."""
from typing import Any, Dict, Optional, Tuple

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    image_size: Tuple[int, int] = (256, 256),
    aug_cfg: Optional[Dict[str, Any]] = None,
) -> A.Compose:
    """Build training augmentation pipeline from config."""
    transforms = [
        A.Resize(height=image_size[0], width=image_size[1],
                  interpolation=1),  # bilinear for image
    ]

    if aug_cfg and aug_cfg.get("enabled", True):
        train_cfg = aug_cfg.get("train", {})

        if "shift_scale_rotate" in train_cfg:
            c = train_cfg["shift_scale_rotate"]
            transforms.append(A.ShiftScaleRotate(
                shift_limit=c.get("shift_limit", 0.05),
                scale_limit=c.get("scale_limit", 0.1),
                rotate_limit=c.get("rotate_limit", 15),
                p=c.get("p", 0.5),
                border_mode=0,
            ))

        if "horizontal_flip" in train_cfg:
            c = train_cfg["horizontal_flip"]
            transforms.append(A.HorizontalFlip(p=c.get("p", 0.5)))

        if "elastic_transform" in train_cfg:
            c = train_cfg["elastic_transform"]
            transforms.append(A.ElasticTransform(
                alpha=c.get("alpha", 50),
                sigma=c.get("sigma", 10),
                p=c.get("p", 0.2),
                border_mode=0,
            ))

        if "brightness_contrast" in train_cfg:
            c = train_cfg["brightness_contrast"]
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=c.get("brightness_limit", 0.1),
                contrast_limit=c.get("contrast_limit", 0.1),
                p=c.get("p", 0.3),
            ))

        if "gaussian_noise" in train_cfg:
            c = train_cfg["gaussian_noise"]
            transforms.append(A.GaussNoise(
                var_limit=tuple(c.get("var_limit", [5.0, 20.0])),
                p=c.get("p", 0.2),
            ))

    transforms.append(ToTensorV2())
    return A.Compose(transforms)


def get_val_transforms(
    image_size: Tuple[int, int] = (256, 256),
) -> A.Compose:
    """Build validation transform pipeline (resize only, no augmentation)."""
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1],
                  interpolation=1),
        ToTensorV2(),
    ])
