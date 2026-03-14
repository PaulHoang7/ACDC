"""Augmentation pipeline using albumentations. Identical across all models."""
from typing import Any, Dict, Optional, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    image_size: Tuple[int, int] = (256, 256),
    aug_cfg: Optional[Dict[str, Any]] = None,
) -> A.Compose:
    """Build training augmentation pipeline from config.

    Augmentations applied (all mild, medical-imaging safe):
        - ShiftScaleRotate: translation + scaling + rotation jitter
        - HorizontalFlip
        - ElasticTransform: slight deformation
        - RandomBrightnessContrast: intensity jitter
        - GaussNoise: additive noise
    """
    transforms: list = []

    if aug_cfg and aug_cfg.get("enabled", True):
        train_cfg = aug_cfg.get("train", {})

        if "shift_scale_rotate" in train_cfg:
            c = train_cfg["shift_scale_rotate"]
            transforms.append(A.Affine(
                translate_percent={"x": (-c.get("shift_limit", 0.05), c.get("shift_limit", 0.05)),
                                   "y": (-c.get("shift_limit", 0.05), c.get("shift_limit", 0.05))},
                scale=(1 - c.get("scale_limit", 0.1), 1 + c.get("scale_limit", 0.1)),
                rotate=(-c.get("rotate_limit", 15), c.get("rotate_limit", 15)),
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
            # albumentations >= 2.0: std_range replaces var_limit
            var_limit = c.get("var_limit", [5.0, 20.0])
            std_lo = var_limit[0] ** 0.5 / 255.0
            std_hi = var_limit[1] ** 0.5 / 255.0
            transforms.append(A.GaussNoise(
                std_range=(std_lo, std_hi),
                p=c.get("p", 0.2),
            ))

    transforms.append(ToTensorV2())
    return A.Compose(transforms)


def get_val_transforms(
    image_size: Tuple[int, int] = (256, 256),
) -> A.Compose:
    """Validation/test transforms: tensor conversion only, no augmentation.

    Note: if using cached backend, slices are already resized to target_size
    during preprocessing, so no Resize is needed here.
    """
    return A.Compose([ToTensorV2()])
