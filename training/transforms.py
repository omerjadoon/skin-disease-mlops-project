"""
Torchvision transforms for training and validation/test pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torchvision.transforms as T


@dataclass
class AugmentationConfig:
    """Configuration for training augmentations."""

    image_size: int = 224
    random_horizontal_flip: bool = True
    random_rotation_degrees: int = 15
    color_jitter: dict[str, float] = field(
        default_factory=lambda: {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1,
        }
    )
    random_resized_crop_scale: tuple[float, float] = (0.7, 1.0)
    random_resized_crop_ratio: tuple[float, float] = (0.75, 1.33)
    normalize_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


def get_train_transforms(config: AugmentationConfig | None = None) -> T.Compose:
    """
    Build training augmentation pipeline.

    Includes: RandomResizedCrop, RandomHorizontalFlip, RandomRotation,
              ColorJitter, ToTensor, Normalize
    """
    if config is None:
        config = AugmentationConfig()

    transforms_list: list[Any] = [
        T.RandomResizedCrop(
            config.image_size,
            scale=config.random_resized_crop_scale,
            ratio=config.random_resized_crop_ratio,
        ),
    ]

    if config.random_horizontal_flip:
        transforms_list.append(T.RandomHorizontalFlip(p=0.5))

    if config.random_rotation_degrees > 0:
        transforms_list.append(T.RandomRotation(degrees=config.random_rotation_degrees))

    if config.color_jitter:
        transforms_list.append(
            T.ColorJitter(
                brightness=config.color_jitter.get("brightness", 0.2),
                contrast=config.color_jitter.get("contrast", 0.2),
                saturation=config.color_jitter.get("saturation", 0.2),
                hue=config.color_jitter.get("hue", 0.1),
            )
        )

    transforms_list.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ]
    )

    return T.Compose(transforms_list)


def get_val_transforms(
    image_size: int = 224,
    normalize_mean: list[float] | None = None,
    normalize_std: list[float] | None = None,
) -> T.Compose:
    """
    Build validation/test transform pipeline.

    Includes: Resize, CenterCrop, ToTensor, Normalize
    """
    if normalize_mean is None:
        normalize_mean = [0.485, 0.456, 0.406]
    if normalize_std is None:
        normalize_std = [0.229, 0.224, 0.225]

    return T.Compose(
        [
            T.Resize(int(image_size * 1.14)),  # ~256 for 224
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=normalize_mean, std=normalize_std),
        ]
    )


def get_inference_transforms(
    image_size: int = 224,
    normalize_mean: list[float] | None = None,
    normalize_std: list[float] | None = None,
) -> T.Compose:
    """Transforms used during inference (same as val, no augmentation)."""
    return get_val_transforms(image_size, normalize_mean, normalize_std)


def build_transforms_from_config(cfg: dict[str, Any]) -> tuple[T.Compose, T.Compose]:
    """
    Build train and val transforms from a config dictionary.

    Args:
        cfg: Training config dict (from configs/train.yaml)

    Returns:
        (train_transform, val_transform)
    """
    image_size = cfg.get("image_size", 224)
    normalize = cfg.get("normalize", {})
    mean = normalize.get("mean", [0.485, 0.456, 0.406])
    std = normalize.get("std", [0.229, 0.224, 0.225])
    aug = cfg.get("augmentations", {})

    aug_config = AugmentationConfig(
        image_size=image_size,
        random_horizontal_flip=aug.get("random_horizontal_flip", True),
        random_rotation_degrees=aug.get("random_rotation_degrees", 15),
        color_jitter=aug.get("color_jitter", {}),
        random_resized_crop_scale=tuple(
            aug.get("random_resized_crop", {}).get("scale", [0.7, 1.0])
        ),
        random_resized_crop_ratio=tuple(
            aug.get("random_resized_crop", {}).get("ratio", [0.75, 1.33])
        ),
        normalize_mean=mean,
        normalize_std=std,
    )

    train_tf = get_train_transforms(aug_config)
    val_tf = get_val_transforms(image_size, mean, std)
    return train_tf, val_tf
