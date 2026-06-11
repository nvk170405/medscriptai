"""Albumentations augmentation pipeline with curriculum learning support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import yaml

from medscript.utils.logging import get_logger

logger = get_logger(__name__)


def get_normalize_transform(
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> A.Normalize:
    """Get ImageNet normalization transform."""
    return A.Normalize(mean=list(mean), std=list(std), max_pixel_value=255.0)


def get_resize_transform(
    height: int = 960,
    width: int = 1280,
    keep_aspect_ratio: bool = True,
) -> A.BasicTransform:
    """Get resize transform with optional aspect ratio preservation."""
    if keep_aspect_ratio:
        return A.LongestMaxSize(max_size=max(height, width))
    return A.Resize(height=height, width=width)


def get_light_augmentation() -> A.Compose:
    """Light augmentations for early curriculum stage."""
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.3,
        ),
        A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),
        A.Rotate(limit=5, border_mode=cv2.BORDER_CONSTANT, value=255, p=0.2),
    ])


def get_medium_augmentation() -> A.Compose:
    """Medium augmentations for mid curriculum stage."""
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, value=255, p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.05,
            rotate_limit=5,
            border_mode=cv2.BORDER_CONSTANT,
            value=255,
            p=0.3,
        ),
    ])


def get_heavy_augmentation() -> A.Compose:
    """Heavy augmentations for late curriculum stage (messy handwriting)."""
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5,
        ),
        A.GaussNoise(var_limit=(20.0, 80.0), p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=255, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.4),
        A.ElasticTransform(alpha=1.0, sigma=50, p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            value=255,
            p=0.4,
        ),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        A.Downscale(scale_range=(0.5, 0.75), p=0.2),
        A.ImageCompression(quality_range=(50, 90), p=0.2),
    ])


def get_validation_transform(
    height: int = 960,
    width: int = 1280,
) -> A.Compose:
    """Validation/test transform — resize and normalize only."""
    return A.Compose([
        A.LongestMaxSize(max_size=max(height, width)),
        A.PadIfNeeded(
            min_height=height,
            min_width=width,
            border_mode=cv2.BORDER_CONSTANT,
            value=255,
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
    ])


# ── Augmentation Level Registry ─────────────────────────────────────────────

AUGMENTATION_LEVELS = {
    "light": get_light_augmentation,
    "medium": get_medium_augmentation,
    "heavy": get_heavy_augmentation,
}


def get_train_transform(
    level: str = "medium",
    height: int = 960,
    width: int = 1280,
) -> A.Compose:
    """
    Get training transform for a given curriculum level.

    Args:
        level: "light", "medium", or "heavy"
        height: Target image height
        width: Target image width
    """
    if level not in AUGMENTATION_LEVELS:
        raise ValueError(f"Unknown augmentation level: {level}. Choose from {list(AUGMENTATION_LEVELS)}")

    aug_fn = AUGMENTATION_LEVELS[level]

    return A.Compose([
        # Resize
        A.LongestMaxSize(max_size=max(height, width)),
        A.PadIfNeeded(
            min_height=height,
            min_width=width,
            border_mode=cv2.BORDER_CONSTANT,
            value=255,
        ),
        # Augmentation
        aug_fn(),
        # Normalize
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
    ])


class CurriculumAugmenter:
    """
    Manages augmentation level transitions during curriculum learning.

    Progressively increases augmentation difficulty as training progresses.
    """

    def __init__(
        self,
        stages: list[dict[str, Any]] | None = None,
        height: int = 960,
        width: int = 1280,
    ) -> None:
        if stages is None:
            stages = [
                {"name": "easy", "epochs": [0, 10], "augmentation_level": "light"},
                {"name": "medium", "epochs": [10, 30], "augmentation_level": "medium"},
                {"name": "hard", "epochs": [30, 50], "augmentation_level": "heavy"},
            ]

        self.stages = stages
        self.height = height
        self.width = width
        self._current_level = stages[0]["augmentation_level"]
        self._current_transform = get_train_transform(self._current_level, height, width)

    def update_epoch(self, epoch: int) -> str:
        """
        Update augmentation level based on current epoch.

        Returns the current augmentation level name.
        """
        for stage in self.stages:
            start, end = stage["epochs"]
            if start <= epoch < end:
                new_level = stage["augmentation_level"]
                if new_level != self._current_level:
                    logger.info(
                        "curriculum_level_change",
                        epoch=epoch,
                        old_level=self._current_level,
                        new_level=new_level,
                    )
                    self._current_level = new_level
                    self._current_transform = get_train_transform(
                        new_level, self.height, self.width
                    )
                return new_level

        # Past all stages — use heaviest
        return self._current_level

    @property
    def transform(self) -> A.Compose:
        """Get current augmentation transform."""
        return self._current_transform

    @property
    def current_level(self) -> str:
        """Get current augmentation level name."""
        return self._current_level
