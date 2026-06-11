"""PyTorch Lightning DataModule for MedScript AI."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from medscript.data.augmentations import CurriculumAugmenter, get_train_transform, get_validation_transform
from medscript.data.dataset import MedScriptDataset, collate_fn
from medscript.utils.logging import get_logger

logger = get_logger(__name__)


class MedScriptDataModule(pl.LightningDataModule):
    """
    Lightning DataModule managing train/val/test splits and augmentation.

    Supports:
    - Multiple data directories (synthetic + HF + custom)
    - Automatic train/val/test splitting
    - Curriculum learning augmentation
    """

    def __init__(
        self,
        data_dirs: list[str | Path] | None = None,
        split_file: str | Path | None = None,
        batch_size: int = 2,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        image_height: int = 960,
        image_width: int = 1280,
        max_text_length: int = 768,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        augmentation_level: str = "medium",
        curriculum_enabled: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dirs = [Path(d) for d in (data_dirs or ["data/synthetic"])]
        self.split_file = Path(split_file) if split_file else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.image_height = image_height
        self.image_width = image_width
        self.max_text_length = max_text_length
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.augmentation_level = augmentation_level
        self.curriculum_enabled = curriculum_enabled
        self.seed = seed

        # Curriculum augmenter
        self.curriculum_augmenter = CurriculumAugmenter(
            height=image_height,
            width=image_width,
        ) if curriculum_enabled else None

        # Dataset references (populated in setup())
        self.train_dataset: MedScriptDataset | None = None
        self.val_dataset: MedScriptDataset | None = None
        self.test_dataset: MedScriptDataset | None = None

    def _merge_annotations(self) -> tuple[Path, list[dict[str, Any]]]:
        """Merge annotations from all data directories."""
        all_annotations: list[dict[str, Any]] = []
        primary_dir = self.data_dirs[0]

        for data_dir in self.data_dirs:
            ann_file = data_dir / "annotations.json"
            if ann_file.exists():
                with open(ann_file, "r", encoding="utf-8") as f:
                    annotations = json.load(f)

                # Fix relative paths
                for ann in annotations:
                    if "image_path" in ann:
                        # Make paths relative to primary dir
                        abs_path = data_dir / ann["image_path"]
                        try:
                            ann["image_path"] = str(abs_path.relative_to(primary_dir))
                        except ValueError:
                            ann["image_path"] = str(abs_path)
                        ann["_data_dir"] = str(data_dir)

                all_annotations.extend(annotations)
                logger.info(
                    "loaded_annotations",
                    source=str(data_dir),
                    count=len(annotations),
                )

        return primary_dir, all_annotations

    def _create_splits(
        self, annotations: list[dict[str, Any]]
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Split annotations into train/val/test."""
        rng = random.Random(self.seed)
        rng.shuffle(annotations)

        n = len(annotations)
        train_end = int(n * self.train_split)
        val_end = train_end + int(n * self.val_split)

        train = annotations[:train_end]
        val = annotations[train_end:val_end]
        test = annotations[val_end:]

        logger.info(
            "data_splits",
            train=len(train),
            val=len(val),
            test=len(test),
        )

        return train, val, test

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for training/validation/testing."""
        primary_dir, all_annotations = self._merge_annotations()

        if not all_annotations:
            raise ValueError(
                f"No annotations found in {self.data_dirs}. "
                "Run `make generate-synthetic` or `make download-data` first."
            )

        train_anns, val_anns, test_anns = self._create_splits(all_annotations)

        # Save split files for reproducibility
        splits_dir = primary_dir / "splits"
        splits_dir.mkdir(exist_ok=True)

        for name, anns in [("train", train_anns), ("val", val_anns), ("test", test_anns)]:
            split_path = splits_dir / f"{name}.json"
            with open(split_path, "w", encoding="utf-8") as f:
                json.dump(anns, f, indent=2, ensure_ascii=False)

        # Training transform
        if self.curriculum_enabled and self.curriculum_augmenter:
            train_transform = self.curriculum_augmenter.transform
        else:
            train_transform = get_train_transform(
                self.augmentation_level, self.image_height, self.image_width
            )

        val_transform = get_validation_transform(self.image_height, self.image_width)

        # Create datasets
        if stage == "fit" or stage is None:
            # Write temp annotation files for each split
            train_ann_path = splits_dir / "train.json"
            val_ann_path = splits_dir / "val.json"

            self.train_dataset = MedScriptDataset(
                data_dir=primary_dir,
                annotations_file=train_ann_path,
                transform=train_transform,
                max_text_length=self.max_text_length,
            )
            self.val_dataset = MedScriptDataset(
                data_dir=primary_dir,
                annotations_file=val_ann_path,
                transform=val_transform,
                max_text_length=self.max_text_length,
            )

        if stage == "test" or stage is None:
            test_ann_path = splits_dir / "test.json"
            self.test_dataset = MedScriptDataset(
                data_dir=primary_dir,
                annotations_file=test_ann_path,
                transform=val_transform,
                max_text_length=self.max_text_length,
            )

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def update_augmentation_epoch(self, epoch: int) -> None:
        """Update curriculum learning augmentation level for new epoch."""
        if self.curriculum_augmenter and self.train_dataset:
            level = self.curriculum_augmenter.update_epoch(epoch)
            self.train_dataset.transform = self.curriculum_augmenter.transform
