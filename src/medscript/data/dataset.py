"""PyTorch Dataset classes for MedScript AI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from medscript.utils.logging import get_logger

logger = get_logger(__name__)


class MedScriptDataset(Dataset):
    """
    PyTorch Dataset for medical prescription images.

    Supports:
    - Synthetic prescription images (from SyntheticPrescriptionGenerator)
    - HuggingFace downloaded datasets (from download.py)
    - Custom annotated datasets

    Expected annotation format:
    {
        "image_path": "images/synth_000001.png",
        "text": "Amoxicillin 500mg TID for 7 days",
        "lines": [...],   # Optional structured entities
    }
    """

    def __init__(
        self,
        data_dir: str | Path,
        annotations_file: str | Path | None = None,
        transform: Any = None,
        max_text_length: int = 768,
        charset: str | None = None,
    ) -> None:
        """
        Args:
            data_dir: Directory containing images and annotations
            annotations_file: Path to annotations JSON (default: data_dir/annotations.json)
            transform: Albumentations transform pipeline
            max_text_length: Maximum text length for CTC targets
            charset: Character set for encoding text → indices
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.max_text_length = max_text_length

        # Load annotations
        if annotations_file is None:
            annotations_file = self.data_dir / "annotations.json"

        with open(annotations_file, "r", encoding="utf-8") as f:
            self.annotations: list[dict[str, Any]] = json.load(f)

        # Build character-to-index mapping
        if charset is None:
            charset = (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789 .,;:!?-/()'\"@#%&+=[]{}|\\<>$^*~`_"
            )

        self.charset = charset
        self.char_to_idx: dict[str, int] = {"<blank>": 0, "<pad>": 1, "<unk>": 2}
        for i, char in enumerate(charset, start=3):
            self.char_to_idx[char] = i
        self.idx_to_char: dict[int, str] = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

        logger.info(
            "dataset_loaded",
            data_dir=str(self.data_dir),
            num_samples=len(self.annotations),
            vocab_size=self.vocab_size,
        )

    def __len__(self) -> int:
        return len(self.annotations)

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text string to tensor of character indices."""
        indices = []
        for char in text:
            idx = self.char_to_idx.get(char, self.char_to_idx["<unk>"])
            indices.append(idx)

        # Truncate if too long
        if len(indices) > self.max_text_length:
            indices = indices[:self.max_text_length]

        return torch.tensor(indices, dtype=torch.long)

    def decode_indices(self, indices: torch.Tensor | list[int]) -> str:
        """Decode tensor of character indices back to text."""
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        chars = []
        prev_idx = -1
        for idx in indices:
            if idx == 0:  # <blank>
                prev_idx = idx
                continue
            if idx == 1:  # <pad>
                break
            if idx == prev_idx:  # Repeated char (CTC)
                prev_idx = idx
                continue
            char = self.idx_to_char.get(idx, "")
            chars.append(char)
            prev_idx = idx

        return "".join(chars)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a single sample.

        Returns:
            Dict with keys:
            - image: (C, H, W) float tensor
            - text: raw text string
            - target: encoded text tensor
            - target_length: length of encoded text
            - metadata: dict with source info, entities, etc.
        """
        ann = self.annotations[idx]

        # Load image
        image_path = self.data_dir / ann["image_path"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentation/transform
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        # Convert to tensor (H, W, C) → (C, H, W)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
            if image.dim() == 3 and image.shape[-1] in (1, 3):
                image = image.permute(2, 0, 1)  # (H, W, C) → (C, H, W)

        # Encode text target
        text = ann.get("text", "") or ann.get("full_text", "")

        # For multi-line prescriptions, use the prescription body
        if "lines" in ann and ann["lines"]:
            line_texts = [line.get("raw_text", "") for line in ann["lines"]]
            text = " | ".join(line_texts)

        target = self.encode_text(text)
        target_length = len(target)

        # Metadata
        metadata = {
            "image_path": str(image_path),
            "source": ann.get("source", "unknown"),
        }

        # Add entity info if available
        if "lines" in ann:
            metadata["entities"] = ann["lines"]

        return {
            "image": image,
            "text": text,
            "target": target,
            "target_length": target_length,
            "metadata": metadata,
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Custom collate function for variable-length targets.

    Pads targets to the same length within a batch.
    """
    images = torch.stack([item["image"] for item in batch])
    texts = [item["text"] for item in batch]
    target_lengths = torch.tensor([item["target_length"] for item in batch], dtype=torch.long)
    metadata = [item["metadata"] for item in batch]

    # Pad targets to max length in batch
    max_target_len = target_lengths.max().item()
    padded_targets = torch.full(
        (len(batch), max_target_len),
        fill_value=1,  # <pad> token
        dtype=torch.long,
    )
    for i, item in enumerate(batch):
        length = item["target_length"]
        padded_targets[i, :length] = item["target"]

    return {
        "images": images,
        "texts": texts,
        "targets": padded_targets,
        "target_lengths": target_lengths,
        "metadata": metadata,
    }
