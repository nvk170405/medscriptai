"""Test fixtures for MedScript AI tests."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a dummy prescription image for testing."""
    # White background with some black text-like marks
    image = np.full((960, 1280, 3), 240, dtype=np.uint8)
    # Add some dark "text" rectangles
    for y in range(200, 800, 60):
        x_start = np.random.randint(100, 200)
        width = np.random.randint(200, 600)
        image[y:y + 15, x_start:x_start + width] = np.random.randint(10, 50, (15, width, 3))
    return image


@pytest.fixture
def sample_image_pil(sample_image: np.ndarray) -> Image.Image:
    """Create a PIL Image from sample image."""
    return Image.fromarray(sample_image)


@pytest.fixture
def sample_image_path(sample_image: np.ndarray, tmp_path: Path) -> Path:
    """Save sample image to disk and return path."""
    import cv2
    path = tmp_path / "test_prescription.png"
    cv2.imwrite(str(path), sample_image)
    return path


@pytest.fixture
def sample_annotations(tmp_path: Path) -> Path:
    """Create sample annotations JSON."""
    annotations = [
        {
            "image_id": 0,
            "image_path": "images/test_000000.png",
            "text": "Amoxicillin 500mg TID for 7 days",
            "lines": [
                {
                    "medicine": "Amoxicillin",
                    "dosage_form": "Tab",
                    "dosage": "500mg",
                    "frequency": "TID",
                    "duration": "7 days",
                    "instruction": "",
                    "raw_text": "Amoxicillin Tab 500mg TID x 7 days",
                }
            ],
        },
        {
            "image_id": 1,
            "image_path": "images/test_000001.png",
            "text": "Paracetamol 650mg SOS",
            "lines": [
                {
                    "medicine": "Paracetamol",
                    "dosage_form": "Tab",
                    "dosage": "650mg",
                    "frequency": "SOS",
                    "duration": "5 days",
                    "instruction": "after food",
                    "raw_text": "Paracetamol Tab 650mg SOS x 5 days (after food)",
                }
            ],
        },
    ]

    ann_path = tmp_path / "annotations.json"
    with open(ann_path, "w") as f:
        json.dump(annotations, f)

    # Create dummy images
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for ann in annotations:
        img = np.full((960, 1280, 3), 240, dtype=np.uint8)
        import cv2
        cv2.imwrite(str(tmp_path / ann["image_path"]), img)

    return ann_path


@pytest.fixture
def sample_pixel_values() -> torch.Tensor:
    """Create a batch of dummy pixel values for model testing."""
    return torch.randn(2, 3, 960, 1280)


@pytest.fixture
def model_config() -> dict[str, Any]:
    """Default model configuration for testing."""
    return {
        "encoder_output_dim": 64,  # Small for testing
        "bilstm_hidden_size": 32,
        "bilstm_num_layers": 1,
        "vocab_size": 95,
        "bilstm_dropout": 0.1,
    }
