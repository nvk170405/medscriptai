"""Unit tests for image preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from medscript.data.preprocessing import (
    deskew,
    enhance_contrast,
    binarize,
    denoise,
    resize_with_aspect_ratio,
    preprocess_image,
)


class TestPreprocessing:
    """Tests for preprocessing functions."""

    def test_deskew_returns_same_shape(self, sample_image: np.ndarray) -> None:
        """Deskew should preserve image dimensions."""
        result = deskew(sample_image)
        assert result.shape == sample_image.shape

    def test_enhance_contrast_returns_same_shape(self, sample_image: np.ndarray) -> None:
        """CLAHE should preserve image dimensions."""
        result = enhance_contrast(sample_image)
        assert result.shape == sample_image.shape

    def test_binarize_otsu(self, sample_image: np.ndarray) -> None:
        """Otsu binarization should return 2D binary image."""
        result = binarize(sample_image, method="otsu")
        assert len(result.shape) == 2
        assert set(np.unique(result)).issubset({0, 255})

    def test_binarize_adaptive(self, sample_image: np.ndarray) -> None:
        """Adaptive binarization should return 2D binary image."""
        result = binarize(sample_image, method="adaptive")
        assert len(result.shape) == 2

    def test_denoise_returns_same_shape(self, sample_image: np.ndarray) -> None:
        """Denoising should preserve image dimensions."""
        result = denoise(sample_image)
        assert result.shape == sample_image.shape

    def test_resize_with_aspect_ratio(self, sample_image: np.ndarray) -> None:
        """Resize should produce target dimensions with padding."""
        result = resize_with_aspect_ratio(sample_image, 480, 640)
        assert result.shape[0] == 480
        assert result.shape[1] == 640

    def test_preprocess_pipeline(self, sample_image: np.ndarray) -> None:
        """Full pipeline should produce correctly sized output."""
        result = preprocess_image(sample_image, target_height=480, target_width=640)
        assert result.shape[0] == 480
        assert result.shape[1] == 640

    def test_preprocess_no_enhance(self, sample_image: np.ndarray) -> None:
        """Pipeline should work with enhancement disabled."""
        result = preprocess_image(
            sample_image, enhance=False, do_deskew=False, do_denoise=False
        )
        assert result.shape[0] == 960
        assert result.shape[1] == 1280
