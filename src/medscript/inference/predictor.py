"""Inference predictor — loads trained model and runs prediction on images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from medscript.data.preprocessing import preprocess_image
from medscript.models.medscript_model import MedScriptModel, TranscriptionResult
from medscript.utils.logging import get_logger

logger = get_logger(__name__)


class MedScriptPredictor:
    """
    Production inference predictor.

    Loads a trained model checkpoint and provides a simple predict() API
    for transcribing prescription images.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu",
        idx_to_char: dict[int, str] | None = None,
        target_height: int = 960,
        target_width: int = 1280,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.device = torch.device(device)
        self.target_height = target_height
        self.target_width = target_width
        self.confidence_threshold = confidence_threshold

        # Character mapping
        if idx_to_char is None:
            charset = (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789 .,;:!?-/()'\"@#%&+=[]{}|\\<>$^*~`_"
            )
            self.idx_to_char = {0: "", 1: "", 2: "?"}  # blank, pad, unk
            for i, char in enumerate(charset, start=3):
                self.idx_to_char[i] = char
        else:
            self.idx_to_char = idx_to_char

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        # Load model
        self.model: MedScriptModel | None = None
        if checkpoint_path:
            self.load_model(checkpoint_path)

    def load_model(self, checkpoint_path: str | Path) -> None:
        """Load model from checkpoint."""
        logger.info("loading_model", checkpoint=str(checkpoint_path))

        # Initialize model
        self.model = MedScriptModel(use_pretrained=False)

        # Load checkpoint
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)

        if "state_dict" in checkpoint:
            # Lightning checkpoint
            state_dict = {
                k.replace("model.", "", 1): v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith("model.")
            }
            self.model.load_state_dict(state_dict, strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)

        self.model.to(self.device)
        self.model.eval()
        logger.info("model_loaded")

        # Warmup
        self._warmup()

    def _warmup(self) -> None:
        """Run a warmup inference to initialize CUDA kernels."""
        if self.model is None:
            return
        dummy = torch.randn(1, 3, self.target_height, self.target_width).to(self.device)
        with torch.no_grad():
            _ = self.model.forward_encoder_decoder(dummy)
        logger.info("model_warmup_complete")

    def _preprocess(self, image: np.ndarray | Image.Image | str | Path) -> torch.Tensor:
        """Preprocess image for model input."""
        # Load image
        if isinstance(image, (str, Path)):
            image_np = cv2.imread(str(image))
            if image_np is None:
                raise FileNotFoundError(f"Could not load image: {image}")
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image_np = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
                image_np = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                image_np = image
            else:
                image_np = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Preprocess (deskew, enhance, resize)
        processed = preprocess_image(
            cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR),
            target_height=self.target_height,
            target_width=self.target_width,
        )
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        # Normalize
        normalized = processed.astype(np.float32) / 255.0
        normalized = (normalized - self.mean) / self.std

        # To tensor (H, W, C) → (1, C, H, W)
        tensor = torch.from_numpy(normalized).float().permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def predict(
        self,
        image: np.ndarray | Image.Image | str | Path,
        run_ner: bool = True,
    ) -> TranscriptionResult:
        """
        Transcribe a single prescription image.

        Args:
            image: Input image (path, PIL Image, or numpy array)
            run_ner: Whether to run entity extraction

        Returns:
            TranscriptionResult with transcription, entities, and confidences
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Preprocess
        pixel_values = self._preprocess(image)

        # Inference
        results = self.model.transcribe(
            pixel_values,
            idx_to_char=self.idx_to_char,
            run_ner=run_ner,
        )

        result = results[0]

        # Flag low-confidence words
        if result.word_confidences:
            low_confidence_count = sum(
                1 for c in result.word_confidences if c < self.confidence_threshold
            )
            if low_confidence_count > 0:
                logger.info(
                    "low_confidence_words",
                    count=low_confidence_count,
                    threshold=self.confidence_threshold,
                )

        return result

    def predict_batch(
        self,
        images: list[np.ndarray | Image.Image | str | Path],
        run_ner: bool = True,
    ) -> list[TranscriptionResult]:
        """Predict on a batch of images."""
        results = []
        for img in images:
            result = self.predict(img, run_ner=run_ner)
            results.append(result)
        return results
