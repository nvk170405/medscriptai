"""Inference predictor — uses EasyOCR + BiomedBERT NER for prescription analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from medscript.models.medscript_model import TranscriptionResult
from medscript.utils.logging import get_logger

logger = get_logger(__name__)


class MedScriptPredictor:
    """
    Production inference predictor.

    Uses EasyOCR (pre-trained) for text extraction and BiomedBERT NER
    for structured entity extraction. No custom training required.

    Pipeline:
        Image → EasyOCR → raw text → BiomedBERT NER → structured entities
    """

    def __init__(
        self,
        device: str = "cpu",
        languages: list[str] | None = None,
        confidence_threshold: float = 0.3,
        **kwargs: Any,
    ) -> None:
        self.device = device
        self.confidence_threshold = confidence_threshold
        self._languages = languages or ["en"]

        # Initialize EasyOCR reader
        self._reader = None
        self._ner = None

        self._init_ocr()
        self._init_ner()

    def _init_ocr(self) -> None:
        """Initialize EasyOCR reader."""
        import easyocr

        gpu = self.device != "cpu"
        logger.info("initializing_easyocr", languages=self._languages, gpu=gpu)
        self._reader = easyocr.Reader(
            self._languages,
            gpu=gpu,
            verbose=False,
        )
        logger.info("easyocr_ready")

    def _init_ner(self) -> None:
        """Initialize rule-based entity extractor."""
        from medscript.inference.entity_extractor import extract_entities

        self._extract_entities = extract_entities
        logger.info("entity_extractor_ready", engine="rule-based")

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
        if self._reader is None:
            raise RuntimeError("EasyOCR not initialized.")

        # Convert image to numpy array for EasyOCR
        image_input = self._prepare_image(image)

        # Run EasyOCR
        logger.info("running_easyocr_inference")
        ocr_results = self._reader.readtext(image_input)

        # Parse OCR results
        text_parts: list[str] = []
        confidences: list[float] = []

        for bbox, text, conf in ocr_results:
            if conf >= self.confidence_threshold:
                text_parts.append(text)
                confidences.append(float(conf))

        transcription = " ".join(text_parts)
        logger.info(
            "ocr_complete",
            text_length=len(transcription),
            num_segments=len(text_parts),
        )

        # Build result
        result = TranscriptionResult(
            transcription=transcription,
            word_confidences=confidences,
            model_version="medscript-ai-v0.2-easyocr",
        )

        # Run entity extraction
        if run_ner and hasattr(self, '_extract_entities') and transcription.strip():
            try:
                entities = self._extract_entities(transcription)
                result.entities = entities
                logger.info("ner_complete", num_entities=len(entities))
            except Exception as e:
                logger.warning("entity_extraction_failed", error=str(e))

        return result

    def predict_batch(
        self,
        images: list[np.ndarray | Image.Image | str | Path],
        run_ner: bool = True,
    ) -> list[TranscriptionResult]:
        """Predict on a batch of images."""
        return [self.predict(img, run_ner=run_ner) for img in images]

    @staticmethod
    def _prepare_image(image: np.ndarray | Image.Image | str | Path) -> np.ndarray:
        """Convert any image input to a numpy array for EasyOCR."""
        import cv2

        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise FileNotFoundError(f"Could not load image: {image}")
            return img
        elif isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            return image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
