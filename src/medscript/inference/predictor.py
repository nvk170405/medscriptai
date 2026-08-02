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
        self.idx_to_char = {}
        self.vocab_size = 95
        if idx_to_char is not None:
            self.idx_to_char = idx_to_char
            self.vocab_size = len(idx_to_char)
        elif checkpoint_path:
            import json
            vocab_path = Path(checkpoint_path).parent / "vocab.json"
            if vocab_path.exists():
                with open(vocab_path, "r", encoding="utf-8") as f:
                    vocab_data = json.load(f)
                    if "idx_to_char" in vocab_data:
                        self.idx_to_char = {int(k): v for k, v in vocab_data["idx_to_char"].items()}
                        self.vocab_size = vocab_data.get("vocab_size", len(self.idx_to_char))
                    else:
                        self.idx_to_char = {int(v): k for k, v in vocab_data.items()}
                        self.vocab_size = len(vocab_data)
            else:
                # fallback
                charset = (
                    "abcdefghijklmnopqrstuvwxyz"
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "0123456789 .,;:!?-/()'\"@#%&+=[]{}|\\<>$^*~`_"
                )
                self.idx_to_char = {0: "", 1: "", 2: "?"}
                for i, char in enumerate(charset, start=3):
                    self.idx_to_char[i] = char
                self.vocab_size = 95

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

        # Load config to get the correct architecture params
        import yaml
        config_path = Path("configs/model_config.yaml")
        donut_cfg, bilstm_cfg, bert_cfg = {}, {}, {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                donut_cfg = cfg.get("donut", {})
                bilstm_cfg = cfg.get("bilstm", {})
                bert_cfg = cfg.get("medical_bert", {})

        # Initialize model with config params
        self.model = MedScriptModel(
            pretrained_donut=donut_cfg.get("pretrained_model", "naver-clova-ix/donut-base"),
            encoder_output_dim=bilstm_cfg.get("input_dim", 1024),
            bilstm_hidden_size=bilstm_cfg.get("hidden_size", 256),
            bilstm_num_layers=bilstm_cfg.get("num_layers", 2),
            bilstm_dropout=bilstm_cfg.get("dropout", 0.3),
            vocab_size=self.vocab_size,
            pretrained_bert=bert_cfg.get("pretrained_model", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"),
            num_ner_labels=bert_cfg.get("num_labels", 11),
            use_pretrained=False,
        )

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
