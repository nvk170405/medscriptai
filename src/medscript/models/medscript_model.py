"""MedScript end-to-end pipeline — orchestrates Donut → BiLSTM-CTC → Medical BERT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from medscript.models.donut_encoder import DonutVisionEncoder
from medscript.models.bilstm_ctc import BiLSTMCTCDecoder
from medscript.models.medical_bert import MedicalBERTNER
from medscript.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TranscriptionResult:
    """Structured output from the MedScript pipeline."""
    transcription: str = ""
    entities: list[dict[str, Any]] = field(default_factory=list)
    word_confidences: list[float] = field(default_factory=list)
    model_version: str = "medscript-ai-v0.1"
    raw_log_probs: torch.Tensor | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "transcription": self.transcription,
            "entities": self.entities,
            "word_confidences": self.word_confidences,
            "model_version": self.model_version,
        }


class MedScriptModel(nn.Module):
    """
    End-to-end MedScript AI pipeline.

    Combines three models into a single orchestrated pipeline:
    1. Donut Vision Encoder → extract spatial features from prescription image
    2. BiLSTM-CTC Decoder → decode features into text
    3. Medical BERT NER → extract structured entities from text

    Supports both:
    - Joint training (all components trained together)
    - Staged training (freeze encoder, train decoder, then fine-tune all)
    """

    def __init__(
        self,
        # Donut config
        pretrained_donut: str = "naver-clova-ix/donut-base",
        encoder_output_dim: int = 512,
        freeze_encoder: bool = False,
        # BiLSTM config
        bilstm_hidden_size: int = 256,
        bilstm_num_layers: int = 2,
        bilstm_dropout: float = 0.3,
        vocab_size: int = 95,
        # Medical BERT config
        pretrained_bert: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        num_ner_labels: int = 11,
        # General
        use_pretrained: bool = True,
    ) -> None:
        super().__init__()

        # Stage 1: Vision encoder
        self.encoder = DonutVisionEncoder(
            pretrained_model=pretrained_donut,
            output_dim=encoder_output_dim,
            freeze_encoder=freeze_encoder,
            use_pretrained=use_pretrained,
        )

        # Stage 2: Sequence decoder
        self.decoder = BiLSTMCTCDecoder(
            input_dim=encoder_output_dim,
            hidden_size=bilstm_hidden_size,
            num_layers=bilstm_num_layers,
            vocab_size=vocab_size,
            dropout=bilstm_dropout,
        )

        # Stage 3: NER (not part of the main gradient graph — trained separately)
        self.ner = MedicalBERTNER(
            pretrained_model=pretrained_bert,
            num_labels=num_ner_labels,
            use_pretrained=use_pretrained,
        )

        self.vocab_size = vocab_size

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "medscript_model_initialized",
            total_params=total_params,
            encoder_params=self.encoder.num_parameters,
            decoder_params=sum(p.numel() for p in self.decoder.parameters()),
            ner_params=sum(p.numel() for p in self.ner.parameters()),
        )

    def forward_encoder_decoder(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through encoder + decoder only (for CTC training).

        Args:
            pixel_values: (B, 3, H, W) normalized images

        Returns:
            log_probs: (B, T, vocab_size) log probabilities
        """
        # Stage 1: Encode image
        features = self.encoder(pixel_values)  # (B, T, encoder_output_dim)

        # Stage 2: Decode to character probabilities
        log_probs = self.decoder(features)  # (B, T, vocab_size)

        return log_probs

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for training (encoder + decoder only)."""
        return self.forward_encoder_decoder(pixel_values)

    @torch.no_grad()
    def transcribe(
        self,
        pixel_values: torch.Tensor,
        idx_to_char: dict[int, str] | None = None,
        run_ner: bool = True,
    ) -> list[TranscriptionResult]:
        """
        Full inference pipeline: image → text → entities.

        Args:
            pixel_values: (B, 3, H, W) normalized images
            idx_to_char: Character index to character mapping
            run_ner: Whether to run NER entity extraction

        Returns:
            List of TranscriptionResult objects (one per image)
        """
        self.eval()

        # Stage 1 + 2: Image → log probs → decoded text
        log_probs = self.forward_encoder_decoder(pixel_values)
        decoded_indices = self.decoder.greedy_decode(log_probs)
        confidence_scores = self.decoder.get_confidence_scores(log_probs)

        results = []
        for batch_idx, (indices, confidences) in enumerate(
            zip(decoded_indices, confidence_scores)
        ):
            # Decode indices to text
            if idx_to_char:
                text = "".join(
                    idx_to_char.get(idx, "") for idx in indices
                )
            else:
                text = str(indices)  # Fallback — raw indices

            result = TranscriptionResult(
                transcription=text,
                word_confidences=confidences,
            )

            # Stage 3: NER entity extraction
            if run_ner and text.strip():
                try:
                    entities = self.ner.predict(text)
                    result.entities = entities
                except Exception as e:
                    logger.warning(
                        "ner_prediction_failed",
                        batch_idx=batch_idx,
                        error=str(e),
                    )

            results.append(result)

        return results

    def set_training_stage(self, stage: str) -> None:
        """
        Configure model for different training stages.

        Stages:
        - "encoder_decoder": Train encoder + decoder with CTC loss (freeze NER)
        - "decoder_only": Freeze encoder, train decoder only
        - "ner_only": Freeze encoder + decoder, train NER only
        - "full": Train all components
        """
        if stage == "encoder_decoder":
            self.encoder.unfreeze()
            for p in self.decoder.parameters():
                p.requires_grad = True
            for p in self.ner.parameters():
                p.requires_grad = False

        elif stage == "decoder_only":
            self.encoder.freeze()
            for p in self.decoder.parameters():
                p.requires_grad = True
            for p in self.ner.parameters():
                p.requires_grad = False

        elif stage == "ner_only":
            self.encoder.freeze()
            for p in self.decoder.parameters():
                p.requires_grad = False
            for p in self.ner.parameters():
                p.requires_grad = True

        elif stage == "full":
            self.encoder.unfreeze()
            for p in self.decoder.parameters():
                p.requires_grad = True
            for p in self.ner.parameters():
                p.requires_grad = True

        else:
            raise ValueError(f"Unknown training stage: {stage}")

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("training_stage_set", stage=stage, trainable_params=trainable)
