"""PyTorch Lightning module for training MedScript models."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import pytorch_lightning as pl

from medscript.models.medscript_model import MedScriptModel
from medscript.training.metrics import word_error_rate, character_error_rate
from medscript.utils.logging import get_logger

logger = get_logger(__name__)


class MedScriptLightningModule(pl.LightningModule):
    """
    Lightning module for training the MedScript encoder-decoder pipeline.

    Handles:
    - CTC loss for handwriting recognition
    - AdamW optimizer with cosine annealing warmup
    - WER/CER metric logging
    - Gradient clipping
    - Curriculum learning epoch hooks
    """

    def __init__(
        self,
        # Model config
        pretrained_donut: str = "naver-clova-ix/donut-base",
        encoder_output_dim: int = 512,
        bilstm_hidden_size: int = 256,
        bilstm_num_layers: int = 2,
        bilstm_dropout: float = 0.3,
        vocab_size: int = 95,
        freeze_encoder: bool = False,
        # Training config
        learning_rate: float = 3e-5,
        bilstm_learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_epochs: int = 50,
        # Misc
        idx_to_char: dict[int, str] | None = None,
        use_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["idx_to_char"])

        self.model = MedScriptModel(
            pretrained_donut=pretrained_donut,
            encoder_output_dim=encoder_output_dim,
            bilstm_hidden_size=bilstm_hidden_size,
            bilstm_num_layers=bilstm_num_layers,
            bilstm_dropout=bilstm_dropout,
            vocab_size=vocab_size,
            freeze_encoder=freeze_encoder,
            use_pretrained=use_pretrained,
        )

        self.ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        self.idx_to_char = idx_to_char or {}
        self.learning_rate = learning_rate
        self.bilstm_learning_rate = bilstm_learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model.forward_encoder_decoder(pixel_values)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        images = batch["images"]
        targets = batch["targets"]
        target_lengths = batch["target_lengths"]

        # Forward pass
        log_probs = self.forward(images)  # (B, T, vocab_size)

        # CTC loss expects (T, B, C) format
        log_probs_ctc = log_probs.permute(1, 0, 2)  # (T, B, vocab_size)
        input_lengths = torch.full(
            (log_probs_ctc.size(1),),
            log_probs_ctc.size(0),
            dtype=torch.long,
            device=self.device,
        )

        loss = self.ctc_loss(log_probs_ctc, targets, input_lengths, target_lengths)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        images = batch["images"]
        targets = batch["targets"]
        target_lengths = batch["target_lengths"]
        texts = batch["texts"]

        # Forward pass
        log_probs = self.forward(images)

        # CTC loss
        log_probs_ctc = log_probs.permute(1, 0, 2)
        input_lengths = torch.full(
            (log_probs_ctc.size(1),),
            log_probs_ctc.size(0),
            dtype=torch.long,
            device=self.device,
        )
        loss = self.ctc_loss(log_probs_ctc, targets, input_lengths, target_lengths)

        # Decode predictions for metrics
        decoded_indices = self.model.decoder.greedy_decode(log_probs)
        predictions = []
        for indices in decoded_indices:
            text = "".join(self.idx_to_char.get(idx, "") for idx in indices)
            predictions.append(text)

        # Compute metrics
        wer = word_error_rate(predictions, texts)
        cer = character_error_rate(predictions, texts)

        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/wer", wer, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/cer", cer, prog_bar=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure AdamW with separate LR for encoder and decoder."""
        # Parameter groups with different learning rates
        encoder_params = list(self.model.encoder.parameters())
        decoder_params = list(self.model.decoder.parameters())

        param_groups = [
            {
                "params": encoder_params,
                "lr": self.learning_rate,
                "name": "encoder",
            },
            {
                "params": decoder_params,
                "lr": self.bilstm_learning_rate,
                "name": "decoder",
            },
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.weight_decay,
        )

        # Cosine annealing with warmup
        def lr_lambda(current_step: int) -> float:
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = float(current_step - self.warmup_steps) / float(
                max(1, self.trainer.estimated_stepping_batches - self.warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + __import__("math").cos(3.14159 * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
