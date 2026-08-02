"""Donut Vision Encoder — wraps naver-clova-ix/donut-base for feature extraction."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import DonutSwinModel, DonutSwinConfig

from medscript.utils.logging import get_logger

logger = get_logger(__name__)


class DonutVisionEncoder(nn.Module):
    """
    Vision encoder based on Donut's Swin Transformer.

    Extracts spatial features from prescription images without explicit OCR
    preprocessing. The Swin Transformer processes the image as patches and
    outputs feature maps that are reshaped for sequential processing by
    the downstream BiLSTM-CTC decoder.

    Architecture:
        Input image (B, 3, H, W)
        → Swin Transformer encoder
        → Feature maps (B, num_patches, hidden_dim)
        → Projection layer
        → Sequential features (B, T, output_dim) for BiLSTM
    """

    def __init__(
        self,
        pretrained_model: str = "naver-clova-ix/donut-base",
        output_dim: int = 512,
        freeze_encoder: bool = False,
        use_pretrained: bool = True,
    ) -> None:
        """
        Args:
            pretrained_model: HuggingFace model identifier
            output_dim: Output feature dimension for downstream BiLSTM
            freeze_encoder: If True, freeze encoder weights (for staged training)
            use_pretrained: If True, load pretrained weights
        """
        super().__init__()

        self.output_dim = output_dim

        # Load Swin Transformer encoder
        if use_pretrained:
            logger.info("loading_pretrained_donut", model=pretrained_model)
            self.encoder = DonutSwinModel.from_pretrained(pretrained_model)
        else:
            config = DonutSwinConfig.from_pretrained(pretrained_model)
            self.encoder = DonutSwinModel(config)

        # Get hidden dimension from encoder config
        self.hidden_dim = self.encoder.config.hidden_size  # Typically 768 or 1024

        # Enable gradient checkpointing to save massive amounts of VRAM!
        # This is critical for training at 960x1280 on GPUs with < 24GB VRAM
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
            logger.info("donut_gradient_checkpointing_enabled")

        # Projection layer: map encoder features to output_dim
        self.projection = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Optionally freeze encoder
        if freeze_encoder:
            self.freeze()

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "donut_encoder_initialized",
            hidden_dim=self.hidden_dim,
            output_dim=output_dim,
            total_params=total_params,
            trainable_params=trainable_params,
        )

    def freeze(self) -> None:
        """Freeze encoder weights for staged training."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("encoder_frozen")

    def unfreeze(self) -> None:
        """Unfreeze encoder weights for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("encoder_unfrozen")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract features from prescription images.

        Args:
            pixel_values: (B, 3, H, W) normalized image tensor

        Returns:
            features: (B, T, output_dim) sequential features
                      where T = number of spatial patches
        """
        # Swin encoder output
        encoder_output = self.encoder(pixel_values=pixel_values)

        # last_hidden_state shape: (B, num_patches, hidden_dim)
        features = encoder_output.last_hidden_state

        # Project to output dimension
        features = self.projection(features)

        return features

    @property
    def num_parameters(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
