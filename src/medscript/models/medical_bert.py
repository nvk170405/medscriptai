"""Medical BERT NER — Named Entity Recognition for medical prescriptions."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from medscript.utils.logging import get_logger

logger = get_logger(__name__)

# Default BIO label map
DEFAULT_LABEL_MAP = {
    "O": 0,
    "B-MEDICINE": 1, "I-MEDICINE": 2,
    "B-DOSAGE": 3, "I-DOSAGE": 4,
    "B-FREQUENCY": 5, "I-FREQUENCY": 6,
    "B-DURATION": 7, "I-DURATION": 8,
    "B-INSTRUCTION": 9, "I-INSTRUCTION": 10,
}

ID_TO_LABEL = {v: k for k, v in DEFAULT_LABEL_MAP.items()}


class MedicalBERTNER(nn.Module):
    """
    Medical BERT-based Named Entity Recognition model.

    Extracts structured entities (medicine, dosage, frequency, duration,
    instruction) from transcribed prescription text using BIO tagging.

    Architecture:
        Text input
        → BiomedBERT tokenizer
        → BiomedBERT encoder
        → Token classification head
        → BIO tags per token
        → Entity extraction
    """

    def __init__(
        self,
        pretrained_model: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        num_labels: int = 11,
        label_map: dict[str, int] | None = None,
        dropout: float = 0.1,
        use_pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.num_labels = num_labels
        self.label_map = label_map or DEFAULT_LABEL_MAP
        self.id_to_label = {v: k for k, v in self.label_map.items()}

        # Load BERT encoder
        if use_pretrained:
            logger.info("loading_medical_bert", model=pretrained_model)
            self.encoder = AutoModel.from_pretrained(pretrained_model)
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        else:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(pretrained_model)
            self.encoder = AutoModel.from_config(config)
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

        hidden_size = self.encoder.config.hidden_size

        # Token classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )

        logger.info(
            "medical_bert_initialized",
            hidden_size=hidden_size,
            num_labels=num_labels,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: (B, L) tokenized input
            attention_mask: (B, L) attention mask
            labels: (B, L) ground truth BIO labels (optional, for training)

        Returns:
            Dict with 'logits', and optionally 'loss' if labels provided
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Token-level representations
        sequence_output = outputs.last_hidden_state  # (B, L, hidden_size)

        # Classify each token
        logits = self.classifier(sequence_output)  # (B, L, num_labels)

        result: dict[str, torch.Tensor] = {"logits": logits}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            result["loss"] = loss

        return result

    def predict(self, text: str) -> list[dict[str, Any]]:
        """
        Run NER prediction on a text string.

        Args:
            text: Transcribed prescription text

        Returns:
            List of extracted entities with type, value, and confidence
        """
        self.eval()

        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            return_offsets_mapping=True,
        )

        offset_mapping = encoding.pop("offset_mapping")[0]
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Move to same device as model
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Predict
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)

        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)[0]  # (L,)
        confidences = probs.max(dim=-1).values[0]  # (L,)

        # Extract entities from BIO tags
        entities = self._extract_entities(
            text=text,
            predictions=predictions.cpu().tolist(),
            confidences=confidences.cpu().tolist(),
            offset_mapping=offset_mapping.cpu().tolist(),
        )

        return entities

    def _extract_entities(
        self,
        text: str,
        predictions: list[int],
        confidences: list[float],
        offset_mapping: list[list[int]],
    ) -> list[dict[str, Any]]:
        """Extract entities from BIO tag predictions."""
        entities: list[dict[str, Any]] = []
        current_entity: dict[str, Any] | None = None

        for idx, (pred, conf, offsets) in enumerate(
            zip(predictions, confidences, offset_mapping)
        ):
            if offsets == [0, 0]:  # Special tokens
                continue

            label = self.id_to_label.get(pred, "O")

            if label.startswith("B-"):
                # Save previous entity
                if current_entity:
                    entities.append(current_entity)

                entity_type = label[2:]  # Remove "B-" prefix
                start, end = offsets
                current_entity = {
                    "type": entity_type.lower(),
                    "value": text[start:end],
                    "start": start,
                    "end": end,
                    "confidence": conf,
                    "token_confidences": [conf],
                }

            elif label.startswith("I-") and current_entity:
                entity_type = label[2:]
                if entity_type.lower() == current_entity["type"]:
                    # Extend current entity
                    start, end = offsets
                    current_entity["end"] = end
                    current_entity["value"] = text[current_entity["start"]:end]
                    current_entity["token_confidences"].append(conf)
                    # Average confidence across tokens
                    current_entity["confidence"] = sum(
                        current_entity["token_confidences"]
                    ) / len(current_entity["token_confidences"])

            else:
                # O label — end current entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        # Don't forget the last entity
        if current_entity:
            entities.append(current_entity)

        # Clean up — remove internal tracking fields
        for entity in entities:
            entity.pop("token_confidences", None)
            entity.pop("start", None)
            entity.pop("end", None)

        return entities
