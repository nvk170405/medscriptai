"""Configuration loader — reads YAML config files into typed dataclasses."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ── Config Root ──────────────────────────────────────────────────────────────

CONFIG_DIR = Path(__file__).resolve().parents[4] / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ── Model Config ─────────────────────────────────────────────────────────────


@dataclass
class DonutConfig:
    pretrained_model: str = "naver-clova-ix/donut-base"
    input_size: list[int] = field(default_factory=lambda: [1280, 960])
    max_length: int = 768
    align_long_axis: bool = False
    encoder_layer: int = -1


@dataclass
class BiLSTMConfig:
    input_dim: int = 1024
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    output_projection: bool = True


@dataclass
class CTCConfig:
    blank_token_id: int = 0
    beam_width: int = 10
    use_word_beam_search: bool = True


@dataclass
class MedicalBERTConfig:
    pretrained_model: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    max_seq_length: int = 256
    num_labels: int = 11
    label_map: dict[str, int] = field(default_factory=lambda: {
        "O": 0,
        "B-MEDICINE": 1, "I-MEDICINE": 2,
        "B-DOSAGE": 3, "I-DOSAGE": 4,
        "B-FREQUENCY": 5, "I-FREQUENCY": 6,
        "B-DURATION": 7, "I-DURATION": 8,
        "B-INSTRUCTION": 9, "I-INSTRUCTION": 10,
    })


@dataclass
class VocabularyConfig:
    charset: str = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789 .,;:!?-/()'\"@#%&+=[]{}|\\<>$^*~`_"
    )
    vocab_size: int = 95
    blank_token: str = "<blank>"
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"


@dataclass
class ModelConfig:
    donut: DonutConfig = field(default_factory=DonutConfig)
    bilstm: BiLSTMConfig = field(default_factory=BiLSTMConfig)
    ctc: CTCConfig = field(default_factory=CTCConfig)
    medical_bert: MedicalBERTConfig = field(default_factory=MedicalBERTConfig)
    vocabulary: VocabularyConfig = field(default_factory=VocabularyConfig)

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> ModelConfig:
        """Load model config from YAML file."""
        if path is None:
            path = CONFIG_DIR / "model_config.yaml"
        data = load_yaml(path)
        return cls(
            donut=DonutConfig(**data.get("donut", {})),
            bilstm=BiLSTMConfig(**data.get("bilstm", {})),
            ctc=CTCConfig(**data.get("ctc", {})),
            medical_bert=MedicalBERTConfig(**data.get("medical_bert", {})),
            vocabulary=VocabularyConfig(**data.get("vocabulary", {})),
        )


# ── Training Config ──────────────────────────────────────────────────────────


@dataclass
class EarlyStoppingConfig:
    monitor: str = "val/wer"
    patience: int = 5
    mode: str = "min"
    min_delta: float = 0.001


@dataclass
class CheckpointConfig:
    monitor: str = "val/wer"
    mode: str = "min"
    save_top_k: int = 3
    save_last: bool = True
    dirpath: str = "checkpoints/"


@dataclass
class CurriculumStage:
    name: str = "easy"
    epochs: list[int] = field(default_factory=lambda: [0, 10])
    augmentation_level: str = "light"


@dataclass
class TrainingConfig:
    experiment_name: str = "medscript-v1"
    seed: int = 42
    deterministic: bool = True

    # Data
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True

    # Training
    batch_size: int = 2
    max_epochs: int = 50
    gradient_accumulation_steps: int = 8
    gradient_clip_val: float = 5.0

    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 3e-5
    bilstm_learning_rate: float = 1e-4
    weight_decay: float = 0.01

    # Scheduler
    scheduler: str = "cosine_warmup"
    warmup_steps: int = 500
    min_learning_rate: float = 1e-6

    # Precision
    precision: str = "16-mixed"

    # Early stopping & checkpointing
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    # Curriculum
    curriculum_enabled: bool = True
    curriculum_stages: list[CurriculumStage] = field(default_factory=lambda: [
        CurriculumStage("easy", [0, 10], "light"),
        CurriculumStage("medium", [10, 30], "medium"),
        CurriculumStage("hard", [30, 50], "heavy"),
    ])

    # NER training
    ner_batch_size: int = 16
    ner_max_epochs: int = 20
    ner_learning_rate: float = 2e-5

    # Logging
    log_every_n_steps: int = 10
    val_check_interval: float = 0.5

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> TrainingConfig:
        """Load training config from YAML file."""
        if path is None:
            path = CONFIG_DIR / "training_config.yaml"
        data = load_yaml(path)
        training = data.get("training", {})
        ner = data.get("ner_training", {})
        data_cfg = data.get("data", {})
        curriculum = data.get("curriculum", {})
        logging_cfg = data.get("logging", {})

        return cls(
            experiment_name=data.get("experiment_name", "medscript-v1"),
            seed=data.get("seed", 42),
            deterministic=data.get("deterministic", True),
            train_split=data_cfg.get("train_split", 0.8),
            val_split=data_cfg.get("val_split", 0.1),
            test_split=data_cfg.get("test_split", 0.1),
            num_workers=data_cfg.get("num_workers", 4),
            pin_memory=data_cfg.get("pin_memory", True),
            batch_size=training.get("batch_size", 2),
            max_epochs=training.get("max_epochs", 50),
            gradient_accumulation_steps=training.get("gradient_accumulation_steps", 8),
            gradient_clip_val=training.get("gradient_clip_val", 5.0),
            optimizer=training.get("optimizer", "adamw"),
            learning_rate=training.get("learning_rate", 3e-5),
            bilstm_learning_rate=training.get("bilstm_learning_rate", 1e-4),
            weight_decay=training.get("weight_decay", 0.01),
            scheduler=training.get("scheduler", "cosine_warmup"),
            warmup_steps=training.get("warmup_steps", 500),
            min_learning_rate=training.get("min_learning_rate", 1e-6),
            precision=training.get("precision", "16-mixed"),
            early_stopping=EarlyStoppingConfig(
                **training.get("early_stopping", {})
            ),
            checkpoint=CheckpointConfig(**training.get("checkpoint", {})),
            curriculum_enabled=curriculum.get("enabled", True),
            ner_batch_size=ner.get("batch_size", 16),
            ner_max_epochs=ner.get("max_epochs", 20),
            ner_learning_rate=ner.get("learning_rate", 2e-5),
            log_every_n_steps=logging_cfg.get("log_every_n_steps", 10),
            val_check_interval=logging_cfg.get("val_check_interval", 0.5),
        )


# ── Convenience Loaders ──────────────────────────────────────────────────────


def get_model_config(path: str | Path | None = None) -> ModelConfig:
    """Get model configuration, using environment override if set."""
    env_path = os.environ.get("MEDSCRIPT_MODEL_CONFIG")
    return ModelConfig.from_yaml(env_path or path)


def get_training_config(path: str | Path | None = None) -> TrainingConfig:
    """Get training configuration, using environment override if set."""
    env_path = os.environ.get("MEDSCRIPT_TRAINING_CONFIG")
    return TrainingConfig.from_yaml(env_path or path)
