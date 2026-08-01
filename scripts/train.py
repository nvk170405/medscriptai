#!/usr/bin/env python3
"""MedScript AI — Training CLI.

Usage:
    python scripts/train.py                          # Default config
    python scripts/train.py --epochs 20 --batch-size 4
    python scripts/train.py --stage decoder_only     # Freeze encoder
    python scripts/train.py --generate-data 5000     # Generate synthetic data first
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medscript.data.datamodule import MedScriptDataModule
from medscript.training.lightning_module import MedScriptLightningModule
from medscript.utils.logging import get_logger

logger = get_logger(__name__)


def load_configs() -> tuple[dict, dict]:
    """Load model and training configs."""
    with open(PROJECT_ROOT / "configs" / "model_config.yaml") as f:
        model_config = yaml.safe_load(f)
    with open(PROJECT_ROOT / "configs" / "training_config.yaml") as f:
        training_config = yaml.safe_load(f)
    return model_config, training_config


def build_vocab(model_config: dict) -> tuple[dict, dict, int]:
    """Build character vocabulary from config."""
    charset = model_config["vocabulary"]["charset"]
    char_to_idx = {char: idx + 1 for idx, char in enumerate(charset)}
    idx_to_char = {idx + 1: char for idx, char in enumerate(charset)}
    idx_to_char[0] = ""  # CTC blank
    vocab_size = len(charset) + 1
    return char_to_idx, idx_to_char, vocab_size


def generate_synthetic_data(num_samples: int, output_dir: Path) -> None:
    """Generate synthetic prescription images."""
    from medscript.data.synthetic import SyntheticPrescriptionGenerator

    generator = SyntheticPrescriptionGenerator(
        output_dir=str(output_dir),
        image_height=960,
        image_width=1280,
    )
    logger.info("generating_synthetic_data", num_samples=num_samples)
    generator.generate_dataset(num_samples=num_samples, seed=42)
    logger.info("synthetic_data_generated", output_dir=str(output_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MedScript AI models")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--stage", choices=["encoder_decoder", "decoder_only", "ner_only", "full"], default="encoder_decoder")
    parser.add_argument("--generate-data", type=int, default=0, help="Generate N synthetic samples before training")
    parser.add_argument("--data-dir", type=str, default="data/synthetic", help="Training data directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--precision", type=str, default=None, help="Override precision (16-mixed, 32, bf16-mixed)")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Load configs
    model_config, training_config = load_configs()

    # Apply CLI overrides
    epochs = args.epochs or training_config["training"]["max_epochs"]
    batch_size = args.batch_size or training_config["training"]["batch_size"]
    lr = args.lr or training_config["training"]["learning_rate"]
    precision = args.precision or training_config["training"]["precision"]
    grad_accum = training_config["training"]["gradient_accumulation_steps"]

    # Seed
    pl.seed_everything(args.seed, workers=True)

    # Generate synthetic data if requested
    data_dir = PROJECT_ROOT / args.data_dir
    if args.generate_data > 0:
        generate_synthetic_data(args.generate_data, data_dir)

    # Build vocabulary
    char_to_idx, idx_to_char, vocab_size = build_vocab(model_config)

    # Save vocab for later use
    ckpt_dir = PROJECT_ROOT / args.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "vocab.json", "w") as f:
        json.dump({
            "char_to_idx": char_to_idx,
            "idx_to_char": {str(k): v for k, v in idx_to_char.items()},
            "vocab_size": vocab_size,
        }, f, indent=2)

    # DataModule
    datamodule = MedScriptDataModule(
        data_dirs=[str(data_dir)],
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=True,
        image_height=960,
        image_width=1280,
        max_text_length=model_config["donut"]["max_length"],
        train_split=0.85,
        val_split=0.15,
        test_split=0.0,
        augmentation_level="medium",
        curriculum_enabled=True,
        seed=args.seed,
    )

    # Model
    model = MedScriptLightningModule(
        pretrained_donut=model_config["donut"]["pretrained_model"],
        encoder_output_dim=model_config["bilstm"]["input_dim"],
        bilstm_hidden_size=model_config["bilstm"]["hidden_size"],
        bilstm_num_layers=model_config["bilstm"]["num_layers"],
        bilstm_dropout=model_config["bilstm"]["dropout"],
        vocab_size=vocab_size,
        freeze_encoder=(args.stage == "decoder_only"),
        learning_rate=lr,
        bilstm_learning_rate=training_config["training"]["bilstm_learning_rate"],
        weight_decay=training_config["training"]["weight_decay"],
        warmup_steps=training_config["training"]["warmup_steps"],
        max_epochs=epochs,
        idx_to_char=idx_to_char,
        use_pretrained=True,
    )

    # Set training stage
    model.model.set_training_stage(args.stage)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="medscript-{epoch:02d}-{val_wer:.4f}",
            monitor="val/wer",
            mode="min",
            save_top_k=3,
            save_last=True,
            verbose=True,
        ),
        EarlyStopping(
            monitor="val/wer",
            patience=5,
            mode="min",
            min_delta=0.001,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Determine accelerator
    if torch.cuda.is_available():
        accelerator = "gpu"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        accelerator = "mps"
    else:
        accelerator = "cpu"
        logger.warning("no_gpu_detected", message="Training on CPU will be very slow!")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=min(args.gpus, torch.cuda.device_count()) if accelerator == "gpu" else 1,
        precision=precision if accelerator != "cpu" else "32-true",
        gradient_clip_val=training_config["training"]["gradient_clip_val"],
        accumulate_grad_batches=grad_accum,
        val_check_interval=0.5,
        log_every_n_steps=10,
        callbacks=callbacks,
        deterministic=True,
    )

    # Print training info
    print("\n" + "=" * 60)
    print("  MedScript AI — Training")
    print("=" * 60)
    print(f"  Stage:       {args.stage}")
    print(f"  Accelerator: {accelerator}")
    print(f"  Precision:   {precision}")
    print(f"  Batch:       {batch_size} x {grad_accum} accum = {batch_size * grad_accum} effective")
    print(f"  Epochs:      {epochs}")
    print(f"  LR (enc):    {lr}")
    print(f"  LR (dec):    {training_config['training']['bilstm_learning_rate']}")
    print(f"  Data:        {data_dir}")
    print(f"  Checkpoints: {ckpt_dir}")
    print("=" * 60 + "\n")

    # Train
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)

    print(f"\n✅ Training complete!")
    print(f"   Best checkpoint: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
