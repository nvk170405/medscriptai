"""Training entry point — CLI for launching MedScript training."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from medscript.data.datamodule import MedScriptDataModule
from medscript.training.lightning_module import MedScriptLightningModule
from medscript.utils.config import get_model_config, get_training_config
from medscript.utils.logging import setup_logging, get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MedScript AI model")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--model-config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--data-dirs", nargs="+", default=["data/synthetic"])
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--gpus", type=int, default=0)
    args = parser.parse_args()

    setup_logging(level="INFO")
    logger = get_logger("train")

    # Load configs
    train_cfg = get_training_config(args.config)
    model_cfg = get_model_config(args.model_config)

    logger.info("training_config", experiment=train_cfg.experiment_name)
    logger.info("model_config", donut=model_cfg.donut.pretrained_model)

    # Seed
    pl.seed_everything(train_cfg.seed, workers=True)

    # Data
    data_module = MedScriptDataModule(
        data_dirs=args.data_dirs,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        image_height=model_cfg.donut.input_size[0],
        image_width=model_cfg.donut.input_size[1],
        train_split=train_cfg.train_split,
        val_split=train_cfg.val_split,
        test_split=train_cfg.test_split,
        curriculum_enabled=train_cfg.curriculum_enabled,
        seed=train_cfg.seed,
    )
    data_module.setup()

    # Build idx_to_char from dataset
    idx_to_char = data_module.train_dataset.idx_to_char if data_module.train_dataset else {}

    # Model
    model = MedScriptLightningModule(
        pretrained_donut=model_cfg.donut.pretrained_model,
        encoder_output_dim=model_cfg.bilstm.input_dim,
        bilstm_hidden_size=model_cfg.bilstm.hidden_size,
        bilstm_num_layers=model_cfg.bilstm.num_layers,
        bilstm_dropout=model_cfg.bilstm.dropout,
        vocab_size=model_cfg.vocabulary.vocab_size,
        learning_rate=train_cfg.learning_rate,
        bilstm_learning_rate=train_cfg.bilstm_learning_rate,
        weight_decay=train_cfg.weight_decay,
        warmup_steps=train_cfg.warmup_steps,
        max_epochs=train_cfg.max_epochs,
        idx_to_char=idx_to_char,
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=train_cfg.checkpoint.dirpath,
            monitor=train_cfg.checkpoint.monitor,
            mode=train_cfg.checkpoint.mode,
            save_top_k=train_cfg.checkpoint.save_top_k,
            save_last=train_cfg.checkpoint.save_last,
            filename="medscript-{epoch:02d}-{val/wer:.4f}",
        ),
        EarlyStopping(
            monitor=train_cfg.early_stopping.monitor,
            patience=train_cfg.early_stopping.patience,
            mode=train_cfg.early_stopping.mode,
            min_delta=train_cfg.early_stopping.min_delta,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=train_cfg.max_epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=max(1, args.gpus),
        precision=train_cfg.precision,
        gradient_clip_val=train_cfg.gradient_clip_val,
        accumulate_grad_batches=train_cfg.gradient_accumulation_steps,
        callbacks=callbacks,
        log_every_n_steps=train_cfg.log_every_n_steps,
        val_check_interval=train_cfg.val_check_interval,
        deterministic=train_cfg.deterministic,
    )

    logger.info("starting_training", max_epochs=train_cfg.max_epochs)
    trainer.fit(model, data_module, ckpt_path=args.resume)
    logger.info("training_complete")

    # Test
    if data_module.test_dataset:
        trainer.test(model, data_module)


if __name__ == "__main__":
    main()
