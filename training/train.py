"""
Main training script for skin severity classification.

Usage:
    python training/train.py --config configs/train.yaml

DISCLAIMER: For educational/research purposes only. Not a medical diagnosis tool.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def merge_configs(*configs: dict) -> dict:
    """Merge multiple config dicts (later configs override earlier ones)."""
    merged: dict = {}
    for cfg in configs:
        merged.update(cfg)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Train skin severity classifier")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--model-config", type=str, default="configs/model.yaml")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    args = parser.parse_args()

    # Load configs
    train_cfg = load_config(args.config)
    model_cfg = load_config(args.model_config) if Path(args.model_config).exists() else {}
    data_cfg = load_config(args.data_config) if Path(args.data_config).exists() else {}
    cfg = merge_configs(data_cfg, model_cfg, train_cfg)

    # CLI overrides
    if args.epochs is not None:
        cfg["max_epochs"] = args.epochs
    if args.lr is not None:
        cfg["learning_rate"] = args.lr
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    print("=" * 60)
    print("  Skin Severity MLOps — Training")
    print("  DISCLAIMER: Educational/research use only.")
    print("=" * 60)
    print(f"  Backbone:    {cfg.get('backbone', 'resnet18')}")
    print(f"  Epochs:      {cfg.get('max_epochs', 20)}")
    print(f"  Batch size:  {cfg.get('batch_size', 16)}")
    print(f"  LR:          {cfg.get('learning_rate', 1e-3)}")
    print(f"  Classes:     {cfg.get('class_names', ['mild', 'moderate', 'severe'])}")
    print("=" * 60)

    # Import here to avoid slow startup for --help
    import torch

    try:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import (
            EarlyStopping,
            LearningRateMonitor,
            ModelCheckpoint,
        )
        from pytorch_lightning.loggers import MLFlowLogger
    except ImportError:
        import lightning as pl  # type: ignore[no-redef]
        from lightning.pytorch.callbacks import (  # type: ignore[no-redef]
            EarlyStopping,
            LearningRateMonitor,
            ModelCheckpoint,
        )
        from lightning.pytorch.loggers import MLFlowLogger  # type: ignore[no-redef]

    import mlflow

    from training.datamodule import SkinDataModule
    from training.model import SkinSeverityClassifier

    # Reproducibility
    pl.seed_everything(cfg.get("seed", 42), workers=True)

    # DataModule
    datamodule = SkinDataModule(cfg)

    # Model
    model = SkinSeverityClassifier(
        backbone=cfg.get("backbone", "resnet18"),
        num_classes=cfg.get("num_classes", 3),
        pretrained=cfg.get("pretrained", True),
        dropout=cfg.get("dropout", 0.3),
        learning_rate=cfg.get("learning_rate", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-4),
        optimizer=cfg.get("optimizer", "adam"),
        scheduler=cfg.get("scheduler", "cosine"),
        class_names=cfg.get("class_names", ["mild", "moderate", "severe"]),
        freeze_backbone=cfg.get("freeze_backbone", False),
    )

    # Artifact directory
    artifact_dir = Path(cfg.get("checkpoint_dir", "artifacts"))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(artifact_dir),
        filename="model-{epoch:02d}-{val/f1_macro:.3f}",
        monitor=cfg.get("monitor_metric", "val/f1_macro"),
        mode=cfg.get("monitor_mode", "max"),
        save_top_k=cfg.get("save_top_k", 1),
        save_last=True,
        verbose=True,
    )
    early_stopping = EarlyStopping(
        monitor=cfg.get("monitor_metric", "val/f1_macro"),
        mode=cfg.get("monitor_mode", "max"),
        patience=cfg.get("patience", 5),
        min_delta=cfg.get("min_delta", 0.001),
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks = [checkpoint_callback, early_stopping, lr_monitor]

    # MLflow setup
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        mlflow_cfg.get("tracking_uri", "http://localhost:5000"),
    )

    logger = None
    run_id = None

    if not args.no_mlflow:
        try:
            mlflow.set_tracking_uri(tracking_uri)
            experiment_name = cfg.get("experiment_name", "skin-severity-experiment")
            mlflow.set_experiment(experiment_name)

            logger = MLFlowLogger(
                experiment_name=experiment_name,
                tracking_uri=tracking_uri,
                run_name=cfg.get("run_name", "resnet18-severity-v1"),
                log_model=True,
            )
            print(f"✓ MLflow connected: {tracking_uri}")
        except Exception as e:
            print(f"⚠ MLflow not available ({e}). Continuing without MLflow.")
            logger = None

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.get("max_epochs", 20),
        accelerator="auto",
        devices=1,
        precision=cfg.get("precision", "32"),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.get("log_every_n_steps", 5),
        deterministic=cfg.get("deterministic", True),
        enable_progress_bar=True,
    )

    # Log hyperparams to MLflow
    if not args.no_mlflow and logger is not None:
        try:
            with mlflow.start_run(run_id=logger.run_id):
                mlflow.log_params(
                    {
                        "backbone": cfg.get("backbone", "resnet18"),
                        "num_classes": cfg.get("num_classes", 3),
                        "pretrained": cfg.get("pretrained", True),
                        "dropout": cfg.get("dropout", 0.3),
                        "learning_rate": cfg.get("learning_rate", 1e-3),
                        "weight_decay": cfg.get("weight_decay", 1e-4),
                        "max_epochs": cfg.get("max_epochs", 20),
                        "batch_size": cfg.get("batch_size", 16),
                        "optimizer": cfg.get("optimizer", "adam"),
                        "scheduler": cfg.get("scheduler", "cosine"),
                        "image_size": cfg.get("image_size", 224),
                    }
                )
                run_id = mlflow.active_run().info.run_id
        except Exception as e:
            print(f"⚠ Could not log params to MLflow: {e}")

    # Train
    print("\n🚀 Starting training...\n")
    trainer.fit(model, datamodule=datamodule)

    # Save best checkpoint path as symlink
    best_ckpt = checkpoint_callback.best_model_path
    if best_ckpt:
        best_link = artifact_dir / "model.ckpt"
        if best_link.exists() or best_link.is_symlink():
            best_link.unlink()
        import shutil
        shutil.copy2(best_ckpt, best_link)
        print(f"\n✓ Best checkpoint saved to: {best_link}")
    else:
        # Save last checkpoint
        last_ckpt = artifact_dir / "last.ckpt"
        if last_ckpt.exists():
            import shutil
            shutil.copy2(last_ckpt, artifact_dir / "model.ckpt")

    # Run test evaluation
    print("\n📊 Running test evaluation...")
    try:
        trainer.test(model, datamodule=datamodule)
    except Exception as e:
        print(f"⚠ Test evaluation failed: {e}")

    # Save metrics JSON
    if trainer.logged_metrics:
        metrics_out = {
            k: float(v) if hasattr(v, "item") else v
            for k, v in trainer.logged_metrics.items()
        }
        metrics_path = artifact_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_out, f, indent=2)
        print(f"✓ Metrics saved to: {metrics_path}")

    # Save run_id for DVC register stage
    if run_id:
        with open(artifact_dir / "run_id.txt", "w") as f:
            f.write(run_id)
        print(f"✓ MLflow run ID: {run_id}")

        # Register model in MLflow registry
        if mlflow_cfg.get("register_model", True):
            try:
                with mlflow.start_run(run_id=run_id):
                    model_name = mlflow_cfg.get("model_name", "skin-severity-classifier")
                    mlflow.pytorch.log_model(model, artifact_path="model")
                    registered = mlflow.register_model(
                        model_uri=f"runs:/{run_id}/model",
                        name=model_name,
                    )
                    print(f"✓ Model registered: {model_name} v{registered.version}")
            except Exception as e:
                print(f"⚠ Model registration failed: {e}")

    print("\n✅ Training complete!")
    print("   DISCLAIMER: This model provides AI-assisted estimates only.")
    print("   It is NOT a medical diagnosis tool.")


if __name__ == "__main__":
    main()
