"""
Model evaluation script — loads a registered model and runs on val/test data.

Usage:
    python training/evaluate.py --config configs/train.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate skin severity classifier")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .ckpt file")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg_path = "configs/data.yaml"
    if Path(data_cfg_path).exists():
        data_cfg = load_config(data_cfg_path)
        cfg = {**data_cfg, **cfg}

    try:
        import pytorch_lightning as pl
    except ImportError:
        import lightning as pl  # type: ignore[no-redef]

    import mlflow

    from training.datamodule import SkinDataModule
    from training.model import SkinSeverityClassifier

    pl.seed_everything(cfg.get("seed", 42))

    # Determine checkpoint path
    ckpt_path = args.checkpoint or cfg.get("model_local_path", "artifacts/model.ckpt")

    if not Path(ckpt_path).exists():
        # Try to load from MLflow
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        model_name = cfg.get("mlflow", {}).get("model_name", "skin-severity-classifier")
        try:
            mlflow.set_tracking_uri(tracking_uri)
            model = mlflow.pytorch.load_model(f"models:/{model_name}/Production")
            print(f"✓ Loaded model from MLflow registry: {model_name}")
        except Exception as e:
            print(f"✗ Could not load model from MLflow: {e}")
            print(f"✗ Checkpoint not found at: {ckpt_path}")
            sys.exit(1)
    else:
        model = SkinSeverityClassifier.load_from_checkpoint(ckpt_path)
        print(f"✓ Loaded checkpoint: {ckpt_path}")

    datamodule = SkinDataModule(cfg)
    datamodule.setup()

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        logger=False,
    )

    print(f"\n📊 Evaluating on {args.split} split...")

    if args.split == "test":
        results = trainer.test(model, dataloaders=datamodule.test_dataloader())
    else:
        results = trainer.validate(model, dataloaders=datamodule.val_dataloader())

    metrics = results[0] if results else {}
    print("\n── Evaluation Results ──")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save evaluation metrics
    artifact_dir = Path(cfg.get("checkpoint_dir", "artifacts"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    eval_path = artifact_dir / "eval_metrics.json"
    with open(eval_path, "w") as f:
        json.dump({k: float(v) if hasattr(v, "item") else v for k, v in metrics.items()}, f, indent=2)
    print(f"\n✓ Evaluation metrics saved: {eval_path}")

    # Log to MLflow if available
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    try:
        mlflow.set_tracking_uri(tracking_uri)
        experiment_name = cfg.get("experiment_name", "skin-severity-experiment")
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="evaluation"):
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))
            mlflow.log_artifact(str(eval_path))
        print("✓ Evaluation metrics logged to MLflow")
    except Exception as e:
        print(f"⚠ Could not log to MLflow: {e}")


if __name__ == "__main__":
    main()
