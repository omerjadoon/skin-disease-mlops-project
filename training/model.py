"""
PyTorch Lightning model for skin disease severity classification.

DISCLAIMER: For educational/research purposes only. Not a medical diagnosis tool.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning as pl  # type: ignore[no-redef]


SUPPORTED_BACKBONES = {
    "resnet18": (tv_models.resnet18, tv_models.ResNet18_Weights.DEFAULT),
    "resnet50": (tv_models.resnet50, tv_models.ResNet50_Weights.DEFAULT),
    "efficientnet_b0": (tv_models.efficientnet_b0, tv_models.EfficientNet_B0_Weights.DEFAULT),
    "efficientnet_b2": (tv_models.efficientnet_b2, tv_models.EfficientNet_B2_Weights.DEFAULT),
}


def build_backbone(backbone_name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Build a backbone model and return (model_without_head, feature_dim)."""
    if backbone_name not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone '{backbone_name}'. "
            f"Choose from: {list(SUPPORTED_BACKBONES.keys())}"
        )

    model_fn, weights = SUPPORTED_BACKBONES[backbone_name]
    model = model_fn(weights=weights if pretrained else None)

    # Extract feature dimension and remove head
    if backbone_name.startswith("resnet"):
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()  # type: ignore[assignment]
    elif backbone_name.startswith("efficientnet"):
        feature_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()  # type: ignore[assignment]
    else:
        raise ValueError(f"Unknown backbone architecture: {backbone_name}")

    return model, feature_dim


class SkinSeverityClassifier(pl.LightningModule):
    """
    PyTorch Lightning module for skin disease severity classification.

    Classifies images into: mild | moderate | severe

    DISCLAIMER: This model provides AI-assisted severity estimates only.
    It is NOT a medical diagnosis tool.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer: str = "adam",
        scheduler: str = "cosine",
        class_names: list[str] | None = None,
        freeze_backbone: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.class_names = class_names or ["mild", "moderate", "severe"]
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler

        # Build backbone + classifier head
        self.backbone, feature_dim = build_backbone(backbone, pretrained)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        metric_kwargs = {"num_classes": num_classes, "average": "macro"}
        self.train_metrics = MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=num_classes),
                "f1_macro": MulticlassF1Score(**metric_kwargs),
            },
            prefix="train/",
        )
        self.val_metrics = MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=num_classes),
                "f1_macro": MulticlassF1Score(**metric_kwargs),
                "precision": MulticlassPrecision(**metric_kwargs),
                "recall": MulticlassRecall(**metric_kwargs),
                "auroc": MulticlassAUROC(num_classes=num_classes),
            },
            prefix="val/",
        )
        self.test_metrics = MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=num_classes),
                "f1_macro": MulticlassF1Score(**metric_kwargs),
                "precision": MulticlassPrecision(**metric_kwargs),
                "recall": MulticlassRecall(**metric_kwargs),
                "auroc": MulticlassAUROC(num_classes=num_classes),
            },
            prefix="test/",
        )
        self.val_confusion = MulticlassConfusionMatrix(num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def _shared_step(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = torch.softmax(logits, dim=1)
        return loss, preds, labels

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss, preds, labels = self._shared_step(batch)
        metrics = self.train_metrics(preds, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        loss, preds, labels = self._shared_step(batch)
        self.val_metrics.update(preds, labels)
        self.val_confusion.update(preds.argmax(dim=1), labels)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, prog_bar=True)
        self.val_metrics.reset()
        self.val_confusion.reset()

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        loss, preds, labels = self._shared_step(batch)
        self.test_metrics.update(preds, labels)
        self.log("test/loss", loss, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        self.log_dict(metrics)
        self.test_metrics.reset()

    def configure_optimizers(self) -> Any:
        params = filter(lambda p: p.requires_grad, self.parameters())

        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                params, lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                params, lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                params, lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        if self.scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs if self.trainer else 20
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return optimizer

    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata for API /model-info endpoint."""
        return {
            "backbone": self.hparams.get("backbone", "resnet18"),
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "input_size": [224, 224],
        }
