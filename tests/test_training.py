"""
Tests for the training pipeline: dataset loading, transforms, model forward pass.
"""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_synthetic_dataset(root_dir: Path, images_per_class: int = 5) -> None:
    """Create a minimal synthetic dataset in folder structure."""
    for severity in ["mild", "moderate", "severe"]:
        class_dir = root_dir / severity
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(images_per_class):
            img = Image.new("RGB", (224, 224), color=(100 + i * 20, 80, 60))
            img.save(class_dir / f"{severity}_{i:04d}.jpg", "JPEG")


# ─── Transform Tests ──────────────────────────────────────────

class TestTransforms:
    def test_train_transforms_output_shape(self):
        from training.transforms import get_train_transforms
        transform = get_train_transforms()
        img = Image.new("RGB", (300, 300), color=(150, 100, 80))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_val_transforms_output_shape(self):
        from training.transforms import get_val_transforms
        transform = get_val_transforms()
        img = Image.new("RGB", (300, 300), color=(150, 100, 80))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_transforms_from_config(self):
        from training.transforms import build_transforms_from_config
        cfg = {
            "image_size": 224,
            "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "augmentations": {"random_horizontal_flip": True, "random_rotation_degrees": 10},
        }
        train_tf, val_tf = build_transforms_from_config(cfg)
        img = Image.new("RGB", (256, 256))
        t = train_tf(img)
        v = val_tf(img)
        assert t.shape == (3, 224, 224)
        assert v.shape == (3, 224, 224)

    def test_inference_transforms(self):
        from training.transforms import get_inference_transforms
        transform = get_inference_transforms()
        img = Image.new("RGB", (400, 300))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32


# ─── Dataset Tests ────────────────────────────────────────────

class TestFolderDataset:
    def test_loads_images_from_folder(self, tmp_path):
        create_synthetic_dataset(tmp_path, images_per_class=5)
        from training.datamodule import FolderSkinDataset
        ds = FolderSkinDataset(tmp_path, ["mild", "moderate", "severe"])
        assert len(ds) == 15

    def test_returns_correct_labels(self, tmp_path):
        create_synthetic_dataset(tmp_path, images_per_class=3)
        from training.datamodule import FolderSkinDataset
        ds = FolderSkinDataset(tmp_path, ["mild", "moderate", "severe"])
        labels = [ds[i][1] for i in range(len(ds))]
        assert set(labels) == {0, 1, 2}

    def test_handles_missing_class_dir(self, tmp_path):
        # Only create 'mild' folder
        (tmp_path / "mild").mkdir()
        img = Image.new("RGB", (224, 224))
        img.save(tmp_path / "mild" / "test.jpg")

        from training.datamodule import FolderSkinDataset
        ds = FolderSkinDataset(tmp_path, ["mild", "moderate", "severe"])
        assert len(ds) == 1  # Only mild images

    def test_with_transform(self, tmp_path):
        create_synthetic_dataset(tmp_path, images_per_class=2)
        from training.datamodule import FolderSkinDataset
        from training.transforms import get_val_transforms
        tf = get_val_transforms()
        ds = FolderSkinDataset(tmp_path, ["mild", "moderate", "severe"], transform=tf)
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 224, 224)


# ─── DataModule Tests ─────────────────────────────────────────

class TestSkinDataModule:
    def test_setup_folder_mode(self, tmp_path):
        create_synthetic_dataset(tmp_path, images_per_class=10)
        from training.datamodule import SkinDataModule
        cfg = {
            "raw_data_dir": str(tmp_path),
            "processed_data_dir": str(tmp_path / "processed"),
            "dataset_mode": "folder",
            "batch_size": 4,
            "num_workers": 0,
            "image_size": 224,
            "val_split": 0.2,
            "test_split": 0.1,
            "seed": 42,
        }
        dm = SkinDataModule(cfg)
        dm.setup()
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert dm.test_dataset is not None

    def test_dataloaders_return_batches(self, tmp_path):
        create_synthetic_dataset(tmp_path, images_per_class=10)
        from training.datamodule import SkinDataModule
        cfg = {
            "raw_data_dir": str(tmp_path),
            "processed_data_dir": str(tmp_path / "processed"),
            "dataset_mode": "folder",
            "batch_size": 4,
            "num_workers": 0,
            "image_size": 224,
            "val_split": 0.2,
            "test_split": 0.1,
            "seed": 42,
        }
        dm = SkinDataModule(cfg)
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        images, labels = batch
        assert images.shape[1:] == (3, 224, 224)
        assert labels.shape[0] == images.shape[0]


# ─── Model Tests ──────────────────────────────────────────────

class TestSkinSeverityClassifier:
    def test_model_forward_pass(self):
        from training.model import SkinSeverityClassifier
        model = SkinSeverityClassifier(backbone="resnet18", num_classes=3, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 3)

    def test_model_training_step(self):
        from training.model import SkinSeverityClassifier
        model = SkinSeverityClassifier(backbone="resnet18", num_classes=3, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        y = torch.tensor([0, 2])
        loss = model.training_step((x, y), batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0

    def test_model_output_probabilities(self):
        from training.model import SkinSeverityClassifier
        model = SkinSeverityClassifier(backbone="resnet18", num_classes=3, pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
        assert abs(probs.sum().item() - 1.0) < 1e-5
        assert probs.shape == (1, 3)

    def test_model_get_model_info(self):
        from training.model import SkinSeverityClassifier
        model = SkinSeverityClassifier(
            backbone="resnet18",
            num_classes=3,
            pretrained=False,
            class_names=["mild", "moderate", "severe"],
        )
        info = model.get_model_info()
        assert info["backbone"] == "resnet18"
        assert info["num_classes"] == 3
        assert info["class_names"] == ["mild", "moderate", "severe"]

    def test_invalid_backbone_raises(self):
        from training.model import build_backbone
        with pytest.raises(ValueError, match="Unsupported backbone"):
            build_backbone("invalid_backbone")
