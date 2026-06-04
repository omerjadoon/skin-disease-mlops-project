"""
PyTorch Lightning DataModule supporting both folder-based and CSV-based datasets.

Expected folder structure:
    data/raw/
    ├── mild/
    ├── moderate/
    └── severe/

Expected CSV structure:
    image_path,severity,disease_label,patient_id,split
    image_001.jpg,mild,acne,p001,train
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning as pl  # type: ignore[no-redef]

import torchvision.transforms as T

from training.transforms import build_transforms_from_config

CLASS_NAMES = ["mild", "moderate", "severe"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


class FolderSkinDataset(Dataset):
    """Dataset for folder-organized images: root/class_name/image.jpg"""

    def __init__(
        self,
        root_dir: str | Path,
        class_names: list[str],
        transform: T.Compose | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                for img_path in class_dir.glob(ext):
                    self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class CSVSkinDataset(Dataset):
    """Dataset loading images from a CSV manifest file."""

    def __init__(
        self,
        csv_path: str | Path,
        class_names: list[str],
        split: str | None = None,
        transform: T.Compose | None = None,
        image_root: str | Path | None = None,
    ) -> None:
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.transform = transform
        self.image_root = Path(image_root) if image_root else None

        df = pd.read_csv(csv_path)
        if split and "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)

        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])
        if self.image_root and not img_path.is_absolute():
            img_path = self.image_root / img_path

        label_str = str(row["severity"]).lower()
        label = self.class_to_idx.get(label_str, 0)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class SkinDataModule(pl.LightningDataModule):
    """
    LightningDataModule for skin severity classification.

    Supports both folder-based and CSV-based dataset loading.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.class_names: list[str] = cfg.get("class_names", CLASS_NAMES)
        self.batch_size: int = cfg.get("batch_size", 16)
        self.num_workers: int = cfg.get("num_workers", 2)
        self.image_size: int = cfg.get("image_size", 224)
        self.val_split: float = cfg.get("val_split", 0.15)
        self.test_split: float = cfg.get("test_split", 0.10)
        self.seed: int = cfg.get("seed", 42)
        self.dataset_mode: str = cfg.get("dataset_mode", "folder")
        self.raw_data_dir: str = cfg.get("raw_data_dir", "data/raw")
        self.processed_data_dir: str = cfg.get("processed_data_dir", "data/processed")
        self.csv_manifest: str | None = cfg.get("csv_manifest")

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

        # Build transforms
        self.train_transform, self.val_transform = build_transforms_from_config(cfg)

    def setup(self, stage: str | None = None) -> None:
        if self.dataset_mode == "csv" and self.csv_manifest:
            self._setup_csv(stage)
        else:
            self._setup_folder(stage)

    def _setup_folder(self, stage: str | None = None) -> None:
        """Load from folder structure: raw_data_dir/class_name/image.jpg"""
        # Try processed dir first, fall back to raw
        data_dir = self.processed_data_dir
        if not Path(data_dir).exists() or not any(
            (Path(data_dir) / c).exists() for c in self.class_names
        ):
            data_dir = self.raw_data_dir

        full_dataset = FolderSkinDataset(data_dir, self.class_names, transform=None)
        total = len(full_dataset)

        if total == 0:
            raise RuntimeError(
                f"No images found in {data_dir}. Run 'make seed-data' to generate sample data."
            )

        n_test = max(1, int(total * self.test_split))
        n_val = max(1, int(total * self.val_split))
        n_train = total - n_val - n_test

        generator = torch.Generator().manual_seed(self.seed)
        train_ds, val_ds, test_ds = random_split(
            full_dataset, [n_train, n_val, n_test], generator=generator
        )

        # Apply transforms via wrapper
        self.train_dataset = _TransformDataset(train_ds, self.train_transform)
        self.val_dataset = _TransformDataset(val_ds, self.val_transform)
        self.test_dataset = _TransformDataset(test_ds, self.val_transform)

    def _setup_csv(self, stage: str | None = None) -> None:
        """Load from CSV manifest with pre-defined splits."""
        image_root = self.processed_data_dir or self.raw_data_dir
        self.train_dataset = CSVSkinDataset(
            self.csv_manifest,  # type: ignore[arg-type]
            self.class_names,
            split="train",
            transform=self.train_transform,
            image_root=image_root,
        )
        self.val_dataset = CSVSkinDataset(
            self.csv_manifest,  # type: ignore[arg-type]
            self.class_names,
            split="val",
            transform=self.val_transform,
            image_root=image_root,
        )
        self.test_dataset = CSVSkinDataset(
            self.csv_manifest,  # type: ignore[arg-type]
            self.class_names,
            split="test",
            transform=self.val_transform,
            image_root=image_root,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.cfg.get("pin_memory", False),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class _TransformDataset(Dataset):
    """Wrapper that applies a transform to a Subset/Dataset."""

    def __init__(self, dataset: Dataset, transform: T.Compose) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        image, label = self.dataset[idx]
        if not isinstance(image, torch.Tensor) and self.transform:
            image = self.transform(image)
        return image, label
