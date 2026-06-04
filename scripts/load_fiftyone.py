"""
Load the skin severity dataset into FiftyOne for visual exploration.

Usage:
    python scripts/load_fiftyone.py
    python scripts/load_fiftyone.py --dataset-name my-skin-data --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fiftyone as fo

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "raw"
CLASS_NAMES = ["mild", "moderate", "severe"]
DEFAULT_DATASET_NAME = "skin-severity"


def load_dataset(
    dataset_name: str = DEFAULT_DATASET_NAME,
    data_dir: Path = DATA_DIR,
    overwrite: bool = False,
) -> fo.Dataset:
    """Load skin severity images into a FiftyOne dataset with classification labels."""

    # Delete existing dataset if overwrite is requested
    if fo.dataset_exists(dataset_name):
        if overwrite:
            fo.delete_dataset(dataset_name)
            print(f"  🗑  Deleted existing dataset '{dataset_name}'")
        else:
            print(f"  ℹ  Dataset '{dataset_name}' already exists — loading it.")
            return fo.load_dataset(dataset_name)

    print("=" * 55)
    print(f"  Loading dataset: {dataset_name}")
    print(f"  Source: {data_dir}")
    print("=" * 55)

    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True  # survives app restarts

    samples = []
    total = 0

    for severity in CLASS_NAMES:
        class_dir = data_dir / severity
        if not class_dir.exists():
            print(f"  ⚠  {class_dir} not found — skipping")
            continue

        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        print(f"  Adding {len(images)} images for class: {severity}")

        # Determine split (mirrors seed_sample_data.py logic)
        n = len(images)
        for i, img_path in enumerate(sorted(images)):
            if i < int(n * 0.7):
                split = "train"
            elif i < int(n * 0.85):
                split = "val"
            else:
                split = "test"

            sample = fo.Sample(
                filepath=str(img_path),
                ground_truth=fo.Classification(label=severity),
                severity=severity,
                split=split,
                tags=[severity, split],
            )
            samples.append(sample)
            total += 1

    dataset.add_samples(samples)

    # Tag by split for easy filtering
    print(f"\n  ✅ Loaded {total} samples into dataset '{dataset_name}'")
    print(f"     Splits: train={sum(1 for s in dataset if s.split=='train')}, "
          f"val={sum(1 for s in dataset if s.split=='val')}, "
          f"test={sum(1 for s in dataset if s.split=='test')}")
    print(f"\n  Class breakdown:")
    for cls in CLASS_NAMES:
        count = len(dataset.match(fo.ViewField("severity") == cls))
        print(f"    {cls:10s}: {count} images")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Load skin severity dataset into FiftyOne")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete and recreate dataset if it exists")
    parser.add_argument("--launch", action="store_true",
                        help="Launch FiftyOne app after loading")
    args = parser.parse_args()

    dataset = load_dataset(
        dataset_name=args.dataset_name,
        data_dir=Path(args.data_dir),
        overwrite=args.overwrite,
    )

    print(f"\n  📊 Open FiftyOne at: http://localhost:5151")
    print(f"  Or run: python3 -c \"import fiftyone as fo; fo.launch_app(fo.load_dataset('{args.dataset_name}'))\"")

    if args.launch:
        session = fo.launch_app(dataset)
        session.wait()


if __name__ == "__main__":
    main()
