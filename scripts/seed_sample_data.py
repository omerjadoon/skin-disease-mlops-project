"""
Synthetic sample data generator for skin severity MLOps project.

Generates placeholder images for testing the pipeline without real medical data.

DISCLAIMER: These are synthetic test images, NOT real medical data.
Real training requires a validated, ethically-approved medical image dataset.

Run:
    python scripts/seed_sample_data.py
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).parent.parent))

CLASS_NAMES = ["mild", "moderate", "severe"]
DISEASE_LABELS = {
    "mild": ["acne", "eczema"],
    "moderate": ["psoriasis", "dermatitis"],
    "severe": ["melanoma-like lesion", "psoriasis"],
}

# Color palettes per severity (synthetic, NOT medically meaningful)
CLASS_COLOR_PALETTES = {
    "mild": [(255, 220, 200), (240, 180, 160), (255, 200, 180)],
    "moderate": [(220, 140, 120), (200, 100, 90), (230, 120, 100)],
    "severe": [(180, 60, 60), (160, 40, 40), (200, 80, 70)],
}


def generate_synthetic_skin_image(
    severity: str,
    img_size: int = 224,
    add_noise: bool = True,
    seed: int | None = None,
) -> Image.Image:
    """
    Generate a synthetic placeholder skin image.

    Creates a textured image with color and pattern characteristics per class.
    These are NOT medically meaningful — for pipeline testing only.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    palette = CLASS_COLOR_PALETTES.get(severity, [(200, 180, 160)])
    base_color = random.choice(palette)

    # Create base image with slight color variation
    img_array = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    for c in range(3):
        noise = np.random.normal(0, 10, (img_size, img_size))
        img_array[:, :, c] = np.clip(base_color[c] + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(img_array, mode="RGB")

    # Add some texture patterns (simulate skin texture, NOT medically accurate)
    draw = ImageDraw.Draw(img)
    num_spots = {"mild": 3, "moderate": 8, "severe": 15}.get(severity, 5)

    for _ in range(num_spots):
        x = random.randint(10, img_size - 10)
        y = random.randint(10, img_size - 10)
        r = random.randint(5, 20)
        spot_color = tuple(
            max(0, min(255, c + random.randint(-40, -10))) for c in base_color
        )
        draw.ellipse([x - r, y - r, x + r, y + r], fill=spot_color)

    # Apply blur for realistic look
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))

    if add_noise:
        noise_arr = np.random.normal(0, 8, (img_size, img_size, 3))
        result = np.clip(np.array(img).astype(float) + noise_arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(result)

    return img


def seed_sample_data(
    output_dir: str = "data/raw",
    sample_dir: str = "data/sample",
    images_per_class: int = 30,
    img_size: int = 224,
    create_csv: bool = True,
    csv_path: str = "data/processed/manifest.csv",
) -> None:
    """Generate synthetic images in folder structure and optional CSV manifest."""
    output_path = Path(output_dir)
    sample_path = Path(sample_dir)
    sample_path.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  Synthetic Sample Data Generator")
    print("  DISCLAIMER: NOT real medical data.")
    print("  For pipeline testing only.")
    print("=" * 55)

    csv_rows = []
    total = 0

    for severity in CLASS_NAMES:
        class_dir = output_path / severity
        class_dir.mkdir(parents=True, exist_ok=True)

        diseases = DISEASE_LABELS.get(severity, ["unknown"])
        print(f"\n  Generating {images_per_class} images for class: {severity}")

        for i in range(images_per_class):
            img = generate_synthetic_skin_image(
                severity=severity,
                img_size=img_size,
                add_noise=True,
                seed=hash(f"{severity}_{i}") % (2**31),
            )

            fname = f"{severity}_{i:04d}.jpg"
            img_path = class_dir / fname
            img.save(img_path, "JPEG", quality=90)

            # Determine split
            if i < int(images_per_class * 0.7):
                split = "train"
            elif i < int(images_per_class * 0.85):
                split = "val"
            else:
                split = "test"

            disease = diseases[i % len(diseases)]
            patient_id = f"synthetic_p{total + 1:04d}"

            csv_rows.append({
                "image_path": str(img_path),
                "severity": severity,
                "disease_label": disease,
                "patient_id": patient_id,
                "split": split,
            })
            total += 1

        print(f"    ✓ {images_per_class} images saved to {class_dir}")

    # Save a test image in sample dir
    test_img = generate_synthetic_skin_image("moderate", img_size=img_size, seed=42)
    test_path = sample_path / "test_image.jpg"
    test_img.save(test_path, "JPEG")
    print(f"\n  ✓ Test image saved: {test_path}")

    # Write CSV manifest
    if create_csv:
        csv_out = Path(csv_path)
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_out, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["image_path", "severity", "disease_label", "patient_id", "split"]
            )
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"  ✓ CSV manifest saved: {csv_out}")

    print(f"\n  ✅ Done! Generated {total} synthetic images.")
    print("     DISCLAIMER: These are NOT real medical images.")
    print("     Real training requires a validated medical dataset.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic skin severity sample data")
    parser.add_argument("--output-dir", type=str, default="data/raw")
    parser.add_argument("--sample-dir", type=str, default="data/sample")
    parser.add_argument("--images-per-class", type=int, default=30)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()

    seed_sample_data(
        output_dir=args.output_dir,
        sample_dir=args.sample_dir,
        images_per_class=args.images_per_class,
        img_size=args.img_size,
        create_csv=not args.no_csv,
    )


if __name__ == "__main__":
    main()
