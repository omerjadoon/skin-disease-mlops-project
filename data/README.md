# Dataset Directory

## Structure

This project supports two dataset loading modes:

### 1. Folder-Based (default)

```
data/raw/
├── mild/
│   ├── image_001.jpg
│   └── image_002.jpg
├── moderate/
│   ├── image_003.jpg
│   └── image_004.jpg
└── severe/
    ├── image_005.jpg
    └── image_006.jpg
```

### 2. CSV-Based

```
data/processed/manifest.csv
```

```csv
image_path,severity,disease_label,patient_id,split
data/raw/mild/image_001.jpg,mild,acne,p001,train
data/raw/moderate/image_003.jpg,moderate,eczema,p002,val
data/raw/severe/image_005.jpg,severe,psoriasis,p003,test
```

## Quick Start (Synthetic Data)

```bash
make seed-data
# or
python scripts/seed_sample_data.py --images-per-class 30
```

## Supported Image Formats

- JPEG / JPG
- PNG

## Disease Labels (Optional Metadata)

| Label | Severity |
|-------|----------|
| acne | mild |
| eczema | mild / moderate |
| psoriasis | moderate / severe |
| dermatitis | moderate |
| melanoma-like lesion | severe |
| unknown | any |

## ⚠ Disclaimer

**This project does NOT include real medical data.**

- The `data/` directory contains synthetic placeholder images for pipeline testing only.
- Real training requires a validated, ethically-approved medical image dataset.
- All outputs are AI-assisted estimates — NOT medical diagnoses.
- Do NOT use this system for clinical decision-making.

## DVC Tracking

Data files are tracked with DVC:

```bash
# After adding real data
dvc add data/raw
dvc push
```

Remote storage is MinIO (`s3://dvc-storage`).
