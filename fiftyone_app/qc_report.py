"""
FiftyOne QC report — detects corrupt images, duplicates, missing labels.

Run:
    python fiftyone_app/qc_report.py --data-dir data/raw --output data/qc_report.html
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CLASS_NAMES = ["mild", "moderate", "severe"]


def run_qc(data_dir: str, class_names: list[str] | None = None) -> dict:
    """Run quality control checks on the raw dataset directory."""
    from PIL import Image as PILImage

    if class_names is None:
        class_names = CLASS_NAMES

    data_path = Path(data_dir)
    report: dict = {
        "data_dir": str(data_path.resolve()),
        "class_names": class_names,
        "classes": {},
        "issues": {
            "corrupt_images": [],
            "duplicate_filenames": [],
            "missing_classes": [],
            "empty_classes": [],
        },
        "summary": {},
    }

    seen_filenames: dict[str, str] = {}
    seen_hashes: dict[str, str] = {}
    total_images = 0

    for class_name in class_names:
        class_dir = data_path / class_name
        if not class_dir.exists():
            report["issues"]["missing_classes"].append(class_name)
            report["classes"][class_name] = {"count": 0, "corrupt": 0}
            continue

        img_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            img_files.extend(class_dir.glob(ext))

        if not img_files:
            report["issues"]["empty_classes"].append(class_name)
            report["classes"][class_name] = {"count": 0, "corrupt": 0}
            continue

        corrupt_count = 0
        for img_path in img_files:
            # Check duplicate filenames
            fname = img_path.name
            if fname in seen_filenames:
                report["issues"]["duplicate_filenames"].append(
                    {"file": str(img_path), "duplicate_of": seen_filenames[fname]}
                )
            else:
                seen_filenames[fname] = str(img_path)

            # Check corrupt images
            try:
                with PILImage.open(img_path) as img:
                    img.verify()
                # Check duplicate content (hash)
                with open(img_path, "rb") as f:
                    img_hash = hashlib.md5(f.read()).hexdigest()
                if img_hash in seen_hashes:
                    report["issues"]["duplicate_filenames"].append(
                        {"file": str(img_path), "duplicate_of": seen_hashes[img_hash], "type": "content"}
                    )
                else:
                    seen_hashes[img_hash] = str(img_path)
            except Exception as e:
                corrupt_count += 1
                report["issues"]["corrupt_images"].append(
                    {"file": str(img_path), "error": str(e)}
                )

        report["classes"][class_name] = {
            "count": len(img_files),
            "corrupt": corrupt_count,
            "valid": len(img_files) - corrupt_count,
        }
        total_images += len(img_files)

    report["summary"] = {
        "total_images": total_images,
        "total_corrupt": len(report["issues"]["corrupt_images"]),
        "total_duplicate_filenames": len(report["issues"]["duplicate_filenames"]),
        "missing_classes": report["issues"]["missing_classes"],
        "empty_classes": report["issues"]["empty_classes"],
        "qc_passed": (
            len(report["issues"]["corrupt_images"]) == 0
            and len(report["issues"]["missing_classes"]) == 0
            and total_images > 0
        ),
    }

    return report


def export_html_report(report: dict, output_path: str) -> None:
    """Export QC results as an HTML file."""
    summary = report["summary"]
    classes = report["classes"]
    issues = report["issues"]

    status_color = "#22c55e" if summary.get("qc_passed") else "#ef4444"
    status_text = "PASSED ✓" if summary.get("qc_passed") else "ISSUES FOUND ✗"

    class_rows = ""
    for cls, info in classes.items():
        class_rows += (
            f"<tr><td>{cls}</td><td>{info.get('count',0)}</td>"
            f"<td>{info.get('valid',0)}</td>"
            f"<td style='color:{'red' if info.get('corrupt',0) > 0 else 'green'}'>"
            f"{info.get('corrupt',0)}</td></tr>"
        )

    corrupt_list = "".join(
        f"<li><code>{c['file']}</code>: {c.get('error','unknown')}</li>"
        for c in issues["corrupt_images"]
    ) or "<li>None</li>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Skin Severity Dataset QC Report</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; }}
  h1 {{ color: #1e293b; }}
  .badge {{ display: inline-block; padding: 6px 16px; border-radius: 6px;
    background: {status_color}; color: white; font-weight: bold; font-size: 1.1em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ border: 1px solid #e2e8f0; padding: 8px 12px; text-align: left; }}
  th {{ background: #f8fafc; }}
  .disclaimer {{ background: #fef9c3; border-left: 4px solid #eab308;
    padding: 12px; margin: 20px 0; border-radius: 4px; }}
</style></head>
<body>
<h1>Skin Severity Dataset — QC Report</h1>
<div class="disclaimer">⚠ DISCLAIMER: This dataset is for educational/research purposes only.
It must not be used for medical diagnosis.</div>
<p>Data directory: <code>{report['data_dir']}</code></p>
<p>QC Status: <span class="badge">{status_text}</span></p>
<h2>Summary</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Total Images</td><td>{summary['total_images']}</td></tr>
  <tr><td>Corrupt Images</td><td style="color:{'red' if summary['total_corrupt']>0 else 'green'}">{summary['total_corrupt']}</td></tr>
  <tr><td>Duplicate Issues</td><td>{summary['total_duplicate_filenames']}</td></tr>
  <tr><td>Missing Classes</td><td>{', '.join(summary['missing_classes']) or 'None'}</td></tr>
</table>
<h2>Per-Class Breakdown</h2>
<table><tr><th>Class</th><th>Total</th><th>Valid</th><th>Corrupt</th></tr>
{class_rows}</table>
<h2>Corrupt Images</h2><ul>{corrupt_list}</ul>
<p><em>Generated by skin-severity-mlops QC pipeline</em></p>
</body></html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"✓ QC report saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QC on skin severity dataset")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--output", type=str, default="data/qc_report.html")
    parser.add_argument("--json-output", type=str, default=None)
    args = parser.parse_args()

    print(f"Running QC on: {args.data_dir}")
    report = run_qc(args.data_dir)

    print("\n── QC Summary ──────────────────────────────")
    print(f"  Total images:    {report['summary']['total_images']}")
    print(f"  Corrupt:         {report['summary']['total_corrupt']}")
    print(f"  Duplicate files: {report['summary']['total_duplicate_filenames']}")
    print(f"  Missing classes: {report['summary']['missing_classes']}")
    print(f"  QC passed:       {report['summary']['qc_passed']}")

    export_html_report(report, args.output)

    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"✓ JSON report saved: {args.json_output}")


if __name__ == "__main__":
    main()
