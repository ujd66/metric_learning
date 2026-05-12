"""Export a deployable inference bundle.

Bundles model checkpoint, prototypes, threshold, config, and class names
into a self-contained directory for deployment.

Usage:
    python scripts/export_inference_bundle.py \
        --config configs/config.yaml \
        --checkpoint outputs/checkpoints/best.pt \
        --prototypes outputs/prototypes/baseline_prototypes.pt \
        --threshold-json outputs/prototypes/baseline_threshold_p05.json \
        --class-names configs/class_names.json \
        --output outputs/deploy/pointnet_baseline_bundle
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime


def get_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


INFERENCE_README = """# PointNet Baseline Inference Bundle

## Overview

This bundle contains everything needed for point cloud classification inference
with a unified 3-stage decision pipeline:
  1. Classifier negative detection
  2. Prototype-based unknown rejection
  3. Gallery retrieval for evidence (optional)

## Decision Logic (Priority Order)

### Step 1: Classifier Negative Detection (HIGHEST PRIORITY)
If the classifier predicts label 19 (negative) → **"negative"**
This is the first priority. If the classifier is confident the sample is negative,
no further checks are needed.

### Step 2: Prototype OOD Rejection (PRIMARY unknown mechanism)
If NOT classified as negative:
- Compute cosine similarity between embedding and all known class prototypes
- If nearest prototype similarity < threshold → **"unknown"**
- This is the PRIMARY mechanism for unknown/out-of-distribution rejection
- Recommended threshold: 0.90~0.91

### Step 3: Known Class Output
If the sample passes prototype threshold:
- final_type = "known"
- final_label = nearest prototype's class name

### Step 4: Gallery Evidence (AUXILIARY ONLY)
If gallery.pt is provided:
- Retrieve top-K most similar gallery samples
- Output as evidence for human review
- **Gallery retrieval does NOT override the final_label**
- Gallery nearest-neighbor similarity should NOT be used as the primary
  unknown rejection score

### Risk Levels

| Risk Level | Meaning |
|------------|---------|
| `safe` | Normal classification, far from threshold |
| `prototype_gallery_conflict` | Prototype class != gallery top1 class. Review recommended. |
| `near_threshold_boundary` | Prototype similarity within 0.03 of threshold. Low confidence. |
| `unsupported_class_no_training_data` | Classifier predicted an unsupported class with no training data. |

## Current Version Limitations

**This bundle is for internal testing, not final production.**

1. **class_014 (shenggaozuofalan) is unsupported** — no training samples available.
   The system cannot reliably identify this class. It will be detected as
   `unsupported_known_class` with `risk_level = "unsupported_class_no_training_data"`.
2. **Negative/unknown rejection is preliminary** — limited negative data means
   threshold calibration is not fully validated.
3. **Prototype similarity is the main OOD score** — do not use gallery similarity for rejection.
4. **Gallery retrieval is evidence only** — provides similar samples for human review.
5. **Deployment status: `limited_internal_prototype`** — requires more data before production use.

### What happens with unsupported classes

If the classifier predicts label 14 (shenggaozuofalan):
- `final_type = "unsupported_known_class"`
- `reason = "classifier_predicted_unsupported_known_class"`
- `risk_level = "unsupported_class_no_training_data"`
- The system cannot provide prototype or gallery evidence for this class

## Files

| File | Description |
|------|-------------|
| `model.pt` | Trained model checkpoint |
| `prototypes.pt` | Per-class prototype vectors (19 known classes) |
| `threshold.json` | Selected similarity threshold and metadata |
| `class_names.json` | Label-to-class-name mapping |
| `config.yaml` | Model and inference configuration |
| `version.json` | Bundle metadata |
| `gallery.pt` | (Optional) Gallery embeddings for evidence retrieval |
| `class_coverage_report.json` | (Optional) Class coverage validation |
| `final_infer.py` | Final inference script |

## Usage

### Final Inference (recommended)

```bash
python final_infer.py \\
    --config config.yaml \\
    --checkpoint model.pt \\
    --prototypes prototypes.pt \\
    --threshold-json threshold.json \\
    --gallery gallery.pt \\
    --input /path/to/sample.pcd
```

Input supports `.pcd` and `.npy` formats.

### Output JSON

```json
{
  "final_type": "known|negative|unknown|unsupported_known_class",
  "final_label": "class_name|negative|unknown",
  "reason": "matched_known_prototype|classified_as_negative|far_from_known_prototypes|classifier_predicted_unsupported_known_class",
  "risk_level": "safe|prototype_gallery_conflict|near_threshold_boundary|unsupported_class_no_training_data",

  "classifier": {
    "pred_label": 5,
    "pred_class_name": "class_name",
    "confidence": 0.92,
    "top5": [...]
  },

  "prototype": {
    "nearest_label": 5,
    "nearest_class_name": "class_name",
    "nearest_similarity": 0.95,
    "threshold": 0.91,
    "top5": [...]
  },

  "gallery": {
    "enabled": true,
    "top1_label": 5,
    "top1_class_name": "class_name",
    "top1_similarity": 0.98,
    "top5": [...]
  }
}
```

## Threshold Adjustment

The threshold controls the tradeoff between:

- **Higher threshold** (e.g., 0.94): More conservative, rejects more as unknown
- **Lower threshold** (e.g., 0.85): More permissive, accepts more samples

**Recommended threshold: 0.90~0.91** (known_quantile P05 strategy)

To adjust:
1. Edit `threshold.json`: change `selected_threshold`
2. Or pass `--threshold-json` with a different threshold file

## Important Notes

1. **Classifier negative is the first priority** — no prototype/gallery check needed
2. **Prototype threshold is the primary OOD mechanism** — use this for unknown rejection
3. **Gallery retrieval is auxiliary evidence only** — do not use for OOD rejection
4. **If gallery top1 and prototype class disagree**, manual review is recommended
5. **If a known class has no train samples**, the bundle is incomplete (check coverage report)

## Negative vs Unknown

- **Negative**: Classifier explicitly predicts label 19. The sample looks like the
  "other/negative" class seen during training.
- **Unknown**: Sample doesn't match any known prototype well enough (similarity < threshold).
  Could be a new class never seen during training.

## Deployment Checklist

Before deploying this bundle:

1. Run `check_class_coverage.py` to verify all known classes have train samples
2. Verify threshold is set correctly for your risk tolerance
3. Verify no known class is missing from prototypes
4. If using gallery, verify it includes all known classes
5. Test with representative samples from each class
"""


def main():
    parser = argparse.ArgumentParser(description="Export deployable inference bundle")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--prototypes", type=str, default="outputs/prototypes/baseline_prototypes.pt")
    parser.add_argument("--threshold-json", type=str, default="outputs/prototypes/baseline_threshold_p05.json")
    parser.add_argument("--class-names", type=str, default="configs/class_names.json")
    parser.add_argument("--gallery", type=str, default=None,
                        help="Path to gallery.pt to include in bundle")
    parser.add_argument("--coverage-report", type=str, default=None,
                        help="Path to class_coverage_report.json to include in bundle")
    parser.add_argument("--deployment-report", type=str, default=None,
                        help="Path to limited_deployment_report.html to include in bundle")
    parser.add_argument("--output", type=str, default="outputs/deploy/pointnet_baseline_bundle")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output, exist_ok=True)

    # Load threshold for metadata
    with open(args.threshold_json) as f:
        threshold_data = json.load(f)

    # Copy files
    print(f"Exporting inference bundle to {args.output} ...")

    dest_model = os.path.join(args.output, "model.pt")
    shutil.copy2(args.checkpoint, dest_model)
    print(f"  Copied: model.pt ({os.path.getsize(dest_model) / 1024:.1f} KB)")

    dest_proto = os.path.join(args.output, "prototypes.pt")
    shutil.copy2(args.prototypes, dest_proto)
    print(f"  Copied: prototypes.pt ({os.path.getsize(dest_proto) / 1024:.1f} KB)")

    dest_threshold = os.path.join(args.output, "threshold.json")
    shutil.copy2(args.threshold_json, dest_threshold)
    print(f"  Copied: threshold.json")

    dest_classnames = os.path.join(args.output, "class_names.json")
    shutil.copy2(args.class_names, dest_classnames)
    print(f"  Copied: class_names.json")

    dest_config = os.path.join(args.output, "config.yaml")
    shutil.copy2(args.config, dest_config)
    print(f"  Copied: config.yaml")

    # Optional: gallery
    if args.gallery and os.path.exists(args.gallery):
        dest_gallery = os.path.join(args.output, "gallery.pt")
        shutil.copy2(args.gallery, dest_gallery)
        print(f"  Copied: gallery.pt ({os.path.getsize(dest_gallery) / 1024:.1f} KB)")

    # Optional: coverage report
    if args.coverage_report and os.path.exists(args.coverage_report):
        dest_coverage = os.path.join(args.output, "class_coverage_report.json")
        shutil.copy2(args.coverage_report, dest_coverage)
        print(f"  Copied: class_coverage_report.json")

    # Optional: deployment report
    if args.deployment_report and os.path.exists(args.deployment_report):
        dest_dep = os.path.join(args.output, "limited_deployment_report.html")
        shutil.copy2(args.deployment_report, dest_dep)
        print(f"  Copied: limited_deployment_report.html")

    # Copy final_infer.py reference
    final_infer_src = os.path.join(os.path.dirname(__file__), "final_infer.py")
    if os.path.exists(final_infer_src):
        dest_infer = os.path.join(args.output, "final_infer.py")
        shutil.copy2(final_infer_src, dest_infer)
        print(f"  Copied: final_infer.py")

    # version.json
    git_commit = get_git_commit()
    supported_cfg = cfg.get("supported_classes", {})
    version = {
        "created_at": datetime.now().isoformat(),
        "checkpoint": os.path.basename(args.checkpoint),
        "threshold": threshold_data["selected_threshold"],
        "threshold_strategy": threshold_data.get("selection_strategy", "unknown"),
        "num_known_classes": cfg["num_known_classes"],
        "negative_label": cfg["negative_label"],
        "embedding_dim": cfg["embedding_dim"],
        "num_points": cfg["num_points"],
        "input_channels": cfg["input_channels"],
        "supported_known_labels": supported_cfg.get("supported_known_labels", list(range(cfg["num_known_classes"]))),
        "unsupported_known_labels": supported_cfg.get("unsupported_known_labels", []),
        "incomplete_known_class_coverage": bool(supported_cfg.get("unsupported_known_labels", [])),
        "git_commit": git_commit,
        "notes": "PointNet baseline with prototype-based unknown rejection (limited deployment)",
    }
    version_path = os.path.join(args.output, "version.json")
    with open(version_path, "w") as f:
        json.dump(version, f, indent=2, ensure_ascii=False)
    print(f"  Created: version.json")

    # README
    readme_path = os.path.join(args.output, "README_INFERENCE.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(INFERENCE_README)
    print(f"  Created: README_INFERENCE.md")

    print(f"\nBundle exported successfully!")
    print(f"  Location: {args.output}")
    print(f"  Threshold: {threshold_data['selected_threshold']:.2f} ({threshold_data.get('selection_strategy', 'unknown')})")
    print(f"  Classes: {cfg['num_known_classes']} known + 1 negative")
    if args.gallery and os.path.exists(args.gallery):
        print(f"  Gallery: included")
    if git_commit:
        print(f"  Git commit: {git_commit}")


if __name__ == "__main__":
    main()
