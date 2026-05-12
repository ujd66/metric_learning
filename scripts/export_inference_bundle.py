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
with prototype-based unknown/negative rejection.

## Files

| File | Description |
|------|-------------|
| `model.pt` | Trained model checkpoint |
| `prototypes.pt` | Per-class prototype vectors (19 known classes) |
| `threshold.json` | Selected similarity threshold and metadata |
| `class_names.json` | Label-to-class-name mapping |
| `config.yaml` | Model and inference configuration |
| `version.json` | Bundle metadata |

## Usage

### Single sample inference

```bash
python scripts/infer.py \\
    --config config.yaml \\
    --checkpoint model.pt \\
    --prototypes prototypes.pt \\
    --threshold-json threshold.json \\
    --input /path/to/sample.pcd
```

Input supports `.pcd` and `.npy` formats.

### Output JSON

```json
{
    "final_type": "known|negative|unknown",
    "final_label": "class_name|negative|unknown",
    "reason": "matched_known_prototype|classified_as_negative|far_from_all_known_prototypes",
    "classifier_pred": 5,
    "classifier_confidence": 0.92,
    "nearest_known_class": "class_name",
    "nearest_similarity": 0.95,
    "similarity_threshold": 0.91,
    "top5_classifier_probs": [...],
    "top5_prototype_similarities": [...]
}
```

## Decision Logic

1. **Classifier prediction**: If classifier predicts label 19 → `"negative"`
2. **Prototype similarity**: If nearest prototype similarity < threshold → `"unknown"`
3. **Known class**: Otherwise → matched prototype's class

## Threshold Adjustment

The threshold controls the tradeoff between:

- **Higher threshold** (e.g., 0.94): More conservative, rejects more samples as unknown
  - Lower known_accept_rate (~92%)
  - Higher known classification accuracy on accepted samples
  - Better at catching unknown/negative samples

- **Lower threshold** (e.g., 0.85): More permissive, accepts more samples
  - Higher known_accept_rate (~99%)
  - May let some unknown/negative pass through

To adjust:
1. Edit `threshold.json`: change `selected_threshold`
2. Or pass `--threshold-json` with a different threshold file

## Negative vs Unknown

- **Negative**: Classifier explicitly predicts label 19. The sample looks like the
  "other/negative" class seen during training.
- **Unknown**: Sample doesn't match any known prototype well enough (similarity < threshold).
  Could be a new class never seen during training.
"""


def main():
    parser = argparse.ArgumentParser(description="Export deployable inference bundle")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--prototypes", type=str, default="outputs/prototypes/baseline_prototypes.pt")
    parser.add_argument("--threshold-json", type=str, default="outputs/prototypes/baseline_threshold_p05.json")
    parser.add_argument("--class-names", type=str, default="configs/class_names.json")
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

    # version.json
    git_commit = get_git_commit()
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
        "git_commit": git_commit,
        "notes": "PointNet baseline with prototype-based unknown rejection",
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
    if git_commit:
        print(f"  Git commit: {git_commit}")


if __name__ == "__main__":
    main()
