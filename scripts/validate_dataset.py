import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import load_config


def collect_class_counts(dataset_root, split):
    """Return {class_dir: count} for a given split."""
    split_dir = os.path.join(dataset_root, split)
    counts = {}
    if not os.path.isdir(split_dir):
        return counts
    for class_dir in sorted(os.listdir(split_dir)):
        class_path = os.path.join(split_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        npy_files = [f for f in os.listdir(class_path) if f.endswith(".npy")]
        if npy_files:
            counts[class_dir] = len(npy_files)
    return counts


def validate_dataset(dataset_root, config_path):
    cfg = load_config(config_path)

    num_known_classes = cfg.get("num_known_classes", 19)
    negative_label = cfg.get("negative_label", 19)
    num_classes = cfg.get("num_classes", num_known_classes + 1)

    label_mapping_cfg = cfg.get("label_mapping", {})
    negative_names = set(label_mapping_cfg.get("negative_names", ["negative"]))

    splits = ["train", "val", "test"]
    report = {
        "dataset_root": os.path.abspath(dataset_root),
        "config": os.path.abspath(config_path),
        "num_known_classes": num_known_classes,
        "negative_label": negative_label,
        "num_classes": num_classes,
        "per_split": {},
        "per_class": {},
        "warnings": [],
        "errors": [],
    }

    # Collect counts per split
    all_classes = set()
    for split in splits:
        counts = collect_class_counts(dataset_root, split)
        report["per_split"][split] = {
            "total": sum(counts.values()),
            "classes": counts,
        }
        all_classes.update(counts.keys())

    # Check splits non-empty
    for split in splits:
        total = report["per_split"][split]["total"]
        if total == 0:
            report["errors"].append(f"{split} split is EMPTY")

    # Build per-class summary
    for class_dir in sorted(all_classes):
        entry = {"train": 0, "val": 0, "test": 0}
        for split in splits:
            entry[split] = report["per_split"][split]["classes"].get(class_dir, 0)
        entry["total"] = sum(entry[s] for s in splits)
        report["per_class"][class_dir] = entry

    # Identify negative class directory
    negative_dirs = []
    for class_dir in all_classes:
        if class_dir == "negative" or class_dir.lower() in negative_names:
            negative_dirs.append(class_dir)

    # Also check class_mapping.json for class names that match negative_names
    class_mapping_path = os.path.join("raw_pcd_dataset", "class_mapping.json")
    negative_mapped_from = set()  # class_xxx dirs whose class_name is in negative_names
    if os.path.exists(class_mapping_path):
        with open(class_mapping_path) as f:
            class_mapping = json.load(f)
        for class_dir_key, class_name in class_mapping.items():
            if class_name.lower() in negative_names:
                negative_mapped_from.add(class_dir_key)
                if class_dir in all_classes and class_dir == class_dir_key:
                    if class_dir not in negative_dirs:
                        negative_dirs.append(class_dir)

    # "negative" dir is always negative
    if "negative" not in negative_dirs and "negative" in all_classes:
        negative_dirs.append("negative")

    report["negative_class_dirs"] = negative_dirs
    if not negative_dirs:
        report["warnings"].append("No negative class directory found")

    # Check every known class has train samples
    train_classes = set(report["per_split"]["train"]["classes"].keys())
    expected_known = set(f"class_{i:03d}" for i in range(num_known_classes + 1))
    # Remove negative dirs from expected known
    if negative_dirs:
        expected_known -= set(negative_dirs)
    # Also remove class dirs that were mapped to negative (e.g., class_014=qita)
    expected_known -= negative_mapped_from
    # Only keep classes that actually exist in the dataset
    expected_known &= all_classes

    missing_train = expected_known - train_classes
    if missing_train:
        report["warnings"].append(f"Classes missing from train: {sorted(missing_train)}")

    # Check val/test coverage
    val_classes = set(report["per_split"]["val"]["classes"].keys())
    test_classes = set(report["per_split"]["test"]["classes"].keys())
    missing_val = train_classes - val_classes
    missing_test = train_classes - test_classes
    if missing_val:
        report["warnings"].append(f"Classes present in train but missing from val: {sorted(missing_val)}")
    if missing_test:
        report["warnings"].append(f"Classes present in train but missing from test: {sorted(missing_test)}")

    # Imbalance analysis
    train_counts = report["per_split"]["train"]["classes"]
    if train_counts:
        counts_list = list(train_counts.values())
        max_count = max(counts_list)
        min_count = min(counts_list)
        imbalance_ratio = max_count / max(min_count, 1)

        largest_class = max(train_counts, key=train_counts.get)
        largest_ratio = max_count / report["per_split"]["train"]["total"]

        # Smallest non-empty class
        non_empty = {k: v for k, v in train_counts.items() if v > 0}
        if non_empty:
            smallest_class = min(non_empty, key=non_empty.get)
            smallest_count = non_empty[smallest_class]
        else:
            smallest_class = "N/A"
            smallest_count = 0

        report["imbalance"] = {
            "max_count": max_count,
            "min_count": min_count,
            "imbalance_ratio": round(imbalance_ratio, 2),
            "largest_class": largest_class,
            "largest_class_count": max_count,
            "largest_class_ratio": round(largest_ratio, 4),
            "smallest_nonempty_class": smallest_class,
            "smallest_nonempty_count": smallest_count,
        }

        if imbalance_ratio > 10:
            report["warnings"].append(
                f"Severe class imbalance: ratio={imbalance_ratio:.1f} "
                f"({largest_class}={max_count} vs min={min_count})"
            )

        # Metric learning suitability per class
        report["metric_learning_suitability"] = {}
        for class_dir, count in sorted(train_counts.items()):
            if count >= 10:
                suitability = "good"
            elif count >= 3:
                suitability = "weak"
            else:
                suitability = "not_suitable"
            report["metric_learning_suitability"][class_dir] = {
                "count": count,
                "suitability": suitability,
            }

    # Missing classes per split
    all_existing = set()
    for split in splits:
        all_existing.update(report["per_split"][split]["classes"].keys())
    report["missing_per_split"] = {}
    for split in splits:
        present = set(report["per_split"][split]["classes"].keys())
        missing = all_existing - present
        if missing:
            report["missing_per_split"][split] = sorted(missing)
        else:
            report["missing_per_split"][split] = []

        # Warn about classes with very few samples
        for class_dir, count in sorted(train_counts.items(), key=lambda x: x[1]):
            if count < 3:
                msg = f"{class_dir} has only {count} train samples — cannot reliably train/evaluate"
                if msg not in report["warnings"]:
                    report["warnings"].append(msg)
            elif count < 10:
                msg = f"{class_dir} has only {count} train samples — may be unreliable"
                if msg not in report["warnings"]:
                    report["warnings"].append(msg)

    # Print summary
    print("=" * 60)
    print("Dataset Validation Report")
    print("=" * 60)
    print(f"\nDataset: {os.path.abspath(dataset_root)}")
    print(f"Known classes: {num_known_classes}, Negative label: {negative_label}")

    print("\n--- Per-Split Summary ---")
    for split in splits:
        info = report["per_split"][split]
        print(f"  {split}: {info['total']} samples, {len(info['classes'])} classes")

    print("\n--- Per-Class Detail ---")
    header = f"  {'Class':<20} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for class_dir, entry in sorted(report["per_class"].items()):
        marker = " *" if class_dir in negative_dirs else ""
        print(f"  {class_dir:<20} {entry['train']:>6} {entry['val']:>6} {entry['test']:>6} {entry['total']:>6}{marker}")
    if negative_dirs:
        print(f"  (* = negative class)")

    if "imbalance" in report:
        imb = report["imbalance"]
        print(f"\n--- Imbalance ---")
        print(f"  Largest class: {imb['largest_class']} ({imb['largest_class_count']}, {imb['largest_class_ratio']:.1%} of train)")
        print(f"  Smallest non-empty class: {imb['smallest_nonempty_class']} ({imb['smallest_nonempty_count']})")
        print(f"  Imbalance ratio (max/min): {imb['imbalance_ratio']}")

    # Missing classes per split
    if report.get("missing_per_split"):
        print(f"\n--- Missing Classes per Split ---")
        for split in splits:
            missing = report["missing_per_split"].get(split, [])
            if missing:
                print(f"  {split}: {missing}")
            else:
                print(f"  {split}: (none)")

    # Metric learning suitability
    if "metric_learning_suitability" in report:
        print(f"\n--- Metric Learning Suitability ---")
        for class_dir, info in report["metric_learning_suitability"].items():
            suit = info["suitability"]
            marker = {"good": "OK", "weak": "WEAK", "not_suitable": "BAD"}[suit]
            print(f"  {class_dir:<20} {info['count']:>4} samples  [{marker}]")

    if report["warnings"]:
        print(f"\n--- Warnings ({len(report['warnings'])}) ---")
        for w in report["warnings"]:
            print(f"  [WARN] {w}")

    if report["errors"]:
        print(f"\n--- Errors ({len(report['errors'])}) ---")
        for e in report["errors"]:
            print(f"  [ERROR] {e}")

    # Save report
    os.makedirs("outputs/reports", exist_ok=True)
    out_path = "outputs/reports/dataset_validation.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {out_path}")

    return len(report["errors"]) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate dataset structure and report issues")
    parser.add_argument("--dataset-root", type=str, default="dataset", help="Root directory of dataset")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    ok = validate_dataset(args.dataset_root, args.config)
    sys.exit(0 if ok else 1)
