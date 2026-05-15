import argparse
import json
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import load_config


def resolve_label(class_dir, label_mapping_cfg):
    """Determine the output label and directory name for a class directory.

    Returns (output_class_dir, is_negative).
    """
    negative_names = set(label_mapping_cfg.get("negative_names", ["negative"]))
    force_neg_label = label_mapping_cfg.get("force_negative_label", 19)

    # Check if class_dir name itself is a negative name
    if class_dir.lower() in negative_names:
        return "negative", True

    # Check class_mapping.json: if class_name is in negative_names
    # Prefer label/class_mapping.json (authoritative) over raw_pcd_dataset/
    class_mapping_path = os.path.join("label", "class_mapping.json")
    if not os.path.exists(class_mapping_path):
        class_mapping_path = os.path.join("raw_pcd_dataset", "class_mapping.json")
    if os.path.exists(class_mapping_path):
        with open(class_mapping_path) as f:
            class_mapping = json.load(f)
        class_name = class_mapping.get(class_dir, "")
        if class_name.lower() in negative_names:
            return "negative", True

    return class_dir, False


def stratified_split(samples, train_ratio, val_ratio, test_ratio, seed):
    """Split samples into train/val/test respecting ratios.

    For very small counts:
      - count >= 3: at least 1 per split
      - count == 2: train 1, val 1, test 0 (warning)
      - count == 1: train 1, val 0, test 0 (warning)
    """
    rng = random.Random(seed)
    n = len(samples)
    shuffled = samples.copy()
    rng.shuffle(shuffled)

    if n == 0:
        return [], [], []
    if n == 1:
        return shuffled[:1], [], []
    if n == 2:
        return shuffled[:1], shuffled[1:], []

    # Ensure at least 1 per split
    train_n = max(1, round(n * train_ratio))
    val_n = max(1, round(n * val_ratio))
    test_n = n - train_n - val_n

    # If test_n <= 0 after ensuring train/val each get 1, redistribute
    if test_n <= 0:
        if n >= 3:
            train_n = max(1, n - 2)
            val_n = 1
            test_n = 1
        else:
            # Should not reach here (handled above), but be safe
            train_n = n
            val_n = 0
            test_n = 0

    # Recompute to ensure sums match
    total = train_n + val_n + test_n
    if total != n:
        train_n += n - total  # adjust train to make sum exact

    train = shuffled[:train_n]
    val = shuffled[train_n:train_n + val_n]
    test = shuffled[train_n + val_n:train_n + val_n + test_n]

    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Rebuild train/val/test split from a flat dataset")
    parser.add_argument("--input-root", type=str, default="dataset_all",
                        help="Input directory with flat class subdirs (no split)")
    parser.add_argument("--output-root", type=str, default="dataset",
                        help="Output directory with train/val/test subdirs")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mapping", type=str, default=None,
                        help="Optional sample_mapping.json to use instead of auto-split")
    args = parser.parse_args()

    cfg = load_config(args.config)
    label_mapping_cfg = cfg.get("label_mapping", {})

    summary = {
        "input_root": os.path.abspath(args.input_root),
        "output_root": os.path.abspath(args.output_root),
        "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "seed": args.seed,
        "per_class": {},
        "warnings": [],
    }

    if args.mapping:
        # Use mapping file to determine splits
        with open(args.mapping) as f:
            mappings = json.load(f)
        # TODO: implement mapping-based split if needed
        print("Mapping-based split not yet implemented, using auto-split")
        # Fall through to auto-split

    # Scan input root for class directories
    input_root = args.input_root
    if not os.path.isdir(input_root):
        print(f"[ERROR] Input directory not found: {input_root}")
        sys.exit(1)

    class_dirs = sorted([
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    ])

    total_split_counts = {"train": 0, "val": 0, "test": 0}

    for class_dir in class_dirs:
        class_path = os.path.join(input_root, class_dir)
        files = sorted([
            f for f in os.listdir(class_path)
            if f.endswith(".npy")
        ])

        if not files:
            print(f"[SKIP] {class_dir}: no .npy files")
            continue

        # Resolve label: check if this class should be mapped to negative
        out_class_dir, is_negative = resolve_label(class_dir, label_mapping_cfg)
        if is_negative:
            print(f"  [MAP] {class_dir} -> negative")

        n = len(files)
        train_files, val_files, test_files = stratified_split(
            files, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
        )

        # Warnings for small classes
        if n == 1:
            msg = f"{class_dir} ({n} sample): only train, no val/test — cannot reliably evaluate"
            summary["warnings"].append(msg)
            print(f"  [WARN] {msg}")
        elif n == 2:
            msg = f"{class_dir} ({n} samples): train + val only, no test"
            summary["warnings"].append(msg)
            print(f"  [WARN] {msg}")

        # Copy files
        for split_name, split_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            dest_dir = os.path.join(args.output_root, split_name, out_class_dir)
            os.makedirs(dest_dir, exist_ok=True)
            for fname in split_files:
                src = os.path.join(class_path, fname)
                dst = os.path.join(dest_dir, fname)
                shutil.copy2(src, dst)
            total_split_counts[split_name] += len(split_files)

        summary["per_class"][class_dir] = {
            "output_dir": out_class_dir,
            "total": n,
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
            "is_negative": is_negative,
        }

        print(f"  {class_dir} -> {out_class_dir}: "
              f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)} (total={n})")

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"Split Summary")
    print(f"{'=' * 50}")
    for split_name in ("train", "val", "test"):
        print(f"  {split_name}: {total_split_counts[split_name]} samples")
    print(f"  Total: {sum(total_split_counts.values())}")

    if summary["warnings"]:
        print(f"\nWarnings ({len(summary['warnings'])}):")
        for w in summary["warnings"]:
            print(f"  [WARN] {w}")

    # Save summary
    summary["total"] = total_split_counts
    os.makedirs("outputs/reports", exist_ok=True)
    out_path = "outputs/reports/split_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSplit summary saved to {out_path}")


if __name__ == "__main__":
    main()
