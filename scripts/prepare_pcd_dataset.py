import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.utils.pcd_io import read_pcd, read_pcd_with_intensity, get_pcd_info

_SUPPORTED_EXTS = (".pcd",)


def _parse_label_from_dirname(dirname):
    if dirname == "negative":
        return -1  # will be resolved later
    m = re.match(r"class_(\d+)", dirname)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse label from directory name: {dirname}")


def resolve_class_dir(class_dir, label_mapping_cfg):
    """Check if a class_dir should be mapped to negative.

    Returns (output_class_dir, label).
    """
    negative_names = set(label_mapping_cfg.get("negative_names", ["negative"]))
    force_neg_label = label_mapping_cfg.get("force_negative_label", 19)

    # Check if class_dir itself is a negative name
    if class_dir.lower() in negative_names:
        return "negative", force_neg_label

    # Check class_mapping.json
    class_mapping_path = os.path.join("raw_pcd_dataset", "class_mapping.json")
    if os.path.exists(class_mapping_path):
        with open(class_mapping_path) as f:
            class_mapping = json.load(f)
        class_name = class_mapping.get(class_dir, "")
        if class_name.lower() in negative_names:
            return "negative", force_neg_label

    # Normal class
    try:
        label = _parse_label_from_dirname(class_dir)
    except ValueError:
        return None, None
    return class_dir, label


def convert_single(pcd_path, label, class_name, sample_id, use_intensity=False):
    if use_intensity:
        points = read_pcd_with_intensity(pcd_path)
    else:
        points = read_pcd(pcd_path)

    pcd_info = get_pcd_info(pcd_path)

    record = {
        "points": points,
        "label": label,
        "class_name": class_name,
        "sample_id": sample_id,
        "source_pcd": os.path.abspath(pcd_path),
        "heightmap_path": pcd_info["heightmap_path"],
        "info_path": pcd_info["info_path"],
        "transform_path": pcd_info["transform_path"],
    }
    return record


def convert_flat(input_root, output_root, label_mapping_cfg, use_intensity=False):
    """Convert all PCDs from input_root into a flat output (no split subdirs).

    Supports two input layouts:
    1. raw_pcd_dataset/ with train/val/test subdirs containing class_xxx/
    2. Flat layout with class dirs directly (e.g., label/ with class name dirs)

    Applies label mapping: negative names -> negative dir with force_negative_label.
    """
    total = 0
    errors = []

    # Detect input layout: does it have split subdirs?
    top_dirs = sorted([
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    ])

    # Check if top-level dirs look like splits (train/val/test) or like classes
    split_names = {"train", "val", "test"}
    has_splits = any(d in split_names for d in top_dirs)
    has_class_dirs = any(d.startswith("class_") or d == "negative" for d in top_dirs)

    if has_splits and not has_class_dirs:
        # Layout 1: input_root/train/class_xxx/*.pcd
        scan_roots = []
        for split_name in ("train", "val", "test"):
            split_path = os.path.join(input_root, split_name)
            if os.path.isdir(split_path):
                scan_roots.append(split_path)
    else:
        # Layout 2: input_root/class_xxx/*.pcd (possibly nested)
        scan_roots = [input_root]

    # Collect class dirs from all scan roots (allow duplicates across splits)
    class_entries = []  # (class_dir, class_path)

    for scan_root in scan_roots:
        for entry in sorted(os.listdir(scan_root)):
            entry_path = os.path.join(scan_root, entry)
            if not os.path.isdir(entry_path):
                continue
            class_entries.append((entry, entry_path))

    for class_dir, class_path in class_entries:
        out_class_dir, label = resolve_class_dir(class_dir, label_mapping_cfg)

        if out_class_dir is None:
            print(f"[SKIP] Cannot resolve class: {class_dir}")
            continue

        out_path_dir = os.path.join(output_root, out_class_dir)
        os.makedirs(out_path_dir, exist_ok=True)

        # Scan for PCD files (may be in subdirectories like 000018/000018_0001_crop.pcd)
        pcd_files = []
        for root, dirs, files in os.walk(class_path):
            for fname in files:
                if fname.endswith(_SUPPORTED_EXTS):
                    pcd_files.append(os.path.join(root, fname))
        pcd_files.sort()

        converted = 0
        for pcd_path in pcd_files:
            sample_id = os.path.splitext(os.path.basename(pcd_path))[0]
            out_file = os.path.join(out_path_dir, f"{sample_id}.npy")
            # Handle filename collision across splits
            if os.path.exists(out_file):
                # Derive a short hash from the source path for disambiguation
                import hashlib
                h = hashlib.md5(pcd_path.encode()).hexdigest()[:6]
                sample_id = f"{sample_id}_{h}"
                out_file = os.path.join(out_path_dir, f"{sample_id}.npy")

            try:
                record = convert_single(
                    pcd_path, label, out_class_dir, sample_id,
                    use_intensity=use_intensity,
                )
                np.save(out_file, record)
                converted += 1
            except Exception as e:
                errors.append((pcd_path, str(e)))
                print(f"  [ERROR] {pcd_path}: {e}")

        print(f"[{class_dir} -> {out_class_dir}] converted {converted} files")
        total += converted

    return total, errors


def convert_split(input_dir, output_dir, splits, use_intensity=False):
    """Original mode: convert from raw_pcd_dataset/<split>/<class>/ to dataset/<split>/<class>/."""
    total = 0
    errors = []

    for split in splits:
        split_in = os.path.join(input_dir, split)
        split_out = os.path.join(output_dir, split)

        if not os.path.isdir(split_in):
            print(f"[SKIP] {split_in} does not exist")
            continue

        for class_dir in sorted(os.listdir(split_in)):
            class_path_in = os.path.join(split_in, class_dir)
            if not os.path.isdir(class_path_in):
                continue

            try:
                label = _parse_label_from_dirname(class_dir)
            except ValueError as e:
                print(f"[SKIP] {e}")
                continue

            class_path_out = os.path.join(split_out, class_dir)
            os.makedirs(class_path_out, exist_ok=True)

            converted = 0
            for fname in sorted(os.listdir(class_path_in)):
                if not fname.endswith(_SUPPORTED_EXTS):
                    continue

                pcd_path = os.path.join(class_path_in, fname)
                sample_id = os.path.splitext(fname)[0]
                out_path = os.path.join(class_path_out, f"{sample_id}.npy")

                try:
                    record = convert_single(
                        pcd_path, label, class_dir, sample_id,
                        use_intensity=use_intensity,
                    )
                    np.save(out_path, record)
                    converted += 1
                except Exception as e:
                    errors.append((pcd_path, str(e)))
                    print(f"  [ERROR] {pcd_path}: {e}")

            print(f"[{split}/{class_dir}] converted {converted} files")
            total += converted

    return total, errors


def main():
    parser = argparse.ArgumentParser(description="Convert PCD dataset to .npy dict format")
    parser.add_argument(
        "--input-dir", type=str, default=None,
        help="Input directory (raw_pcd_dataset with splits) — legacy mode",
    )
    parser.add_argument(
        "--input-root", type=str, default=None,
        help="Input root directory for --flat mode (e.g., raw_pcd_dataset)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory — legacy mode",
    )
    parser.add_argument(
        "--output-root", type=str, default=None,
        help="Output root directory — flat mode",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        help="Which splits to convert (legacy mode only)",
    )
    parser.add_argument(
        "--use_intensity", action="store_true",
        help="Include intensity channel (xyz + intensity -> 4 channels)",
    )
    parser.add_argument(
        "--flat", action="store_true",
        help="Flat mode: convert all PCDs into a single pool without split subdirs",
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config.yaml (for label_mapping)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    label_mapping_cfg = cfg.get("label_mapping", {})

    if args.flat:
        input_root = args.input_root or args.input_dir or "raw_pcd_dataset"
        output_root = args.output_root or args.output_dir or "dataset_all"
        print(f"Flat mode: {input_root} -> {output_root}")
        total, errors = convert_flat(
            input_root, output_root, label_mapping_cfg,
            use_intensity=args.use_intensity,
        )
    else:
        input_dir = args.input_dir or "raw_pcd_dataset"
        output_dir = args.output_dir or args.output_root or "dataset"
        print(f"Split mode: {input_dir} -> {output_dir}")
        total, errors = convert_split(
            input_dir, output_dir, args.splits,
            use_intensity=args.use_intensity,
        )

    print(f"\nDone. Total: {total} files converted.")
    if errors:
        print(f"Errors: {len(errors)}")
        for path, err in errors:
            print(f"  {path}: {err}")


if __name__ == "__main__":
    main()
