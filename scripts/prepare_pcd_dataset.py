import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.pcd_io import read_pcd, read_pcd_with_intensity, get_pcd_info

_SUPPORTED_EXTS = (".pcd",)


def _parse_label_from_dirname(dirname):
    if dirname == "negative":
        return 19
    m = re.match(r"class_(\d+)", dirname)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse label from directory name: {dirname}")


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


def main():
    parser = argparse.ArgumentParser(description="Convert PCD dataset to .npy dict format")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="raw_pcd_dataset",
        help="Root directory of raw PCD dataset (contains train/val/test splits)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset",
        help="Output directory for .npy files",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Which splits to convert",
    )
    parser.add_argument(
        "--use_intensity",
        action="store_true",
        help="Include intensity channel (xyz + intensity -> 4 channels)",
    )
    args = parser.parse_args()

    total = 0
    errors = []

    for split in args.splits:
        split_in = os.path.join(args.input_dir, split)
        split_out = os.path.join(args.output_dir, split)

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
                        use_intensity=args.use_intensity,
                    )
                    np.save(out_path, record)
                    converted += 1
                except Exception as e:
                    errors.append((pcd_path, str(e)))
                    print(f"  [ERROR] {pcd_path}: {e}")

            print(f"[{split}/{class_dir}] converted {converted} files")
            total += converted

    print(f"\nDone. Total: {total} files converted.")
    if errors:
        print(f"Errors: {len(errors)}")
        for path, err in errors:
            print(f"  {path}: {err}")


if __name__ == "__main__":
    main()
