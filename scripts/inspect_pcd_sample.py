import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.pcd_io import read_pcd, read_pcd_with_intensity, get_pcd_info, read_pcd_header


def main():
    parser = argparse.ArgumentParser(description="Inspect a single PCD sample")
    parser.add_argument("--input", type=str, required=True, help="Path to .pcd file")
    args = parser.parse_args()

    path = args.input
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"=== PCD File: {path} ===\n")

    header = read_pcd_header(path)
    print("--- Header ---")
    for k, v in header.items():
        print(f"  {k}: {v}")
    print()

    points = read_pcd(path)
    print(f"--- Points (XYZ only) ---")
    print(f"  Shape: {points.shape}")
    print(f"  Dtype: {points.dtype}")
    print(f"  Min:   {points.min(axis=0)}")
    print(f"  Max:   {points.max(axis=0)}")
    print(f"  Mean:  {points.mean(axis=0)}")
    print(f"  Std:   {points.std(axis=0)}")
    print()

    try:
        points_i = read_pcd_with_intensity(path)
        if points_i.shape[1] == 4:
            print(f"--- Intensity ---")
            print(f"  Min:  {points_i[:, 3].min():.6f}")
            print(f"  Max:  {points_i[:, 3].max():.6f}")
            print(f"  Mean: {points_i[:, 3].mean():.6f}")
            print()
    except Exception as e:
        print(f"  [WARN] Could not read intensity: {e}\n")

    info = get_pcd_info(path)
    print("--- Companion Files ---")
    for key in ("heightmap_path", "info_path", "transform_path"):
        val = info[key]
        status = val if val else "(not found)"
        print(f"  {key}: {status}")
    if info["info_text"]:
        print(f"\n--- Info Text ---")
        print(info["info_text"])
    if info["transform"]:
        import json
        print(f"\n--- Transform ---")
        print(json.dumps(info["transform"], indent=2))


if __name__ == "__main__":
    main()
