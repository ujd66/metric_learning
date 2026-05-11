"""Organize labeled PCD dataset into train/val/test split with class_xxx naming.

Reads from dataset/label/{chinese_class_name}/{scene_id}/*_crop.pcd
Writes to   raw_pcd_dataset/{train|val|test}/{class_xxx}/*.pcd

Split ratio: 80% train, 10% val, 10% test (per-class, stratified).
Classes with < 3 samples go entirely to train.
"""

import json
import os
import random
import shutil

LABEL_DIR = "dataset/label"
OUTPUT_DIR = "raw_pcd_dataset"
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# TEST_RATIO = 1 - TRAIN_RATIO - VAL_RATIO = 0.1


def main():
    random.seed(SEED)

    # Sorted class names -> class_xxx mapping
    class_names = sorted(os.listdir(LABEL_DIR))
    name_to_id = {}
    for idx, name in enumerate(class_names):
        name_to_id[name] = idx

    print(f"Found {len(class_names)} classes:")
    for name in class_names:
        cid = name_to_id[name]
        print(f"  class_{cid:03d} -> {name}")
    print()

    # Collect all samples per class
    class_samples = {}
    for class_name in class_names:
        class_dir = os.path.join(LABEL_DIR, class_name)
        samples = []
        for scene_dir in sorted(os.listdir(class_dir)):
            scene_path = os.path.join(class_dir, scene_dir)
            if not os.path.isdir(scene_path):
                continue
            for fname in sorted(os.listdir(scene_path)):
                if fname.endswith("_crop.pcd"):
                    samples.append(os.path.join(scene_path, fname))
        class_samples[class_name] = samples

    # Print distribution
    total = 0
    for name in class_names:
        n = len(class_samples[name])
        total += n
        print(f"  {name}: {n}")
    print(f"  Total: {total}\n")

    # Split per class
    split_counts = {"train": 0, "val": 0, "test": 0}
    mapping = []

    for class_name in class_names:
        cid = name_to_id[class_name]
        class_dir_name = f"class_{cid:03d}"
        samples = class_samples[class_name]
        random.shuffle(samples)
        n = len(samples)

        if n < 3:
            train_s, val_s, test_s = samples, [], []
        else:
            n_train = max(1, round(n * TRAIN_RATIO))
            n_val = max(1, round(n * VAL_RATIO))
            # Ensure at least 1 in test if enough samples
            if n - n_train - n_val < 1 and n >= 3:
                n_val = n - n_train - 1
            train_s = samples[:n_train]
            val_s = samples[n_train:n_train + n_val]
            test_s = samples[n_train + n_val:]

        for split, split_samples in [("train", train_s), ("val", val_s), ("test", test_s)]:
            out_dir = os.path.join(OUTPUT_DIR, split, class_dir_name)
            os.makedirs(out_dir, exist_ok=True)
            for src_path in split_samples:
                src_fname = os.path.basename(src_path)
                dst_path = os.path.join(out_dir, src_fname)
                shutil.copy2(src_path, dst_path)
                mapping.append({
                    "source_pcd": os.path.abspath(src_path),
                    "dest_pcd": os.path.abspath(dst_path),
                    "class_name": class_name,
                    "class_id": cid,
                    "class_dir": class_dir_name,
                    "split": split,
                })
                # Copy companion files (png, json) if they exist
                base = src_path.replace("_crop.pcd", "")
                for ext in ["_crop.png", "_meta.json"]:
                    companion = base + ext.replace("_crop", "")
                    # Correct companion naming
                    if ext == "_crop.png":
                        companion = src_path.replace("_crop.pcd", "_crop.png")
                    else:
                        companion = src_path.replace("_crop.pcd", "_meta.json")
                    if os.path.exists(companion):
                        dst_companion = os.path.join(out_dir, os.path.basename(companion))
                        shutil.copy2(companion, dst_companion)

            split_counts[split] += len(split_samples)

    # Save mapping
    mapping_path = os.path.join(OUTPUT_DIR, "class_mapping.json")
    class_mapping = {f"class_{name_to_id[n]:03d}": n for n in class_names}
    with open(mapping_path, "w") as f:
        json.dump(class_mapping, f, indent=2, ensure_ascii=False)
    print(f"Saved class mapping to {mapping_path}")

    # Save sample mapping
    sample_map_path = os.path.join(OUTPUT_DIR, "sample_mapping.json")
    with open(sample_map_path, "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"\nSplit results:")
    print(f"  train: {split_counts['train']}")
    print(f"  val:   {split_counts['val']}")
    print(f"  test:  {split_counts['test']}")
    print(f"  total: {sum(split_counts.values())}")
    print(f"\nDone. Output in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
