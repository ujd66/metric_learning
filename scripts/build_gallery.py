"""Build a gallery of embeddings for retrieval-based inference.

Extracts embeddings from a split (default: train) and saves them as a
gallery file for use in query-gallery retrieval evaluation.

Usage:
    python scripts/build_gallery.py \
        --config configs/config.yaml \
        --checkpoint outputs/checkpoints/best.pt \
        --split train \
        --output outputs/gallery/baseline_train_gallery.pt
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.pointcloud_dataset import PointCloudDataset
from src.models.model_factory import build_model_from_checkpoint
from src.utils.config import load_config


def collate_fn(batch):
    points = torch.stack([b["points"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    sample_ids = [b["sample_id"] for b in batch]
    class_names = [b["class_name"] for b in batch]
    return {"points": points, "label": labels, "sample_id": sample_ids, "class_name": class_names}


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    all_embeddings = []
    all_labels = []
    all_sample_ids = []
    all_class_names = []
    for batch in tqdm(loader, desc="Extract"):
        points = batch["points"].to(device)
        out = model(points)
        all_embeddings.append(out["embedding"].cpu())
        all_labels.extend(batch["label"].tolist())
        all_sample_ids.extend(batch["sample_id"])
        all_class_names.extend(batch["class_name"])
    embeddings = torch.cat(all_embeddings, dim=0)
    return embeddings, all_labels, all_sample_ids, all_class_names


def main():
    parser = argparse.ArgumentParser(description="Build gallery of embeddings for retrieval")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: outputs/gallery/<checkpoint_name>_<split>_gallery.pt)")
    parser.add_argument("--include-negative", action="store_true", default=False,
                        help="Include negative samples in gallery (default: exclude)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    num_known_classes = cfg["num_known_classes"]
    negative_label = cfg["negative_label"]

    # Supported / unsupported classes
    supported_cfg = cfg.get("supported_classes", {})
    supported_known_labels = set(supported_cfg.get("supported_known_labels", list(range(num_known_classes))))
    unsupported_known_labels = supported_cfg.get("unsupported_known_labels", [])

    if args.output is None:
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output = f"outputs/gallery/{ckpt_name}_{args.split}_gallery.pt"

    # Load model
    print(f"Loading model from {args.checkpoint} ...")
    model, _ = build_model_from_checkpoint(cfg, args.checkpoint)
    model = model.to(device)

    # Load dataset
    dataset = PointCloudDataset(
        root_dir=cfg["data"]["root"],
        split=args.split,
        num_points=cfg["num_points"],
        input_channels=cfg["input_channels"],
        augmentation_config={},
    )
    if len(dataset) == 0:
        print(f"[ERROR] No samples found in {args.split} split")
        sys.exit(1)

    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        num_workers=4, collate_fn=collate_fn)

    # Extract embeddings
    print(f"Extracting embeddings from {len(dataset)} samples ({args.split} split) ...")
    embeddings, labels, sample_ids, class_names = extract_embeddings(model, loader, device)

    # L2 normalize
    norms = embeddings.norm(dim=1, keepdim=True)
    norms = norms.clamp(min=1e-8)
    embeddings = embeddings / norms

    print(f"  Total samples extracted: {len(labels)}")

    # Filter by class
    if not args.include_negative:
        keep_mask = [l < num_known_classes and l in supported_known_labels for l in labels]
        n_negative = sum(1 for l in labels if l == negative_label)
        n_unsupported = sum(1 for l in labels if l < num_known_classes and l not in supported_known_labels)
        n_removed = sum(1 for k in keep_mask if not k)
    else:
        keep_mask = [l not in unsupported_known_labels or l >= num_known_classes for l in labels]
        n_negative = sum(1 for l in labels if l == negative_label)
        n_unsupported = sum(1 for l in labels if l in unsupported_known_labels)
        n_removed = sum(1 for k in keep_mask if not k)

    embeddings = embeddings[keep_mask]
    labels_filtered = [l for l, k in zip(labels, keep_mask) if k]
    sample_ids_filtered = [s for s, k in zip(sample_ids, keep_mask) if k]
    class_names_filtered = [c for c, k in zip(class_names, keep_mask) if k]

    print(f"  Gallery samples (known only): {len(labels_filtered)}")
    if n_removed > 0:
        print(f"  Excluded {n_removed} samples (negative + unsupported)")
    if unsupported_known_labels:
        print(f"  Unsupported classes excluded: {unsupported_known_labels}")

    # Class distribution
    class_counts = {}
    for l in labels_filtered:
        class_counts[l] = class_counts.get(l, 0) + 1
    print(f"  Classes in gallery: {len(class_counts)}")
    for c in sorted(class_counts.keys()):
        cn = class_names_filtered[labels_filtered.index(c)] if c in labels_filtered else f"class_{c:03d}"
        print(f"    {cn}: {class_counts[c]}")

    # Save
    gallery = {
        "embeddings": embeddings,
        "labels": torch.tensor(labels_filtered, dtype=torch.long),
        "class_names": class_names_filtered,
        "sample_ids": sample_ids_filtered,
        "source_paths": [],  # not available from dataset
        "num_known_classes": num_known_classes,
        "negative_label": negative_label,
        "embedding_dim": cfg["embedding_dim"],
        "checkpoint": args.checkpoint,
        "split": args.split,
        "include_negative": args.include_negative,
        "created_at": datetime.now().isoformat(),
        "supported_known_labels": sorted(supported_known_labels),
        "unsupported_known_labels": unsupported_known_labels,
        "incomplete_known_class_coverage": len(unsupported_known_labels) > 0,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(gallery, args.output)
    print(f"\nSaved gallery: {args.output}")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Labels shape: {gallery['labels'].shape}")


if __name__ == "__main__":
    main()
