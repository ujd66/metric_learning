"""Build per-class prototypes from a trained model's embeddings.

Extracts embeddings from the specified split (e.g. train), computes the
mean embedding for each known class (0..num_known_classes-1), L2-normalizes
the result, and saves as prototypes.pt.

Usage:
    python scripts/build_prototypes.py \
        --config configs/config.yaml \
        --checkpoint outputs/checkpoints/best.pt \
        --split train \
        --output outputs/prototypes/baseline_prototypes.pt
"""

import argparse
import json
import os
import sys
import warnings
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
    return {"points": points, "label": labels}


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    all_embeddings = []
    all_labels = []
    for batch in tqdm(loader, desc="Extract"):
        points = batch["points"].to(device)
        out = model(points)
        all_embeddings.append(out["embedding"].cpu().numpy())
        all_labels.extend(batch["label"].numpy())
    return np.concatenate(all_embeddings, axis=0), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description="Build class prototypes from model embeddings")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: outputs/prototypes/<checkpoint_name>_prototypes.pt)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    num_known_classes = cfg["num_known_classes"]
    negative_label = cfg["negative_label"]
    embedding_dim = cfg["embedding_dim"]

    # Supported / unsupported classes
    supported_cfg = cfg.get("supported_classes", {})
    supported_known_labels = supported_cfg.get("supported_known_labels", list(range(num_known_classes)))
    unsupported_known_labels = supported_cfg.get("unsupported_known_labels", [])

    # Determine output path
    if args.output is None:
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output = f"outputs/prototypes/{ckpt_name}_prototypes.pt"

    # Load model
    model, _ = build_model_from_checkpoint(cfg, args.checkpoint)
    model = model.to(device)

    # Load dataset
    dataset = PointCloudDataset(
        root_dir=cfg["data"]["root"],
        split=args.split,
        num_points=cfg["num_points"],
        input_channels=cfg["input_channels"],
        augmentation_config={},  # no augmentation for prototype building
    )
    if len(dataset) == 0:
        print(f"[ERROR] No samples found in {args.split} split")
        sys.exit(1)

    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        num_workers=4, collate_fn=collate_fn)

    # Extract embeddings
    print(f"Extracting embeddings from {len(dataset)} samples ({args.split} split) ...")
    embeddings, labels = extract_embeddings(model, loader, device)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Build prototypes for known classes only
    # Only build prototypes for supported known labels
    num_supported = len(supported_known_labels)
    print(f"Building prototypes for {num_supported} supported known classes (out of {num_known_classes} total) ...")

    # Load class names early for logging
    class_names_path = os.path.join(os.path.dirname(args.config), "class_names.json")
    if os.path.exists(class_names_path):
        with open(class_names_path) as f:
            name_map = json.load(f)
        class_names = [name_map.get(str(i), f"class_{i:03d}") for i in range(num_known_classes)]
    else:
        class_names = [f"class_{i:03d}" for i in range(num_known_classes)]

    if unsupported_known_labels:
        unsup_names = [class_names[c] for c in unsupported_known_labels]
        print(f"  Unsupported known classes: {unsupported_known_labels} ({unsup_names})")

    prototypes = np.zeros((num_known_classes, embedding_dim), dtype=np.float32)
    class_support = {}
    incomplete = False

    for c in range(num_known_classes):
        if c in unsupported_known_labels:
            class_support[c] = 0
            print(f"  [SKIP] Class {c} ({class_names[c]}) is unsupported — no prototype built")
            continue
        mask = labels == c
        count = int(mask.sum())
        class_support[c] = count
        if count == 0:
            warnings.warn(f"Class {c} ({class_names[c]}) has no samples in {args.split} — prototype will be zero vector")
            incomplete = True
            continue
        if count == 1:
            warnings.warn(f"Class {c} ({class_names[c]}) has only 1 sample — prototype may be unreliable")
        prototypes[c] = embeddings[mask].mean(axis=0)

    # L2 normalize prototypes
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    prototypes = prototypes / norms

    # Unsupported class names
    unsupported_class_names = [class_names[c] for c in unsupported_known_labels]
    if unsupported_known_labels:
        incomplete = True

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        "prototypes": torch.tensor(prototypes, dtype=torch.float32),
        "class_names": class_names,
        "num_known_classes": num_known_classes,
        "negative_label": negative_label,
        "source_split": args.split,
        "checkpoint": args.checkpoint,
        "class_support": class_support,
        "created_at": datetime.now().isoformat(),
        "similarity_threshold": None,
        "per_class_thresholds": None,
        "supported_known_labels": supported_known_labels,
        "unsupported_known_labels": unsupported_known_labels,
        "unsupported_class_names": unsupported_class_names,
        "incomplete_known_class_coverage": incomplete,
    }, args.output)

    # Summary
    print(f"\nPrototypes saved to: {args.output}")
    print(f"  Known classes: {num_known_classes}")
    print(f"  Embedding dim: {embedding_dim}")
    for c in range(num_known_classes):
        status = "OK" if class_support[c] >= 2 else ("1 sample" if class_support[c] == 1 else "EMPTY")
        print(f"  Class {c} ({class_names[c]}): {class_support[c]} samples [{status}]")


if __name__ == "__main__":
    main()
