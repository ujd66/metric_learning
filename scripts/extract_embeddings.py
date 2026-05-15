"""Extract embeddings from a dataset split and save as .npz.

Used by the regression pipeline to generate embeddings for evaluation.

Usage:
    python scripts/extract_embeddings.py \
        --config configs/config.yaml \
        --checkpoint outputs/checkpoints/best.pt \
        --split test \
        --output outputs/embeddings/test_embeddings.npz
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.pointcloud_dataset import PointCloudDataset
from src.models.metric_model import MetricPointNet
from src.utils.checkpoint import load_checkpoint
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
        all_embeddings.append(out["embedding"].cpu().numpy())
        all_labels.extend(batch["label"].tolist())
        all_sample_ids.extend(batch["sample_id"])
        all_class_names.extend(batch["class_name"])
    return (np.concatenate(all_embeddings, axis=0),
            np.array(all_labels),
            np.array(all_sample_ids),
            list(all_class_names))


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from a dataset split")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", type=str, required=True, help="Output .npz path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    model = MetricPointNet(
        input_channels=cfg["input_channels"],
        num_classes=cfg["num_classes"],
        embedding_dim=cfg["embedding_dim"],
    ).to(device)
    load_checkpoint(args.checkpoint, model)

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

    print(f"Extracting embeddings from {len(dataset)} samples ({args.split}) ...")
    embeddings, labels, sample_ids, class_names = extract_embeddings(model, loader, device)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output,
             embeddings=embeddings,
             labels=labels,
             sample_ids=sample_ids,
             class_names=class_names)
    print(f"Saved: {args.output} ({embeddings.shape[0]} samples, dim={embeddings.shape[1]})")


if __name__ == "__main__":
    main()
