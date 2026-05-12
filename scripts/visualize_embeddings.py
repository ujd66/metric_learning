import argparse
import json
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


def plot_tsne(embeddings, labels, class_names, negative_label, save_path):
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    coords = tsne.fit_transform(embeddings)

    unique_labels = sorted(set(labels))

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot known classes first
    for lbl in unique_labels:
        mask = labels == lbl
        name = class_names.get(str(lbl), f"class_{lbl}")
        is_neg = (lbl == negative_label)

        if is_neg:
            # Plot negative with different marker
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       label=f"{name} (negative)", marker="x", s=80, c="gray", alpha=0.7, zorder=5)
        else:
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       label=name, s=30, alpha=0.7)

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    ax.set_title("t-SNE Embedding Visualization")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"t-SNE plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize embeddings with t-SNE")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    num_classes = cfg["num_classes"]
    negative_label = cfg["negative_label"]

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

    print(f"Extracting embeddings from {len(dataset)} samples ({args.split} split)")

    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=4, collate_fn=collate_fn)

    model = MetricPointNet(
        input_channels=cfg["input_channels"],
        num_classes=num_classes,
        embedding_dim=cfg["embedding_dim"],
    ).to(device)
    load_checkpoint(args.checkpoint, model)

    embeddings, labels = extract_embeddings(model, loader, device)

    # Load class names
    class_names_path = os.path.join(os.path.dirname(args.config), "class_names.json")
    if os.path.exists(class_names_path):
        with open(class_names_path) as f:
            class_names = json.load(f)
    else:
        class_names = {str(i): f"class_{i:03d}" for i in range(num_classes)}

    # Save embeddings
    os.makedirs("outputs/reports", exist_ok=True)
    np.save("outputs/reports/embedding_features.npy", embeddings)
    np.save("outputs/reports/embedding_labels.npy", labels)
    print(f"Embeddings saved: features {embeddings.shape}, labels {labels.shape}")

    # Plot
    save_path = f"outputs/reports/embedding_tsne_{args.split}.png"
    plot_tsne(embeddings, labels, class_names, negative_label, save_path)


if __name__ == "__main__":
    main()
