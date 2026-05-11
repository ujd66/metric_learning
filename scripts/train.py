import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.pointcloud_dataset import PointCloudDataset
from src.metrics.classification_metrics import compute_metrics
from src.models.metric_model import MetricPointNet
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def collate_fn(batch):
    points = torch.stack([b["points"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"points": points, "label": labels}


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    num_samples = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        points = batch["points"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        out = model(points)
        loss = criterion(out["logits"], labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * points.size(0)
        num_samples += points.size(0)
    return total_loss / max(num_samples, 1)


@torch.no_grad()
def evaluate(model, loader, device, num_known_classes, negative_label):
    model.eval()
    all_labels = []
    all_preds = []
    for batch in tqdm(loader, desc="Val", leave=False):
        points = batch["points"].to(device)
        out = model(points)
        preds = out["logits"].argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch["label"].numpy())
    metrics = compute_metrics(
        all_labels, all_preds,
        num_known_classes=num_known_classes,
        negative_label=negative_label,
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train PointNet metric learning model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    logger = get_logger("train", "outputs/logs/train.log")

    logger.info(f"Config: {cfg}")
    logger.info(f"Device: {device}")

    train_ds = PointCloudDataset(
        root_dir=cfg["data"]["root"],
        split=cfg["data"]["train_split"],
        num_points=cfg["num_points"],
        input_channels=cfg["input_channels"],
        augmentation_config=cfg.get("augmentation", {}),
    )
    val_ds = PointCloudDataset(
        root_dir=cfg["data"]["root"],
        split=cfg["data"]["val_split"],
        num_points=cfg["num_points"],
        input_channels=cfg["input_channels"],
        augmentation_config={},
    )
    logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=4, collate_fn=collate_fn)

    model = MetricPointNet(
        input_channels=cfg["input_channels"],
        num_classes=cfg["num_classes"],
        embedding_dim=cfg["embedding_dim"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    start_epoch = 0

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device, cfg["num_known_classes"], cfg["negative_label"])

        logger.info(
            f"Epoch {epoch+1}/{cfg['train']['epochs']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_metrics['overall_accuracy']:.4f} | "
            f"Val Macro F1: {val_metrics['macro_f1']:.4f} | "
            f"Known Acc: {val_metrics['known_class_accuracy']:.4f} | "
            f"Neg Acc: {val_metrics['negative_accuracy']:.4f}"
        )

        save_checkpoint(model, optimizer, epoch + 1, "outputs/checkpoints/last.pt", metrics=val_metrics)

        if val_metrics["overall_accuracy"] > best_acc:
            best_acc = val_metrics["overall_accuracy"]
            save_checkpoint(model, optimizer, epoch + 1, "outputs/checkpoints/best.pt", metrics=val_metrics)
            logger.info(f"New best model saved with accuracy {best_acc:.4f}")

    logger.info(f"Training complete. Best accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
