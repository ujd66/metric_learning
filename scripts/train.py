import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
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


def compute_class_weights(dataset, num_classes, max_weight=10.0):
    """Compute inverse-frequency class weights with clipping.

    weight[i] = total_samples / (num_classes * count[i])
    Clipped to [1.0, max_weight].
    """
    labels = [dataset.samples[i]["label"] for i in range(len(dataset))]
    counts = Counter(labels)
    total = len(labels)

    weights = []
    for c in range(num_classes):
        cnt = counts.get(c, 0)
        if cnt == 0:
            weights.append(1.0)
        else:
            w = total / (num_classes * cnt)
            w = min(w, max_weight)
            w = max(w, 1.0)
            weights.append(w)

    return weights, counts


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
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
        preds = out["logits"].argmax(dim=1)
        correct += (preds == labels).sum().item()
        num_samples += points.size(0)
    avg_loss = total_loss / max(num_samples, 1)
    accuracy = correct / max(num_samples, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_known_classes, negative_label):
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0.0
    num_samples = 0
    for batch in tqdm(loader, desc="Val", leave=False):
        points = batch["points"].to(device)
        labels = batch["label"].to(device)
        out = model(points)
        loss = criterion(out["logits"], labels)
        total_loss += loss.item() * points.size(0)
        num_samples += points.size(0)
        preds = out["logits"].argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(num_samples, 1)
    metrics = compute_metrics(
        all_labels, all_preds,
        num_known_classes=num_known_classes,
        negative_label=negative_label,
    )
    metrics["val_loss"] = avg_loss
    return metrics


def save_per_class_metrics(metrics, num_known_classes, negative_label, class_names_path, save_path):
    """Save per-class metrics to JSON, handling missing classes gracefully."""
    per_class = {}
    num_classes = num_known_classes + 1

    class_name_map = {}
    if os.path.exists(class_names_path):
        with open(class_names_path) as f:
            class_name_map = json.load(f)

    pc = metrics.get("per_class", {})
    for c in range(num_classes):
        name = class_name_map.get(str(c), f"class_{c:03d}")
        entry = pc.get(str(c), {})
        is_negative = (c == negative_label)

        per_class[name] = {
            "label": c,
            "is_negative": is_negative,
            "support": entry.get("support", 0),
            "precision": entry.get("precision"),
            "recall": entry.get("recall"),
            "f1": entry.get("f1"),
        }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(per_class, f, indent=2, ensure_ascii=False)


def plot_training_curves(history, save_path):
    """Plot training curves: loss, accuracy, F1."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train Loss")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, [h["train_acc"] for h in history], label="Train Acc")
    axes[1].plot(epochs, [h["val_acc"] for h in history], label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1
    axes[2].plot(epochs, [h["val_macro_f1"] for h in history], label="Val Macro F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1")
    axes[2].set_title("Val Macro F1")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train PointNet metric learning model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--overfit-small-batch", action="store_true",
                        help="Overfit a small batch to sanity-check the pipeline")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    logger = get_logger("train", "outputs/logs/train.log")

    logger.info(f"Config: {cfg}")
    logger.info(f"Device: {device}")

    # --- Dataset ---
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

    # --- Overfit small batch mode ---
    overfit_mode = args.overfit_small_batch
    if overfit_mode:
        num_overfit = min(32, len(train_ds))
        indices = list(range(num_overfit))
        train_ds = Subset(train_ds, indices)
        # Disable augmentation for overfit test
        logger.info(f"[OVERFIT MODE] Using {num_overfit} samples, no augmentation")
        epochs = 100
        batch_size = min(32, num_overfit)
    else:
        epochs = cfg["train"]["epochs"]
        batch_size = cfg["train"]["batch_size"]

    logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=4, collate_fn=collate_fn,
    )

    # --- Model ---
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

    # --- Class weight ---
    use_class_weight = cfg.get("train", {}).get("use_class_weight", False)
    max_weight = cfg.get("train", {}).get("class_weight_max", 10.0)
    # For overfit mode, use the underlying dataset if it's a Subset
    weight_dataset = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
    if use_class_weight:
        weights, counts = compute_class_weights(
            weight_dataset, cfg["num_classes"], max_weight=max_weight,
        )
        class_weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
        logger.info("Class weights (inverse frequency, clipped):")
        for c in range(cfg["num_classes"]):
            cnt = counts.get(c, 0)
            logger.info(f"  class_{c:03d}: weight={weights[c]:.2f}, count={cnt}")
    else:
        criterion = nn.CrossEntropyLoss()

    # --- Resume ---
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        epoch_loaded, _ = load_checkpoint(args.resume, model, optimizer)
        start_epoch = epoch_loaded
        logger.info(f"Resumed from epoch {start_epoch}")

    # --- Training loop ---
    best_val_f1 = 0.0
    history = []

    neg_label = cfg["negative_label"]
    num_known = cfg["num_known_classes"]

    for epoch in range(start_epoch, epochs):
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        if overfit_mode:
            # Also evaluate on same small set
            val_loss, val_acc = 0.0, 0.0
            val_metrics = {
                "val_loss": train_loss,
                "overall_accuracy": train_acc,
                "macro_precision": 0,
                "macro_recall": 0,
                "macro_f1": 0,
                "known_class_accuracy": train_acc,
                "negative_accuracy": 0,
                "per_class": {},
            }
            # Re-evaluate for actual metrics on the same data
            model.eval()
            all_l, all_p = [], []
            with torch.no_grad():
                for batch in train_loader:
                    pts = batch["points"].to(device)
                    lbls = batch["label"].to(device)
                    out = model(pts)
                    all_p.extend(out["logits"].argmax(1).cpu().numpy())
                    all_l.extend(lbls.cpu().numpy())
            val_metrics = compute_metrics(all_l, all_p, num_known_classes=num_known, negative_label=neg_label)
            val_metrics["val_loss"] = train_loss
            val_acc = val_metrics["overall_accuracy"]

            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Acc: {val_acc:.4f} | "
                f"LR: {lr:.6f}"
            )
        else:
            val_metrics = evaluate(model, val_loader, criterion, device, num_known, neg_label)
            val_acc = val_metrics["overall_accuracy"]
            val_loss = val_metrics["val_loss"]

            # Check if val has negative samples
            neg_acc = val_metrics.get("negative_accuracy", 0)
            neg_acc_str = f"{neg_acc:.4f}" if neg_acc > 0 or any(
                l == neg_label for l in [
                    val_ds.samples[i]["label"] for i in range(min(len(val_ds), 1))
                ]
            ) else "N/A"

            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                f"Macro P: {val_metrics['macro_precision']:.4f} "
                f"R: {val_metrics['macro_recall']:.4f} "
                f"F1: {val_metrics['macro_f1']:.4f} | "
                f"Known Acc: {val_metrics['known_class_accuracy']:.4f} | "
                f"Neg Acc: {neg_acc_str} | "
                f"LR: {lr:.6f}"
            )

        # Record history
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics.get("val_loss", 0),
            "val_acc": val_metrics["overall_accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_known_acc": val_metrics["known_class_accuracy"],
            "val_neg_acc": val_metrics.get("negative_accuracy", 0),
            "lr": lr,
        })

        # Save last checkpoint
        save_checkpoint(model, optimizer, epoch + 1, "outputs/checkpoints/last.pt", metrics=val_metrics)

        # Save best checkpoint by val macro F1 (or train acc in overfit mode)
        current_metric = val_metrics["overall_accuracy"] if overfit_mode else val_metrics["macro_f1"]
        if current_metric > best_val_f1:
            best_val_f1 = current_metric
            save_checkpoint(model, optimizer, epoch + 1, "outputs/checkpoints/best.pt", metrics=val_metrics)

            # Save per-class metrics for best checkpoint
            class_names_path = os.path.join(os.path.dirname(args.config), "class_names.json")
            save_per_class_metrics(
                val_metrics, num_known, neg_label,
                class_names_path, "outputs/reports/best_val_per_class_metrics.json",
            )

            logger.info(f"New best model saved (F1={'overfit' if overfit_mode else 'val'}={best_val_f1:.4f})")

    # --- Save training curves ---
    if history:
        plot_training_curves(history, "outputs/reports/training_curves.png")
        logger.info("Training curves saved to outputs/reports/training_curves.png")

        # Save history as JSON
        with open("outputs/reports/training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    if overfit_mode:
        final_acc = history[-1]["val_acc"] if history else 0
        logger.info(f"Overfit test complete. Final accuracy on {num_overfit} samples: {final_acc:.4f}")
        if final_acc > 0.9:
            logger.info("PASS: Model can overfit small batch (acc > 90%)")
        elif final_acc > 0.5:
            logger.info("PARTIAL: Model is learning but not fully converging. Check lr / epochs / augmentation.")
        else:
            logger.info("FAIL: Model cannot overfit small batch. Check data labels, preprocessing, or model code.")
    else:
        logger.info(f"Training complete. Best val macro F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
