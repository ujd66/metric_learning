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
from src.models.model_factory import build_model
from src.utils.checkpoint import load_checkpoint, save_checkpoint, save_checkpoint_with_config
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


def get_output_dir(cfg):
    """Get experiment-specific output directory."""
    experiment = cfg.get("experiment", {})
    subdir = experiment.get("output_subdir", "")
    if subdir:
        return os.path.join("outputs", "runs", subdir)
    return "outputs"


def train_one_epoch(
    model, loader, ce_criterion, optimizer, device,
    metric_loss_fn=None, ce_weight=1.0, metric_weight=0.0,
):
    """Train one epoch.

    Returns:
        (avg_total_loss, avg_ce_loss, avg_metric_loss, accuracy)
    """
    model.train()
    total_loss_sum = 0.0
    ce_loss_sum = 0.0
    metric_loss_sum = 0.0
    correct = 0
    num_samples = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        points = batch["points"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()

        out = model(points)
        ce_loss = ce_criterion(out["logits"], labels)

        metric_loss = torch.tensor(0.0, device=device)
        if metric_loss_fn is not None and metric_weight > 0:
            metric_loss = metric_loss_fn(out["embedding"], labels)

        loss = ce_weight * ce_loss + metric_weight * metric_loss

        # Guard against NaN
        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        optimizer.step()

        batch_size = points.size(0)
        total_loss_sum += loss.item() * batch_size
        ce_loss_sum += ce_loss.item() * batch_size
        metric_loss_sum += (metric_loss.item() if metric_loss.requires_grad else metric_loss.item()) * batch_size
        preds = out["logits"].argmax(dim=1)
        correct += (preds == labels).sum().item()
        num_samples += batch_size

    avg_total = total_loss_sum / max(num_samples, 1)
    avg_ce = ce_loss_sum / max(num_samples, 1)
    avg_metric = metric_loss_sum / max(num_samples, 1)
    accuracy = correct / max(num_samples, 1)
    return avg_total, avg_ce, avg_metric, accuracy


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


def plot_training_curves(history, save_path, has_metric_loss=False):
    """Plot training curves: loss, accuracy, F1, and optionally metric loss."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]

    ncols = 4 if has_metric_loss else 3
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

    # Loss
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train Loss")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Total Loss")
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

    # Metric loss (if applicable)
    if has_metric_loss:
        metric_losses = [h.get("train_metric_loss", 0) for h in history]
        axes[3].plot(epochs, metric_losses, label="Metric Loss", color="orange")
        axes[3].set_xlabel("Epoch")
        axes[3].set_ylabel("Loss")
        axes[3].set_title("Metric Loss")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

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

    # Output directories
    output_dir = get_output_dir(cfg)
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger("train", os.path.join(log_dir, "train.log"))

    # Also keep old-style output dirs for backward compat
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    # --- Config parsing ---
    ml_cfg = cfg.get("metric_learning", {})
    loss_cfg = cfg.get("loss", {})
    sampler_cfg = cfg.get("sampler", {})
    experiment_cfg = cfg.get("experiment", {})

    metric_learning_enabled = ml_cfg.get("enabled", False)
    ce_weight = loss_cfg.get("ce_weight", 1.0)
    metric_weight_target = loss_cfg.get("metric_weight", 0.05) if metric_learning_enabled else 0.0
    temperature = loss_cfg.get("temperature", 0.07)
    include_neg_in_metric = loss_cfg.get("include_negative_in_metric_loss", False)
    warmup_epochs = loss_cfg.get("warmup_epochs_for_metric_loss", 10)
    neg_label = cfg["negative_label"]
    num_known = cfg["num_known_classes"]

    use_pk_sampler = sampler_cfg.get("use_pk_sampler", False)
    pk_P = sampler_cfg.get("classes_per_batch", 8)
    pk_K = sampler_cfg.get("samples_per_class", 4)
    pk_include_neg = sampler_cfg.get("include_negative", False)
    pk_drop_singleton = sampler_cfg.get("drop_singleton_classes", True)

    logger.info(f"Config: {cfg}")
    logger.info(f"Device: {device}")
    backbone = cfg.get("model", {}).get("backbone", "pointnet")
    logger.info(f"Backbone: {backbone}")
    logger.info(f"Metric learning: {'ENABLED' if metric_learning_enabled else 'DISABLED'}")
    if metric_learning_enabled:
        logger.info(f"  CE weight={ce_weight}, metric weight={metric_weight_target}, "
                     f"temperature={temperature}, warmup={warmup_epochs} epochs")
        logger.info(f"  Include negative in metric loss: {include_neg_in_metric}")
        logger.info(f"  PK sampler: {'ENABLED' if use_pk_sampler else 'DISABLED'}")

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
        # Disable metric learning and PK sampler in overfit mode
        logger.info(f"[OVERFIT MODE] Using {num_overfit} samples, no augmentation")
        metric_learning_enabled = False
        metric_weight_target = 0.0
        use_pk_sampler = False
        epochs = 100
        batch_size = min(32, num_overfit)
    else:
        epochs = cfg["train"]["epochs"]
        batch_size = cfg["train"]["batch_size"]

    logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # --- DataLoader ---
    # For overfit mode, use the underlying dataset
    weight_dataset = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
    labels_list = [weight_dataset.samples[i]["label"] for i in range(len(weight_dataset))]

    if use_pk_sampler and not overfit_mode and metric_learning_enabled:
        from src.datasets.samplers import PKSampler
        pk_sampler = PKSampler(
            labels=labels_list,
            classes_per_batch=pk_P,
            samples_per_class=pk_K,
            include_negative=pk_include_neg,
            negative_label=neg_label,
            drop_singleton_classes=pk_drop_singleton,
        )
        effective_batch_size = pk_P * pk_K
        train_loader = DataLoader(
            train_ds, batch_size=effective_batch_size,
            sampler=pk_sampler,
            num_workers=4, collate_fn=collate_fn, drop_last=True,
        )
        logger.info(f"PKSampler: P={pk_P}, K={pk_K}, batch_size={effective_batch_size}")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4, collate_fn=collate_fn, drop_last=False,
        )

    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=4, collate_fn=collate_fn,
    )

    # --- Model ---
    model = build_model(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    # --- Class weight ---
    use_class_weight = cfg.get("train", {}).get("use_class_weight", False)
    max_weight = cfg.get("train", {}).get("class_weight_max", 10.0)

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

    # --- Metric loss ---
    metric_loss_fn = None
    if metric_learning_enabled:
        from src.losses.metric_losses import SupervisedContrastiveLoss
        metric_loss_fn = SupervisedContrastiveLoss(
            temperature=temperature,
            negative_label=neg_label,
            include_negative=include_neg_in_metric,
        )
        logger.info(f"SupervisedContrastiveLoss: temperature={temperature}, "
                     f"include_negative={include_neg_in_metric}")

    # --- Resume ---
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        epoch_loaded, _ = load_checkpoint(args.resume, model, optimizer)
        start_epoch = epoch_loaded
        logger.info(f"Resumed from epoch {start_epoch}")

    # --- Training loop ---
    best_val_f1 = 0.0
    history = []

    for epoch in range(start_epoch, epochs):
        lr = optimizer.param_groups[0]["lr"]

        # Warmup metric weight
        if metric_learning_enabled:
            if epoch < warmup_epochs:
                metric_weight_current = 0.0
            else:
                metric_weight_current = metric_weight_target
        else:
            metric_weight_current = 0.0

        train_loss, train_ce_loss, train_metric_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            metric_loss_fn=metric_loss_fn,
            ce_weight=ce_weight,
            metric_weight=metric_weight_current,
        )

        if overfit_mode:
            # Re-evaluate on same data
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

            neg_acc = val_metrics.get("negative_accuracy", 0)
            neg_acc_str = f"{neg_acc:.4f}" if neg_acc > 0 or any(
                l == neg_label for l in [
                    val_ds.samples[i]["label"] for i in range(min(len(val_ds), 1))
                ]
            ) else "N/A"

            metric_info = ""
            if metric_learning_enabled:
                metric_info = f"MetricLoss: {train_metric_loss:.4f} (w={metric_weight_current:.4f}) | "

            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} (CE: {train_ce_loss:.4f}) Acc: {train_acc:.4f} | "
                f"{metric_info}"
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
            "train_ce_loss": train_ce_loss,
            "train_metric_loss": train_metric_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics.get("val_loss", 0),
            "val_acc": val_metrics["overall_accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_known_acc": val_metrics["known_class_accuracy"],
            "val_neg_acc": val_metrics.get("negative_accuracy", 0),
            "lr": lr,
            "metric_weight": metric_weight_current,
        })

        # Save last checkpoint (with config for backbone recovery)
        last_ckpt_path = os.path.join(ckpt_dir, "last.pt")
        save_checkpoint_with_config(model, optimizer, epoch + 1, last_ckpt_path, cfg, metrics=val_metrics)
        # Also save to old location for backward compat
        save_checkpoint(model, optimizer, epoch + 1, "outputs/checkpoints/last.pt", metrics=val_metrics)

        # Save best checkpoint by val macro F1 (or train acc in overfit mode)
        current_metric = val_metrics["overall_accuracy"] if overfit_mode else val_metrics["macro_f1"]
        if current_metric > best_val_f1:
            best_val_f1 = current_metric
            best_ckpt_path = os.path.join(ckpt_dir, "best.pt")
            save_checkpoint_with_config(model, optimizer, epoch + 1, best_ckpt_path, cfg, metrics=val_metrics)
            # Also save to old location
            save_checkpoint(model, optimizer, epoch + 1, "outputs/checkpoints/best.pt", metrics=val_metrics)

            # Save per-class metrics for best checkpoint
            class_names_path = os.path.join(os.path.dirname(args.config), "class_names.json")
            per_class_path = os.path.join(output_dir, "best_val_per_class_metrics.json")
            save_per_class_metrics(
                val_metrics, num_known, neg_label,
                class_names_path, per_class_path,
            )
            # Also to old location
            save_per_class_metrics(
                val_metrics, num_known, neg_label,
                class_names_path, "outputs/reports/best_val_per_class_metrics.json",
            )

            logger.info(f"New best model saved (F1={'overfit' if overfit_mode else 'val'}={best_val_f1:.4f})")

    # --- Save training curves ---
    if history:
        curves_path = os.path.join(output_dir, "training_curves.png")
        plot_training_curves(history, curves_path, has_metric_loss=metric_learning_enabled)
        logger.info(f"Training curves saved to {curves_path}")

        # Also save to old location
        plot_training_curves(history, "outputs/reports/training_curves.png", has_metric_loss=metric_learning_enabled)

        # Save history as JSON
        hist_path = os.path.join(output_dir, "training_history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        # Also old location
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
        logger.info(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
