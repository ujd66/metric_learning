import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.pointcloud_dataset import PointCloudDataset
from src.metrics.classification_metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_normalized_confusion_matrix,
)
from src.models.model_factory import build_model_from_checkpoint
from src.utils.config import load_config


def collate_fn(batch):
    points = torch.stack([b["points"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"points": points, "label": labels}


@torch.no_grad()
def run_evaluation(model, loader, device, num_known_classes, negative_label):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    for batch in tqdm(loader, desc="Evaluate"):
        points = batch["points"].to(device)
        out = model(points)
        probs = torch.softmax(out["logits"], dim=1)
        preds = out["logits"].argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch["label"].numpy())
        all_probs.extend(probs.cpu().numpy())
    metrics = compute_metrics(
        all_labels, all_preds, all_probs,
        num_known_classes=num_known_classes,
        negative_label=negative_label,
    )
    return metrics, all_labels, all_preds


def load_class_names(config_path, num_classes):
    class_names_path = os.path.join(os.path.dirname(config_path), "class_names.json")
    if os.path.exists(class_names_path):
        with open(class_names_path) as f:
            name_map = json.load(f)
        return [name_map.get(str(i), f"class_{i:03d}") for i in range(num_classes)]
    return [f"class_{i:03d}" for i in range(num_classes)]


def save_per_class_csv(metrics, class_names, num_classes, save_path, split):
    """Save per-class metrics as CSV."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pc = metrics.get("per_class", {})

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "label", "split", "precision", "recall", "f1", "support"])
        for i in range(num_classes):
            name = class_names[i] if i < len(class_names) else f"class_{i:03d}"
            entry = pc.get(str(i), {})
            support = entry.get("support", 0)
            prec = entry.get("precision", 0)
            rec = entry.get("recall", 0)
            f1 = entry.get("f1", 0)
            writer.writerow([name, i, split, f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", support])


def main():
    parser = argparse.ArgumentParser(description="Evaluate PointNet model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: outputs/reports)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    num_classes = cfg["num_classes"]
    num_known_classes = cfg["num_known_classes"]
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

    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=4, collate_fn=collate_fn)

    model, _ = build_model_from_checkpoint(cfg, args.checkpoint)
    model = model.to(device)

    metrics, y_true, y_pred = run_evaluation(model, loader, device, num_known_classes, negative_label)

    report_dir = args.output_dir or "outputs/reports"
    os.makedirs(report_dir, exist_ok=True)

    # Save metrics JSON
    eval_path = os.path.join(report_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to {eval_path}")

    # Class names
    class_names = load_class_names(args.config, num_classes)

    # Confusion matrix
    cm_path = os.path.join(report_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    # Normalized confusion matrix
    cmn_path = os.path.join(report_dir, "confusion_matrix_normalized.png")
    plot_normalized_confusion_matrix(y_true, y_pred, class_names, cmn_path)
    print(f"Normalized confusion matrix saved to {cmn_path}")

    # Per-class CSV
    pc_csv_path = os.path.join(report_dir, "per_class_metrics.csv")
    save_per_class_csv(metrics, class_names, num_classes, pc_csv_path, args.split)
    print(f"Per-class metrics saved to {pc_csv_path}")

    # Check for missing classes in this split
    present_labels = set(y_true)
    missing_classes = []
    for i in range(num_classes):
        if i not in present_labels:
            name = class_names[i] if i < len(class_names) else f"class_{i:03d}"
            missing_classes.append(name)
    if missing_classes:
        print(f"\n[WARN] Classes not present in {args.split}: {missing_classes}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"Evaluation Results ({args.split})")
    print(f"{'='*50}")
    print(f"Samples: {len(y_true)}")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Known-class Accuracy: {metrics['known_class_accuracy']:.4f}")
    print(f"Negative Accuracy: {metrics['negative_accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")

    print(f"\n--- Per-Class ---")
    for i in range(num_classes):
        name = class_names[i] if i < len(class_names) else f"class_{i:03d}"
        entry = metrics.get("per_class", {}).get(str(i), {})
        support = entry.get("support", 0)
        prec = entry.get("precision", 0)
        rec = entry.get("recall", 0)
        f1 = entry.get("f1", 0)
        print(f"  {name:<30s} P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  support={support}")


if __name__ == "__main__":
    main()
