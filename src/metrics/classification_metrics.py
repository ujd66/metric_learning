import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred, y_prob=None, num_known_classes=19, negative_label=19):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    num_classes = num_known_classes + 1

    overall_acc = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    known_mask = y_true != negative_label
    neg_mask = y_true == negative_label

    known_acc = accuracy_score(y_true[known_mask], y_pred[known_mask]) if known_mask.any() else 0.0
    neg_acc = accuracy_score(y_true[neg_mask], y_pred[neg_mask]) if neg_mask.any() else 0.0

    per_class_precision = precision_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)

    per_class = {}
    for i in range(num_classes):
        mask = y_true == i
        support = int(mask.sum())
        per_class[str(i)] = {
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1": float(per_class_f1[i]),
            "support": support,
        }

    return {
        "overall_accuracy": float(overall_acc),
        "known_class_accuracy": float(known_acc),
        "negative_accuracy": float(neg_acc),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
    }


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    all_labels = sorted(set(list(y_true) + list(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    names = [class_names[i] if i < len(class_names) else f"class_{i}" for i in all_labels]
    fig, ax = plt.subplots(figsize=(max(10, len(all_labels) * 0.6), max(8, len(all_labels) * 0.5)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(all_labels)),
           yticks=np.arange(len(all_labels)),
           xticklabels=names,
           yticklabels=names,
           ylabel="True label",
           xlabel="Predicted label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
    plt.setp(ax.get_yticklabels(), fontsize=6)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_normalized_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot row-normalized confusion matrix (each row sums to 1)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    all_labels = sorted(set(list(y_true) + list(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    names = [class_names[i] if i < len(class_names) else f"class_{i}" for i in all_labels]
    # Row-normalize
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid div by zero
    cm_norm = cm.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(max(10, len(all_labels) * 0.6), max(8, len(all_labels) * 0.5)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(all_labels)),
           yticks=np.arange(len(all_labels)),
           xticklabels=names,
           yticklabels=names,
           ylabel="True label",
           xlabel="Predicted label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
    plt.setp(ax.get_yticklabels(), fontsize=6)

    # Add text annotations
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=5)

    ax.set_title("Normalized Confusion Matrix (row)")
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
