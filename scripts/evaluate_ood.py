"""Evaluate OOD / unknown / negative rejection on a given split.

Loads prototypes and threshold, runs full inference on the test split,
and generates comprehensive metrics, plots, and HTML report.

Supports multi-threshold sweep via --thresholds.

Usage:
    python scripts/evaluate_ood.py \
        --config configs/config.yaml \
        --checkpoint outputs/checkpoints/best.pt \
        --prototypes outputs/prototypes/baseline_prototypes.pt \
        --threshold-json outputs/prototypes/baseline_threshold.json \
        --split test \
        --output-dir outputs/reports/ood_eval_baseline_test

    # With multi-threshold sweep:
    python scripts/evaluate_ood.py \
        ... \
        --thresholds 0.68,0.75,0.80,0.85,0.90,0.92,0.94,0.96
"""

import argparse
import csv
import json
import math
import os
import sys
import warnings

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
    return {"points": points, "label": labels, "sample_id": sample_ids}


def compute_quantiles(data, quantiles=None):
    """Compute quantiles for a numpy array."""
    if quantiles is None:
        quantiles = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
    if len(data) == 0:
        return {f"p{int(q*100):02d}": None for q in quantiles}
    return {f"p{int(q*100):02d}": float(np.quantile(data, q)) for q in quantiles}


@torch.no_grad()
def extract_predictions(model, loader, device, prototypes, proto_class_names,
                        similarity_threshold, num_known_classes, negative_label):
    """Extract full predictions including prototype matching for all samples."""
    model.eval()
    all_results = []

    for batch in tqdm(loader, desc="Predict"):
        points = batch["points"].to(device)
        out = model(points)
        embeddings = out["embedding"].cpu().numpy()  # [B, D]
        logits = out["logits"]
        probs = torch.softmax(logits, dim=1).cpu().numpy()  # [B, C]
        labels = batch["label"].numpy()
        sample_ids = batch["sample_id"]

        # Prototype similarities
        emb_t = torch.tensor(embeddings, dtype=torch.float32)
        sims = emb_t @ prototypes.T  # [B, num_known]
        sims_np = sims.numpy()

        for i in range(len(labels)):
            pred_label = int(probs[i].argmax())
            confidence = float(probs[i].max())
            label = int(labels[i])

            nearest_sim = float(sims_np[i].max())
            nearest_idx = int(sims_np[i].argmax())
            nearest_class = proto_class_names[nearest_idx]

            # Decision logic
            if pred_label == negative_label:
                final_type = "negative"
                final_label_str = "negative"
                reason = "classified_as_negative"
            elif nearest_sim < similarity_threshold:
                final_type = "unknown"
                final_label_str = "unknown"
                reason = "far_from_all_known_prototypes"
            else:
                final_type = "known"
                final_label_str = nearest_class
                reason = "matched_known_prototype"

            margin_to_threshold = nearest_sim - similarity_threshold

            # Risk level
            if label < num_known_classes:  # known sample
                if nearest_sim < similarity_threshold:
                    risk_level = "rejected_known"
                elif 0 <= margin_to_threshold < 0.03:
                    risk_level = "near_boundary_known"
                else:
                    risk_level = "safe_known"
            else:  # negative sample
                if nearest_sim >= similarity_threshold:
                    risk_level = "false_known_negative"
                elif -0.03 < margin_to_threshold < 0:
                    risk_level = "near_boundary_negative"
                else:
                    risk_level = "safe_rejected_negative"

            all_results.append({
                "sample_id": sample_ids[i],
                "true_label": label,
                "pred_label": pred_label,
                "confidence": confidence,
                "nearest_known_label": nearest_idx,
                "nearest_known_class": nearest_class,
                "nearest_similarity": nearest_sim,
                "final_type": final_type,
                "final_label": final_label_str,
                "reason": reason,
                "margin_to_threshold": margin_to_threshold,
                "risk_level": risk_level,
            })

    return all_results


def compute_auroc(known_sims, negative_sims):
    """Compute AUROC treating known as positive, negative as negative."""
    from sklearn.metrics import roc_auc_score
    y_true = np.concatenate([np.ones(len(known_sims)), np.zeros(len(negative_sims))])
    y_score = np.concatenate([known_sims, negative_sims])
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def compute_ood_metrics(results, similarity_threshold, num_known_classes, negative_label):
    """Compute OOD metrics from a list of prediction results."""
    known_results = [r for r in results if r["true_label"] < num_known_classes]
    negative_results = [r for r in results if r["true_label"] == negative_label]

    n_known = len(known_results)
    n_negative = len(negative_results)

    known_accepted = [r for r in known_results if r["final_type"] == "known"]
    known_rejected = [r for r in known_results if r["final_type"] != "known"]

    known_accept_rate = len(known_accepted) / n_known if n_known > 0 else 0.0
    known_reject_rate = len(known_rejected) / n_known if n_known > 0 else 0.0

    known_correct_after_accept = sum(
        1 for r in known_accepted if r["nearest_known_label"] == r["true_label"]
    )
    known_class_acc_after_accept = (
        known_correct_after_accept / len(known_accepted) if known_accepted else 0.0
    )

    known_overall_correct_rate = known_correct_after_accept / n_known if n_known > 0 else 0.0

    avg_sim_known = np.mean([r["nearest_similarity"] for r in known_results]) if known_results else 0.0
    avg_sim_negative = np.mean([r["nearest_similarity"] for r in negative_results]) if negative_results else None

    neg_reject_rate = None
    neg_false_known_rate = None
    neg_classified_neg_rate = None
    neg_classified_known_rate = None

    if n_negative > 0:
        neg_rejected = [r for r in negative_results if r["final_type"] != "known"]
        neg_false_known = [r for r in negative_results if r["final_type"] == "known"]
        neg_classified_neg = [r for r in negative_results if r["pred_label"] == negative_label]
        neg_reject_rate = len(neg_rejected) / n_negative
        neg_false_known_rate = len(neg_false_known) / n_negative
        neg_classified_neg_rate = len(neg_classified_neg) / n_negative
        neg_classified_known_rate = len(neg_false_known) / n_negative

    auroc = None
    if n_known > 0 and n_negative > 0:
        known_sims = np.array([r["nearest_similarity"] for r in known_results])
        neg_sims = np.array([r["nearest_similarity"] for r in negative_results])
        auroc = compute_auroc(known_sims, neg_sims)

    # Per-class metrics
    per_class_rows = []
    for c in range(num_known_classes):
        class_results = [r for r in known_results if r["true_label"] == c]
        support = len(class_results)
        if support == 0:
            continue
        accepted = [r for r in class_results if r["final_type"] == "known"]
        accept_rate = len(accepted) / support
        correct = sum(1 for r in accepted if r["nearest_known_label"] == c)
        acc_after_accept = correct / len(accepted) if accepted else 0.0
        avg_sim = np.mean([r["nearest_similarity"] for r in class_results])
        per_class_rows.append({
            "label": c,
            "support": support,
            "accept_rate": accept_rate,
            "accuracy_after_accept": acc_after_accept,
            "avg_similarity": float(avg_sim),
        })

    # Final macro F1
    from sklearn.metrics import f1_score
    final_known_f1 = None
    if known_accepted:
        y_true = [r["true_label"] for r in known_accepted]
        y_pred = [r["nearest_known_label"] for r in known_accepted]
        final_known_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Balanced score
    balanced_score = known_accept_rate + (neg_reject_rate if neg_reject_rate is not None else 0.0)

    return {
        "threshold": similarity_threshold,
        "num_samples": len(results),
        "num_known_samples": n_known,
        "num_negative_samples": n_negative,
        "known_accept_rate": known_accept_rate,
        "known_reject_rate": known_reject_rate,
        "known_classification_accuracy_after_accept": known_class_acc_after_accept,
        "known_overall_correct_rate": known_overall_correct_rate,
        "negative_reject_rate": neg_reject_rate,
        "false_known_rate": neg_false_known_rate,
        "negative_classified_as_negative_rate": neg_classified_neg_rate,
        "negative_classified_as_known_rate": neg_classified_known_rate,
        "final_known_accuracy": known_class_acc_after_accept,
        "final_macro_f1_on_known_classes": final_known_f1,
        "average_nearest_similarity_known": float(avg_sim_known),
        "average_nearest_similarity_negative": float(avg_sim_negative) if avg_sim_negative is not None else None,
        "auroc": auroc,
        "balanced_score": balanced_score,
    }, per_class_rows


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_nearest_similarity_histogram(results, output_path, class_names_map, threshold=None):
    """Histogram of nearest known similarity for known vs negative samples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    known_sims = [r["nearest_similarity"] for r in results if r["true_label"] < 19]
    negative_sims = [r["nearest_similarity"] for r in results if r["true_label"] == 19]

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1, 50)
    if known_sims:
        ax.hist(known_sims, bins=bins, alpha=0.6, label=f"Known (n={len(known_sims)})",
                color="#3b82f6", edgecolor="white")
    if negative_sims:
        ax.hist(negative_sims, bins=bins, alpha=0.6, label=f"Negative (n={len(negative_sims)})",
                color="#ef4444", edgecolor="white")
    if threshold is not None:
        ax.axvline(x=threshold, color="#10b981", linestyle="--", linewidth=2,
                   label=f"Threshold: {threshold:.2f}")
    ax.set_xlabel("Nearest Known Prototype Similarity", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Nearest Similarity Distribution: Known vs Negative", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_curve(threshold_curve, selected_threshold, output_path,
                         known_p05=None, manual_threshold=None):
    """Plot threshold curve with known_accept_rate and negative_reject_rate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = [tc["threshold"] for tc in threshold_curve]
    ka = [tc["known_accept_rate"] for tc in threshold_curve]
    kr = [tc["known_reject_rate"] for tc in threshold_curve]
    nr = [tc["negative_reject_rate"] for tc in threshold_curve]
    fk = [tc["false_known_rate"] for tc in threshold_curve]
    bs = [tc["balanced_score"] for tc in threshold_curve]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ts, ka, label="Known Accept Rate", color="#3b82f6", linewidth=2)
    ax.plot(ts, kr, label="Known Reject Rate", color="#8b5cf6", linewidth=1.5, linestyle="--")
    has_negative = any(v is not None for v in nr)
    if has_negative:
        nr_plot = [v if v is not None else 0 for v in nr]
        fk_plot = [v if v is not None else 0 for v in fk]
        ax.plot(ts, nr_plot, label="Negative Reject Rate", color="#10b981", linewidth=2)
        ax.plot(ts, fk_plot, label="False Known Rate", color="#ef4444", linewidth=1.5, linestyle="--")
    ax.plot(ts, bs, label="Balanced Score", color="#f59e0b", linewidth=1.5, linestyle=":")

    # Vertical lines
    ax.axvline(x=selected_threshold, color="#ef4444", linestyle="-.", linewidth=2,
               label=f"Selected: {selected_threshold:.2f}")
    if known_p05 is not None and abs(known_p05 - selected_threshold) > 0.005:
        ax.axvline(x=known_p05, color="#8b5cf6", linestyle=":", linewidth=1.5,
                   label=f"Known P05: {known_p05:.2f}")
    if manual_threshold is not None and abs(manual_threshold - selected_threshold) > 0.005:
        ax.axvline(x=manual_threshold, color="#6b7280", linestyle=":", linewidth=1.5,
                   label=f"Manual: {manual_threshold:.2f}")

    ax.set_xlabel("Similarity Threshold", fontsize=12)
    ax.set_ylabel("Rate", fontsize=12)
    ax.set_title("Threshold Search Curve", fontsize=14)
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    ax.set_xlim(ts[0], ts[-1])
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_sweep(sweep_metrics, output_path):
    """Plot multi-threshold sweep results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = [m["threshold"] for m in sweep_metrics]
    ka = [m["known_accept_rate"] for m in sweep_metrics]
    kr = [m["known_reject_rate"] for m in sweep_metrics]
    acc = [m["known_classification_accuracy_after_accept"] for m in sweep_metrics]
    bs = [m["balanced_score"] for m in sweep_metrics]

    has_neg = any(m.get("negative_reject_rate") is not None for m in sweep_metrics)
    nr = [m.get("negative_reject_rate") or 0 for m in sweep_metrics]
    fk = [m.get("false_known_rate") or 0 for m in sweep_metrics]

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(ts, ka, "o-", label="Known Accept Rate", color="#3b82f6", linewidth=2, markersize=6)
    ax1.plot(ts, acc, "s-", label="Known Acc (after accept)", color="#10b981", linewidth=2, markersize=6)
    if has_neg:
        ax1.plot(ts, nr, "^-", label="Negative Reject Rate", color="#f59e0b", linewidth=2, markersize=6)
        ax1.plot(ts, fk, "v-", label="False Known Rate", color="#ef4444", linewidth=1.5, markersize=5)
    ax1.plot(ts, bs, "D--", label="Balanced Score", color="#8b5cf6", linewidth=1.5, markersize=5)

    ax1.axhline(y=0.95, color="#3b82f6", linestyle=":", alpha=0.5, label="95% known accept target")
    ax1.set_xlabel("Similarity Threshold", fontsize=12)
    ax1.set_ylabel("Rate", fontsize=12)
    ax1.set_title("Multi-Threshold Sweep", fontsize=14)
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Annotate each point
    for i, t in enumerate(ts):
        ax1.annotate(f"{t:.2f}", (t, ka[i]), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8)

    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_final_confusion_matrix(results, class_names_map, num_known_classes, output_path):
    """Confusion matrix with known classes + negative + unknown."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    LABEL_UNKNOWN = 100

    y_true = []
    y_pred = []
    for r in results:
        true_lbl = r["true_label"]
        if r["final_type"] == "unknown":
            pred_lbl = LABEL_UNKNOWN
        elif r["final_type"] == "negative":
            pred_lbl = 19
        else:
            pred_lbl = r["nearest_known_label"]
        y_true.append(true_lbl)
        y_pred.append(pred_lbl)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    present_true = sorted(set(y_true.tolist()))
    present_pred = sorted(set(y_pred.tolist()))
    all_labels = sorted(set(present_true + present_pred))

    display_names = []
    for lbl in all_labels:
        if lbl == LABEL_UNKNOWN:
            display_names.append("unknown")
        elif lbl == 19:
            display_names.append("negative")
        else:
            display_names.append(class_names_map.get(str(lbl), f"class_{lbl:03d}"))

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    fig, ax = plt.subplots(figsize=(max(10, len(all_labels) * 0.6), max(8, len(all_labels) * 0.5)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_xticks(range(len(all_labels)))
    ax.set_yticks(range(len(all_labels)))
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(display_names, fontsize=7)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title("Final Confusion Matrix (Classifier + Prototype Rejection)", fontsize=14)

    for i in range(len(all_labels)):
        for j in range(len(all_labels)):
            val = cm[i, j]
            if val > 0:
                color = "white" if val > cm.max() / 2 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=6)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# HTML report builders
# ---------------------------------------------------------------------------

def build_ood_html_report(
    checkpoint, prototypes_path, threshold, split,
    num_samples, num_known, num_negative,
    metrics, per_class_rows, warnings_list,
    sim_hist_path, threshold_curve_path, cm_path,
    class_names_map, threshold_data,
    known_quantiles=None, negative_quantiles=None,
):
    """Build self-contained HTML report for OOD evaluation."""
    import base64
    import html as html_mod
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _img_to_base64(path):
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            data = f.read()
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        mime = {"png": "image/png", "jpg": "image/jpeg"}.get(ext, "image/png")
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

    def _fmt(val, decimals=4):
        if val is None:
            return "N/A"
        if isinstance(val, float) and math.isnan(val):
            return "N/A"
        return f"{val:.{decimals}f}"

    def _card_cls(val, higher_better=True, good=0.9, bad=0.5):
        if val is None:
            return ""
        if higher_better:
            if val >= good: return "good"
            if val >= bad: return "warn"
            return "bad"
        else:
            if val <= bad: return "good"
            if val <= good: return "warn"
            return "bad"

    CSS = """
:root {
    --bg: #f5f7fa; --card-bg: #ffffff; --text: #1a1a2e; --text-secondary: #555;
    --border: #e0e4e8; --accent: #3b82f6; --accent-light: #dbeafe;
    --success: #10b981; --warning: #f59e0b; --danger: #ef4444;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: var(--bg); color: var(--text); line-height: 1.6; padding: 24px; }
.container { max-width: 1200px; margin: 0 auto; }
h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 8px; }
h2 { font-size: 1.3rem; font-weight: 600; margin-top: 32px; margin-bottom: 16px;
     padding-bottom: 8px; border-bottom: 2px solid var(--accent); }
.subtitle { color: var(--text-secondary); margin-bottom: 24px; font-size: 0.95rem; }
.info-bar { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 24px; padding: 16px;
            background: var(--card-bg); border-radius: 8px; border: 1px solid var(--border); }
.info-item { padding: 4px 12px; background: var(--accent-light); border-radius: 4px; font-size: 0.85rem; }
.info-item span { font-weight: 600; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 16px; margin-bottom: 24px; }
.card { background: var(--card-bg); border-radius: 10px; padding: 20px 16px; text-align: center;
        border: 1px solid var(--border); box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.card-value { font-size: 1.5rem; font-weight: 700; margin-bottom: 4px; }
.card-value.good { color: var(--success); } .card-value.warn { color: var(--warning); }
.card-value.bad { color: var(--danger); } .card-value { color: var(--accent); }
.card-label { font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; }
.table-wrap { overflow-x: auto; margin-bottom: 24px; background: var(--card-bg);
              border-radius: 8px; border: 1px solid var(--border); }
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
th { background: #f8f9fb; padding: 10px 12px; text-align: left; font-weight: 600;
     border-bottom: 2px solid var(--border); white-space: nowrap; }
td { padding: 8px 12px; border-bottom: 1px solid var(--border); white-space: nowrap; }
tr:last-child td { border-bottom: none; } tr:hover td { background: #f8f9fb; }
.img-section { margin-bottom: 24px; background: var(--card-bg); border-radius: 8px;
               border: 1px solid var(--border); padding: 16px; text-align: center; }
.img-section img { max-width: 100%; height: auto; }
.warning-banner { background: #fef3c7; border: 1px solid #f59e0b; border-radius: 6px;
                  padding: 12px 16px; margin-bottom: 16px; color: #92400e; font-size: 0.9rem; }
.quantile-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); gap: 8px; margin-bottom: 24px; }
.quantile-card { background: var(--card-bg); border-radius: 8px; padding: 12px; text-align: center;
                 border: 1px solid var(--border); }
.quantile-label { font-size: 0.75rem; color: var(--text-secondary); }
.quantile-value { font-size: 1.1rem; font-weight: 600; color: var(--accent); }
footer { margin-top: 40px; text-align: center; color: var(--text-secondary); font-size: 0.8rem; }
"""

    # Info bar
    info_items = [
        f"Checkpoint: <span>{html_mod.escape(os.path.basename(checkpoint))}</span>",
        f"Prototypes: <span>{html_mod.escape(os.path.basename(prototypes_path))}</span>",
        f"Threshold: <span>{_fmt(threshold)}</span>",
        f"Split: <span>{html_mod.escape(split)}</span>",
        f"Samples: <span>{num_samples}</span>",
        f"Known: <span>{num_known}</span>",
        f"Negative: <span>{num_negative}</span>",
    ]
    info_bar = "\n".join(f'<div class="info-item">{item}</div>' for item in info_items)

    # Metric cards
    card_data = [
        ("Known Accept Rate", metrics.get("known_accept_rate"), _card_cls(metrics.get("known_accept_rate"), True, 0.95, 0.8)),
        ("Known Reject Rate", metrics.get("known_reject_rate"), _card_cls(metrics.get("known_reject_rate"), False, 0.05, 0.2)),
        ("Known Acc (after accept)", metrics.get("known_classification_accuracy_after_accept"), _card_cls(metrics.get("known_classification_accuracy_after_accept"), True, 0.9, 0.7)),
        ("Negative Reject Rate", metrics.get("negative_reject_rate"), _card_cls(metrics.get("negative_reject_rate"), True, 0.8, 0.5) if metrics.get("negative_reject_rate") is not None else ""),
        ("False Known Rate", metrics.get("false_known_rate"), _card_cls(metrics.get("false_known_rate"), False, 0.1, 0.3) if metrics.get("false_known_rate") is not None else ""),
        ("AUROC", metrics.get("auroc"), _card_cls(metrics.get("auroc"), True, 0.9, 0.7) if metrics.get("auroc") is not None else ""),
    ]
    cards = ""
    for label, val, cls in card_data:
        cards += f'<div class="card"><div class="card-value {cls}">{_fmt(val) if val is not None else "N/A"}</div><div class="card-label">{html_mod.escape(label)}</div></div>\n'

    # Warnings
    warns_html = ""
    for w in warnings_list:
        warns_html += f'<div class="warning-banner">{html_mod.escape(w)}</div>\n'

    # Similarity quantiles section
    quantiles_html = ""
    if known_quantiles:
        q_cards = ""
        for k, v in known_quantiles.items():
            val_str = f"{v:.4f}" if v is not None else "N/A"
            q_cards += f'<div class="quantile-card"><div class="quantile-value">{val_str}</div><div class="quantile-label">Known {k.upper()}</div></div>\n'
        quantiles_html += f'<h2>Known Nearest Similarity Quantiles</h2><div class="quantile-grid">{q_cards}</div>\n'

    if negative_quantiles:
        q_cards = ""
        for k, v in negative_quantiles.items():
            val_str = f"{v:.4f}" if v is not None else "N/A"
            q_cards += f'<div class="quantile-card"><div class="quantile-value">{val_str}</div><div class="quantile-label">Negative {k.upper()}</div></div>\n'
        quantiles_html += f'<h2>Negative Nearest Similarity Quantiles</h2><div class="quantile-grid">{q_cards}</div>\n'

    # Per-class table
    pc_header = "<th>Class</th><th>Support</th><th>Accept Rate</th><th>Acc (accepted)</th><th>Avg Similarity</th>"
    pc_rows = ""
    for row in per_class_rows:
        cname = row.get("class_name", class_names_map.get(str(row["label"]), f"class_{row['label']:03d}"))
        pc_rows += f'<tr>'
        pc_rows += f'<td>{html_mod.escape(cname)}</td>'
        pc_rows += f'<td>{row["support"]}</td>'
        pc_rows += f'<td>{_fmt(row["accept_rate"])}</td>'
        pc_rows += f'<td>{_fmt(row["accuracy_after_accept"])}</td>'
        pc_rows += f'<td>{_fmt(row["avg_similarity"])}</td>'
        pc_rows += '</tr>\n'

    per_class_table = f'<div class="table-wrap"><table><tr>{pc_header}</tr>{pc_rows}</table></div>'

    # Images
    def _img_section(title, path):
        uri = _img_to_base64(path)
        if uri:
            return f'<div class="img-section"><h3 style="margin-bottom:12px;font-size:1.05rem;">{html_mod.escape(title)}</h3><img src="{uri}"></div>'
        return f'<div class="img-section"><h3 style="margin-bottom:12px;">{html_mod.escape(title)}</h3><div class="warning-banner">Image not found</div></div>'

    sim_img = _img_section("Nearest Similarity Distribution", sim_hist_path)
    tc_img = _img_section("Threshold Search Curve", threshold_curve_path)
    cm_img = _img_section("Final Confusion Matrix", cm_path)

    report = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OOD Evaluation Report</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">
<h1>OOD / Unknown Rejection Evaluation Report</h1>
<p class="subtitle">Generated at {html_mod.escape(timestamp)}</p>

<h2>Basic Information</h2>
<div class="info-bar">{info_bar}</div>

{warns_html}

<h2>Core Metrics</h2>
<div class="cards">{cards}</div>

{quantiles_html}

<h2>Per-class Metrics</h2>
{per_class_table}

<h2>Visualizations</h2>
{sim_img}
{tc_img}
{cm_img}

<footer>OOD Evaluation Report &mdash; pointcloud_metric_learning</footer>
</div>
</body>
</html>"""
    return report


def build_threshold_sweep_html(sweep_metrics, output_dir):
    """Build HTML report for multi-threshold sweep."""
    import base64
    import html as html_mod
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Find best thresholds
    best_ka_ge_095 = [m for m in sweep_metrics if m["known_accept_rate"] >= 0.95]
    best_neg_reject = max(sweep_metrics, key=lambda m: m.get("negative_reject_rate") or 0)
    best_balanced = max(sweep_metrics, key=lambda m: m["balanced_score"])

    # Recommended: highest neg_reject among ka >= 0.95, fallback to best balanced
    if best_ka_ge_095:
        recommended = max(best_ka_ge_095, key=lambda m: m.get("negative_reject_rate") or 0)
    else:
        recommended = best_balanced

    CSS = """
:root {
    --bg: #f5f7fa; --card-bg: #ffffff; --text: #1a1a2e; --text-secondary: #555;
    --border: #e0e4e8; --accent: #3b82f6; --accent-light: #dbeafe;
    --success: #10b981; --warning: #f59e0b; --danger: #ef4444;
    --highlight: #fef3c7;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: var(--bg); color: var(--text); line-height: 1.6; padding: 24px; }
.container { max-width: 1400px; margin: 0 auto; }
h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 8px; }
h2 { font-size: 1.3rem; font-weight: 600; margin-top: 32px; margin-bottom: 16px;
     padding-bottom: 8px; border-bottom: 2px solid var(--accent); }
.subtitle { color: var(--text-secondary); margin-bottom: 24px; font-size: 0.95rem; }
.recommendation { background: #ecfdf5; border: 2px solid var(--success); border-radius: 10px;
                  padding: 20px; margin-bottom: 24px; }
.recommendation h3 { color: #065f46; margin-bottom: 8px; }
.table-wrap { overflow-x: auto; margin-bottom: 24px; background: var(--card-bg);
              border-radius: 8px; border: 1px solid var(--border); }
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
th { background: #f8f9fb; padding: 10px 12px; text-align: left; font-weight: 600;
     border-bottom: 2px solid var(--border); white-space: nowrap; }
td { padding: 8px 12px; border-bottom: 1px solid var(--border); white-space: nowrap; }
tr:last-child td { border-bottom: none; }
tr.highlight-recommended td { background: #ecfdf5; font-weight: 600; }
tr.highlight-best-nr td { background: #fef3c7; }
tr.highlight-best-bs td { background: #dbeafe; }
tr:hover td { background: #f8f9fb; }
.img-section { margin-bottom: 24px; background: var(--card-bg); border-radius: 8px;
               border: 1px solid var(--border); padding: 16px; text-align: center; }
.img-section img { max-width: 100%; height: auto; }
footer { margin-top: 40px; text-align: center; color: var(--text-secondary); font-size: 0.8rem; }
"""

    # Recommendation box
    rec_html = f"""<div class="recommendation">
<h3>Recommended Threshold: {recommended['threshold']:.2f}</h3>
<p>Known Accept Rate: {recommended['known_accept_rate']:.4f} |
   Known Acc (after accept): {recommended['known_classification_accuracy_after_accept']:.4f} |
   Negative Reject Rate: {recommended.get('negative_reject_rate', 'N/A')} |
   Balanced Score: {recommended['balanced_score']:.4f}</p>
</div>"""

    # Table
    header = ("<tr><th>Threshold</th><th>Known Accept</th><th>Known Reject</th>"
              "<th>Known Acc</th><th>Known Overall</th>"
              "<th>Neg Reject</th><th>False Known</th>"
              "<th>Avg Sim Known</th><th>Avg Sim Neg</th>"
              "<th>Macro F1</th><th>Balanced Score</th><th>Notes</th></tr>")

    rows_html = ""
    for m in sweep_metrics:
        notes = []
        row_cls = ""
        if m["threshold"] == recommended["threshold"]:
            notes.append("RECOMMENDED")
            row_cls = "highlight-recommended"
        if m["threshold"] == best_neg_reject["threshold"]:
            notes.append("BEST_NEG_REJECT")
            if row_cls == "":
                row_cls = "highlight-best-nr"
        if m["threshold"] == best_balanced["threshold"]:
            notes.append("BEST_BALANCED")
            if row_cls == "":
                row_cls = "highlight-best-bs"
        if m["known_accept_rate"] >= 0.95:
            notes.append("KA>=0.95")

        def _f(v, d=4):
            return f"{v:.{d}f}" if v is not None else "N/A"

        rows_html += f'<tr class="{row_cls}">'
        rows_html += f'<td>{m["threshold"]:.2f}</td>'
        rows_html += f'<td>{_f(m["known_accept_rate"])}</td>'
        rows_html += f'<td>{_f(m["known_reject_rate"])}</td>'
        rows_html += f'<td>{_f(m["known_classification_accuracy_after_accept"])}</td>'
        rows_html += f'<td>{_f(m["known_overall_correct_rate"])}</td>'
        rows_html += f'<td>{_f(m.get("negative_reject_rate"))}</td>'
        rows_html += f'<td>{_f(m.get("false_known_rate"))}</td>'
        rows_html += f'<td>{_f(m["average_nearest_similarity_known"])}</td>'
        rows_html += f'<td>{_f(m.get("average_nearest_similarity_negative"))}</td>'
        rows_html += f'<td>{_f(m.get("final_macro_f1_on_known_classes"))}</td>'
        rows_html += f'<td>{_f(m["balanced_score"])}</td>'
        rows_html += f'<td>{", ".join(notes)}</td>'
        rows_html += '</tr>\n'

    # Image
    sweep_img_path = os.path.join(output_dir, "threshold_sweep_plot.png")
    img_section = ""
    if os.path.exists(sweep_img_path):
        with open(sweep_img_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("ascii")
        img_section = f"""<div class="img-section"><h3 style="margin-bottom:12px;">Threshold Sweep Plot</h3>
<img src="data:image/png;base64,{img_data}"></div>"""

    report = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Threshold Sweep Report</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">
<h1>Multi-Threshold Sweep Report</h1>
<p class="subtitle">Generated at {html_mod.escape(timestamp)}</p>

{rec_html}

<h2>Threshold Comparison</h2>
<div class="table-wrap">
<table>
{header}
{rows_html}
</table>
</div>

<h2>Sweep Visualization</h2>
{img_section}

<footer>Threshold Sweep Report &mdash; pointcloud_metric_learning</footer>
</div>
</body>
</html>"""
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate OOD / unknown rejection")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--prototypes", type=str, required=True)
    parser.add_argument("--threshold-json", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Comma-separated threshold values for multi-threshold sweep "
                             "(e.g., '0.68,0.75,0.80,0.85,0.90')")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    num_known_classes = cfg["num_known_classes"]
    negative_label = cfg["negative_label"]

    if args.output_dir is None:
        args.output_dir = f"outputs/reports/ood_eval_{args.split}"

    # Load prototypes
    print(f"Loading prototypes from {args.prototypes} ...")
    proto_data = torch.load(args.prototypes, map_location="cpu", weights_only=False)
    prototypes = proto_data["prototypes"]  # [num_known, D]
    proto_class_names = proto_data["class_names"]

    # Load threshold
    print(f"Loading threshold from {args.threshold_json} ...")
    with open(args.threshold_json) as f:
        threshold_data = json.load(f)
    similarity_threshold = threshold_data["selected_threshold"]
    threshold_curve = threshold_data.get("threshold_curve", [])
    print(f"  Similarity threshold: {similarity_threshold:.2f}")
    print(f"  Selection strategy: {threshold_data.get('selection_strategy', 'unknown')}")

    # Get quantile info from threshold_data
    known_quantiles = threshold_data.get("known_similarity_quantiles", {})
    negative_quantiles = threshold_data.get("negative_similarity_quantiles", {})
    manual_threshold = cfg.get("ood", {}).get("similarity_threshold", 0.65)
    known_p05 = known_quantiles.get("p05")

    # Load model
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

    # Load class names
    class_names_path = os.path.join(os.path.dirname(args.config), "class_names.json")
    class_names_map = {}
    if os.path.exists(class_names_path):
        with open(class_names_path) as f:
            class_names_map = json.load(f)

    # =========================================================================
    # Main evaluation with selected threshold
    # =========================================================================
    print(f"\nEvaluating {len(dataset)} samples ({args.split} split) with threshold={similarity_threshold:.2f} ...")
    results = extract_predictions(
        model, loader, device,
        prototypes, proto_class_names,
        similarity_threshold, num_known_classes, negative_label,
    )

    # Compute metrics
    ood_metrics, per_class_rows = compute_ood_metrics(
        results, similarity_threshold, num_known_classes, negative_label,
    )

    # Add per-class names
    for row in per_class_rows:
        c = row["label"]
        row["class_name"] = class_names_map.get(str(c), proto_class_names[c] if c < len(proto_class_names) else f"class_{c:03d}")

    # Compute quantiles from actual results
    known_sims = np.array([r["nearest_similarity"] for r in results if r["true_label"] < num_known_classes])
    neg_sims = np.array([r["nearest_similarity"] for r in results if r["true_label"] == negative_label])
    result_known_quantiles = compute_quantiles(known_sims)
    result_negative_quantiles = compute_quantiles(neg_sims) if len(neg_sims) > 0 else {}

    n_known = ood_metrics["num_known_samples"]
    n_negative = ood_metrics["num_negative_samples"]

    # Rejection rate
    all_rejected = [r for r in results if r["final_type"] in ("unknown", "negative")]
    ood_metrics["rejection_rate"] = len(all_rejected) / len(results) if results else 0.0

    # Warnings
    warnings_list = []
    if n_negative == 0:
        warnings_list.append("No negative samples in this split; negative rejection metrics are unavailable.")
    elif n_negative < 10:
        warnings_list.append(
            f"Only {n_negative} negative sample(s) in this split; negative-related metrics are unreliable."
        )
    warnings_list.append(
        "Current negative samples are very limited. Threshold calibration for negative rejection is unreliable. "
        "Focus on known_accept_rate and nearest_similarity distribution. Collect more negative/unknown samples for robust calibration."
    )

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    # ood_metrics.json
    metrics_path = os.path.join(args.output_dir, "ood_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(ood_metrics, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved: {metrics_path}")

    # per_sample_predictions.csv (with risk fields)
    csv_path = os.path.join(args.output_dir, "per_sample_predictions.csv")
    fieldnames = ["sample_id", "true_label", "pred_label", "confidence",
                  "nearest_known_label", "nearest_known_class", "nearest_similarity",
                  "final_type", "final_label", "reason",
                  "margin_to_threshold", "risk_level"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Saved: {csv_path}")

    # per_class_ood_metrics.csv
    pc_csv_path = os.path.join(args.output_dir, "per_class_ood_metrics.csv")
    with open(pc_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "class_name", "support",
                                                "accept_rate", "accuracy_after_accept", "avg_similarity"])
        writer.writeheader()
        writer.writerows(per_class_rows)
    print(f"Saved: {pc_csv_path}")

    # --- Hard negative / near boundary reports ---
    # hard_negative_samples.csv
    hard_neg = [r for r in results if r["true_label"] == negative_label and r["nearest_similarity"] >= similarity_threshold]
    if hard_neg:
        hn_path = os.path.join(args.output_dir, "hard_negative_samples.csv")
        hn_fields = ["sample_id", "nearest_known_class", "nearest_similarity",
                      "pred_label", "confidence", "risk_level"]
        with open(hn_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=hn_fields)
            writer.writeheader()
            for r in hard_neg:
                writer.writerow({k: r.get(k, "") for k in hn_fields})
        print(f"Saved: {hn_path} ({len(hard_neg)} hard negatives)")

    # near_boundary_known_samples.csv (top 50 known by smallest margin)
    boundary_known = [r for r in results if r["true_label"] < num_known_classes]
    boundary_known.sort(key=lambda r: r["margin_to_threshold"])
    boundary_known = boundary_known[:50]
    if boundary_known:
        nb_path = os.path.join(args.output_dir, "near_boundary_known_samples.csv")
        nb_fields = ["sample_id", "true_label", "nearest_known_class", "nearest_similarity",
                      "margin_to_threshold", "risk_level", "final_type"]
        with open(nb_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=nb_fields)
            writer.writeheader()
            for r in boundary_known:
                writer.writerow({k: r.get(k, "") for k in nb_fields})
        print(f"Saved: {nb_path} ({len(boundary_known)} near-boundary known)")

    # Plots
    sim_hist_path = os.path.join(args.output_dir, "nearest_similarity_histogram.png")
    plot_nearest_similarity_histogram(results, sim_hist_path, class_names_map, threshold=similarity_threshold)
    print(f"Saved: {sim_hist_path}")

    tc_path = os.path.join(args.output_dir, "threshold_curve.png")
    plot_threshold_curve(threshold_curve, similarity_threshold, tc_path,
                         known_p05=known_p05, manual_threshold=manual_threshold)
    print(f"Saved: {tc_path}")

    cm_path = os.path.join(args.output_dir, "final_confusion_matrix.png")
    plot_final_confusion_matrix(results, class_names_map, num_known_classes, cm_path)
    print(f"Saved: {cm_path}")

    # HTML report
    report_html = build_ood_html_report(
        checkpoint=args.checkpoint,
        prototypes_path=args.prototypes,
        threshold=similarity_threshold,
        split=args.split,
        num_samples=len(results),
        num_known=n_known,
        num_negative=n_negative,
        metrics=ood_metrics,
        per_class_rows=per_class_rows,
        warnings_list=warnings_list,
        sim_hist_path=sim_hist_path,
        threshold_curve_path=tc_path,
        cm_path=cm_path,
        class_names_map=class_names_map,
        threshold_data=threshold_data,
        known_quantiles=result_known_quantiles,
        negative_quantiles=result_negative_quantiles if result_negative_quantiles else None,
    )
    report_path = os.path.join(args.output_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_html)
    print(f"Saved: {report_path}")

    # =========================================================================
    # Multi-threshold sweep (if --thresholds provided)
    # =========================================================================
    if args.thresholds:
        sweep_thresholds = [float(t.strip()) for t in args.thresholds.split(",")]
        print(f"\n{'='*60}")
        print(f"MULTI-THRESHOLD SWEEP: {sweep_thresholds}")
        print(f"{'='*60}")

        # We need raw embeddings to re-evaluate at different thresholds
        # Re-extract just the raw data we need
        model.eval()
        all_embeddings = []
        all_labels = []
        all_sample_ids = []
        all_preds = []
        all_confs = []

        loader_sweep = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=False,
                                  num_workers=4, collate_fn=collate_fn)

        with torch.no_grad():
            for batch in tqdm(loader_sweep, desc="Extract for sweep"):
                points = batch["points"].to(device)
                out = model(points)
                all_embeddings.append(out["embedding"].cpu().numpy())
                logits = out["logits"]
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_labels.extend(batch["label"].numpy())
                all_sample_ids.extend(batch["sample_id"])
                all_preds.extend(probs.argmax(axis=1))
                all_confs.extend(probs.max(axis=1))

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_confs = np.array(all_confs)

        # Compute prototype similarities once
        emb_t = torch.tensor(all_embeddings, dtype=torch.float32)
        sims_all = (emb_t @ prototypes.T).numpy()  # [N, num_known]
        nearest_sims_all = sims_all.max(axis=1)
        nearest_idx_all = sims_all.argmax(axis=1)

        sweep_metrics = []

        for t in sweep_thresholds:
            # Build results for this threshold
            sweep_results = []
            for i in range(len(all_labels)):
                pred_label = int(all_preds[i])
                label = int(all_labels[i])
                nearest_sim = float(nearest_sims_all[i])
                nearest_idx = int(nearest_idx_all[i])
                nearest_class = proto_class_names[nearest_idx]

                if pred_label == negative_label:
                    final_type = "negative"
                    reason = "classified_as_negative"
                elif nearest_sim < t:
                    final_type = "unknown"
                    reason = "far_from_all_known_prototypes"
                else:
                    final_type = "known"
                    reason = "matched_known_prototype"

                margin = nearest_sim - t
                if label < num_known_classes:
                    if nearest_sim < t:
                        risk = "rejected_known"
                    elif 0 <= margin < 0.03:
                        risk = "near_boundary_known"
                    else:
                        risk = "safe_known"
                else:
                    if nearest_sim >= t:
                        risk = "false_known_negative"
                    elif -0.03 < margin < 0:
                        risk = "near_boundary_negative"
                    else:
                        risk = "safe_rejected_negative"

                sweep_results.append({
                    "sample_id": all_sample_ids[i],
                    "true_label": label,
                    "pred_label": pred_label,
                    "confidence": float(all_confs[i]),
                    "nearest_known_label": nearest_idx,
                    "nearest_known_class": nearest_class,
                    "nearest_similarity": nearest_sim,
                    "final_type": final_type,
                    "final_label": nearest_class if final_type == "known" else final_type,
                    "reason": reason,
                    "margin_to_threshold": margin,
                    "risk_level": risk,
                })

            t_metrics, _ = compute_ood_metrics(
                sweep_results, t, num_known_classes, negative_label,
            )
            sweep_metrics.append(t_metrics)

        # Save sweep results
        sweep_csv_path = os.path.join(args.output_dir, "threshold_sweep_metrics.csv")
        sweep_fields = ["threshold", "known_accept_rate", "known_reject_rate",
                        "known_classification_accuracy_after_accept", "known_overall_correct_rate",
                        "negative_reject_rate", "false_known_rate",
                        "average_nearest_similarity_known", "average_nearest_similarity_negative",
                        "final_macro_f1_on_known_classes", "balanced_score"]
        with open(sweep_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sweep_fields)
            writer.writeheader()
            for m in sweep_metrics:
                writer.writerow({k: m.get(k) for k in sweep_fields})
        print(f"Saved: {sweep_csv_path}")

        sweep_json_path = os.path.join(args.output_dir, "threshold_sweep_metrics.json")
        with open(sweep_json_path, "w") as f:
            json.dump(sweep_metrics, f, indent=2, ensure_ascii=False, default=str)
        print(f"Saved: {sweep_json_path}")

        # Sweep plot
        sweep_plot_path = os.path.join(args.output_dir, "threshold_sweep_plot.png")
        plot_threshold_sweep(sweep_metrics, sweep_plot_path)
        print(f"Saved: {sweep_plot_path}")

        # Sweep HTML report
        sweep_html = build_threshold_sweep_html(sweep_metrics, args.output_dir)
        sweep_html_path = os.path.join(args.output_dir, "threshold_sweep_report.html")
        with open(sweep_html_path, "w", encoding="utf-8") as f:
            f.write(sweep_html)
        print(f"Saved: {sweep_html_path}")

        # Print sweep summary
        print(f"\n  Threshold Sweep Summary:")
        print(f"  {'Threshold':>10} {'KA':>8} {'KR':>8} {'Acc':>8} {'NR':>8} {'FK':>8} {'BS':>8}")
        print(f"  {'-'*58}")
        for m in sweep_metrics:
            nr_str = f"{m['negative_reject_rate']:.4f}" if m.get('negative_reject_rate') is not None else "N/A"
            fk_str = f"{m['false_known_rate']:.4f}" if m.get('false_known_rate') is not None else "N/A"
            print(f"  {m['threshold']:>10.2f} {m['known_accept_rate']:>8.4f} {m['known_reject_rate']:>8.4f} "
                  f"{m['known_classification_accuracy_after_accept']:>8.4f} {nr_str:>8} {fk_str:>8} "
                  f"{m['balanced_score']:>8.4f}")

    # Console summary
    print(f"\n{'='*60}")
    print("OOD EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Split: {args.split}, Samples: {len(results)} (known={n_known}, negative={n_negative})")
    print(f"Strategy: {threshold_data.get('selection_strategy', 'unknown')}")
    print(f"Similarity threshold: {similarity_threshold:.2f}")
    print(f"Known accept rate:        {ood_metrics['known_accept_rate']:.4f}")
    print(f"Known reject rate:        {ood_metrics['known_reject_rate']:.4f}")
    print(f"Known acc (after accept): {ood_metrics['known_classification_accuracy_after_accept']:.4f}")
    if ood_metrics.get("negative_reject_rate") is not None:
        print(f"Negative reject rate:     {ood_metrics['negative_reject_rate']:.4f}")
        print(f"False known rate:         {ood_metrics['false_known_rate']:.4f}")
    if ood_metrics.get("auroc") is not None:
        print(f"AUROC:                    {ood_metrics['auroc']:.4f}")
    if ood_metrics.get("final_macro_f1_on_known_classes") is not None:
        print(f"Final Macro F1 (known):   {ood_metrics['final_macro_f1_on_known_classes']:.4f}")
    print(f"Avg sim (known):          {ood_metrics['average_nearest_similarity_known']:.4f}")
    if ood_metrics.get("average_nearest_similarity_negative") is not None:
        print(f"Avg sim (negative):       {ood_metrics['average_nearest_similarity_negative']:.4f}")

    # Quantile summary
    if result_known_quantiles:
        print(f"\nKnown similarity quantiles:")
        for k, v in result_known_quantiles.items():
            print(f"  {k}: {v:.6f}" if v is not None else f"  {k}: N/A")
    if result_negative_quantiles:
        print(f"Negative similarity quantiles:")
        for k, v in result_negative_quantiles.items():
            print(f"  {k}: {v:.6f}" if v is not None else f"  {k}: N/A")

    # Risk summary
    risk_counts = {}
    for r in results:
        rl = r["risk_level"]
        risk_counts[rl] = risk_counts.get(rl, 0) + 1
    print(f"\nRisk level distribution:")
    for rl, cnt in sorted(risk_counts.items()):
        print(f"  {rl}: {cnt}")

    for w in warnings_list:
        print(f"WARNING: {w}")
    print("=" * 60)


if __name__ == "__main__":
    main()
