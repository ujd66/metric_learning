"""Evaluate OOD / unknown / negative rejection on a given split.

Loads prototypes and threshold, runs full inference on the test split,
and generates comprehensive metrics, plots, and HTML report.

Usage:
    python scripts/evaluate_ood.py \
        --config configs/config.yaml \
        --checkpoint outputs/checkpoints/best.pt \
        --prototypes outputs/prototypes/baseline_prototypes.pt \
        --threshold-json outputs/prototypes/baseline_threshold.json \
        --split test \
        --output-dir outputs/reports/ood_eval_baseline_test
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
from src.models.metric_model import MetricPointNet
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config


def collate_fn(batch):
    points = torch.stack([b["points"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    sample_ids = [b["sample_id"] for b in batch]
    return {"points": points, "label": labels, "sample_id": sample_ids}


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


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_nearest_similarity_histogram(results, output_path, class_names_map):
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
    ax.set_xlabel("Nearest Known Prototype Similarity", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Nearest Similarity Distribution: Known vs Negative", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_curve(threshold_curve, selected_threshold, output_path):
    """Plot threshold curve with known_accept_rate and negative_reject_rate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = [tc["threshold"] for tc in threshold_curve]
    ka = [tc["known_accept_rate"] for tc in threshold_curve]
    nr = [tc["negative_reject_rate"] for tc in threshold_curve]
    bs = [tc["balanced_score"] for tc in threshold_curve]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ts, ka, label="Known Accept Rate", color="#3b82f6", linewidth=2)
    has_negative = any(v is not None for v in nr)
    if has_negative:
        nr_plot = [v if v is not None else 0 for v in nr]
        ax.plot(ts, nr_plot, label="Negative Reject Rate", color="#10b981", linewidth=2)
    ax.plot(ts, bs, label="Balanced Score", color="#f59e0b", linewidth=1.5, linestyle="--")
    ax.axvline(x=selected_threshold, color="#ef4444", linestyle=":", linewidth=2,
               label=f"Selected: {selected_threshold:.2f}")
    ax.set_xlabel("Similarity Threshold", fontsize=12)
    ax.set_ylabel("Rate", fontsize=12)
    ax.set_title("Threshold Search Curve", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(ts[0], ts[-1])
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_final_confusion_matrix(results, class_names_map, num_known_classes, output_path):
    """Confusion matrix with known classes + negative + unknown."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    # Build label set: known classes (0..18) + negative (19) + unknown (100)
    LABEL_UNKNOWN = 100

    y_true = []
    y_pred = []
    for r in results:
        true_lbl = r["true_label"]
        if r["final_type"] == "unknown":
            pred_lbl = LABEL_UNKNOWN
        elif r["final_type"] == "negative":
            pred_lbl = 19  # negative label
        else:
            pred_lbl = r["nearest_known_label"]
        y_true.append(true_lbl)
        y_pred.append(pred_lbl)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Determine present labels
    present_true = sorted(set(y_true.tolist()))
    present_pred = sorted(set(y_pred.tolist()))
    all_labels = sorted(set(present_true + present_pred))

    # Build display names
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

    # Annotate cells
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
# HTML report builder
# ---------------------------------------------------------------------------

def build_ood_html_report(
    checkpoint, prototypes_path, threshold, split,
    num_samples, num_known, num_negative,
    metrics, per_class_rows, warnings_list,
    sim_hist_path, threshold_curve_path, cm_path,
    class_names_map, threshold_data,
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

    # Per-class table
    pc_header = "<th>Class</th><th>Support</th><th>Accept Rate</th><th>Acc (accepted)</th><th>Avg Similarity</th>"
    pc_rows = ""
    for row in per_class_rows:
        pc_rows += f'<tr>'
        pc_rows += f'<td>{html_mod.escape(row["class_name"])}</td>'
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
    threshold_curve = threshold_data["threshold_curve"]
    print(f"  Similarity threshold: {similarity_threshold:.2f}")

    # Load model
    model = MetricPointNet(
        input_channels=cfg["input_channels"],
        num_classes=cfg["num_classes"],
        embedding_dim=cfg["embedding_dim"],
    ).to(device)
    load_checkpoint(args.checkpoint, model)

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

    # Extract predictions
    print(f"\nEvaluating {len(dataset)} samples ({args.split} split) ...")
    results = extract_predictions(
        model, loader, device,
        prototypes, proto_class_names,
        similarity_threshold, num_known_classes, negative_label,
    )

    # Load class names
    class_names_path = os.path.join(os.path.dirname(args.config), "class_names.json")
    class_names_map = {}
    if os.path.exists(class_names_path):
        with open(class_names_path) as f:
            class_names_map = json.load(f)

    # Compute metrics
    known_results = [r for r in results if r["true_label"] < num_known_classes]
    negative_results = [r for r in results if r["true_label"] == negative_label]

    n_known = len(known_results)
    n_negative = len(negative_results)

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

    # Known metrics
    known_accepted = [r for r in known_results if r["final_type"] == "known"]
    known_rejected = [r for r in known_results if r["final_type"] != "known"]

    known_accept_rate = len(known_accepted) / n_known if n_known > 0 else 0.0
    known_reject_rate = len(known_rejected) / n_known if n_known > 0 else 0.0

    # Known classification accuracy among accepted
    known_correct_after_accept = sum(
        1 for r in known_accepted if r["nearest_known_label"] == r["true_label"]
    )
    known_class_acc_after_accept = (
        known_correct_after_accept / len(known_accepted) if known_accepted else 0.0
    )

    # Known overall correct (accepted + correct)
    known_overall_correct = known_correct_after_accept
    known_overall_correct_rate = known_overall_correct / n_known if n_known > 0 else 0.0

    # Average similarity
    avg_sim_known = np.mean([r["nearest_similarity"] for r in known_results]) if known_results else 0.0
    avg_sim_negative = np.mean([r["nearest_similarity"] for r in negative_results]) if negative_results else None

    # Negative metrics
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

    # AUROC
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
        cname = class_names_map.get(str(c), proto_class_names[c] if c < len(proto_class_names) else f"class_{c:03d}")
        per_class_rows.append({
            "label": c,
            "class_name": cname,
            "support": support,
            "accept_rate": accept_rate,
            "accuracy_after_accept": acc_after_accept,
            "avg_similarity": float(avg_sim),
        })

    # Final macro F1 on known classes (for accepted known samples)
    from sklearn.metrics import f1_score
    final_known_f1 = None
    if known_accepted:
        y_true = [r["true_label"] for r in known_accepted]
        y_pred = [r["nearest_known_label"] for r in known_accepted]
        final_known_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Rejection rate (all samples)
    all_rejected = [r for r in results if r["final_type"] in ("unknown", "negative")]
    rejection_rate = len(all_rejected) / len(results) if results else 0.0

    # Build metrics dict
    ood_metrics = {
        "similarity_threshold": similarity_threshold,
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
        "rejection_rate": rejection_rate,
        "average_nearest_similarity_known": float(avg_sim_known),
        "average_nearest_similarity_negative": float(avg_sim_negative) if avg_sim_negative is not None else None,
        "auroc": auroc,
    }

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    # ood_metrics.json
    metrics_path = os.path.join(args.output_dir, "ood_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(ood_metrics, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved: {metrics_path}")

    # per_sample_predictions.csv
    csv_path = os.path.join(args.output_dir, "per_sample_predictions.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved: {csv_path}")

    # per_class_ood_metrics.csv
    pc_csv_path = os.path.join(args.output_dir, "per_class_ood_metrics.csv")
    with open(pc_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "class_name", "support",
                                                "accept_rate", "accuracy_after_accept", "avg_similarity"])
        writer.writeheader()
        writer.writerows(per_class_rows)
    print(f"Saved: {pc_csv_path}")

    # Plots
    sim_hist_path = os.path.join(args.output_dir, "nearest_similarity_histogram.png")
    plot_nearest_similarity_histogram(results, sim_hist_path, class_names_map)
    print(f"Saved: {sim_hist_path}")

    tc_path = os.path.join(args.output_dir, "threshold_curve.png")
    plot_threshold_curve(threshold_curve, similarity_threshold, tc_path)
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
    )
    report_path = os.path.join(args.output_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_html)
    print(f"Saved: {report_path}")

    # Console summary
    print("\n" + "=" * 60)
    print("OOD EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Split: {args.split}, Samples: {len(results)} (known={n_known}, negative={n_negative})")
    print(f"Similarity threshold: {similarity_threshold:.2f}")
    print(f"Known accept rate:        {known_accept_rate:.4f}")
    print(f"Known reject rate:        {known_reject_rate:.4f}")
    print(f"Known acc (after accept): {known_class_acc_after_accept:.4f}")
    if neg_reject_rate is not None:
        print(f"Negative reject rate:     {neg_reject_rate:.4f}")
        print(f"False known rate:         {neg_false_known_rate:.4f}")
    if auroc is not None:
        print(f"AUROC:                    {auroc:.4f}")
    if final_known_f1 is not None:
        print(f"Final Macro F1 (known):   {final_known_f1:.4f}")
    print(f"Avg sim (known):          {avg_sim_known:.4f}")
    if avg_sim_negative is not None:
        print(f"Avg sim (negative):       {avg_sim_negative:.4f}")
    for w in warnings_list:
        print(f"WARNING: {w}")
    print("=" * 60)


if __name__ == "__main__":
    main()
