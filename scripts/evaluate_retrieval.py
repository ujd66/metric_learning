"""Query-gallery retrieval evaluation for OOD / unknown rejection.

Loads a gallery of embeddings and evaluates retrieval-based inference
on a query split, with optional multi-threshold sweep.

Usage:
    python scripts/evaluate_retrieval.py \
        --config configs/config.yaml \
        --checkpoint outputs/checkpoints/best.pt \
        --gallery outputs/gallery/baseline_train_gallery.pt \
        --threshold-json outputs/prototypes/baseline_threshold_p05.json \
        --split test \
        --output-dir outputs/reports/retrieval_eval_baseline_test

    # With multi-threshold sweep:
    python scripts/evaluate_retrieval.py ... --thresholds 0.85,0.90,0.91,0.92,0.94
"""

import argparse
import csv
import json
import math
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
    for batch in tqdm(loader, desc="Extract queries"):
        points = batch["points"].to(device)
        out = model(points)
        emb = out["embedding"].cpu()
        # L2 normalize
        norms = emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
        emb = emb / norms
        all_embeddings.append(emb)
        all_labels.extend(batch["label"].tolist())
        all_sample_ids.extend(batch["sample_id"])
        all_class_names.extend(batch["class_name"])
    return torch.cat(all_embeddings, dim=0), all_labels, all_sample_ids, all_class_names


def compute_auroc(known_sims, negative_sims):
    from sklearn.metrics import roc_auc_score
    y_true = np.concatenate([np.ones(len(known_sims)), np.zeros(len(negative_sims))])
    y_score = np.concatenate([known_sims, negative_sims])
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def evaluate_at_threshold(query_labels, query_sample_ids, gallery_labels, gallery_sample_ids, gallery_class_names,
                          topk_indices, topk_sims, nearest_sims, threshold,
                          num_known_classes, negative_label, class_names_map):
    """Compute all retrieval metrics at a given threshold."""
    query_results = []

    for i in range(len(query_labels)):
        qlabel = query_labels[i]
        sim = float(nearest_sims[i])
        top1_idx = int(topk_indices[i, 0])
        top1_label = gallery_labels[top1_idx]
        top1_sim = float(topk_sims[i, 0])
        top1_sample_id = gallery_sample_ids[top1_idx]
        top1_class_name = gallery_class_names[top1_idx]

        top5_labels = [int(gallery_labels[int(topk_indices[i, j])]) for j in range(min(5, topk_indices.shape[1]))]
        top5_sims = [float(topk_sims[i, j]) for j in range(min(5, topk_sims.shape[1]))]

        # Decision
        if sim < threshold:
            final_type = "unknown"
            final_label = "unknown"
        else:
            final_type = "known"
            final_label = class_names_map.get(str(top1_label), f"class_{top1_label:03d}")

        # Correctness
        if qlabel < num_known_classes:  # known query
            if final_type == "unknown":
                correct = False
            else:
                correct = (top1_label == qlabel)
        else:  # negative query
            correct = (final_type == "unknown")

        # Risk level
        margin = sim - threshold
        if qlabel < num_known_classes:
            if final_type == "unknown":
                risk_level = "rejected_known"
            elif not correct:
                risk_level = "misclassified_known"
            elif abs(margin) < 0.03:
                risk_level = "near_boundary"
            else:
                risk_level = "safe"
        else:
            if final_type == "known":
                risk_level = "false_known_negative"
            elif abs(margin) < 0.03:
                risk_level = "near_boundary"
            else:
                risk_level = "safe"

        query_results.append({
            "query_sample_id": query_sample_ids[i],
            "true_label": qlabel,
            "true_class_name": class_names_map.get(str(qlabel), "negative" if qlabel == negative_label else f"class_{qlabel:03d}"),
            "final_type": final_type,
            "final_label": final_label,
            "nearest_similarity": sim,
            "top1_neighbor_sample_id": top1_sample_id,
            "top1_neighbor_label": top1_label,
            "top1_neighbor_class_name": top1_class_name,
            "top1_neighbor_source_path": "",
            "top5_neighbor_labels": ";".join(str(l) for l in top5_labels),
            "top5_neighbor_similarities": ";".join(f"{s:.4f}" for s in top5_sims),
            "correct": correct,
            "risk_level": risk_level,
        })

    # Compute metrics
    known_q = [r for r in query_results if r["true_label"] < num_known_classes]
    neg_q = [r for r in query_results if r["true_label"] == negative_label]
    n_known = len(known_q)
    n_neg = len(neg_q)

    known_accepted = [r for r in known_q if r["final_type"] == "known"]
    known_rejected = [r for r in known_q if r["final_type"] == "unknown"]

    ka = len(known_accepted) / n_known if n_known > 0 else 0.0
    kr = len(known_rejected) / n_known if n_known > 0 else 0.0

    # Top1 accuracy on accepted known
    correct_accepted = sum(1 for r in known_accepted if r["correct"])
    top1_acc = correct_accepted / len(known_accepted) if known_accepted else 0.0

    # TopK recall for accepted known
    topk_metrics = {}
    for k_val in [1, 3, 5, 10]:
        recall_sum = 0
        precision_sum = 0
        for r in known_accepted:
            top5_l = [int(x) for x in r["top5_neighbor_labels"].split(";")]
            topk_l = top5_l[:k_val]
            true_l = r["true_label"]
            matches = sum(1 for l in topk_l if l == true_l)
            recall_sum += min(matches, 1)  # binary: at least one match
            precision_sum += matches / k_val
        topk_metrics[f"recall@{k_val}"] = recall_sum / len(known_accepted) if known_accepted else 0.0
        topk_metrics[f"precision@{k_val}"] = precision_sum / len(known_accepted) if known_accepted else 0.0

    # Negative metrics
    neg_reject = None
    false_known = None
    neg_sim_mean = None
    neg_sim_std = None
    if n_neg > 0:
        neg_rejected = [r for r in neg_q if r["final_type"] == "unknown"]
        neg_false = [r for r in neg_q if r["final_type"] == "known"]
        neg_reject = len(neg_rejected) / n_neg
        false_known = len(neg_false) / n_neg
        neg_sims = [r["nearest_similarity"] for r in neg_q]
        neg_sim_mean = float(np.mean(neg_sims))
        neg_sim_std = float(np.std(neg_sims))

    # Overall accuracy with unknown
    all_correct = sum(1 for r in query_results if r["correct"])
    overall_acc = all_correct / len(query_results) if query_results else 0.0

    # Final accuracy: accepted known that are correct + rejected negatives
    final_correct = sum(1 for r in query_results
                       if (r["true_label"] < num_known_classes and r["final_type"] == "known" and r["correct"])
                       or (r["true_label"] == negative_label and r["final_type"] == "unknown"))
    final_acc_with_unknown = final_correct / len(query_results) if query_results else 0.0

    # Macro F1 on known classes (accepted only)
    from sklearn.metrics import f1_score
    macro_f1 = None
    if known_accepted:
        y_true = [r["true_label"] for r in known_accepted]
        y_pred = [r["top1_neighbor_label"] for r in known_accepted]
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # AUROC
    auroc = None
    if n_known > 0 and n_neg > 0:
        known_sims = np.array([r["nearest_similarity"] for r in known_q])
        neg_sims_arr = np.array([r["nearest_similarity"] for r in neg_q])
        auroc = compute_auroc(known_sims, neg_sims_arr)

    # Balanced score
    balanced = ka + (neg_reject if neg_reject is not None else 0.0)

    # Rejection rate
    all_rejected = [r for r in query_results if r["final_type"] == "unknown"]
    rejection_rate = len(all_rejected) / len(query_results) if query_results else 0.0

    metrics = {
        "threshold": threshold,
        "num_queries": len(query_results),
        "num_known_queries": n_known,
        "num_negative_queries": n_neg,
        "known_accept_rate": ka,
        "known_reject_rate": kr,
        "top1_accuracy_on_accepted": top1_acc,
        "negative_reject_rate": neg_reject,
        "false_known_rate": false_known,
        "nearest_gallery_similarity_mean_negative": neg_sim_mean,
        "nearest_gallery_similarity_std_negative": neg_sim_std,
        "overall_accuracy": overall_acc,
        "final_accuracy_with_unknown": final_acc_with_unknown,
        "macro_f1_known_classes": macro_f1,
        "auroc": auroc,
        "balanced_score": balanced,
        "rejection_rate": rejection_rate,
    }
    metrics.update(topk_metrics)

    # Per-class metrics
    per_class = []
    for c in range(num_known_classes):
        cq = [r for r in known_q if r["true_label"] == c]
        support = len(cq)
        if support == 0:
            continue
        accepted = [r for r in cq if r["final_type"] == "known"]
        accept_rate = len(accepted) / support
        correct_c = sum(1 for r in accepted if r["correct"])
        top1_acc_c = correct_c / len(accepted) if accepted else 0.0
        per_class.append({
            "label": c,
            "class_name": class_names_map.get(str(c), f"class_{c:03d}"),
            "support": support,
            "accept_rate": accept_rate,
            "top1_accuracy": top1_acc_c,
        })

    return metrics, per_class, query_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_similarity_histogram(results, output_path, threshold, num_known_classes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    known_sims = [r["nearest_similarity"] for r in results if r["true_label"] < num_known_classes]
    neg_sims = [r["nearest_similarity"] for r in results if r["true_label"] >= num_known_classes]

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1, 50)
    if known_sims:
        ax.hist(known_sims, bins=bins, alpha=0.6, label=f"Known (n={len(known_sims)})",
                color="#3b82f6", edgecolor="white")
    if neg_sims:
        ax.hist(neg_sims, bins=bins, alpha=0.6, label=f"Negative (n={len(neg_sims)})",
                color="#ef4444", edgecolor="white")
    if threshold is not None:
        ax.axvline(x=threshold, color="#10b981", linestyle="--", linewidth=2,
                   label=f"Threshold: {threshold:.2f}")
    ax.set_xlabel("Nearest Gallery Similarity", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Nearest Gallery Similarity: Query Distribution", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(results, class_names_map, num_known_classes, negative_label, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    LABEL_UNKNOWN = 100
    y_true, y_pred = [], []
    for r in results:
        t = r["true_label"]
        if r["final_type"] == "unknown":
            p = LABEL_UNKNOWN
        else:
            p = r["top1_neighbor_label"]
        y_true.append(t)
        y_pred.append(p)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    all_labels = sorted(set(y_true.tolist() + y_pred.tolist()))

    display_names = []
    for l in all_labels:
        if l == LABEL_UNKNOWN:
            display_names.append("unknown")
        elif l == negative_label:
            display_names.append("negative")
        else:
            display_names.append(class_names_map.get(str(l), f"class_{l:03d}"))

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    fig, ax = plt.subplots(figsize=(max(10, len(all_labels) * 0.6), max(8, len(all_labels) * 0.5)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_xticks(range(len(all_labels)))
    ax.set_yticks(range(len(all_labels)))
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(display_names, fontsize=7)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Retrieval Confusion Matrix")

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


def plot_sweep(sweep_metrics, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = [m["threshold"] for m in sweep_metrics]
    ka = [m["known_accept_rate"] for m in sweep_metrics]
    acc = [m["top1_accuracy_on_accepted"] for m in sweep_metrics]
    fa = [m["final_accuracy_with_unknown"] for m in sweep_metrics]
    bs = [m["balanced_score"] for m in sweep_metrics]

    has_neg = any(m.get("negative_reject_rate") is not None for m in sweep_metrics)
    nr = [m.get("negative_reject_rate") or 0 for m in sweep_metrics]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(ts, ka, "o-", label="Known Accept Rate", color="#3b82f6", linewidth=2, markersize=6)
    ax.plot(ts, acc, "s-", label="Top1 Acc (accepted)", color="#10b981", linewidth=2, markersize=6)
    ax.plot(ts, fa, "D-", label="Final Acc (w/ unknown)", color="#8b5cf6", linewidth=2, markersize=6)
    if has_neg:
        ax.plot(ts, nr, "^-", label="Negative Reject Rate", color="#f59e0b", linewidth=2, markersize=6)
    ax.plot(ts, bs, "v--", label="Balanced Score", color="#ef4444", linewidth=1.5, markersize=5)
    ax.axhline(y=0.95, color="#3b82f6", linestyle=":", alpha=0.5, label="95% KA target")

    for i, t in enumerate(ts):
        ax.annotate(f"{t:.2f}", (t, ka[i]), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8)

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Rate")
    ax.set_title("Retrieval Threshold Sweep")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# HTML reports
# ---------------------------------------------------------------------------

def build_report_html(metrics, per_class, results, sim_hist_path, cm_path,
                      class_names_map, threshold, split):
    import base64
    import html as html_mod
    from datetime import datetime

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _b64(path):
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            data = f.read()
        return f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"

    def _f(v, d=4):
        if v is None: return "N/A"
        return f"{v:.{d}f}"

    CSS = """
:root { --bg:#f5f7fa;--card-bg:#fff;--text:#1a1a2e;--text2:#555;--border:#e0e4e8;
        --accent:#3b82f6;--accent-light:#dbeafe;--success:#10b981;--warning:#f59e0b;--danger:#ef4444; }
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--text);line-height:1.6;padding:24px}
.container{max-width:1200px;margin:0 auto}
h1{font-size:1.8rem;font-weight:700;margin-bottom:8px}h2{font-size:1.3rem;font-weight:600;margin-top:32px;margin-bottom:16px;padding-bottom:8px;border-bottom:2px solid var(--accent)}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:16px;margin-bottom:24px}
.card{background:var(--card-bg);border-radius:10px;padding:16px;text-align:center;border:1px solid var(--border)}
.card-value{font-size:1.4rem;font-weight:700;color:var(--accent)}.card-label{font-size:0.75rem;color:var(--text2);text-transform:uppercase}
.table-wrap{overflow-x:auto;margin-bottom:24px;background:var(--card-bg);border-radius:8px;border:1px solid var(--border)}
table{width:100%;border-collapse:collapse;font-size:0.85rem}th{background:#f8f9fb;padding:10px 12px;text-align:left;font-weight:600;border-bottom:2px solid var(--border)}
td{padding:8px 12px;border-bottom:1px solid var(--border)}tr:last-child td{border-bottom:none}tr:hover td{background:#f8f9fb}
.img-section{margin-bottom:24px;background:var(--card-bg);border-radius:8px;border:1px solid var(--border);padding:16px;text-align:center}
.img-section img{max-width:100%;height:auto}footer{margin-top:40px;text-align:center;color:var(--text2);font-size:0.8rem}
"""

    cards = ""
    for label, key in [("Known Accept Rate", "known_accept_rate"), ("Known Reject Rate", "known_reject_rate"),
                       ("Top1 Acc (accepted)", "top1_accuracy_on_accepted"), ("Negative Reject Rate", "negative_reject_rate"),
                       ("Final Acc (w/ unknown)", "final_accuracy_with_unknown"), ("AUROC", "auroc"),
                       ("Balanced Score", "balanced_score"), ("Macro F1 (known)", "macro_f1_known_classes")]:
        v = metrics.get(key)
        cards += f'<div class="card"><div class="card-value">{_f(v) if v is not None else "N/A"}</div><div class="card-label">{html_mod.escape(label)}</div></div>\n'

    pc_rows = ""
    for row in per_class:
        pc_rows += f'<tr><td>{html_mod.escape(row["class_name"])}</td><td>{row["support"]}</td><td>{_f(row["accept_rate"])}</td><td>{_f(row["top1_accuracy"])}</td></tr>\n'
    pc_table = f'<div class="table-wrap"><table><tr><th>Class</th><th>Support</th><th>Accept Rate</th><th>Top1 Acc</th></tr>{pc_rows}</table></div>'

    def _img(title, path):
        uri = _b64(path)
        if uri:
            return f'<div class="img-section"><h3>{html_mod.escape(title)}</h3><img src="{uri}"></div>'
        return ""

    report = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0"><title>Retrieval Evaluation Report</title>
<style>{CSS}</style></head><body><div class="container">
<h1>Query-Gallery Retrieval Evaluation</h1>
<p style="color:var(--text2);margin-bottom:24px">Generated at {html_mod.escape(ts)} | Split: {html_mod.escape(split)} | Threshold: {_f(threshold)}</p>
<h2>Core Metrics</h2><div class="cards">{cards}</div>
<h2>Per-class Metrics</h2>{pc_table}
<h2>Visualizations</h2>
{_img("Nearest Similarity Distribution", sim_hist_path)}
{_img("Retrieval Confusion Matrix", cm_path)}
<footer>Retrieval Evaluation Report &mdash; pointcloud_metric_learning</footer>
</div></body></html>"""
    return report


def build_sweep_html(sweep_metrics, output_dir):
    import base64
    import html as html_mod
    from datetime import datetime

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Find bests
    best_ka_095 = [m for m in sweep_metrics if m["known_accept_rate"] >= 0.95]
    best_nr = max(sweep_metrics, key=lambda m: m.get("negative_reject_rate") or 0)
    best_bs = max(sweep_metrics, key=lambda m: m["balanced_score"])
    if best_ka_095:
        recommended = max(best_ka_095, key=lambda m: m.get("negative_reject_rate") or 0)
    else:
        recommended = best_bs

    CSS = """
:root{--bg:#f5f7fa;--card-bg:#fff;--text:#1a1a2e;--text2:#555;--border:#e0e4e8;--accent:#3b82f6;--success:#10b981}
*{margin:0;padding:0;box-sizing:border-box}body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--text);line-height:1.6;padding:24px}
.container{max-width:1400px;margin:0 auto}h1{font-size:1.8rem;font-weight:700;margin-bottom:8px}h2{font-size:1.3rem;font-weight:600;margin-top:32px;margin-bottom:16px;padding-bottom:8px;border-bottom:2px solid var(--accent)}
.table-wrap{overflow-x:auto;margin-bottom:24px;background:var(--card-bg);border-radius:8px;border:1px solid var(--border)}
table{width:100%;border-collapse:collapse;font-size:0.85rem}th{background:#f8f9fb;padding:10px 12px;text-align:left;font-weight:600;border-bottom:2px solid var(--border)}
td{padding:8px 12px;border-bottom:1px solid var(--border)}tr:last-child td{border-bottom:none}
tr.rec td{background:#ecfdf5;font-weight:600}tr.best-nr td{background:#fef3c7}tr.best-bs td{background:#dbeafe}
.img-section{margin-bottom:24px;background:var(--card-bg);border-radius:8px;border:1px solid var(--border);padding:16px;text-align:center}
.img-section img{max-width:100%;height:auto}footer{margin-top:40px;text-align:center;color:var(--text2);font-size:0.8rem}
"""

    def _f(v, d=4):
        if v is None: return "N/A"
        return f"{v:.{d}f}"

    rec_html = f"""<div style="background:#ecfdf5;border:2px solid var(--success);border-radius:10px;padding:20px;margin-bottom:24px">
<h3 style="color:#065f46;margin-bottom:8px">Recommended Threshold: {recommended['threshold']:.2f}</h3>
<p>KA={_f(recommended['known_accept_rate'])} | Top1 Acc={_f(recommended['top1_accuracy_on_accepted'])} |
NR={_f(recommended.get('negative_reject_rate'))} | Balanced={_f(recommended['balanced_score'])}</p></div>"""

    header = "<tr><th>Threshold</th><th>KA</th><th>KR</th><th>Top1 Acc</th><th>Final Acc</th><th>NR</th><th>FK</th><th>Balanced</th><th>Notes</th></tr>"
    rows = ""
    for m in sweep_metrics:
        notes = []
        row_cls = ""
        if m["threshold"] == recommended["threshold"]:
            notes.append("RECOMMENDED"); row_cls = "rec"
        if m["threshold"] == best_nr["threshold"]:
            notes.append("BEST_NR")
            if not row_cls: row_cls = "best-nr"
        if m["threshold"] == best_bs["threshold"]:
            notes.append("BEST_BS")
            if not row_cls: row_cls = "best-bs"
        if m["known_accept_rate"] >= 0.95:
            notes.append("KA>=0.95")

        rows += f'<tr class="{row_cls}"><td>{m["threshold"]:.2f}</td><td>{_f(m["known_accept_rate"])}</td><td>{_f(m["known_reject_rate"])}</td>'
        rows += f'<td>{_f(m["top1_accuracy_on_accepted"])}</td><td>{_f(m["final_accuracy_with_unknown"])}</td>'
        rows += f'<td>{_f(m.get("negative_reject_rate"))}</td><td>{_f(m.get("false_known_rate"))}</td>'
        rows += f'<td>{_f(m["balanced_score"])}</td><td>{", ".join(notes)}</td></tr>\n'

    sweep_img_path = os.path.join(output_dir, "threshold_sweep_retrieval_plot.png")
    img_html = ""
    if os.path.exists(sweep_img_path):
        with open(sweep_img_path, "rb") as f:
            img_html = f'<div class="img-section"><h3>Sweep Plot</h3><img src="data:image/png;base64,{base64.b64encode(f.read()).decode("ascii")}"></div>'

    report = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0"><title>Retrieval Threshold Sweep</title>
<style>{CSS}</style></head><body><div class="container">
<h1>Retrieval Threshold Sweep Report</h1>
<p style="color:var(--text2);margin-bottom:24px">Generated at {html_mod.escape(ts)}</p>
{rec_html}<h2>Threshold Comparison</h2>
<div class="table-wrap"><table>{header}{rows}</table></div>
<h2>Sweep Visualization</h2>{img_html}
<footer>Retrieval Threshold Sweep &mdash; pointcloud_metric_learning</footer>
</div></body></html>"""
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Query-gallery retrieval evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--gallery", type=str, required=True, help="Path to gallery .pt file")
    parser.add_argument("--threshold-json", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Comma-separated thresholds for sweep (e.g., '0.85,0.90,0.91,0.92,0.94')")
    parser.add_argument("--topk", type=str, default="1,3,5,10",
                        help="Comma-separated K values for top-K metrics")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    num_known_classes = cfg["num_known_classes"]
    negative_label = cfg["negative_label"]
    topk_values = [int(k) for k in args.topk.split(",")]
    max_k = max(topk_values)

    if args.output_dir is None:
        args.output_dir = f"outputs/reports/retrieval_eval_{args.split}"

    # Load gallery
    print(f"Loading gallery from {args.gallery} ...")
    gallery = torch.load(args.gallery, map_location="cpu", weights_only=False)
    gallery_embeddings = gallery["embeddings"]  # [M, D]
    gallery_labels = gallery["labels"].tolist()
    gallery_sample_ids = gallery["sample_ids"]
    gallery_class_names = gallery.get("class_names", [f"class_{l:03d}" for l in gallery_labels])
    print(f"  Gallery size: {len(gallery_labels)} samples")

    # Load threshold
    print(f"Loading threshold from {args.threshold_json} ...")
    with open(args.threshold_json) as f:
        threshold_data = json.load(f)
    similarity_threshold = threshold_data["selected_threshold"]
    print(f"  Threshold: {similarity_threshold:.2f}")

    # Load class names
    class_names_path = os.path.join(os.path.dirname(args.config), "class_names.json")
    class_names_map = {}
    if os.path.exists(class_names_path):
        with open(class_names_path) as f:
            class_names_map = json.load(f)

    # Load model
    model = MetricPointNet(
        input_channels=cfg["input_channels"],
        num_classes=cfg["num_classes"],
        embedding_dim=cfg["embedding_dim"],
    ).to(device)
    load_checkpoint(args.checkpoint, model)

    # Load query dataset
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

    # Extract query embeddings
    print(f"\nExtracting query embeddings from {len(dataset)} samples ({args.split} split) ...")
    query_embeddings, query_labels, query_sample_ids, query_class_names = extract_embeddings(model, loader, device)
    # Compute all similarities once: [N_queries, M_gallery]
    print(f"Computing similarities: {query_embeddings.shape[0]} queries x {gallery_embeddings.shape[0]} gallery ...")
    sims = query_embeddings @ gallery_embeddings.T  # [N, M]

    # Top-K indices and similarities
    topk_sims, topk_indices = sims.topk(k=max_k, dim=1)  # [N, max_k]
    nearest_sims = topk_sims[:, 0]  # [N]

    # =========================================================================
    # Main evaluation
    # =========================================================================
    print(f"\nEvaluating at threshold={similarity_threshold:.2f} ...")
    metrics, per_class, results = evaluate_at_threshold(
        query_labels, query_sample_ids, gallery_labels, gallery_sample_ids, gallery_class_names,
        topk_indices, topk_sims, nearest_sims, similarity_threshold,
        num_known_classes, negative_label, class_names_map,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "retrieval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved: {metrics_path}")

    # Per-sample CSV
    csv_path = os.path.join(args.output_dir, "per_sample_retrieval.csv")
    fieldnames = ["query_sample_id", "true_label", "true_class_name", "final_type", "final_label",
                  "nearest_similarity", "top1_neighbor_sample_id", "top1_neighbor_label",
                  "top1_neighbor_class_name", "top1_neighbor_source_path",
                  "top5_neighbor_labels", "top5_neighbor_similarities",
                  "correct", "risk_level"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Saved: {csv_path}")

    # Per-class CSV
    pc_path = os.path.join(args.output_dir, "per_class_retrieval_metrics.csv")
    with open(pc_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "class_name", "support", "accept_rate", "top1_accuracy"])
        writer.writeheader()
        writer.writerows(per_class)
    print(f"Saved: {pc_path}")

    # TopK CSV
    topk_path = os.path.join(args.output_dir, "topk_metrics.csv")
    topk_rows = []
    for k in topk_values:
        topk_rows.append({
            "k": k,
            "recall": metrics.get(f"recall@{k}"),
            "precision": metrics.get(f"precision@{k}"),
        })
    with open(topk_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "recall", "precision"])
        writer.writeheader()
        writer.writerows(topk_rows)
    print(f"Saved: {topk_path}")

    # Plots
    sim_hist_path = os.path.join(args.output_dir, "nearest_similarity_histogram.png")
    plot_similarity_histogram(results, sim_hist_path, similarity_threshold, num_known_classes)
    print(f"Saved: {sim_hist_path}")

    cm_path = os.path.join(args.output_dir, "retrieval_confusion_matrix.png")
    plot_confusion_matrix(results, class_names_map, num_known_classes, negative_label, cm_path)
    print(f"Saved: {cm_path}")

    # HTML report
    report_html = build_report_html(
        metrics, per_class, results, sim_hist_path, cm_path,
        class_names_map, similarity_threshold, args.split,
    )
    with open(os.path.join(args.output_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(report_html)
    print(f"Saved: {os.path.join(args.output_dir, 'report.html')}")

    # =========================================================================
    # Multi-threshold sweep
    # =========================================================================
    if args.thresholds:
        sweep_thresholds = [float(t.strip()) for t in args.thresholds.split(",")]
        print(f"\n{'='*60}")
        print(f"RETRIEVAL THRESHOLD SWEEP: {sweep_thresholds}")
        print(f"{'='*60}")

        sweep_metrics = []
        for t in sweep_thresholds:
            t_m, _, _ = evaluate_at_threshold(
                query_labels, query_sample_ids, gallery_labels, gallery_sample_ids, gallery_class_names,
                topk_indices, topk_sims, nearest_sims, t,
                num_known_classes, negative_label, class_names_map,
            )
            sweep_metrics.append(t_m)

        # Save sweep CSV
        sweep_csv_path = os.path.join(args.output_dir, "threshold_sweep_retrieval.csv")
        sweep_fields = ["threshold", "known_accept_rate", "known_reject_rate",
                        "top1_accuracy_on_accepted", "final_accuracy_with_unknown",
                        "negative_reject_rate", "false_known_rate", "balanced_score"]
        with open(sweep_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sweep_fields)
            writer.writeheader()
            for m in sweep_metrics:
                writer.writerow({k: m.get(k) for k in sweep_fields})
        print(f"Saved: {sweep_csv_path}")

        # Sweep plot
        sweep_plot_path = os.path.join(args.output_dir, "threshold_sweep_retrieval_plot.png")
        plot_sweep(sweep_metrics, sweep_plot_path)
        print(f"Saved: {sweep_plot_path}")

        # Sweep HTML
        sweep_html = build_sweep_html(sweep_metrics, args.output_dir)
        sweep_html_path = os.path.join(args.output_dir, "threshold_sweep_retrieval.html")
        with open(sweep_html_path, "w", encoding="utf-8") as f:
            f.write(sweep_html)
        print(f"Saved: {sweep_html_path}")

        # Print summary
        print(f"\n  Sweep Summary:")
        print(f"  {'Threshold':>10} {'KA':>8} {'KR':>8} {'Top1':>8} {'Final':>8} {'NR':>8} {'BS':>8}")
        print(f"  {'-'*58}")
        for m in sweep_metrics:
            nr_str = _f_console(m.get("negative_reject_rate"))
            print(f"  {m['threshold']:>10.2f} {m['known_accept_rate']:>8.4f} {m['known_reject_rate']:>8.4f} "
                  f"{m['top1_accuracy_on_accepted']:>8.4f} {m['final_accuracy_with_unknown']:>8.4f} "
                  f"{nr_str:>8} {m['balanced_score']:>8.4f}")

    # Console summary
    print(f"\n{'='*60}")
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Split: {args.split}, Queries: {len(query_labels)} (known={metrics['num_known_queries']}, neg={metrics['num_negative_queries']})")
    print(f"Gallery: {len(gallery_labels)} samples")
    print(f"Threshold: {similarity_threshold:.2f}")
    print(f"Known accept rate:      {metrics['known_accept_rate']:.4f}")
    print(f"Known reject rate:      {metrics['known_reject_rate']:.4f}")
    print(f"Top1 acc (accepted):    {metrics['top1_accuracy_on_accepted']:.4f}")
    if metrics.get("negative_reject_rate") is not None:
        print(f"Negative reject rate:   {metrics['negative_reject_rate']:.4f}")
        print(f"False known rate:       {metrics['false_known_rate']:.4f}")
    print(f"Final acc (w/ unknown): {metrics['final_accuracy_with_unknown']:.4f}")
    if metrics.get("auroc") is not None:
        print(f"AUROC:                  {metrics['auroc']:.4f}")
    if metrics.get("macro_f1_known_classes") is not None:
        print(f"Macro F1 (known):       {metrics['macro_f1_known_classes']:.4f}")
    print(f"Balanced score:         {metrics['balanced_score']:.4f}")
    for k in topk_values:
        print(f"Recall@{k}:              {metrics.get(f'recall@{k}', 'N/A')}")
    print("=" * 60)


def _f_console(v, d=4):
    if v is None:
        return "N/A"
    return f"{v:.{d}f}"


if __name__ == "__main__":
    main()
