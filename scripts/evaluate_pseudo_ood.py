"""Pseudo-OOD evaluation using leave-one-class-out methodology.

When real negative samples are insufficient, this script simulates unknown
samples by temporarily removing each known class's prototype and treating
its test samples as pseudo-unknown queries.

Usage:
    python scripts/evaluate_pseudo_ood.py \
        --config configs/config.yaml \
        --checkpoint outputs/checkpoints/best.pt \
        --split test \
        --output-dir outputs/reports/pseudo_ood_eval
"""

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
    sample_ids = [b["sample_id"] for b in batch]
    return {"points": points, "label": labels, "sample_id": sample_ids}


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    all_embeddings = []
    all_labels = []
    all_sample_ids = []
    for batch in tqdm(loader, desc="Extract"):
        points = batch["points"].to(device)
        out = model(points)
        all_embeddings.append(out["embedding"].cpu())
        all_labels.extend(batch["label"].tolist())
        all_sample_ids.extend(batch["sample_id"])
    embeddings = torch.cat(all_embeddings, dim=0)
    # L2 normalize
    norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
    embeddings = embeddings / norms
    return embeddings, all_labels, all_sample_ids


def evaluate_leave_one_out(prototypes, proto_class_names, embeddings_np,
                           labels, left_out_class, threshold):
    """Evaluate with one class left out of prototypes."""
    # Create mask for remaining prototypes
    remaining_mask = [i != left_out_class for i in range(prototypes.shape[0])]
    remaining_protos = prototypes[remaining_mask]
    remaining_labels = [i for i in range(prototypes.shape[0]) if i != left_out_class]

    if remaining_protos.shape[0] == 0:
        return None

    # Find samples of the left-out class
    query_mask = np.array([l == left_out_class for l in labels])
    query_indices = np.where(query_mask)[0]
    if len(query_indices) == 0:
        return None

    query_embs = embeddings_np[query_mask]

    # Compute similarities: [num_queries, num_remaining_protos]
    sims = query_embs @ remaining_protos.T  # cosine sim (both L2-normed)
    max_vals = sims.max(axis=1)
    max_idx = sims.argmax(axis=1)
    nearest_labels = [remaining_labels[j] for j in max_idx]

    rejected = int((max_vals < threshold).sum())
    total = len(query_indices)
    reject_rate = rejected / total
    false_known_rate = 1.0 - reject_rate

    # Most common false known class
    false_known_labels = [nearest_labels[i] for i in range(total)
                          if max_vals[i] >= threshold]
    if false_known_labels:
        from collections import Counter
        most_common = Counter(false_known_labels).most_common(1)[0][0]
    else:
        most_common = None

    avg_nearest_sim = float(max_vals.mean())

    return {
        "pseudo_unknown_class": int(left_out_class),
        "support": total,
        "reject_rate": reject_rate,
        "false_known_rate": false_known_rate,
        "most_common_false_known_class": int(most_common) if most_common is not None else None,
        "avg_nearest_similarity_to_remaining": avg_nearest_sim,
    }


def main():
    parser = argparse.ArgumentParser(description="Pseudo-OOD leave-one-class-out evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--prototypes", type=str,
                        default="outputs/prototypes/baseline_prototypes.pt")
    parser.add_argument("--threshold-json", type=str,
                        default="outputs/prototypes/baseline_threshold_p05.json")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--min-support", type=int, default=3,
                        help="Minimum test samples for a class to be included")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/reports/pseudo_ood_eval")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    num_known_classes = cfg["num_known_classes"]
    negative_label = cfg["negative_label"]

    # Supported classes
    supported_cfg = cfg.get("supported_classes", {})
    supported_known_labels = set(supported_cfg.get("supported_known_labels",
                                                    list(range(num_known_classes))))
    unsupported_known_labels = set(supported_cfg.get("unsupported_known_labels", []))

    # Load class names
    class_names_path = os.path.join(os.path.dirname(args.config), "class_names.json")
    class_names = {}
    if os.path.exists(class_names_path):
        with open(class_names_path) as f:
            class_names = json.load(f)

    def get_class_name(label):
        return class_names.get(str(label), f"class_{label:03d}")

    # Load threshold
    with open(args.threshold_json) as f:
        threshold_data = json.load(f)
    threshold = threshold_data["selected_threshold"]

    # Load prototypes
    proto_data = torch.load(args.prototypes, map_location="cpu", weights_only=False)
    prototypes_tensor = proto_data["prototypes"]  # [num_known, D]
    proto_class_names_list = proto_data["class_names"]
    prototypes_np = prototypes_tensor.numpy()

    # Load model and extract embeddings
    print(f"Loading model from {args.checkpoint} ...")
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
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        num_workers=4, collate_fn=collate_fn)

    print(f"Extracting embeddings from {len(dataset)} samples ({args.split}) ...")
    embeddings, labels, sample_ids = extract_embeddings(model, loader, device)
    embeddings_np = embeddings.numpy()
    labels_np = np.array(labels)

    # Count per class in test split
    class_counts = {}
    for l in labels:
        if l < num_known_classes:
            class_counts[l] = class_counts.get(l, 0) + 1

    # Determine which classes to evaluate
    eval_classes = []
    for c in sorted(supported_known_labels):
        if c in unsupported_known_labels:
            continue
        cnt = class_counts.get(c, 0)
        if cnt >= args.min_support:
            eval_classes.append(c)
        else:
            print(f"  Skipping class {c} ({get_class_name(c)}): only {cnt} test samples (< {args.min_support})")

    print(f"\nThreshold: {threshold:.4f}")
    print(f"Classes to evaluate (support >= {args.min_support}): {len(eval_classes)}")

    # Run leave-one-out
    per_class_results = []
    for left_out in eval_classes:
        result = evaluate_leave_one_out(
            prototypes_np, proto_class_names_list, embeddings_np,
            labels, left_out, threshold,
        )
        if result is not None:
            result["class_name"] = get_class_name(left_out)
            per_class_results.append(result)

    # Also evaluate known accept rate when other class is left out
    # (for all non-left-out classes, check they're still accepted)
    known_accept_when_left_out = []
    for left_out in eval_classes:
        remaining_mask = [i != left_out for i in range(prototypes_np.shape[0])]
        remaining_protos = prototypes_np[remaining_mask]
        remaining_labels = [i for i in range(prototypes_np.shape[0]) if i != left_out]

        # Check all other supported classes
        other_known_indices = [i for i, l in enumerate(labels) if l in supported_known_labels and l != left_out and l != negative_label]
        if not other_known_indices:
            continue
        other_embs = embeddings_np[other_known_indices]
        other_labels_list = [labels[i] for i in other_known_indices]

        sims = other_embs @ remaining_protos.T
        max_vals = sims.max(axis=1)
        accepted = int((max_vals >= threshold).sum())
        ka_rate = accepted / len(other_known_indices)
        known_accept_when_left_out.append(ka_rate)

    # Compute summary
    reject_rates = [r["reject_rate"] for r in per_class_results]
    summary = {
        "threshold": threshold,
        "min_support": args.min_support,
        "num_classes_evaluated": len(per_class_results),
        "classes_evaluated": [r["pseudo_unknown_class"] for r in per_class_results],
        "skipped_classes": sorted(supported_known_labels - set(eval_classes)),
        "unsupported_classes": sorted(unsupported_known_labels),
        "mean_pseudo_unknown_reject_rate": float(np.mean(reject_rates)) if reject_rates else None,
        "median_pseudo_unknown_reject_rate": float(np.median(reject_rates)) if reject_rates else None,
        "min_pseudo_unknown_reject_rate": float(np.min(reject_rates)) if reject_rates else None,
        "max_pseudo_unknown_reject_rate": float(np.max(reject_rates)) if reject_rates else None,
        "known_accept_rate_when_other_class_left_out": float(np.mean(known_accept_when_left_out)) if known_accept_when_left_out else None,
        "per_class": per_class_results,
    }

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    # JSON
    with open(os.path.join(args.output_dir, "pseudo_ood_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # CSV
    import csv
    csv_path = os.path.join(args.output_dir, "per_class_pseudo_ood.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pseudo_unknown_class", "class_name", "support", "reject_rate",
            "false_known_rate", "most_common_false_known_class",
            "avg_nearest_similarity_to_remaining",
        ])
        writer.writeheader()
        for r in per_class_results:
            writer.writerow(r)

    # Plots
    _plot_bar(per_class_results, args.output_dir, get_class_name)
    _plot_heatmap(per_class_results, prototypes_np, proto_class_names_list,
                  embeddings_np, labels, args.output_dir, get_class_name)

    # HTML
    html = _build_html(summary, args.output_dir, get_class_name, threshold)
    with open(os.path.join(args.output_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Pseudo-OOD Evaluation Results")
    print(f"{'='*60}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Classes evaluated: {len(per_class_results)}")
    if reject_rates:
        print(f"Mean reject rate: {np.mean(reject_rates):.4f}")
        print(f"Median reject rate: {np.median(reject_rates):.4f}")
        print(f"Min reject rate: {np.min(reject_rates):.4f}")
        print(f"Max reject rate: {np.max(reject_rates):.4f}")
    if known_accept_when_left_out:
        print(f"Known accept rate (other class left out): {np.mean(known_accept_when_left_out):.4f}")

    print(f"\nPer-class results:")
    print(f"  {'Class':<35} {'Support':>7} {'Reject':>7} {'FalseKnown':>11} {'AvgSim':>7}")
    for r in per_class_results:
        cn = r["class_name"]
        print(f"  {cn:<35} {r['support']:>7} {r['reject_rate']:>7.3f} {r['false_known_rate']:>11.3f} {r['avg_nearest_similarity_to_remaining']:>7.4f}")

    print(f"\nOutputs saved to {args.output_dir}")


def _plot_bar(per_class_results, output_dir, get_class_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["class_name"] for r in per_class_results]
    reject_rates = [r["reject_rate"] for r in per_class_results]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.8), 6))
    bars = ax.bar(range(len(names)), reject_rates, color="#3b82f6", edgecolor="white")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Pseudo-Unknown Reject Rate")
    ax.set_title("Leave-One-Class-Out Pseudo-OOD Reject Rate")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=np.mean(reject_rates) if reject_rates else 0, color="red",
               linestyle="--", alpha=0.7, label=f"Mean: {np.mean(reject_rates):.3f}" if reject_rates else "")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pseudo_ood_reject_rate_bar.png"), dpi=150)
    plt.close()


def _plot_heatmap(per_class_results, prototypes_np, proto_class_names_list,
                  embeddings_np, labels, output_dir, get_class_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    left_out_classes = [r["pseudo_unknown_class"] for r in per_class_results]
    if not left_out_classes:
        return

    # Compute similarity matrix: left-out class mean embedding vs remaining prototypes
    sim_matrix = np.zeros((len(left_out_classes), prototypes_np.shape[0]))

    for i, left_out in enumerate(left_out_classes):
        mask = np.array([l == left_out for l in labels])
        if mask.sum() == 0:
            continue
        class_mean = embeddings_np[mask].mean(axis=0)
        class_mean = class_mean / max(np.linalg.norm(class_mean), 1e-8)
        sim_matrix[i] = class_mean @ prototypes_np.T

    fig, ax = plt.subplots(figsize=(max(8, prototypes_np.shape[0] * 0.5),
                                     max(6, len(left_out_classes) * 0.5)))
    im = ax.imshow(sim_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(proto_class_names_list.__len__()))
    ax.set_xticklabels(proto_class_names_list, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(left_out_classes)))
    ax.set_yticklabels([get_class_name(c) for c in left_out_classes], fontsize=8)
    ax.set_xlabel("Prototype Class")
    ax.set_ylabel("Pseudo-Unknown Class (left out)")
    ax.set_title("Similarity: Pseudo-Unknown Class Mean vs All Prototypes")
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pseudo_ood_similarity_heatmap.png"), dpi=150)
    plt.close()


def _build_html(summary, output_dir, get_class_name, threshold):
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

    def _f(v, d=3):
        if v is None:
            return "N/A"
        return f"{v:.{d}f}"

    CSS = """
    :root { --bg:#f5f7fa;--card-bg:#fff;--text:#1a1a2e;--text2:#555;--border:#e0e4e8;
            --accent:#3b82f6;--success:#10b981;--warning:#f59e0b;--danger:#ef4444; }
    *{margin:0;padding:0;box-sizing:border-box}
    body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         background:var(--bg);color:var(--text);line-height:1.6;padding:24px}
    .container{max-width:1200px;margin:0 auto}
    h1{font-size:1.8rem;font-weight:700;margin-bottom:8px}
    h2{font-size:1.3rem;font-weight:600;margin-top:32px;margin-bottom:16px;
       padding-bottom:8px;border-bottom:2px solid var(--accent)}
    .info-box{background:#eff6ff;border:1px solid var(--accent);border-radius:8px;
              padding:12px 16px;margin-bottom:24px;font-size:0.85rem;line-height:1.6}
    .cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:16px;margin-bottom:24px}
    .card{background:var(--card-bg);border-radius:10px;padding:16px;text-align:center;
          border:1px solid var(--border)}
    .card-value{font-size:1.4rem;font-weight:700;color:var(--accent)}
    .card-label{font-size:0.75rem;color:var(--text2);text-transform:uppercase}
    .table-wrap{overflow-x:auto;margin-bottom:24px;background:var(--card-bg);
                border-radius:8px;border:1px solid var(--border)}
    table{width:100%;border-collapse:collapse;font-size:0.85rem}
    th{background:#f8f9fb;padding:10px 12px;text-align:left;font-weight:600;
       border-bottom:2px solid var(--border)}
    td{padding:8px 12px;border-bottom:1px solid var(--border)}
    tr:last-child td{border-bottom:none}tr:hover td{background:#f8f9fb}
    .img-section{margin-bottom:24px;background:var(--card-bg);border-radius:8px;
                 border:1px solid var(--border);padding:16px;text-align:center}
    .img-section img{max-width:100%;height:auto}
    footer{margin-top:40px;text-align:center;color:var(--text2);font-size:0.8rem}
    """

    # Summary cards
    cards_html = ""
    for label, key in [
        ("Mean Reject Rate", "mean_pseudo_unknown_reject_rate"),
        ("Median Reject Rate", "median_pseudo_unknown_reject_rate"),
        ("Min Reject Rate", "min_pseudo_unknown_reject_rate"),
        ("Max Reject Rate", "max_pseudo_unknown_reject_rate"),
        ("Known Accept (other left out)", "known_accept_rate_when_other_class_left_out"),
        ("Classes Evaluated", "num_classes_evaluated"),
    ]:
        v = summary.get(key)
        if isinstance(v, (int, float)):
            cards_html += f'<div class="card"><div class="card-value">{_f(v) if isinstance(v, float) else v}</div><div class="card-label">{html_mod.escape(label)}</div></div>\n'
        else:
            cards_html += f'<div class="card"><div class="card-value">N/A</div><div class="card-label">{html_mod.escape(label)}</div></div>\n'

    # Per-class table
    rows = ""
    for r in summary.get("per_class", []):
        mcfkc = get_class_name(r["most_common_false_known_class"]) if r.get("most_common_false_known_class") is not None else "N/A"
        rows += (f'<tr><td>{html_mod.escape(r["class_name"])}</td><td>{r["support"]}</td>'
                 f'<td>{_f(r["reject_rate"])}</td><td>{_f(r["false_known_rate"])}</td>'
                 f'<td>{html_mod.escape(mcfkc)}</td>'
                 f'<td>{_f(r["avg_nearest_similarity_to_remaining"])}</td></tr>\n')

    pc_table = f"""<div class="table-wrap"><table>
    <tr><th>Class (left out)</th><th>Support</th><th>Reject Rate</th><th>False Known Rate</th>
    <th>Most Common False Known</th><th>Avg Sim to Remaining</th></tr>
    {rows}</table></div>"""

    # Images
    bar_img = ""
    uri = _b64(os.path.join(output_dir, "pseudo_ood_reject_rate_bar.png"))
    if uri:
        bar_img = f'<div class="img-section"><h3>Reject Rate by Class</h3><img src="{uri}"></div>'

    heat_img = ""
    uri2 = _b64(os.path.join(output_dir, "pseudo_ood_similarity_heatmap.png"))
    if uri2:
        heat_img = f'<div class="img-section"><h3>Similarity Heatmap</h3><img src="{uri2}"></div>'

    skipped = summary.get("skipped_classes", [])
    unsupported = summary.get("unsupported_classes", [])

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Pseudo-OOD Evaluation Report</title>
<style>{CSS}</style></head><body><div class="container">
<h1>Pseudo-OOD Evaluation (Leave-One-Class-Out)</h1>
<p style="color:var(--text2);margin-bottom:24px">Generated at {html_mod.escape(ts)} | Threshold: {threshold:.4f}</p>

<div class="info-box">
<strong>Note:</strong> Pseudo-OOD evaluation is not a replacement for real negative data.
It estimates the risk of accepting unseen known-like categories by temporarily treating
each known class as unknown.
</div>

<h2>Summary</h2>
<div class="cards">{cards_html}</div>

<h2>Per-class Results</h2>
{pc_table}

<h2>Visualizations</h2>
{bar_img}
{heat_img}

<h2>Metadata</h2>
<div class="table-wrap"><table>
<tr><th>Item</th><th>Value</th></tr>
<tr><td>Min support threshold</td><td>{summary.get('min_support', 'N/A')}</td></tr>
<tr><td>Skipped classes (low support)</td><td>{html_mod.escape(str(skipped))}</td></tr>
<tr><td>Unsupported classes</td><td>{html_mod.escape(str(unsupported))}</td></tr>
</table></div>

<footer>Pseudo-OOD Evaluation Report &mdash; pointcloud_metric_learning</footer>
</div></body></html>"""


if __name__ == "__main__":
    main()
