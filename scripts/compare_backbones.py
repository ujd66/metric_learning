"""Compare multiple backbone runs side by side.

Reads final_report.json (or individual report JSONs) from each run directory
and produces a comprehensive comparison table across all key metrics.

Usage:
    python scripts/compare_backbones.py \
        --runs \
            outputs/runs/newdata_pointnet_ce_only_v1 \
            outputs/runs/newdata_pointnet_ce_supcon_v1 \
            outputs/runs/newdata_pointnet2_ce_only_v1 \
            outputs/runs/newdata_pointnet2_ce_supcon_v1 \
        --labels \
            "PointNet CE" \
            "PointNet CE+SupCon" \
            "PointNet++ CE" \
            "PointNet++ CE+SupCon" \
        --output-dir outputs/reports/backbone_ablation
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def extract_metrics(run_dir):
    """Extract all relevant metrics from a run directory."""
    result = {}

    # Classification metrics
    eval_data = load_json(os.path.join(run_dir, "evaluation.json"))
    if eval_data:
        result["overall_acc"] = eval_data.get("overall_accuracy")
        result["macro_f1"] = eval_data.get("macro_f1")
        result["macro_precision"] = eval_data.get("macro_precision")
        result["macro_recall"] = eval_data.get("macro_recall")
        result["known_acc"] = eval_data.get("known_class_accuracy")
        result["negative_acc"] = eval_data.get("negative_accuracy")

        # Per-class metrics
        per_class = eval_data.get("per_class", {})
        result["per_class"] = per_class

        # class_014 metrics
        c14 = per_class.get("14", {})
        result["class_014_precision"] = c14.get("precision")
        result["class_014_recall"] = c14.get("recall")
        result["class_014_f1"] = c14.get("f1")

        # Negative metrics
        neg = per_class.get("19", {})
        result["negative_precision"] = neg.get("precision")
        result["negative_recall"] = neg.get("recall")
        result["negative_f1"] = neg.get("f1")

    # Embedding metrics
    emb_data = load_json(os.path.join(run_dir, "embedding_report.json"))
    if emb_data:
        result["similarity_gap"] = emb_data.get("similarity_gap")
        result["recall_1"] = emb_data.get("recall_at_1")
        result["recall_5"] = emb_data.get("recall_at_5")
        result["nn_accuracy"] = emb_data.get("1nn_accuracy")

    # OOD metrics
    ood_data = load_json(os.path.join(run_dir, "ood_report.json"))
    if ood_data:
        result["ood_auroc"] = ood_data.get("auroc")
        result["known_accept_rate"] = ood_data.get("known_accept_rate")
        result["negative_reject_rate"] = ood_data.get("negative_reject_rate")

    # Retrieval metrics
    ret_data = load_json(os.path.join(run_dir, "retrieval_report.json"))
    if ret_data:
        result["retrieval_auroc"] = ret_data.get("retrieval_auroc")
        if result.get("recall_1") is None:
            result["recall_1"] = ret_data.get("recall_at_1")
        if result.get("recall_5") is None:
            result["recall_5"] = ret_data.get("recall_at_5")

    # Threshold info
    thresh_data = load_json(os.path.join(run_dir, "best_threshold.json"))
    if thresh_data:
        result["best_threshold"] = thresh_data.get("threshold")
        result["threshold_method"] = thresh_data.get("method")

    # Checkpoint info (parameter count)
    ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    if os.path.exists(ckpt_path):
        import torch
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            saved_cfg = ckpt.get("config", {})
            result["backbone"] = saved_cfg.get("model", {}).get("backbone", "pointnet")
        except Exception:
            pass

    # Try final_report.json as fallback
    if not result:
        final = load_json(os.path.join(run_dir, "final_report.json"))
        if final:
            for k, v in final.items():
                if k not in result and not isinstance(v, dict):
                    result[k] = v

    return result


def find_worst_class_f1(per_class, num_known=19):
    """Find the worst 5 classes by F1 among known classes."""
    entries = []
    for label_str, metrics in per_class.items():
        label = int(label_str)
        if label >= num_known:
            continue
        f1 = metrics.get("f1", 0) or 0
        entries.append((label, f1))
    entries.sort(key=lambda x: x[1])
    worst = entries[:5]
    return worst


def fmt(val, digits=4):
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{digits}f}"
    return str(val)


def generate_html_report(rows, labels, metrics_list, output_path):
    """Generate HTML comparison report."""
    metric_names = [
        ("Classification", [
            ("Overall Acc", "overall_acc"),
            ("Macro F1", "macro_f1"),
            ("Known Acc", "known_acc"),
            ("Negative Acc", "negative_acc"),
        ]),
        ("Negative Class", [
            ("Neg Precision", "negative_precision"),
            ("Neg Recall", "negative_recall"),
            ("Neg F1", "negative_f1"),
        ]),
        ("class_014", [
            ("class_014 P", "class_014_precision"),
            ("class_014 R", "class_014_recall"),
            ("class_014 F1", "class_014_f1"),
        ]),
        ("Embedding", [
            ("Sim Gap", "similarity_gap"),
            ("Recall@1", "recall_1"),
            ("Recall@5", "recall_5"),
            ("1-NN Acc", "nn_accuracy"),
        ]),
        ("OOD", [
            ("OOD AUROC", "ood_auroc"),
            ("Known Accept", "known_accept_rate"),
            ("Neg Reject", "negative_reject_rate"),
        ]),
        ("Retrieval", [
            ("Retrieval AUROC", "retrieval_auroc"),
        ]),
    ]

    html = ['<!DOCTYPE html><html><head><meta charset="utf-8">']
    html.append('<title>Backbone Ablation Comparison</title>')
    html.append('<style>')
    html.append('body{font-family:monospace;margin:20px;background:#f8f8f8}')
    html.append('h1{color:#333}')
    html.append('table{border-collapse:collapse;width:100%;background:white}')
    html.append('th,td{border:1px solid #ddd;padding:8px 12px;text-align:center}')
    html.append('th{background:#4a90d9;color:white;font-weight:bold}')
    html.append('td.best{background:#d4edda;font-weight:bold}')
    html.append('.section{background:#e8f0fe;font-weight:bold;text-align:left}')
    html.append('</style></head><body>')
    html.append('<h1>Backbone Ablation Comparison</h1>')
    html.append(f'<p>Generated from {len(labels)} runs</p>')
    html.append('<table>')

    # Header
    html.append('<tr><th>Metric</th>')
    for label in labels:
        html.append(f'<th>{label}</th>')
    html.append('</tr>')

    for section_name, metrics_in_section in metric_names:
        html.append(f'<tr><td colspan="{len(labels)+1}" class="section">{section_name}</td></tr>')
        for display_name, key in metrics_in_section:
            values = [m.get(key) for m in metrics_list]
            numeric_vals = [v for v in values if isinstance(v, (int, float))]

            html.append(f'<tr><td>{display_name}</td>')
            for val in values:
                if val is None:
                    html.append('<td>N/A</td>')
                else:
                    is_best = len(numeric_vals) > 1 and isinstance(val, (int, float))
                    # For some metrics higher is better, for others lower
                    if is_best:
                        if "gap" in key.lower():
                            is_best = val == min(numeric_vals)
                        else:
                            is_best = val == max(numeric_vals)
                    cls = ' class="best"' if is_best else ''
                    html.append(f'<td{cls}>{fmt(val)}</td>')
            html.append('</tr>')

        # Worst 5 classes
    for idx, m in enumerate(metrics_list):
        per_class = m.get("per_class", {})
        if per_class:
            worst = find_worst_class_f1(per_class)
            if idx == 0:
                html.append(f'<tr><td colspan="{len(labels)+1}" class="section">Worst 5 Class F1</td></tr>')
            break

    # Per-run worst classes
    for idx, m in enumerate(metrics_list):
        per_class = m.get("per_class", {})
        if per_class:
            worst = find_worst_class_f1(per_class)
            worst_str = ", ".join([f"class_{l}: {fmt(f1)}" for l, f1 in worst])

    html.append('</table>')

    # Per-class F1 table
    any_per_class = any(m.get("per_class") for m in metrics_list)
    if any_per_class:
        html.append('<h2>Per-Class F1</h2>')
        html.append('<table><tr><th>Class</th>')
        for label in labels:
            html.append(f'<th>{label}</th>')
        html.append('</tr>')

        all_labels = set()
        for m in metrics_list:
            all_labels.update(int(k) for k in m.get("per_class", {}).keys())
        all_labels = sorted(all_labels)

        for cl in all_labels:
            html.append(f'<tr><td>class_{cl:03d}</td>')
            for m in metrics_list:
                pc = m.get("per_class", {})
                f1 = pc.get(str(cl), {}).get("f1")
                html.append(f'<td>{fmt(f1)}</td>')
            html.append('</tr>')
        html.append('</table>')

    html.append('</body></html>')

    with open(output_path, "w") as f:
        f.write("\n".join(html))
    print(f"HTML report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare backbone experiment runs")
    parser.add_argument("--runs", nargs="+", required=True, help="Paths to run directories")
    parser.add_argument("--labels", nargs="+", default=None, help="Labels for each run")
    parser.add_argument("--output-dir", default="outputs/reports/backbone_ablation",
                        help="Output directory for comparison reports")
    args = parser.parse_args()

    if len(args.runs) == 0:
        print("[ERROR] No runs specified")
        sys.exit(1)

    if args.labels is None:
        args.labels = [os.path.basename(r) for r in args.runs]

    if len(args.labels) != len(args.runs):
        print(f"[ERROR] Number of labels ({len(args.labels)}) != number of runs ({len(args.runs)})")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Extract metrics
    metrics_list = []
    for run_dir in args.runs:
        m = extract_metrics(run_dir)
        metrics_list.append(m)
        print(f"Loaded metrics from {run_dir} ({len(m)} metrics)")

    # Collect all metric keys for CSV/JSON
    all_keys = []
    seen = set()
    for m in metrics_list:
        for k in m:
            if k not in seen and k != "per_class":
                all_keys.append(k)
                seen.add(k)

    # Save CSV
    csv_path = os.path.join(args.output_dir, "backbone_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + args.labels)
        for key in all_keys:
            row = [key]
            for m in metrics_list:
                row.append(fmt(m.get(key)))
            writer.writerow(row)
    print(f"CSV saved to {csv_path}")

    # Save JSON
    json_data = {}
    for label, m in zip(args.labels, metrics_list):
        json_data[label] = {k: v for k, v in m.items() if k != "per_class"}
    json_path = os.path.join(args.output_dir, "backbone_comparison.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"JSON saved to {json_path}")

    # Generate HTML
    html_path = os.path.join(args.output_dir, "backbone_comparison_report.html")
    generate_html_report(args.runs, args.labels, metrics_list, html_path)

    # Print summary
    print(f"\n{'='*80}")
    print("Backbone Ablation Summary")
    print(f"{'='*80}")

    summary_keys = [
        "backbone", "overall_acc", "macro_f1", "known_acc", "negative_acc",
        "negative_f1", "class_014_f1", "similarity_gap",
        "recall_1", "ood_auroc", "negative_reject_rate", "retrieval_auroc",
    ]
    header = f"{'Metric':<25s}" + "".join(f"{l:>18s}" for l in args.labels)
    print(header)
    print("-" * len(header))
    for key in summary_keys:
        row = f"{key:<25s}"
        for m in metrics_list:
            row += f"{fmt(m.get(key)):>18s}"
        print(row)


if __name__ == "__main__":
    main()
