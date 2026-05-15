"""Compare two or more experiment runs side by side.

Reads final_report.json (or individual report JSONs) from each run directory
and produces a controlled comparison across all key metrics.

Usage:
    python scripts/compare_runs.py \
        --runs outputs/runs/newdata_pointnet_ce_only_v1 \
               outputs/runs/newdata_pointnet_ce_supcon_v1 \
        --output-dir outputs/reports/newdata_pointnet_controlled_comparison

    # With explicit labels:
    python scripts/compare_runs.py \
        --runs outputs/runs/newdata_pointnet_ce_only_v1 \
               outputs/runs/newdata_pointnet_ce_supcon_v1 \
        --labels "CE-only" "CE+SupCon" \
        --output-dir outputs/reports/newdata_pointnet_controlled_comparison
"""

import argparse
import csv
import json
import os
import sys
from collections import OrderedDict
from datetime import datetime


def load_json(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _fv(val, digits=4):
    """Format float or return N/A."""
    if val is None:
        return "N/A"
    return round(val, digits)


def extract_run_metrics(run_dir):
    """Extract all comparable metrics from a run directory.

    Tries final_report.json first, then falls back to individual report files.
    Returns a flat dict of metric_name -> value.
    """
    metrics = OrderedDict()
    run_name = os.path.basename(run_dir)

    # Try final_report.json first
    fr = load_json(os.path.join(run_dir, "final_report.json"))

    if fr:
        metrics["run_name"] = fr.get("run_name", run_name)

        # Classification
        cls = fr.get("classification", {})
        metrics["overall_accuracy"] = cls.get("overall_accuracy")
        metrics["known_class_accuracy"] = cls.get("known_class_accuracy")
        metrics["negative_accuracy"] = cls.get("negative_accuracy")
        metrics["macro_precision"] = cls.get("macro_precision")
        metrics["macro_recall"] = cls.get("macro_recall")
        metrics["macro_f1"] = cls.get("macro_f1")

        # Per-class
        pc = cls.get("per_class", {})
        for label_str, vals in pc.items():
            label = int(label_str)
            prefix = f"class_{label:03d}"
            metrics[f"{prefix}_precision"] = vals.get("precision")
            metrics[f"{prefix}_recall"] = vals.get("recall")
            metrics[f"{prefix}_f1"] = vals.get("f1")
            metrics[f"{prefix}_support"] = vals.get("support")

        # Embedding
        emb = fr.get("embedding", {})
        intra = emb.get("intra_class_similarity", {})
        metrics["intra_class_similarity_macro"] = intra.get("macro_avg")
        metrics["intra_class_similarity_global"] = intra.get("global_avg")
        inter = emb.get("inter_class_similarity", {})
        metrics["inter_class_similarity"] = inter.get("global_avg")
        metrics["similarity_gap"] = emb.get("similarity_gap")

        recall_k = emb.get("recall_at_k", {})
        for k_name, k_val in recall_k.items():
            metrics[f"embedding_{k_name}"] = k_val
        prec_k = emb.get("precision_at_k", {})
        for k_name, k_val in prec_k.items():
            metrics[f"embedding_{k_name}"] = k_val
        metrics["nn_accuracy"] = emb.get("nn_accuracy")

        # OOD
        ood = fr.get("ood", {})
        metrics["ood_threshold"] = ood.get("threshold")
        metrics["known_accept_rate"] = ood.get("known_accept_rate")
        metrics["known_reject_rate"] = ood.get("known_reject_rate")
        metrics["negative_reject_rate"] = ood.get("negative_reject_rate")
        metrics["false_known_rate"] = ood.get("false_known_rate")
        metrics["ood_auroc"] = ood.get("auroc")
        metrics["ood_balanced_score"] = ood.get("balanced_score")
        metrics["final_known_accuracy"] = ood.get("final_known_accuracy")
        metrics["final_macro_f1_on_known"] = ood.get("final_macro_f1_on_known_classes")

        # Retrieval
        ret = fr.get("retrieval", {})
        metrics["retrieval_threshold"] = ret.get("threshold")
        metrics["retrieval_known_accept_rate"] = ret.get("known_accept_rate")
        metrics["retrieval_top1_accuracy"] = ret.get("top1_accuracy_on_accepted")
        metrics["retrieval_negative_reject_rate"] = ret.get("negative_reject_rate")
        metrics["retrieval_overall_accuracy"] = ret.get("overall_accuracy")
        metrics["retrieval_macro_f1_known"] = ret.get("macro_f1_known_classes")
        metrics["retrieval_auroc"] = ret.get("auroc")
        metrics["retrieval_recall@1"] = ret.get("recall@1")
        metrics["retrieval_recall@3"] = ret.get("recall@3")
        metrics["retrieval_recall@5"] = ret.get("recall@5")

        # Top confusing pairs
        confusing = emb.get("top_confusing_pairs", [])
        metrics["top_confusing_pairs"] = confusing

        return metrics

    # Fallback: load individual report files
    metrics["run_name"] = run_name

    # Classification
    cls = load_json(os.path.join(run_dir, "reports", "classification", "evaluation.json"))
    if cls:
        for key in ["overall_accuracy", "known_class_accuracy", "negative_accuracy",
                     "macro_precision", "macro_recall", "macro_f1"]:
            metrics[key] = cls.get(key)
        pc = cls.get("per_class", {})
        for label_str, vals in pc.items():
            label = int(label_str)
            prefix = f"class_{label:03d}"
            metrics[f"{prefix}_precision"] = vals.get("precision")
            metrics[f"{prefix}_recall"] = vals.get("recall")
            metrics[f"{prefix}_f1"] = vals.get("f1")
            metrics[f"{prefix}_support"] = vals.get("support")

    # Embedding
    emb_summary = load_json(os.path.join(run_dir, "reports", "embeddings", "metrics_summary.json"))
    if emb_summary:
        metrics["intra_class_similarity_macro"] = emb_summary.get("intra_class_similarity", {}).get("macro_avg")
        metrics["intra_class_similarity_global"] = emb_summary.get("intra_class_similarity", {}).get("global_avg")
        metrics["inter_class_similarity"] = emb_summary.get("inter_class_similarity", {}).get("global_avg")
        metrics["similarity_gap"] = emb_summary.get("similarity_gap")
        metrics["nn_accuracy"] = emb_summary.get("nn_accuracy")

        recall_k = emb_summary.get("recall_at_k", {})
        for k_name, k_val in recall_k.items():
            metrics[f"embedding_{k_name}"] = k_val
        prec_k = emb_summary.get("precision_at_k", {})
        for k_name, k_val in prec_k.items():
            metrics[f"embedding_{k_name}"] = k_val
        metrics["top_confusing_pairs"] = emb_summary.get("top_confusing_pairs", [])

    # OOD
    ood = load_json(os.path.join(run_dir, "reports", "ood", "ood_metrics.json"))
    if ood:
        metrics["ood_threshold"] = ood.get("threshold")
        metrics["known_accept_rate"] = ood.get("known_accept_rate")
        metrics["known_reject_rate"] = ood.get("known_reject_rate")
        metrics["negative_reject_rate"] = ood.get("negative_reject_rate")
        metrics["false_known_rate"] = ood.get("false_known_rate")
        metrics["ood_auroc"] = ood.get("auroc")
        metrics["ood_balanced_score"] = ood.get("balanced_score")

    # Retrieval
    ret = load_json(os.path.join(run_dir, "reports", "retrieval", "retrieval_metrics.json"))
    if ret:
        metrics["retrieval_threshold"] = ret.get("threshold")
        metrics["retrieval_known_accept_rate"] = ret.get("known_accept_rate")
        metrics["retrieval_top1_accuracy"] = ret.get("top1_accuracy_on_accepted")
        metrics["retrieval_negative_reject_rate"] = ret.get("negative_reject_rate")
        metrics["retrieval_overall_accuracy"] = ret.get("overall_accuracy")
        metrics["retrieval_auroc"] = ret.get("auroc")

    return metrics


def get_comparison_rows(all_metrics, labels):
    """Build ordered comparison rows for CSV/JSON/HTML output.

    Returns list of (metric_display_name, metric_key, [values]).
    """
    rows = []

    def _add(display, key):
        vals = []
        for m in all_metrics:
            v = m.get(key)
            vals.append(_fv(v) if v is not None else None)
        rows.append((display, key, vals))

    def _diff(vals):
        """Compute delta: last - first (assumes exactly 2 runs)."""
        if len(vals) == 2 and vals[0] is not None and vals[1] is not None:
            return _fv(vals[1] - vals[0])
        return None

    # --- Classification ---
    _add("Overall Accuracy", "overall_accuracy")
    _add("Known Class Accuracy", "known_class_accuracy")
    _add("Negative Accuracy", "negative_accuracy")
    _add("Macro Precision", "macro_precision")
    _add("Macro Recall", "macro_recall")
    _add("Macro F1", "macro_f1")

    # --- class_014 specific ---
    _add("class_014 Precision", "class_014_precision")
    _add("class_014 Recall", "class_014_recall")
    _add("class_014 F1", "class_014_f1")

    # --- Negative specific ---
    _add("Negative Precision", "class_019_precision")
    _add("Negative Recall", "class_019_recall")
    _add("Negative F1", "class_019_f1")

    # --- Worst 5 classes by F1 ---
    # Collect all per-class F1 from first run
    class_f1s = []
    for key, val in all_metrics[0].items():
        if key.startswith("class_") and key.endswith("_f1") and val is not None:
            label = key.replace("class_", "").replace("_f1", "")
            if label != "019":  # exclude negative (already shown)
                class_f1s.append((label, val))
    class_f1s.sort(key=lambda x: x[1])
    worst5 = class_f1s[:5]
    for label, f1_val in worst5:
        _add(f"Worst: class_{label} F1", f"class_{label}_f1")

    # --- Embedding ---
    _add("Intra-class Sim (macro)", "intra_class_similarity_macro")
    _add("Intra-class Sim (global)", "intra_class_similarity_global")
    _add("Inter-class Sim", "inter_class_similarity")
    _add("Similarity Gap", "similarity_gap")
    _add("NN Accuracy", "nn_accuracy")
    _add("Embedding Recall@1", "embedding_recall@1")
    _add("Embedding Recall@5", "embedding_recall@5")

    # --- OOD ---
    _add("OOD Threshold", "ood_threshold")
    _add("Known Accept Rate", "known_accept_rate")
    _add("Negative Reject Rate", "negative_reject_rate")
    _add("OOD AUROC", "ood_auroc")
    _add("OOD Balanced Score", "ood_balanced_score")
    _add("Final Known Acc (after OOD)", "final_known_accuracy")

    # --- Retrieval ---
    _add("Retrieval Top1 Acc", "retrieval_top1_accuracy")
    _add("Retrieval Negative Reject", "retrieval_negative_reject_rate")
    _add("Retrieval AUROC", "retrieval_auroc")
    _add("Retrieval Recall@1", "retrieval_recall@1")
    _add("Retrieval Recall@5", "retrieval_recall@5")

    return rows


def write_csv(rows, labels, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric"] + labels + (["Delta"] if len(labels) == 2 else []))
        for display, key, vals in rows:
            str_vals = [f"{v:.4f}" if v is not None else "N/A" for v in vals]
            delta = ""
            if len(vals) == 2 and vals[0] is not None and vals[1] is not None:
                delta = f"{vals[1] - vals[0]:+.4f}"
            writer.writerow([display] + str_vals + ([delta] if len(labels) == 2 else []))
    print(f"CSV saved: {output_path}")


def write_json_report(rows, all_metrics, labels, output_path):
    result = {
        "generated_at": datetime.now().isoformat(),
        "runs": labels,
        "comparison": {},
        "top_confusing_pairs": {},
        "conclusions": {},
    }

    for display, key, vals in rows:
        result["comparison"][key] = {
            "display_name": display,
            "values": {labels[i]: vals[i] for i in range(len(labels))},
        }
        if len(vals) == 2 and vals[0] is not None and vals[1] is not None:
            result["comparison"][key]["delta"] = round(vals[1] - vals[0], 4)

    # Top confusing pairs per run
    for i, m in enumerate(all_metrics):
        pairs = m.get("top_confusing_pairs", [])
        result["top_confusing_pairs"][labels[i]] = pairs[:5]

    # Auto-generate conclusions
    conclusions = generate_conclusions(all_metrics, labels)
    result["conclusions"] = conclusions

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"JSON saved: {output_path}")


def generate_conclusions(all_metrics, labels):
    """Auto-judge whether CE+SupCon is better than CE-only."""
    if len(all_metrics) != 2:
        return {"note": "Conclusions require exactly 2 runs for comparison."}

    m_ce = all_metrics[0]
    m_sc = all_metrics[1]

    conclusions = {}

    # 1. Overall classification
    ce_f1 = m_ce.get("macro_f1") or 0
    sc_f1 = m_sc.get("macro_f1") or 0
    conclusions["ce_only_macro_f1"] = _fv(ce_f1)
    conclusions["ce_supcon_macro_f1"] = _fv(sc_f1)
    conclusions["classification_winner"] = "CE+SupCon" if sc_f1 > ce_f1 else "CE-only"
    conclusions["classification_delta_f1"] = _fv(sc_f1 - ce_f1)

    # 2. Embedding quality
    ce_gap = m_ce.get("similarity_gap") or 0
    sc_gap = m_sc.get("similarity_gap") or 0
    conclusions["ce_only_similarity_gap"] = _fv(ce_gap)
    conclusions["ce_supcon_similarity_gap"] = _fv(sc_gap)
    conclusions["embedding_winner"] = "CE+SupCon" if sc_gap > ce_gap else "CE-only"
    conclusions["embedding_degradation"] = sc_gap < ce_gap

    # 3. Negative recall
    ce_neg_recall = m_ce.get("class_019_recall") or m_ce.get("negative_accuracy") or 0
    sc_neg_recall = m_sc.get("class_019_recall") or m_sc.get("negative_accuracy") or 0
    conclusions["ce_only_negative_recall"] = _fv(ce_neg_recall)
    conclusions["ce_supcon_negative_recall"] = _fv(sc_neg_recall)
    conclusions["negative_recall_dropped"] = sc_neg_recall < ce_neg_recall

    # 4. OOD
    ce_auroc = m_ce.get("ood_auroc") or 0
    sc_auroc = m_sc.get("ood_auroc") or 0
    conclusions["ce_only_ood_auroc"] = _fv(ce_auroc)
    conclusions["ce_supcon_ood_auroc"] = _fv(sc_auroc)
    conclusions["ood_winner"] = "CE+SupCon" if sc_auroc > ce_auroc else "CE-only"

    # 5. class_014
    ce_014_f1 = m_ce.get("class_014_f1") or 0
    sc_014_f1 = m_sc.get("class_014_f1") or 0
    conclusions["ce_only_class_014_f1"] = _fv(ce_014_f1)
    conclusions["ce_supcon_class_014_f1"] = _fv(sc_014_f1)

    # 6. Overall verdict
    wins = 0
    if sc_f1 > ce_f1:
        wins += 1
    if sc_gap > ce_gap:
        wins += 1
    if sc_auroc > ce_auroc:
        wins += 1

    if wins >= 2:
        conclusions["overall_verdict"] = (
            "CE+SupCon is superior to CE-only. "
            "Recommend continuing with SupCon in PointNet++ / PointNeXt experiments."
        )
    elif wins == 1:
        conclusions["overall_verdict"] = (
            "CE+SupCon shows mixed results vs CE-only. "
            "Both approaches are comparable; consider tuning SupCon hyperparameters "
            "(metric_weight, temperature, warmup_epochs) before concluding."
        )
    else:
        conclusions["overall_verdict"] = (
            "CE-only outperforms CE+SupCon on key metrics. "
            "SupCon may need hyperparameter tuning or may not benefit this dataset size. "
            "Consider trying different metric_weight or temperature values."
        )

    conclusions["recommend_supcon_for_pointnet_plus_plus"] = wins >= 2

    return conclusions


def write_html_report(rows, all_metrics, labels, output_path):
    """Generate a comprehensive HTML comparison report."""
    conclusions = generate_conclusions(all_metrics, labels)

    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Phase 3.1 Controlled Comparison: CE-only vs CE+SupCon</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       margin: 20px; background: #f8f9fa; color: #212529; }
h1 { color: #343a40; border-bottom: 2px solid #dee2e6; padding-bottom: 10px; }
h2 { color: #495057; margin-top: 30px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; background: white; }
th, td { border: 1px solid #dee2e6; padding: 8px 12px; text-align: right; }
th { background: #e9ecef; font-weight: 600; }
td:first-child, th:first-child { text-align: left; font-weight: 500; }
.better { background: #d4edda; color: #155724; }
.worse { background: #f8d7da; color: #721c24; }
.neutral { background: #fff3cd; color: #856404; }
.verdict { padding: 20px; margin: 20px 0; border-radius: 8px; font-size: 1.1em; }
.verdict-positive { background: #d4edda; border-left: 5px solid #28a745; }
.verdict-mixed { background: #fff3cd; border-left: 5px solid #ffc107; }
.verdict-negative { background: #f8d7da; border-left: 5px solid #dc3545; }
.section { margin: 20px 0; padding: 15px; background: white; border-radius: 8px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
code { background: #e9ecef; padding: 2px 6px; border-radius: 3px; }
pre { background: #f1f3f5; padding: 15px; border-radius: 5px; overflow-x: auto; }
</style>
</head>
<body>
<h1>Phase 3.1: CE-only vs CE+SupCon Controlled Comparison</h1>
<p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
"""

    # Runs summary
    html += '<div class="section"><h2>Runs</h2><ul>'
    for i, label in enumerate(labels):
        run_name = all_metrics[i].get("run_name", label)
        html += f'<li><strong>{label}</strong>: {run_name}</li>'
    html += '</ul></div>'

    # Comparison table
    html += '<div class="section"><h2>Metric Comparison</h2>'
    html += '<table><tr><th>Metric</th>'
    for label in labels:
        html += f'<th>{label}</th>'
    if len(labels) == 2:
        html += '<th>Delta</th><th>Verdict</th>'
    html += '</tr>'

    # Group metrics
    section_headers = {
        "Overall Accuracy": "Classification",
        "Known Class Accuracy": None,
        "Negative Accuracy": None,
        "Macro Precision": None,
        "Macro Recall": None,
        "Macro F1": None,
        "class_014 Precision": "class_014 (shenggaozuofalan)",
        "class_014 Recall": None,
        "class_014 F1": None,
        "Negative Precision": "Negative (qita)",
        "Negative Recall": None,
        "Negative F1": None,
        "Worst:": "Worst 5 Classes by F1",
        "Intra-class Sim": "Embedding Quality",
        "Inter-class Sim": None,
        "Similarity Gap": None,
        "NN Accuracy": None,
        "Embedding Recall@1": None,
        "Embedding Recall@5": None,
        "OOD Threshold": "OOD Detection",
        "Known Accept Rate": None,
        "Negative Reject Rate": None,
        "OOD AUROC": None,
        "OOD Balanced Score": None,
        "Final Known Acc": None,
        "Retrieval Top1 Acc": "Retrieval",
        "Retrieval Negative Reject": None,
        "Retrieval AUROC": None,
        "Retrieval Recall@1": None,
        "Retrieval Recall@5": None,
    }

    for display, key, vals in rows:
        # Check if we need a section header
        for prefix, header in section_headers.items():
            if display.startswith(prefix) and header:
                html += f'<tr><th colspan="{2 + len(labels)}" style="background:#cfe2ff;text-align:left">{header}</th></tr>'
                break

        str_vals = [f"{v:.4f}" if v is not None else "N/A" for v in vals]
        delta_str = ""
        verdict_str = ""
        css_class = ""

        if len(vals) == 2 and vals[0] is not None and vals[1] is not None:
            delta = vals[1] - vals[0]
            delta_str = f"{delta:+.4f}"
            # For metrics, higher is better (except inter-class sim, reject rates where context matters)
            if key in ("inter_class_similarity",):
                # Lower inter-class is better
                if delta < 0:
                    css_class = "better"
                    verdict_str = "Better"
                elif delta > 0:
                    css_class = "worse"
                    verdict_str = "Worse"
                else:
                    css_class = "neutral"
                    verdict_str = "Same"
            elif delta > 0:
                css_class = "better"
                verdict_str = "Better"
            elif delta < 0:
                css_class = "worse"
                verdict_str = "Worse"
            else:
                css_class = "neutral"
                verdict_str = "Same"

        html += f'<tr><td>{display}</td>'
        for sv in str_vals:
            html += f'<td>{sv}</td>'
        if len(labels) == 2:
            html += f'<td>{delta_str}</td>'
            html += f'<td class="{css_class}">{verdict_str}</td>'
        html += '</tr>'

    html += '</table></div>'

    # Top confusing pairs
    html += '<div class="section"><h2>Top Confusing Pairs</h2>'
    for i, m in enumerate(all_metrics):
        pairs = m.get("top_confusing_pairs", [])
        html += f'<h3>{labels[i]}</h3>'
        if pairs:
            html += '<table><tr><th>#</th><th>Class A</th><th>Class B</th><th>Similarity</th></tr>'
            for j, p in enumerate(pairs[:5]):
                html += (f'<tr><td>{j+1}</td>'
                         f'<td>{p.get("name_i", "N/A")}</td>'
                         f'<td>{p.get("name_j", "N/A")}</td>'
                         f'<td>{p.get("similarity", 0):.4f}</td></tr>')
            html += '</table>'
        else:
            html += '<p>No confusing pairs data.</p>'
    html += '</div>'

    # Conclusions
    html += '<div class="section"><h2>Conclusions</h2>'
    verdict = conclusions.get("overall_verdict", "N/A")
    recommend = conclusions.get("recommend_supcon_for_pointnet_plus_plus", False)

    if recommend:
        html += f'<div class="verdict verdict-positive"><strong>Verdict:</strong> {verdict}</div>'
    else:
        html += f'<div class="verdict verdict-mixed"><strong>Verdict:</strong> {verdict}</div>'

    html += '<table><tr><th>Criterion</th><th>CE-only</th><th>CE+SupCon</th><th>Winner</th></tr>'

    for metric_display, ce_key, sc_key, winner_key in [
        ("Macro F1", "ce_only_macro_f1", "ce_supcon_macro_f1", "classification_winner"),
        ("Similarity Gap", "ce_only_similarity_gap", "ce_supcon_similarity_gap", "embedding_winner"),
        ("OOD AUROC", "ce_only_ood_auroc", "ce_supcon_ood_auroc", "ood_winner"),
    ]:
        ce_val = conclusions.get(ce_key, "N/A")
        sc_val = conclusions.get(sc_key, "N/A")
        winner = conclusions.get(winner_key, "N/A")
        html += f'<tr><td>{metric_display}</td><td>{ce_val}</td><td>{sc_val}</td><td>{winner}</td></tr>'

    html += '</table>'

    html += f'<h3>Detailed Checks</h3><ul>'
    embed_deg = conclusions.get("embedding_degradation", False)
    neg_drop = conclusions.get("negative_recall_dropped", False)
    html += f'<li>Embedding degradation: {"YES - SupCon hurt embedding quality" if embed_deg else "NO - SupCon improved or maintained embedding quality"}</li>'
    html += f'<li>Negative recall drop: {"YES - SupCon reduced negative recall" if neg_drop else "NO - SupCon maintained or improved negative recall"}</li>'
    html += f'<li>Recommend SupCon for PointNet++: {"YES" if recommend else "NO - need more investigation"}</li>'
    html += '</ul></div>'

    # How to proceed
    html += """<div class="section"><h2>How to Proceed</h2>
<ol>
<li>If CE+SupCon wins: Continue with SupCon in PointNet++ / PointNeXt experiments.</li>
<li>If CE-only wins: SupCon hyperparameters may need tuning (metric_weight, temperature, warmup_epochs),
    or the dataset size does not benefit from metric learning at the PointNet level.</li>
<li>If mixed: Focus on specific weakness areas (e.g., class_014, negative recall)
    and tune SupCon accordingly before moving to PointNet++.</li>
</ol>
<p>Regardless of the result, the next step is to introduce PointNet++ and/or PointNeXt
as a new backbone, running the same pipeline with the same data split for fair comparison.</p>
</div>"""

    html += '</body></html>'

    with open(output_path, "w") as f:
        f.write(html)
    print(f"HTML saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two or more experiment runs side by side",
    )
    parser.add_argument(
        "--runs", nargs="+", required=True,
        help="Paths to run directories (e.g., outputs/runs/newdata_pointnet_ce_only_v1)",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Display labels for each run (default: directory names)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="outputs/reports/controlled_comparison",
        help="Output directory for comparison reports",
    )
    args = parser.parse_args()

    if len(args.runs) < 2:
        print("[ERROR] At least 2 run directories required")
        sys.exit(1)

    # Validate run dirs
    for run_dir in args.runs:
        if not os.path.isdir(run_dir):
            print(f"[ERROR] Run directory not found: {run_dir}")
            sys.exit(1)

    # Labels
    labels = args.labels or [os.path.basename(r) for r in args.runs]
    if len(labels) != len(args.runs):
        print(f"[ERROR] Number of labels ({len(labels)}) != number of runs ({len(args.runs)})")
        sys.exit(1)

    print(f"Comparing {len(args.runs)} runs:")
    for i, (r, l) in enumerate(zip(args.runs, labels)):
        print(f"  [{i}] {l}: {r}")

    # Extract metrics
    all_metrics = []
    for run_dir in args.runs:
        print(f"\nExtracting metrics from {run_dir} ...")
        metrics = extract_run_metrics(run_dir)
        all_metrics.append(metrics)
        print(f"  Found {sum(1 for v in metrics.values() if v is not None)} metrics")

    # Build comparison
    rows = get_comparison_rows(all_metrics, labels)

    # Output
    os.makedirs(args.output_dir, exist_ok=True)

    write_csv(rows, labels, os.path.join(args.output_dir, "comparison.csv"))
    write_json_report(rows, all_metrics, labels, os.path.join(args.output_dir, "comparison.json"))
    write_html_report(rows, all_metrics, labels, os.path.join(args.output_dir, "comparison_report.html"))

    # Print summary
    print(f"\n{'=' * 60}")
    print("Comparison Summary")
    print(f"{'=' * 60}")
    for display, key, vals in rows:
        if len(vals) == 2 and vals[0] is not None and vals[1] is not None:
            delta = vals[1] - vals[0]
            arrow = "+" if delta > 0 else ""
            print(f"  {display}: {vals[0]:.4f} -> {vals[1]:.4f} ({arrow}{delta:.4f})")

    conclusions = generate_conclusions(all_metrics, labels)
    print(f"\nVerdict: {conclusions.get('overall_verdict', 'N/A')}")


if __name__ == "__main__":
    main()
