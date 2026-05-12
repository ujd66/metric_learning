"""Compare embedding evaluation reports between baseline and metric learning.

Usage:
    python scripts/compare_embedding_reports.py \
        --baseline-json outputs/reports/embedding_eval/metrics.json \
        --new-json outputs/runs/ce_supcon/embedding_eval_all/metrics.json \
        --output outputs/runs/ce_supcon/comparison_report.html

Input JSON files are the metrics.json produced by evaluate_embeddings.py.
"""

import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _fmt(val, decimals=4):
    if val is None:
        return "N/A"
    if isinstance(val, float) and math.isnan(val):
        return "N/A"
    if isinstance(val, (int, float)):
        return f"{val:.{decimals}f}"
    return str(val)


def _delta_str(baseline, new, decimals=4, higher_better=True):
    """Return formatted delta string and classification."""
    if baseline is None or new is None:
        return "N/A", "unchanged"
    if isinstance(baseline, float) and math.isnan(baseline):
        return "N/A", "unchanged"
    if isinstance(new, float) and math.isnan(new):
        return "N/A", "unchanged"

    delta = new - baseline
    sign = "+" if delta > 0 else ""
    delta_str = f"{sign}{delta:.{decimals}f}"

    threshold = 0.005  # 0.5% absolute threshold for "unchanged"
    if abs(delta) < threshold:
        return delta_str, "unchanged"
    elif (delta > 0 and higher_better) or (delta < 0 and not higher_better):
        return delta_str, "improved"
    else:
        return delta_str, "declined"


def load_metrics(path):
    """Load metrics.json from evaluate_embeddings.py."""
    with open(path, "r") as f:
        return json.load(f)


def extract_comparison_metrics(baseline, new):
    """Extract and compare key metrics."""
    comparisons = []

    # Intra-class similarity
    b_intra = baseline.get("intra_class_similarity", {}).get("macro_avg")
    n_intra = new.get("intra_class_similarity", {}).get("macro_avg")
    delta, status = _delta_str(b_intra, n_intra, higher_better=True)
    comparisons.append({
        "metric": "Intra-class Similarity (macro)",
        "baseline": b_intra, "new": n_intra,
        "delta": delta, "status": status,
    })

    # Inter-class similarity (lower is better)
    b_inter = baseline.get("inter_class_similarity", {}).get("global_avg")
    n_inter = new.get("inter_class_similarity", {}).get("global_avg")
    delta, status = _delta_str(b_inter, n_inter, higher_better=False)
    comparisons.append({
        "metric": "Inter-class Similarity (global)",
        "baseline": b_inter, "new": n_inter,
        "delta": delta, "status": status,
    })

    # Similarity gap
    b_gap = baseline.get("similarity_gap")
    n_gap = new.get("similarity_gap")
    delta, status = _delta_str(b_gap, n_gap, higher_better=True)
    comparisons.append({
        "metric": "Similarity Gap",
        "baseline": b_gap, "new": n_gap,
        "delta": delta, "status": status,
    })

    # Recall@K
    for k_str in sorted(baseline.get("recall_at_k", {}).keys()):
        b_val = baseline["recall_at_k"][k_str]
        n_val = new.get("recall_at_k", {}).get(k_str)
        delta, status = _delta_str(b_val, n_val, higher_better=True)
        comparisons.append({
            "metric": f"Recall@{k_str.replace('recall@', '')}",
            "baseline": b_val, "new": n_val,
            "delta": delta, "status": status,
        })

    # Precision@K
    for k_str in sorted(baseline.get("precision_at_k", {}).keys()):
        b_val = baseline["precision_at_k"][k_str]
        n_val = new.get("precision_at_k", {}).get(k_str)
        delta, status = _delta_str(b_val, n_val, higher_better=True)
        comparisons.append({
            "metric": f"Precision@{k_str.replace('precision@', '')}",
            "baseline": b_val, "new": n_val,
            "delta": delta, "status": status,
        })

    # NN accuracy
    b_nn = baseline.get("nn_accuracy")
    n_nn = new.get("nn_accuracy")
    delta, status = _delta_str(b_nn, n_nn, higher_better=True)
    comparisons.append({
        "metric": "1-NN Accuracy",
        "baseline": b_nn, "new": n_nn,
        "delta": delta, "status": status,
    })

    # Macro F1 (from per_class_nn_accuracy or classification metrics)
    b_f1 = baseline.get("macro_f1")
    n_f1 = new.get("macro_f1")
    if b_f1 is not None or n_f1 is not None:
        delta, status = _delta_str(b_f1, n_f1, higher_better=True)
        comparisons.append({
            "metric": "Macro F1",
            "baseline": b_f1, "new": n_f1,
            "delta": delta, "status": status,
        })

    return comparisons


def extract_focus_pair_metrics(baseline, new, focus_pairs):
    """Extract metrics for focus class pairs."""
    results = []

    for pair_names in focus_pairs:
        if len(pair_names) != 2:
            continue
        name_a, name_b = pair_names

        # Look for this pair in top_confusing_pairs data
        # If the user also provides --baseline-confusing and --new-confusing CSVs
        b_sim = _find_pair_similarity(baseline, name_a, name_b)
        n_sim = _find_pair_similarity(new, name_a, name_b)

        delta, status = _delta_str(b_sim, n_sim, higher_better=False)  # lower similarity = better

        results.append({
            "pair": f"{name_a} <-> {name_b}",
            "baseline_similarity": b_sim,
            "new_similarity": n_sim,
            "delta": delta,
            "status": status,
        })

    return results


def _find_pair_similarity(metrics, name_a, name_b):
    """Try to find pair similarity from confusing_pairs data in metrics."""
    pairs = metrics.get("top_confusing_pairs", [])
    for p in pairs:
        if (p.get("name_i") == name_a and p.get("name_j") == name_b) or \
           (p.get("name_i") == name_b and p.get("name_j") == name_a):
            return p.get("similarity")
    return None


def build_html_report(comparisons, focus_pairs, baseline_path, new_path, output_path):
    """Build comparison HTML report."""
    import html as html_mod

    css = """
    :root {
        --bg: #f5f7fa;
        --card-bg: #ffffff;
        --text: #1a1a2e;
        --text-secondary: #555;
        --border: #e0e4e8;
        --accent: #3b82f6;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background: var(--bg); color: var(--text); line-height: 1.6; padding: 24px;
    }
    .container { max-width: 1000px; margin: 0 auto; }
    h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 8px; }
    h2 { font-size: 1.3rem; font-weight: 600; margin-top: 32px; margin-bottom: 16px;
         padding-bottom: 8px; border-bottom: 2px solid var(--accent); }
    .subtitle { color: var(--text-secondary); margin-bottom: 24px; font-size: 0.95rem; }

    .table-wrap {
        overflow-x: auto; margin-bottom: 24px; background: var(--card-bg);
        border-radius: 8px; border: 1px solid var(--border);
    }
    table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    th { background: #f8f9fb; padding: 10px 12px; text-align: left;
         font-weight: 600; border-bottom: 2px solid var(--border); white-space: nowrap; }
    td { padding: 8px 12px; border-bottom: 1px solid var(--border); white-space: nowrap; }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: #f8f9fb; }

    .improved { color: var(--success); font-weight: 600; }
    .declined { color: var(--danger); font-weight: 600; }
    .unchanged { color: var(--text-secondary); }

    .summary-cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px; }
    .summary-card {
        background: var(--card-bg); border-radius: 10px; padding: 20px 16px;
        text-align: center; border: 1px solid var(--border);
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .summary-card .num { font-size: 2rem; font-weight: 700; }
    .summary-card .label { font-size: 0.85rem; color: var(--text-secondary); margin-top: 4px; }

    .warning-banner {
        background: #fef3c7; border: 1px solid #f59e0b; border-radius: 6px;
        padding: 12px 16px; margin-bottom: 16px; color: #92400e; font-size: 0.9rem;
    }

    footer { margin-top: 40px; text-align: center; color: var(--text-secondary); font-size: 0.8rem; }
    """

    # Count statuses
    improved = sum(1 for c in comparisons if c["status"] == "improved")
    declined = sum(1 for c in comparisons if c["status"] == "declined")
    unchanged = sum(1 for c in comparisons if c["status"] == "unchanged")

    # Summary cards
    summary_html = f"""
    <div class="summary-cards">
        <div class="summary-card">
            <div class="num" style="color: var(--success);">{improved}</div>
            <div class="label">Improved</div>
        </div>
        <div class="summary-card">
            <div class="num" style="color: var(--danger);">{declined}</div>
            <div class="label">Declined</div>
        </div>
        <div class="summary-card">
            <div class="num" style="color: var(--text-secondary);">{unchanged}</div>
            <div class="label">Unchanged</div>
        </div>
    </div>
    """

    # Comparison table
    rows_html = ""
    for c in comparisons:
        cls = c["status"]
        rows_html += f"""<tr>
            <td>{html_mod.escape(c['metric'])}</td>
            <td>{_fmt(c['baseline'])}</td>
            <td>{_fmt(c['new'])}</td>
            <td class="{cls}">{c['delta']}</td>
            <td class="{cls}">{c['status']}</td>
        </tr>\n"""

    table_html = f"""<div class="table-wrap">
    <table>
        <tr><th>Metric</th><th>Baseline</th><th>New</th><th>Delta</th><th>Status</th></tr>
        {rows_html}
    </table>
    </div>"""

    # Focus pairs section
    focus_html = ""
    if focus_pairs:
        focus_rows = ""
        for fp in focus_pairs:
            cls = fp["status"]
            focus_rows += f"""<tr>
                <td>{html_mod.escape(fp['pair'])}</td>
                <td>{_fmt(fp['baseline_similarity'])}</td>
                <td>{_fmt(fp['new_similarity'])}</td>
                <td class="{cls}">{fp['delta']}</td>
                <td class="{cls}">{fp['status']}</td>
            </tr>\n"""

        focus_html = f"""
        <h2>Focus Class Pairs</h2>
        <div class="table-wrap">
        <table>
            <tr><th>Pair</th><th>Baseline Sim</th><th>New Sim</th><th>Delta</th><th>Status</th></tr>
            {focus_rows}
        </table>
        </div>"""

    # Verdict
    verdict = ""
    if declined == 0:
        verdict = '<div style="background: #d1fae5; border: 1px solid #10b981; border-radius: 6px; padding: 16px; margin-bottom: 24px; color: #065f46;">All metrics improved or unchanged. Metric learning appears effective.</div>'
    elif declined <= 2:
        verdict = '<div class="warning-banner">Some metrics declined slightly. Review the specific metrics before proceeding.</div>'
    else:
        verdict = '<div style="background: #fee2e2; border: 1px solid #ef4444; border-radius: 6px; padding: 16px; margin-bottom: 24px; color: #991b1b;">Multiple metrics declined. Consider adjusting metric_weight or temperature.</div>'

    report = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Embedding Comparison Report</title>
<style>{css}</style>
</head>
<body>
<div class="container">
<h1>Embedding Comparison Report</h1>
<p class="subtitle">
    Baseline: {html_mod.escape(os.path.basename(baseline_path))}<br>
    New: {html_mod.escape(os.path.basename(new_path))}
</p>

<h2>Summary</h2>
{summary_html}

{verdict}

<h2>Metric Comparison</h2>
{table_html}

{focus_html}

<h2>Success Criteria</h2>
<div style="background: var(--card-bg); border-radius: 8px; border: 1px solid var(--border); padding: 16px; margin-bottom: 24px; font-size: 0.9rem;">
<ol>
<li>Val Macro F1 not degraded by more than 3 percentage points</li>
<li>Recall@1 not lower than baseline</li>
<li>Similarity gap not lower than baseline</li>
<li>Top confusing pair similarity decreased or at least not increased</li>
<li>Negative accuracy not significantly worsened</li>
<li>Training stable, no NaN</li>
</ol>
</div>

<footer>Comparison Report &mdash; pointcloud_metric_learning</footer>
</div>
</body>
</html>"""

    return report


def main():
    parser = argparse.ArgumentParser(description="Compare embedding evaluation reports")
    parser.add_argument("--baseline-json", required=True, help="Path to baseline metrics.json")
    parser.add_argument("--new-json", required=True, help="Path to new metrics.json")
    parser.add_argument("--output", required=True, help="Output HTML path")
    parser.add_argument("--focus-pairs", nargs="+", default=None,
                        help="Focus class pairs as 'name1:name2' (e.g. changtiaofalan:teshujiaqiangtieduantou)")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Config path for focus_class_pairs")
    args = parser.parse_args()

    baseline = load_metrics(args.baseline_json)
    new = load_metrics(args.new_json)

    comparisons = extract_comparison_metrics(baseline, new)

    # Focus pairs
    focus_pair_names = []
    if args.focus_pairs:
        for pair_str in args.focus_pairs:
            parts = pair_str.split(":")
            if len(parts) == 2:
                focus_pair_names.append(parts)
    else:
        # Try loading from config
        try:
            from src.utils.config import load_config
            cfg = load_config(args.config)
            focus_pair_names = cfg.get("analysis", {}).get("focus_class_pairs", [])
        except Exception:
            pass

    focus_pairs = extract_focus_pair_metrics(baseline, new, focus_pair_names)

    # Build report
    report_html = build_html_report(
        comparisons, focus_pairs,
        args.baseline_json, args.new_json, args.output,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report_html)
    print(f"Comparison report saved to {args.output}")

    # Also save CSV
    csv_path = args.output.replace(".html", "_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "baseline", "new", "delta", "status"])
        for c in comparisons:
            writer.writerow([
                c["metric"],
                _fmt(c["baseline"]),
                _fmt(c["new"]),
                c["delta"],
                c["status"],
            ])
    print(f"Comparison CSV saved to {csv_path}")

    # Console summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    improved = sum(1 for c in comparisons if c["status"] == "improved")
    declined = sum(1 for c in comparisons if c["status"] == "declined")
    unchanged = sum(1 for c in comparisons if c["status"] == "unchanged")
    print(f"  Improved: {improved}, Declined: {declined}, Unchanged: {unchanged}")

    for c in comparisons:
        marker = {"improved": "+", "declined": "-", "unchanged": "="}[c["status"]]
        print(f"  [{marker}] {c['metric']}: {_fmt(c['baseline'])} -> {_fmt(c['new'])} ({c['delta']})")

    if focus_pairs:
        print("\n  Focus Class Pairs:")
        for fp in focus_pairs:
            marker = {"improved": "+", "declined": "-", "unchanged": "="}[fp["status"]]
            print(f"  [{marker}] {fp['pair']}: {_fmt(fp['baseline_similarity'])} -> {_fmt(fp['new_similarity'])} ({fp['delta']})")

    print("=" * 60)


if __name__ == "__main__":
    main()
