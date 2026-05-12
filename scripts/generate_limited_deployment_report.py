"""Generate a limited deployment report summarizing current model capabilities.

Aggregates coverage, OOD, retrieval, pseudo-OOD, and threshold data into a
single report with a clear deployment readiness assessment.

Usage:
    python scripts/generate_limited_deployment_report.py \
        --class-coverage outputs/reports/class_coverage_report.json \
        --ood outputs/reports/ood_eval_baseline_test_p05/ood_metrics.json \
        --retrieval outputs/reports/retrieval_eval_baseline_test/retrieval_metrics.json \
        --pseudo-ood outputs/reports/pseudo_ood_eval/pseudo_ood_metrics.json \
        --threshold outputs/prototypes/baseline_threshold_p05.json \
        --output-dir outputs/reports/limited_deployment
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_json(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate limited deployment report")
    parser.add_argument("--class-coverage", type=str,
                        default="outputs/reports/class_coverage_report.json")
    parser.add_argument("--ood", type=str,
                        default="outputs/reports/ood_eval_baseline_test_p05/ood_metrics.json")
    parser.add_argument("--retrieval", type=str,
                        default="outputs/reports/retrieval_eval_baseline_test/retrieval_metrics.json")
    parser.add_argument("--pseudo-ood", type=str,
                        default="outputs/reports/pseudo_ood_eval/pseudo_ood_metrics.json")
    parser.add_argument("--threshold", type=str,
                        default="outputs/prototypes/baseline_threshold_p05.json")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/reports/limited_deployment")
    args = parser.parse_args()

    coverage = load_json(args.class_coverage)
    ood = load_json(args.ood)
    retrieval = load_json(args.retrieval)
    pseudo_ood = load_json(args.pseudo_ood)
    threshold_data = load_json(args.threshold)

    # Load config for supported classes info
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.utils.config import load_config
    cfg = load_config(args.config)

    supported_cfg = cfg.get("supported_classes", {})
    supported_known_labels = supported_cfg.get("supported_known_labels", [])
    unsupported_known_labels = supported_cfg.get("unsupported_known_labels", [])

    # Load class names
    class_names_path = os.path.join(os.path.dirname(args.config), "class_names.json")
    class_names = {}
    if os.path.exists(class_names_path):
        with open(class_names_path) as f:
            class_names = json.load(f)

    # Determine deployment status
    has_unsupported = len(unsupported_known_labels) > 0
    negative_count = 0
    if coverage:
        for split_data in coverage.get("split_coverage", {}).values():
            negative_count = max(negative_count, split_data.get("19", split_data.get(str(cfg.get("negative_label", 19)), 0)))

    if has_unsupported:
        deployment_status = "limited_internal_prototype"
        deployment_label = "Ready for Internal Testing with Known Limitations"
    else:
        deployment_status = "production_ready"
        deployment_label = "Production Ready"

    # Build report data
    report = {
        "deployment_status": deployment_status,
        "deployment_label": deployment_label,
        "has_unsupported_known_classes": has_unsupported,
        "supported_known_classes": {
            "count": len(supported_known_labels),
            "labels": supported_known_labels,
        },
        "unsupported_known_classes": {
            "count": len(unsupported_known_labels),
            "labels": unsupported_known_labels,
            "names": [class_names.get(str(l), f"class_{l:03d}") for l in unsupported_known_labels],
        },
        "negative_evaluation": {
            "status": "limited",
            "train_samples": negative_count,
            "note": "Negative sample count is too low for statistically reliable evaluation",
        },
        "threshold": {
            "value": threshold_data.get("selected_threshold") if threshold_data else None,
            "strategy": threshold_data.get("selection_strategy") if threshold_data else None,
        },
        "ood_metrics": {
            "known_accept_rate": ood.get("known_accept_rate") if ood else None,
            "negative_reject_rate": ood.get("negative_reject_rate") if ood else None,
            "auroc": ood.get("auroc") if ood else None,
        },
        "retrieval_metrics": {
            "top1_accuracy_on_accepted": retrieval.get("top1_accuracy_on_accepted") if retrieval else None,
            "recall_at_1": retrieval.get("recall@1") if retrieval else None,
        },
        "pseudo_ood_metrics": {
            "mean_reject_rate": pseudo_ood.get("mean_pseudo_unknown_reject_rate") if pseudo_ood else None,
            "median_reject_rate": pseudo_ood.get("median_pseudo_unknown_reject_rate") if pseudo_ood else None,
            "min_reject_rate": pseudo_ood.get("min_pseudo_unknown_reject_rate") if pseudo_ood else None,
            "num_classes_evaluated": pseudo_ood.get("num_classes_evaluated") if pseudo_ood else None,
        },
        "recommended_inference_logic": [
            "Step 1: Classifier negative (label 19) → negative",
            "Step 2: Prototype similarity < threshold → unknown",
            "Step 3: Prototype match → known class",
            "Step 4: Gallery retrieval for evidence only",
        ],
        "required_data_improvements": [
            "Add training samples for class_014 (shenggaozuofalan)",
            "Add more negative/unknown samples for robust OOD evaluation",
            "Add hard negatives similar to existing flange/ring/ear classes",
        ],
    }

    # Save JSON
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "limited_deployment_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Build HTML
    html = _build_html(report, class_names, cfg)
    html_path = os.path.join(args.output_dir, "limited_deployment_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report saved to:")
    print(f"  {json_path}")
    print(f"  {html_path}")
    print(f"\nDeployment Status: {deployment_status}")
    print(f"  {deployment_label}")
    if has_unsupported:
        for l in unsupported_known_labels:
            cn = class_names.get(str(l), f"class_{l:03d}")
            print(f"  UNSUPPORTED: {cn} (label {l}) - no training data")


def _build_html(report, class_names, cfg):
    import html as html_mod
    from datetime import datetime

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    status_color = {
        "limited_internal_prototype": "#f59e0b",
        "production_ready": "#10b981",
    }
    sc = status_color.get(report["deployment_status"], "#555")

    CSS = """
    :root { --bg:#f5f7fa;--card-bg:#fff;--text:#1a1a2e;--text2:#555;--border:#e0e4e8;
            --accent:#3b82f6;--success:#10b981;--warning:#f59e0b;--danger:#ef4444; }
    *{margin:0;padding:0;box-sizing:border-box}
    body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         background:var(--bg);color:var(--text);line-height:1.6;padding:24px}
    .container{max-width:1100px;margin:0 auto}
    h1{font-size:1.8rem;font-weight:700;margin-bottom:8px}
    h2{font-size:1.3rem;font-weight:600;margin-top:32px;margin-bottom:16px;
       padding-bottom:8px;border-bottom:2px solid var(--accent)}
    .badge{display:inline-block;padding:6px 16px;border-radius:20px;font-weight:700;
           font-size:0.95rem;color:#fff;margin-left:12px}
    .status-box{border-radius:12px;padding:20px;margin-bottom:24px;border:2px solid}
    .table-wrap{overflow-x:auto;margin-bottom:24px;background:var(--card-bg);
                border-radius:8px;border:1px solid var(--border)}
    table{width:100%;border-collapse:collapse;font-size:0.85rem}
    th{background:#f8f9fb;padding:10px 12px;text-align:left;font-weight:600;
       border-bottom:2px solid var(--border)}
    td{padding:8px 12px;border-bottom:1px solid var(--border)}
    tr:last-child td{border-bottom:none}tr:hover td{background:#f8f9fb}
    .cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:16px;margin-bottom:24px}
    .card{background:var(--card-bg);border-radius:10px;padding:16px;text-align:center;
          border:1px solid var(--border)}
    .card-value{font-size:1.3rem;font-weight:700;color:var(--accent)}
    .card-label{font-size:0.75rem;color:var(--text2);text-transform:uppercase}
    .error-box{background:#fef2f2;border:1px solid var(--danger);border-radius:8px;
               padding:12px 16px;margin-bottom:8px;font-size:0.85rem}
    .warn-box{background:#fffbeb;border:1px solid var(--warning);border-radius:8px;
              padding:12px 16px;margin-bottom:8px;font-size:0.85rem}
    .info-box{background:#eff6ff;border:1px solid var(--accent);border-radius:8px;
              padding:12px 16px;margin-bottom:16px;font-size:0.85rem;line-height:1.7}
    ol{padding-left:20px;margin-bottom:16px}li{margin-bottom:6px}
    footer{margin-top:40px;text-align:center;color:var(--text2);font-size:0.8rem}
    """

    # Status banner
    status_html = f"""<div class="status-box" style="background:{sc}15;border-color:{sc}">
    <h2 style="border:none;margin-top:0;color:{sc}">Deployment Status: {html_mod.escape(report["deployment_status"])}</h2>
    <p style="font-size:1.1rem;font-weight:600">{html_mod.escape(report["deployment_label"])}</p>
    </div>"""

    # Supported range
    supp = report["supported_known_classes"]
    unsupp = report["unsupported_known_classes"]
    supp_names = [class_names.get(str(l), f"class_{l:03d}") for l in supp["labels"]]

    supported_table = f"""<div class="table-wrap"><table>
    <tr><th>Item</th><th>Detail</th></tr>
    <tr><td>Supported known classes</td><td>{supp['count']}</td></tr>
    <tr><td>Unsupported known classes</td><td>{unsupp['count']}</td></tr>
    <tr><td>Unsupported labels</td><td>{html_mod.escape(str(unsupp['labels']))}</td></tr>
    <tr><td>Unsupported names</td><td>{html_mod.escape(', '.join(unsupp['names']))}</td></tr>
    <tr><td>Negative evaluation</td><td>{html_mod.escape(report['negative_evaluation']['status'])} ({report['negative_evaluation']['train_samples']} train samples)</td></tr>
    </table></div>"""

    # Limitations
    limitations_html = ""
    if unsupp["count"] > 0:
        for name in unsupp["names"]:
            limitations_html += f'<div class="error-box">Cannot reliably identify <strong>{html_mod.escape(name)}</strong> — no training samples available</div>\n'
    limitations_html += '<div class="warn-box">Negative/unknown rejection evaluation is preliminary due to limited negative data</div>\n'

    # Metrics cards
    ood_m = report.get("ood_metrics", {})
    ret_m = report.get("retrieval_metrics", {})
    pseudo_m = report.get("pseudo_ood_metrics", {})
    thresh = report.get("threshold", {})

    def _f(v, d=3):
        if v is None:
            return "N/A"
        return f"{v:.{d}f}"

    metrics_cards = ""
    for label, val in [
        ("Threshold", _f(thresh.get("value"))),
        ("Known Accept Rate", _f(ood_m.get("known_accept_rate"))),
        ("Negative Reject Rate", _f(ood_m.get("negative_reject_rate"))),
        ("AUROC", _f(ood_m.get("auroc"))),
        ("Top1 Acc (accepted)", _f(ret_m.get("top1_accuracy_on_accepted"))),
        ("Pseudo-OOD Mean Reject", _f(pseudo_m.get("mean_reject_rate"))),
    ]:
        metrics_cards += f'<div class="card"><div class="card-value">{val}</div><div class="card-label">{html_mod.escape(label)}</div></div>\n'

    # Inference logic
    logic_html = "<ol>"
    for step in report.get("recommended_inference_logic", []):
        logic_html += f"<li>{html_mod.escape(step)}</li>"
    logic_html += "</ol>"

    # Required improvements
    improve_html = "<ol>"
    for item in report.get("required_data_improvements", []):
        improve_html += f"<li>{html_mod.escape(item)}</li>"
    improve_html += "</ol>"

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Limited Deployment Report</title>
<style>{CSS}</style></head><body><div class="container">
<h1>Limited Deployment Report <span class="badge" style="background:{sc}">{html_mod.escape(report["deployment_status"])}</span></h1>
<p style="color:var(--text2);margin-bottom:24px">Generated at {html_mod.escape(ts)}</p>

{status_html}

<h2>Current Support Range</h2>
{supported_table}

<h2>Limitations</h2>
{limitations_html}

<h2>Key Metrics</h2>
<div class="cards">{metrics_cards}</div>

<h2>Recommended Inference Logic</h2>
<div class="info-box">{logic_html}</div>

<h2>Required Data Improvements</h2>
<div class="info-box">{improve_html}</div>

<footer>Limited Deployment Report &mdash; pointcloud_metric_learning</footer>
</div></body></html>"""


if __name__ == "__main__":
    main()
