"""Generate final regression report from all pipeline outputs.

Aggregates classification, embedding, OOD, retrieval, and coverage results
into a comprehensive HTML and JSON report, with comparison to the previous
run if available.

Usage:
    python scripts/generate_final_report.py \
        --config configs/config.yaml \
        --run-name newdata_pointnet_baseline_v1 \
        --run-dir outputs/runs/newdata_pointnet_baseline_v1 \
        --checkpoint outputs/checkpoints/best.pt \
        --prototypes outputs/runs/newdata_pointnet_baseline_v1/prototypes/baseline_prototypes.pt \
        --threshold-json outputs/runs/newdata_pointnet_baseline_v1/prototypes/baseline_threshold.json \
        --gallery outputs/runs/newdata_pointnet_baseline_v1/gallery/baseline_train_gallery.pt \
        --output-dir outputs/runs/newdata_pointnet_baseline_v1
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config


def load_json(path):
    """Load JSON file, return None if not found."""
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _fmt(val, d=4):
    if val is None:
        return "N/A"
    return f"{val:.{d}f}"


def _safe_div(a, b):
    return a / b if b > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Generate final regression report")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--prototypes", type=str, required=True)
    parser.add_argument("--threshold-json", type=str, required=True)
    parser.add_argument("--gallery", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--previous-run-dir", type=str, default=None,
                        help="Path to previous run dir for comparison")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.run_dir

    cfg = load_config(args.config)
    report_dir = args.run_dir

    # Load all data
    print("Loading pipeline results ...")

    # 1. Classification
    classification = load_json(os.path.join(report_dir, "reports", "classification", "evaluation.json"))

    # 2. Embedding
    embed_metrics = load_json(os.path.join(report_dir, "reports", "embeddings", "metrics_summary.json"))
    embed_per_class = load_json(os.path.join(report_dir, "reports", "embeddings", "metrics.json"))

    # 3. OOD
    ood_metrics = load_json(os.path.join(report_dir, "reports", "ood", "ood_metrics.json"))
    ood_per_class = None
    pc_ood_path = os.path.join(report_dir, "reports", "ood", "per_class_ood_metrics.csv")
    if os.path.exists(pc_ood_path):
        import csv
        with open(pc_ood_path) as f:
            reader = csv.DictReader(f)
            ood_per_class = list(reader)

    # 4. Retrieval
    retrieval_metrics = load_json(os.path.join(report_dir, "reports", "retrieval", "retrieval_metrics.json"))

    # 5. Threshold
    threshold_data = load_json(args.threshold_json)

    # 6. Coverage
    coverage = load_json("outputs/reports/class_coverage_report.json")
    validation = load_json("outputs/reports/dataset_validation.json")
    split_summary = load_json("outputs/reports/split_summary.json")

    # 7. Training history
    training_history = load_json(os.path.join(report_dir, "reports", "classification", "..", "..", "logs",
                                              "training_history.json"))
    if not training_history:
        # Try legacy location
        training_history = load_json("outputs/reports/training_history.json")

    # 8. Prototypes info
    import torch
    proto_info = {}
    if os.path.exists(args.prototypes):
        proto_data = torch.load(args.prototypes, map_location="cpu", weights_only=False)
        proto_info = {
            "num_prototypes": proto_data["prototypes"].shape[0],
            "class_support": proto_data.get("class_support", {}),
            "supported_known_labels": proto_data.get("supported_known_labels", []),
            "unsupported_known_labels": proto_data.get("unsupported_known_labels", []),
        }

    # 9. Gallery info
    gallery_info = {}
    if os.path.exists(args.gallery):
        gal_data = torch.load(args.gallery, map_location="cpu", weights_only=False)
        gallery_labels = gal_data["labels"].tolist()
        label_counts = Counter(gallery_labels)
        gallery_info = {
            "num_samples": len(gallery_labels),
            "num_classes": len(label_counts),
            "per_class": {str(k): v for k, v in sorted(label_counts.items())},
        }

    # Load class names
    class_names_path = os.path.join(os.path.dirname(args.config), "class_names.json")
    class_names = {}
    if os.path.exists(class_names_path):
        with open(class_names_path) as f:
            class_names = json.load(f)

    num_known_classes = cfg["num_known_classes"]
    negative_label = cfg["negative_label"]

    # =========================================================================
    # Build report data
    # =========================================================================
    print("Building report data ...")

    # Data distribution from split_summary
    data_dist = {}
    if split_summary and "per_class" in split_summary:
        for class_dir, info in split_summary["per_class"].items():
            data_dist[class_dir] = {
                "total": info["total"],
                "train": info["train"],
                "val": info["val"],
                "test": info["test"],
            }

    # Class weights (recompute from dataset)
    class_weights_info = {}
    use_cw = cfg.get("train", {}).get("use_class_weight", False)
    cw_max = cfg.get("train", {}).get("class_weight_max", 10.0)
    if data_dist:
        train_counts = {k: v["train"] for k, v in data_dist.items() if v["train"] > 0}
        total_train = sum(train_counts.values())
        num_classes_total = cfg["num_classes"]
        weights = {}
        for c in range(num_classes_total):
            cn = class_names.get(str(c), f"class_{c:03d}")
            cnt_key = cn if cn in train_counts else f"class_{c:03d}"
            cnt = train_counts.get(cnt_key, 0)
            if cnt == 0:
                w = 1.0
            else:
                w = total_train / (num_classes_total * cnt)
                w = min(max(w, 1.0), cw_max)
            weights[str(c)] = {"name": cn, "train_count": cnt, "weight": round(w, 3)}

        class_weights_info = {
            "enabled": use_cw,
            "max_weight": cw_max,
            "per_class": weights,
            "imbalance_ratio": None,
            "largest_class_ratio": None,
        }

        non_zero = [v["train_count"] for v in weights.values() if v["train_count"] > 0]
        if non_zero:
            class_weights_info["imbalance_ratio"] = round(max(non_zero) / min(non_zero), 2)
            class_weights_info["largest_class_ratio"] = round(max(non_zero) / total_train, 4)

    # class_014 specific
    class_014_info = {
        "name": class_names.get("14", "shenggaozuofalan"),
        "supported": 14 not in cfg.get("supported_classes", {}).get("unsupported_known_labels", []),
        "train_samples": data_dist.get("class_014", data_dist.get("shenggaozuofalan", {})).get("train", 0) if data_dist else "N/A",
        "has_prototype": proto_info.get("class_support", {}).get("14", proto_info.get("class_support", {}).get(14, 0)) > 0 if proto_info.get("class_support") else "N/A",
        "has_gallery": str(14) in gallery_info.get("per_class", {}) or 14 in gallery_info.get("per_class", {}) if gallery_info else "N/A",
    }

    # Negative specific
    neg_info = {
        "name": class_names.get(str(negative_label), "qita"),
        "label": negative_label,
        "train_samples": data_dist.get("negative", data_dist.get("qita", {})).get("train", 0) if data_dist else "N/A",
    }
    if classification:
        pc = classification.get("per_class", {}).get(str(negative_label), {})
        neg_info["precision"] = pc.get("precision")
        neg_info["recall"] = pc.get("recall")
        neg_info["f1"] = pc.get("f1")
        neg_info["support"] = pc.get("support")

    # Most similar class pairs from embedding eval
    top_confusing = embed_metrics.get("top_confusing_pairs", []) if embed_metrics else []

    # Recommended threshold
    recommended_threshold = threshold_data.get("selected_threshold") if threshold_data else None
    recommended_strategy = threshold_data.get("selection_strategy") if threshold_data else None

    # =========================================================================
    # Build JSON report
    # =========================================================================
    report_data = {
        "run_name": args.run_name,
        "generated_at": datetime.now().isoformat(),
        "config": {
            "num_known_classes": num_known_classes,
            "negative_label": negative_label,
            "embedding_dim": cfg["embedding_dim"],
            "supported_known_labels": cfg.get("supported_classes", {}).get("supported_known_labels", list(range(num_known_classes))),
            "unsupported_known_labels": cfg.get("supported_classes", {}).get("unsupported_known_labels", []),
        },
        "data_distribution": data_dist,
        "class_weights": class_weights_info,
        "classification": classification,
        "embedding": embed_metrics,
        "ood": ood_metrics,
        "retrieval": retrieval_metrics,
        "threshold": {
            "value": recommended_threshold,
            "strategy": recommended_strategy,
            "data": threshold_data,
        },
        "class_014": class_014_info,
        "negative": neg_info,
        "top_confusing_pairs": top_confusing[:5],
        "coverage_status": coverage.get("status") if coverage else "UNKNOWN",
        "class_names": class_names,
    }

    # Save JSON
    json_path = os.path.join(args.output_dir, "final_report.json")
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved: {json_path}")

    # =========================================================================
    # Build HTML report
    # =========================================================================
    html = _build_html_report(report_data, cfg, ood_per_class)
    html_path = os.path.join(args.output_dir, "final_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {html_path}")

    # =========================================================================
    # Build comparison to previous version (if available)
    # =========================================================================
    prev_dir = args.previous_run_dir
    if not prev_dir:
        # Try to find previous run
        runs_dir = "outputs/runs"
        if os.path.isdir(runs_dir):
            other_runs = sorted([
                d for d in os.listdir(runs_dir)
                if os.path.isdir(os.path.join(runs_dir, d)) and d != args.run_name
            ])
            if other_runs:
                prev_dir = os.path.join(runs_dir, other_runs[-1])
                print(f"Found previous run: {prev_dir}")

    if prev_dir and os.path.isdir(prev_dir):
        prev_json = load_json(os.path.join(prev_dir, "final_report.json"))
        if prev_json:
            comp_html = _build_comparison_html(report_data, prev_json)
            comp_path = os.path.join(args.output_dir, "comparison_to_previous.html")
            with open(comp_path, "w", encoding="utf-8") as f:
                f.write(comp_html)
            print(f"Saved: {comp_path}")
        else:
            print(f"No final_report.json found in {prev_dir}, skipping comparison")
    else:
        # Generate a standalone comparison based on known old values
        # Old config: 18 supported + 1 unsupported (class_014), ~16 negative
        old_snapshot = {
            "run_name": "old_pointnet_baseline (pre-Phase 3.0)",
            "config": {
                "num_known_classes": 19,
                "unsupported_known_labels": [14],
            },
            "classification": {"macro_f1": None, "overall_accuracy": None},
            "ood": {"auroc": None, "known_accept_rate": None},
            "threshold": {"value": 0.91, "strategy": "known_quantile"},
            "negative": {"train_samples": 16},
            "class_014": {"supported": False, "train_samples": 0},
        }
        comp_html = _build_comparison_html(report_data, old_snapshot)
        comp_path = os.path.join(args.output_dir, "comparison_to_previous.html")
        with open(comp_path, "w", encoding="utf-8") as f:
            f.write(comp_html)
        print(f"Saved: {comp_path}")

    # Print console summary
    print(f"\n{'='*60}")
    print(f"FINAL REGRESSION REPORT SUMMARY")
    print(f"{'='*60}")
    print(f"Run: {args.run_name}")
    print(f"Class coverage: {report_data['coverage_status']}")
    print(f"class_014 (shenggaozuofalan): {'SUPPORTED' if class_014_info['supported'] else 'UNSUPPORTED'}")
    print(f"  Train samples: {class_014_info['train_samples']}")
    print(f"Negative (qita): {neg_info['train_samples']} train samples")
    if classification:
        print(f"Classification:")
        print(f"  Overall Acc: {_fmt(classification.get('overall_accuracy'))}")
        print(f"  Macro F1: {_fmt(classification.get('macro_f1'))}")
        print(f"  Known Acc: {_fmt(classification.get('known_class_accuracy'))}")
    if ood_metrics:
        print(f"OOD:")
        print(f"  AUROC: {_fmt(ood_metrics.get('auroc'))}")
        print(f"  Known Accept Rate: {_fmt(ood_metrics.get('known_accept_rate'))}")
        print(f"  Negative Reject Rate: {_fmt(ood_metrics.get('negative_reject_rate'))}")
    if recommended_threshold:
        print(f"Recommended threshold: {recommended_threshold:.2f} ({recommended_strategy})")
    print(f"{'='*60}")


def _build_html_report(report_data, cfg, ood_per_class):
    import html as html_mod
    from datetime import datetime

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
    h3{font-size:1.1rem;font-weight:600;margin-top:20px;margin-bottom:12px;color:var(--accent)}
    .subtitle{color:var(--text2);margin-bottom:24px;font-size:0.95rem}
    .cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:16px;margin-bottom:24px}
    .card{background:var(--card-bg);border-radius:10px;padding:16px;text-align:center;
          border:1px solid var(--border)}
    .card-value{font-size:1.3rem;font-weight:700;color:var(--accent)}
    .card-label{font-size:0.75rem;color:var(--text2);text-transform:uppercase}
    .table-wrap{overflow-x:auto;margin-bottom:24px;background:var(--card-bg);
                border-radius:8px;border:1px solid var(--border)}
    table{width:100%;border-collapse:collapse;font-size:0.85rem}
    th{background:#f8f9fb;padding:10px 12px;text-align:left;font-weight:600;
       border-bottom:2px solid var(--border)}
    td{padding:8px 12px;border-bottom:1px solid var(--border)}
    tr:last-child td{border-bottom:none}tr:hover td{background:#f8f9fb}
    .good{color:var(--success)}.warn{color:var(--warning)}.bad{color:var(--danger)}
    .info-box{background:#eff6ff;border:1px solid var(--accent);border-radius:8px;
              padding:12px 16px;margin-bottom:16px;font-size:0.85rem;line-height:1.7}
    .success-box{background:#ecfdf5;border:1px solid var(--success);border-radius:8px;
                 padding:12px 16px;margin-bottom:16px;font-size:0.85rem;line-height:1.7}
    .warn-box{background:#fffbeb;border:1px solid var(--warning);border-radius:8px;
              padding:12px 16px;margin-bottom:16px;font-size:0.85rem}
    .error-box{background:#fef2f2;border:1px solid var(--danger);border-radius:8px;
               padding:12px 16px;margin-bottom:16px;font-size:0.85rem}
    footer{margin-top:40px;text-align:center;color:var(--text2);font-size:0.8rem}
    """

    # Status banner
    cov = report_data.get("coverage_status", "UNKNOWN")
    cov_color = {"PASS": "var(--success)", "WARNING": "var(--warning)", "FAIL": "var(--danger)"}.get(cov, "var(--accent)")
    c014 = report_data["class_014"]
    status_html = f"""<div style="background:{cov_color}15;border:2px solid {cov_color};border-radius:12px;padding:20px;margin-bottom:24px">
    <h2 style="border:none;margin-top:0;color:{cov_color}">Regression Status: {html_mod.escape(cov)}</h2>
    <p>class_014 ({html_mod.escape(c014['name'])}): {'SUPPORTED' if c014['supported'] else 'UNSUPPORTED'} | Train: {c014['train_samples']}</p>
    <p>Negative (qita): {report_data['negative'].get('train_samples', 'N/A')} train samples</p>
    </div>"""

    # Metric cards
    cl = report_data.get("classification", {}) or {}
    ood = report_data.get("ood", {}) or {}
    ret = report_data.get("retrieval", {}) or {}
    emb = report_data.get("embedding", {}) or {}

    cards_html = ""
    card_items = [
        ("Overall Acc", cl.get("overall_accuracy")),
        ("Macro F1", cl.get("macro_f1")),
        ("Known Acc", cl.get("known_class_accuracy")),
        ("AUROC (OOD)", ood.get("auroc")),
        ("KA Rate", ood.get("known_accept_rate")),
        ("Neg Reject", ood.get("negative_reject_rate")),
        ("Retrieval Top1", ret.get("top1_accuracy_on_accepted")),
        ("Embed NN Acc", emb.get("nn_accuracy")),
        ("Threshold", report_data.get("threshold", {}).get("value")),
    ]
    for label, val in card_items:
        v_str = _fmt(val, 3) if val is not None else "N/A"
        cards_html += f'<div class="card"><div class="card-value">{v_str}</div><div class="card-label">{html_mod.escape(label)}</div></div>\n'

    # Data distribution table
    data_dist = report_data.get("data_distribution", {})
    dist_rows = ""
    if data_dist:
        for class_dir, info in sorted(data_dist.items()):
            cn = class_dir
            dist_rows += f'<tr><td>{html_mod.escape(cn)}</td><td>{info.get("train",0)}</td><td>{info.get("val",0)}</td><td>{info.get("test",0)}</td><td>{info.get("total",0)}</td></tr>\n'

    dist_table = f"""<div class="table-wrap"><table>
    <tr><th>Class</th><th>Train</th><th>Val</th><th>Test</th><th>Total</th></tr>
    {dist_rows}</table></div>""" if dist_rows else ""

    # Class weights table
    cw = report_data.get("class_weights", {})
    cw_rows = ""
    if cw.get("per_class"):
        for label_str, winfo in sorted(cw["per_class"].items(), key=lambda x: int(x[0])):
            cw_rows += f'<tr><td>{html_mod.escape(winfo["name"])}</td><td>{winfo["train_count"]}</td><td>{winfo["weight"]}</td></tr>\n'

    cw_table = ""
    if cw_rows:
        imbalance = cw.get("imbalance_ratio", "N/A")
        largest = cw.get("largest_class_ratio", "N/A")
        cw_table = f"""<div class="info-box">
        <strong>Class weight enabled:</strong> {cw.get('enabled', False)} |
        <strong>Max weight:</strong> {cw.get('max_weight', 'N/A')} |
        <strong>Imbalance ratio:</strong> {imbalance} |
        <strong>Largest class ratio:</strong> {largest}
        </div>
        <div class="table-wrap"><table>
        <tr><th>Class</th><th>Train Count</th><th>Weight</th></tr>
        {cw_rows}</table></div>"""

    # Per-class classification metrics
    pc_rows = ""
    if cl.get("per_class"):
        for label_str in sorted(cl["per_class"].keys(), key=lambda x: int(x)):
            entry = cl["per_class"][label_str]
            name = report_data.get("class_names", {}).get(label_str, f"class_{int(label_str):03d}")
            support = entry.get("support", 0)
            prec = _fmt(entry.get("precision"), 3)
            rec = _fmt(entry.get("recall"), 3)
            f1 = _fmt(entry.get("f1"), 3)
            is_neg = int(label_str) == report_data.get("config", {}).get("negative_label", 19)
            marker = ' style="color:var(--warning)"' if is_neg else ""
            pc_rows += f'<tr{marker}><td>{html_mod.escape(name)}</td><td>{support}</td><td>{prec}</td><td>{rec}</td><td>{f1}</td></tr>\n'

    pc_table = ""
    if pc_rows:
        pc_table = f"""<div class="table-wrap"><table>
        <tr><th>Class</th><th>Support</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
        {pc_rows}</table></div>"""

    # OOD per-class
    ood_pc_rows = ""
    if ood_per_class:
        for row in ood_per_class:
            ood_pc_rows += f'<tr><td>{html_mod.escape(row.get("class_name",""))}</td><td>{row.get("support","")}</td><td>{_fmt(float(row.get("accept_rate",0)),3)}</td><td>{_fmt(float(row.get("accuracy_after_accept",0)),3)}</td><td>{_fmt(float(row.get("avg_similarity",0)))}</td></tr>\n'

    ood_pc_table = ""
    if ood_pc_rows:
        ood_pc_table = f"""<div class="table-wrap"><table>
        <tr><th>Class</th><th>Support</th><th>Accept Rate</th><th>Acc (accepted)</th><th>Avg Sim</th></tr>
        {ood_pc_rows}</table></div>"""

    # class_014 specific
    c014_html = f"""<div class="success-box">
    <strong>class_014 ({html_mod.escape(c014['name'])})</strong><br>
    Supported: {'YES' if c014['supported'] else 'NO'} |
    Train samples: {c014['train_samples']} |
    Has prototype: {c014['has_prototype']} |
    Has gallery: {c014['has_gallery']}
    </div>"""

    # Negative specific
    neg = report_data.get("negative", {})
    neg_html = f"""<div class="info-box">
    <strong>Negative ({html_mod.escape(neg.get('name','qita'))})</strong><br>
    Train samples: {neg.get('train_samples', 'N/A')} |
    Precision: {_fmt(neg.get('precision'),3)} |
    Recall: {_fmt(neg.get('recall'),3)} |
    F1: {_fmt(neg.get('f1'),3)}
    </div>"""

    # Top confusing pairs
    confusing_html = ""
    for pair in report_data.get("top_confusing_pairs", [])[:5]:
        confusing_html += f'<div class="warn-box">{html_mod.escape(pair.get("name_i",""))} <-> {html_mod.escape(pair.get("name_j",""))}: sim={_fmt(pair.get("similarity"),4)}</div>\n'

    # Threshold recommendation
    thresh_html = f"""<div class="info-box">
    <strong>Recommended threshold:</strong> {_fmt(report_data.get('threshold',{}).get('value'),2)} |
    <strong>Strategy:</strong> {html_mod.escape(str(report_data.get('threshold',{}).get('strategy','N/A')))}
    </div>"""

    # Can replace old version?
    replace_verdict = "UNKNOWN"
    replace_reasons = []
    if cl.get("macro_f1") and cl["macro_f1"] > 0.5:
        replace_reasons.append(f"Macro F1 = {_fmt(cl['macro_f1'],3)}")
    if cov == "PASS":
        replace_reasons.append("Class coverage PASS")
    elif cov == "WARNING":
        replace_reasons.append("Class coverage WARNING (minor issues)")
    else:
        replace_reasons.append("Class coverage FAIL")
    if c014["supported"]:
        replace_reasons.append("class_014 now supported")
    if neg.get("train_samples", 0) and neg["train_samples"] >= 100:
        replace_reasons.append(f"Negative samples adequate ({neg['train_samples']})")

    if cov in ("PASS", "WARNING") and cl.get("macro_f1", 0) > 0.5 and c014["supported"]:
        replace_verdict = "YES"
    elif cov == "FAIL":
        replace_verdict = "NO"
    else:
        replace_verdict = "MAYBE"

    replace_color = {"YES": "var(--success)", "NO": "var(--danger)", "MAYBE": "var(--warning)"}.get(replace_verdict, "var(--accent)")
    replace_html = f"""<div style="background:{replace_color}15;border:2px solid {replace_color};border-radius:10px;padding:20px;margin-bottom:24px">
    <h3 style="color:{replace_color};margin-bottom:8px">Can replace old version? {replace_verdict}</h3>
    <ul style="padding-left:20px">{''.join(f'<li>{html_mod.escape(r)}</li>' for r in replace_reasons)}</ul>
    </div>"""

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
    <meta name="viewport" content="width=device-width,initial-scale=1.0">
    <title>Final Regression Report - {html_mod.escape(report_data.get('run_name',''))}</title>
    <style>{CSS}</style></head><body><div class="container">
    <h1>Final Regression Report</h1>
    <p class="subtitle">Run: {html_mod.escape(report_data.get('run_name',''))} | Generated at {html_mod.escape(ts)}</p>

    {status_html}

    <h2>Key Metrics</h2>
    <div class="cards">{cards_html}</div>

    {thresh_html}

    <h2>Data Distribution</h2>
    {dist_table}

    <h2>Class Weights &amp; Imbalance</h2>
    {cw_table}

    <h2>Classification (Per-Class)</h2>
    {pc_table}

    <h2>OOD Metrics (Per-Class)</h2>
    {ood_pc_table}

    <h2>class_014 (shenggaozuofalan)</h2>
    {c014_html}

    <h2>Negative (qita)</h2>
    {neg_html}

    <h2>Most Similar Class Pairs</h2>
    {confusing_html if confusing_html else '<div class="info-box">No confusing pairs data</div>'}

    <h2>Replacement Verdict</h2>
    {replace_html}

    <footer>Final Regression Report &mdash; pointcloud_metric_learning &mdash; {html_mod.escape(ts)}</footer>
    </div></body></html>"""


def _build_comparison_html(current, previous):
    import html as html_mod
    from datetime import datetime

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cur_cl = current.get("classification") or {}
    prev_cl = previous.get("classification") or {}
    cur_ood = current.get("ood") or {}
    prev_ood = previous.get("ood") or {}
    cur_cfg = current.get("config") or {}
    prev_cfg = previous.get("config") or {}

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
    .subtitle{color:var(--text2);margin-bottom:24px;font-size:0.95rem}
    .table-wrap{overflow-x:auto;margin-bottom:24px;background:var(--card-bg);
                border-radius:8px;border:1px solid var(--border)}
    table{width:100%;border-collapse:collapse;font-size:0.85rem}
    th{background:#f8f9fb;padding:10px 12px;text-align:left;font-weight:600;
       border-bottom:2px solid var(--border)}
    td{padding:8px 12px;border-bottom:1px solid var(--border)}
    tr:last-child td{border-bottom:none}tr:hover td{background:#f8f9fb}
    .better{color:var(--success);font-weight:600}.worse{color:var(--danger);font-weight:600}
    .info-box{background:#eff6ff;border:1px solid var(--accent);border-radius:8px;
              padding:12px 16px;margin-bottom:16px;font-size:0.85rem;line-height:1.7}
    footer{margin-top:40px;text-align:center;color:var(--text2);font-size:0.8rem}
    """

    def _delta(cur, prev):
        if cur is None or prev is None:
            return "N/A", ""
        d = cur - prev
        cls = "better" if d > 0.001 else ("worse" if d < -0.001 else "")
        return f"{d:+.4f}", cls

    rows = ""

    # Config comparison
    cur_unsup = cur_cfg.get("unsupported_known_labels", [])
    prev_unsup = prev_cfg.get("unsupported_known_labels", [])
    rows += f'<tr><td>Supported known classes</td><td>{len(cur_cfg.get("supported_known_labels", list(range(19))))}</td><td>{len(prev_cfg.get("supported_known_labels", list(range(19))))}</td><td></td></tr>\n'
    rows += f'<tr><td>Unsupported known classes</td><td>{cur_unsup}</td><td>{prev_unsup}</td><td></td></tr>\n'
    rows += f'<tr><td>Negative train samples</td><td>{current.get("negative",{}).get("train_samples","N/A")}</td><td>{previous.get("negative",{}).get("train_samples","N/A")}</td><td></td></tr>\n'

    # Metrics comparison
    comparisons = [
        ("Overall Accuracy", cur_cl.get("overall_accuracy"), prev_cl.get("overall_accuracy")),
        ("Macro F1", cur_cl.get("macro_f1"), prev_cl.get("macro_f1")),
        ("Known Class Accuracy", cur_cl.get("known_class_accuracy"), prev_cl.get("known_class_accuracy")),
        ("OOD AUROC", cur_ood.get("auroc"), prev_ood.get("auroc")),
        ("Known Accept Rate", cur_ood.get("known_accept_rate"), prev_ood.get("known_accept_rate")),
        ("Negative Reject Rate", cur_ood.get("negative_reject_rate"), prev_ood.get("negative_reject_rate")),
        ("Threshold", current.get("threshold",{}).get("value"), previous.get("threshold",{}).get("value")),
    ]

    for label, cur_v, prev_v in comparisons:
        cur_s = _fmt(cur_v, 3) if cur_v is not None else "N/A"
        prev_s = _fmt(prev_v, 3) if prev_v is not None else "N/A"
        delta_s, delta_cls = _delta(cur_v, prev_v) if cur_v is not None and prev_v is not None else ("N/A", "")
        rows += f'<tr><td>{html_mod.escape(label)}</td><td>{cur_s}</td><td>{prev_s}</td><td class="{delta_cls}">{delta_s}</td></tr>\n'

    table_html = f"""<div class="table-wrap"><table>
    <tr><th>Metric</th><th>New ({html_mod.escape(current.get('run_name',''))})</th><th>Old ({html_mod.escape(previous.get('run_name',''))})</th><th>Delta</th></tr>
    {rows}</table></div>"""

    # Key differences
    diff_items = []
    if cur_unsup != prev_unsup:
        diff_items.append(f"class_014 status changed: {'now SUPPORTED' if not cur_unsup else 'still UNSUPPORTED'}")
    cur_neg = current.get("negative", {}).get("train_samples", 0)
    prev_neg = previous.get("negative", {}).get("train_samples", 0)
    if cur_neg != prev_neg:
        diff_items.append(f"Negative samples: {prev_neg} -> {cur_neg}")

    diff_html = ""
    for item in diff_items:
        diff_html += f'<div class="info-box">{html_mod.escape(item)}</div>\n'

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
    <meta name="viewport" content="width=device-width,initial-scale=1.0">
    <title>Comparison Report</title>
    <style>{CSS}</style></head><body><div class="container">
    <h1>Comparison: New vs Previous Baseline</h1>
    <p class="subtitle">Generated at {html_mod.escape(ts)}</p>

    <h2>Key Differences</h2>
    {diff_html if diff_html else '<div class="info-box">No structural differences detected</div>'}

    <h2>Metric Comparison</h2>
    {table_html}

    <footer>Comparison Report &mdash; pointcloud_metric_learning &mdash; {html_mod.escape(ts)}</footer>
    </div></body></html>"""


if __name__ == "__main__":
    main()
