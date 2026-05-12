"""Check class coverage across config, prototypes, gallery, and dataset splits.

Verifies that all known classes have prototypes, gallery entries, and dataset
samples. Reports missing classes, label holes, and inconsistencies.

Usage:
    python scripts/check_class_coverage.py \
        --dataset-root dataset \
        --config configs/config.yaml \
        --prototypes outputs/prototypes/baseline_prototypes.pt \
        --gallery outputs/gallery/baseline_train_gallery.pt
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config


def count_samples_per_class(dataset_root, split, negative_label=19):
    """Count .npy files in each class_* directory for a given split."""
    split_dir = os.path.join(dataset_root, split)
    counts = {}
    if not os.path.isdir(split_dir):
        return counts
    for entry in sorted(os.listdir(split_dir)):
        cls_dir = os.path.join(split_dir, entry)
        if not os.path.isdir(cls_dir):
            continue
        if entry == "negative":
            label = negative_label
        elif entry.startswith("class_"):
            try:
                label = int(entry.split("_")[1])
            except (ValueError, IndexError):
                continue
        else:
            continue
        npy_files = [f for f in os.listdir(cls_dir) if f.endswith(".npy")]
        # If class_019 and negative both exist, merge counts
        if label in counts:
            counts[label] += len(npy_files)
        else:
            counts[label] = len(npy_files)
    return counts


def main():
    parser = argparse.ArgumentParser(description="Check class coverage across all artifacts")
    parser.add_argument("--dataset-root", type=str, default="dataset")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--prototypes", type=str, default="outputs/prototypes/baseline_prototypes.pt")
    parser.add_argument("--gallery", type=str, default="outputs/gallery/baseline_train_gallery.pt")
    args = parser.parse_args()

    import torch

    cfg = load_config(args.config)
    num_known_classes = cfg["num_known_classes"]
    negative_label = cfg["negative_label"]

    known_labels = list(range(num_known_classes))

    # Load class names
    class_names_path = os.path.join(os.path.dirname(args.config), "class_names.json")
    class_names = {}
    if os.path.exists(class_names_path):
        with open(class_names_path) as f:
            class_names = json.load(f)

    report = {
        "config": {
            "num_known_classes": num_known_classes,
            "negative_label": negative_label,
            "class_names_count": len(class_names),
        },
        "checks": [],
        "errors": [],
        "warnings": [],
        "status": "PASS",
    }

    # --- Check 1: class_names.json length ---
    if len(class_names) != num_known_classes + 1:  # known + negative
        report["checks"].append({
            "item": "class_names.json length",
            "expected": num_known_classes + 1,
            "actual": len(class_names),
            "status": "ERROR",
        })
        report["errors"].append(
            f"class_names.json has {len(class_names)} entries, expected {num_known_classes + 1} "
            f"({num_known_classes} known + 1 negative)"
        )
    else:
        report["checks"].append({
            "item": "class_names.json length",
            "expected": num_known_classes + 1,
            "actual": len(class_names),
            "status": "OK",
        })

    # --- Check 2: class_names.json keys match expected labels ---
    expected_keys = {str(i) for i in range(num_known_classes + 1)}
    actual_keys = set(class_names.keys())
    missing_keys = expected_keys - actual_keys
    extra_keys = actual_keys - expected_keys
    if missing_keys:
        report["errors"].append(f"class_names.json missing keys: {sorted(missing_keys)}")
    if extra_keys:
        report["warnings"].append(f"class_names.json has extra keys: {sorted(extra_keys)}")
    report["checks"].append({
        "item": "class_names.json keys",
        "status": "OK" if not missing_keys else "ERROR",
        "missing": sorted(missing_keys),
        "extra": sorted(extra_keys),
    })

    # --- Check 3: Dataset split coverage ---
    split_coverage = {}
    for split in ["train", "val", "test"]:
        counts = count_samples_per_class(args.dataset_root, split, negative_label)
        split_coverage[split] = counts

        for label in known_labels:
            cnt = counts.get(label, 0)
            if cnt == 0:
                msg = (f"Class {label} ({class_names.get(str(label), f'class_{label:03d}')}) "
                       f"has 0 samples in {split} split")
                if split == "train":
                    report["errors"].append(msg)
                    report["checks"].append({
                        "item": f"{split}/class_{label:03d}",
                        "status": "ERROR",
                        "count": 0,
                        "message": msg,
                    })
                else:
                    report["warnings"].append(msg)
                    report["checks"].append({
                        "item": f"{split}/class_{label:03d}",
                        "status": "WARNING",
                        "count": 0,
                        "message": msg,
                    })
            else:
                report["checks"].append({
                    "item": f"{split}/class_{label:03d}",
                    "status": "OK",
                    "count": cnt,
                })

    report["split_coverage"] = {
        split: {str(k): v for k, v in counts.items()}
        for split, counts in split_coverage.items()
    }

    # --- Check 4: Negative class in dataset ---
    for split in ["train", "val", "test"]:
        neg_cnt = split_coverage[split].get(negative_label, 0)
        report["checks"].append({
            "item": f"{split}/negative (label {negative_label})",
            "status": "OK" if neg_cnt > 0 else "WARNING",
            "count": neg_cnt,
        })
        if neg_cnt == 0:
            report["warnings"].append(f"Negative class has 0 samples in {split} split")

    # --- Check 5: Prototypes coverage ---
    if os.path.exists(args.prototypes):
        proto_data = torch.load(args.prototypes, map_location="cpu", weights_only=False)
        proto_labels_in_file = list(range(proto_data["prototypes"].shape[0]))
        proto_class_names = proto_data.get("class_names", [])
        proto_support = proto_data.get("class_support", {})

        missing_proto = []
        for label in known_labels:
            if label not in proto_labels_in_file:
                missing_proto.append(label)

        if missing_proto:
            report["errors"].append(
                f"Prototypes missing for classes: "
                f"{[class_names.get(str(l), f'class_{l:03d}') for l in missing_proto]}"
            )
            report["checks"].append({
                "item": "prototypes coverage",
                "status": "ERROR",
                "missing": missing_proto,
            })
        else:
            report["checks"].append({
                "item": "prototypes coverage",
                "status": "OK",
                "num_prototypes": len(proto_labels_in_file),
            })

        # Check prototype classes with 0 support
        for label in known_labels:
            support = proto_support.get(str(label), proto_support.get(label, 0))
            if isinstance(support, torch.Tensor):
                support = support.item()
            if support == 0:
                cn = class_names.get(str(label), f"class_{label:03d}")
                report["errors"].append(
                    f"Prototype for {cn} (label {label}) was built with 0 train samples"
                )
    else:
        report["warnings"].append(f"Prototypes file not found: {args.prototypes}")
        report["checks"].append({
            "item": "prototypes file",
            "status": "WARNING",
            "message": f"File not found: {args.prototypes}",
        })

    # --- Check 6: Gallery coverage ---
    if os.path.exists(args.gallery):
        gallery_data = torch.load(args.gallery, map_location="cpu", weights_only=False)
        gallery_labels = gallery_data["labels"].tolist()
        gallery_class_set = set(gallery_labels)

        missing_gallery = []
        for label in known_labels:
            if label not in gallery_class_set:
                missing_gallery.append(label)

        if missing_gallery:
            report["errors"].append(
                f"Gallery missing for classes: "
                f"{[class_names.get(str(l), f'class_{l:03d}') for l in missing_gallery]}"
            )
            report["checks"].append({
                "item": "gallery coverage",
                "status": "ERROR",
                "missing": missing_gallery,
            })
        else:
            report["checks"].append({
                "item": "gallery coverage",
                "status": "OK",
                "num_gallery_samples": len(gallery_labels),
                "num_classes": len(gallery_class_set),
            })
    else:
        report["warnings"].append(f"Gallery file not found: {args.gallery}")
        report["checks"].append({
            "item": "gallery file",
            "status": "WARNING",
            "message": f"File not found: {args.gallery}",
        })

    # --- Determine overall status ---
    if report["errors"]:
        report["status"] = "FAIL"
    elif report["warnings"]:
        report["status"] = "WARNING"
    else:
        report["status"] = "PASS"

    # --- Save JSON ---
    os.makedirs("outputs/reports", exist_ok=True)
    json_path = "outputs/reports/class_coverage_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"JSON report saved to {json_path}")

    # --- Build HTML ---
    html = _build_html(report, class_names, num_known_classes, negative_label)
    html_path = "outputs/reports/class_coverage_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report saved to {html_path}")

    # --- Print summary ---
    print(f"\n{'='*60}")
    print(f"Class Coverage Report: {report['status']}")
    print(f"{'='*60}")
    print(f"Known classes: {num_known_classes}")
    print(f"Negative label: {negative_label}")
    print(f"Errors: {len(report['errors'])}")
    print(f"Warnings: {len(report['warnings'])}")

    for err in report["errors"]:
        print(f"  [ERROR] {err}")
    for warn in report["warnings"]:
        print(f"  [WARN]  {warn}")

    # Print split table
    print(f"\n{'Class':<35} {'Train':>6} {'Val':>6} {'Test':>6}")
    print("-" * 55)
    for label in list(range(num_known_classes)) + [negative_label]:
        cn = class_names.get(str(label), f"class_{label:03d}")
        if label == negative_label:
            cn += " (neg)"
        tr = split_coverage["train"].get(label, 0)
        va = split_coverage["val"].get(label, 0)
        te = split_coverage["test"].get(label, 0)
        marker = " *** NO TRAIN ***" if tr == 0 and label < negative_label else ""
        print(f"  {cn:<33} {tr:>6} {va:>6} {te:>6}{marker}")


def _build_html(report, class_names, num_known_classes, negative_label):
    import html as html_mod
    from datetime import datetime

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    status_color = {"PASS": "#10b981", "WARNING": "#f59e0b", "FAIL": "#ef4444"}
    sc = status_color.get(report["status"], "#555")

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
    .badge{display:inline-block;padding:4px 12px;border-radius:20px;font-weight:700;
           font-size:0.9rem;color:#fff;margin-left:12px}
    .table-wrap{overflow-x:auto;margin-bottom:24px;background:var(--card-bg);
                border-radius:8px;border:1px solid var(--border)}
    table{width:100%;border-collapse:collapse;font-size:0.85rem}
    th{background:#f8f9fb;padding:10px 12px;text-align:left;font-weight:600;
       border-bottom:2px solid var(--border)}
    td{padding:8px 12px;border-bottom:1px solid var(--border)}
    tr:last-child td{border-bottom:none}
    tr:hover td{background:#f8f9fb}
    td.zero{color:var(--danger);font-weight:700}
    .error-box{background:#fef2f2;border:1px solid var(--danger);border-radius:8px;
               padding:12px 16px;margin-bottom:8px;font-size:0.85rem}
    .warn-box{background:#fffbeb;border:1px solid var(--warning);border-radius:8px;
              padding:12px 16px;margin-bottom:8px;font-size:0.85rem}
    footer{margin-top:40px;text-align:center;color:var(--text2);font-size:0.8rem}
    """

    # Status badge
    badge_html = f'<span class="badge" style="background:{sc}">{report["status"]}</span>'

    # Errors
    errors_html = ""
    for e in report["errors"]:
        errors_html += f'<div class="error-box">[ERROR] {html_mod.escape(e)}</div>\n'

    # Warnings
    warns_html = ""
    for w in report["warnings"]:
        warns_html += f'<div class="warn-box">[WARN] {html_mod.escape(w)}</div>\n'

    # Split table
    split_coverage = report.get("split_coverage", {})
    rows = ""
    for label in list(range(num_known_classes)) + [negative_label]:
        cn = class_names.get(str(label), f"class_{label:03d}")
        if label == negative_label:
            cn += " (negative)"
        tr = split_coverage.get("train", {}).get(str(label), 0)
        va = split_coverage.get("val", {}).get(str(label), 0)
        te = split_coverage.get("test", {}).get(str(label), 0)
        tr_cls = ' class="zero"' if tr == 0 and label < negative_label else ""
        va_cls = ' class="zero"' if va == 0 else ""
        te_cls = ' class="zero"' if te == 0 else ""
        marker = ""
        if label < negative_label and tr == 0:
            marker = ' <span style="color:var(--danger)">NO TRAIN</span>'
        rows += (f'<tr><td>{html_mod.escape(cn)}</td>'
                 f'<td{tr_cls}>{tr}</td><td{va_cls}>{va}</td><td{te_cls}>{te}</td>'
                 f'<td>{marker}</td></tr>\n')

    split_table = f"""<div class="table-wrap"><table>
    <tr><th>Class</th><th>Train</th><th>Val</th><th>Test</th><th>Note</th></tr>
    {rows}</table></div>"""

    # Checks table
    check_rows = ""
    for c in report["checks"]:
        status = c.get("status", "?")
        color = {"OK": "var(--success)", "ERROR": "var(--danger)", "WARNING": "var(--warning)"}.get(status, "var(--text2)")
        check_rows += (f'<tr><td>{html_mod.escape(c["item"])}</td>'
                       f'<td style="color:{color};font-weight:600">{status}</td>'
                       f'<td>{html_mod.escape(str(c.get("count", c.get("message", c.get("missing", c.get("num_prototypes", c.get("num_gallery_samples", "")))))))}</td></tr>\n')

    checks_table = f"""<div class="table-wrap"><table>
    <tr><th>Item</th><th>Status</th><th>Detail</th></tr>
    {check_rows}</table></div>"""

    html_out = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Class Coverage Report</title>
<style>{CSS}</style></head><body><div class="container">
<h1>Class Coverage Report {badge_html}</h1>
<p style="color:var(--text2);margin-bottom:24px">Generated at {html_mod.escape(ts)}</p>

<h2>Config Summary</h2>
<div class="table-wrap"><table>
<tr><th>Item</th><th>Value</th></tr>
<tr><td>num_known_classes</td><td>{num_known_classes}</td></tr>
<tr><td>negative_label</td><td>{negative_label}</td></tr>
<tr><td>class_names.json entries</td><td>{report["config"]["class_names_count"]}</td></tr>
</table></div>

{errors_html}{warns_html}

<h2>Dataset Split Coverage</h2>
{split_table}

<h2>Detailed Checks</h2>
{checks_table}

<footer>Class Coverage Report &mdash; pointcloud_metric_learning</footer>
</div></body></html>"""

    return html_out


if __name__ == "__main__":
    main()
