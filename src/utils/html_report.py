"""HTML report builder for embedding evaluation.

Generates a self-contained HTML file with inline CSS and base64-embedded
images. No external network resources required.
"""

import base64
import html
import math
import os
from datetime import datetime


def _img_to_base64(path):
    """Read an image file and return a base64 data URI string."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = f.read()
        ext = os.path.splitext(path)[1].lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext.lstrip("."), "image/png")
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    except Exception:
        return None


def _fmt(val, decimals=4):
    """Format a float, handling NaN."""
    if isinstance(val, float) and math.isnan(val):
        return "N/A"
    if isinstance(val, (int, float)):
        return f"{val:.{decimals}f}"
    return str(val)


def _color_for_value(val, low=0.0, high=1.0, invert=False):
    """Return a green-red CSS color based on value position in [low, high]."""
    if isinstance(val, float) and math.isnan(val):
        return "#888"
    t = (val - low) / max(high - low, 1e-9)
    t = max(0.0, min(1.0, t))
    if invert:
        t = 1.0 - t
    # green = good, red = bad
    r = int(220 * t + 40 * (1 - t))
    g = int(60 * t + 200 * (1 - t))
    b = 60
    # Lighten background
    r = min(255, r + 60)
    g = min(255, g + 60)
    b = min(255, b + 60)
    return f"rgb({r},{g},{b})"


_CSS = """
:root {
    --bg: #f5f7fa;
    --card-bg: #ffffff;
    --text: #1a1a2e;
    --text-secondary: #555;
    --border: #e0e4e8;
    --accent: #3b82f6;
    --accent-light: #dbeafe;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 24px;
}
.container { max-width: 1200px; margin: 0 auto; }
h1 {
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 8px;
    color: var(--text);
}
h2 {
    font-size: 1.3rem;
    font-weight: 600;
    margin-top: 32px;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--accent);
    color: var(--text);
}
.subtitle { color: var(--text-secondary); margin-bottom: 24px; font-size: 0.95rem; }

/* Info bar */
.info-bar {
    display: flex; flex-wrap: wrap; gap: 12px;
    margin-bottom: 24px; padding: 16px;
    background: var(--card-bg); border-radius: 8px;
    border: 1px solid var(--border);
}
.info-item {
    padding: 4px 12px; background: var(--accent-light);
    border-radius: 4px; font-size: 0.85rem; color: var(--text);
}
.info-item span { font-weight: 600; }

/* Metric cards */
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }
.card {
    background: var(--card-bg); border-radius: 10px;
    padding: 20px 16px; text-align: center;
    border: 1px solid var(--border);
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s;
}
.card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
.card-value {
    font-size: 1.6rem; font-weight: 700; color: var(--accent);
    margin-bottom: 4px;
}
.card-value.good { color: var(--success); }
.card-value.warn { color: var(--warning); }
.card-value.bad { color: var(--danger); }
.card-label { font-size: 0.78rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; }

/* Tables */
.table-wrap {
    overflow-x: auto; margin-bottom: 24px;
    background: var(--card-bg); border-radius: 8px;
    border: 1px solid var(--border);
}
table {
    width: 100%; border-collapse: collapse; font-size: 0.85rem;
}
th {
    background: #f8f9fb; padding: 10px 12px; text-align: left;
    font-weight: 600; border-bottom: 2px solid var(--border);
    white-space: nowrap;
}
td {
    padding: 8px 12px; border-bottom: 1px solid var(--border);
    white-space: nowrap;
}
tr:last-child td { border-bottom: none; }
tr:hover td { background: #f8f9fb; }

/* Images */
.img-section {
    margin-bottom: 24px; background: var(--card-bg);
    border-radius: 8px; border: 1px solid var(--border);
    padding: 16px; text-align: center;
}
.img-section img {
    max-width: 100%; height: auto; border-radius: 4px;
}

/* Warning banner */
.warning-banner {
    background: #fef3c7; border: 1px solid #f59e0b; border-radius: 6px;
    padding: 12px 16px; margin-bottom: 16px;
    color: #92400e; font-size: 0.9rem;
}

/* Explanation section */
.explanation {
    background: var(--card-bg); border-radius: 8px;
    border: 1px solid var(--border); padding: 20px;
    font-size: 0.9rem;
}
.explanation dt { font-weight: 600; margin-top: 12px; color: var(--text); }
.explanation dd { margin-left: 16px; color: var(--text-secondary); margin-bottom: 4px; }

/* Confusing pairs highlight */
.pair-high { color: var(--danger); font-weight: 600; }
.pair-mid { color: var(--warning); font-weight: 600; }

/* Two-column layout */
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }
@media (max-width: 800px) { .two-col { grid-template-columns: 1fr; } }

footer { margin-top: 40px; text-align: center; color: var(--text-secondary); font-size: 0.8rem; }
"""


def build_report(
    input_path,
    embeddings_shape,
    num_classes,
    do_normalize,
    exclude_negative,
    negative_label,
    ks,
    intra,
    inter,
    gap,
    retrieval,
    nn,
    top_confusing,
    per_class_rows,
    class_names,
    unique_labels,
    counts,
    sim_png_path,
    cm_png_path,
    tsne_png_path,
):
    """Build the full HTML report string."""

    N, D = embeddings_shape
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- Metric cards ---
    def _card_class(val, higher_better=True, threshold_good=0.7, threshold_bad=0.3):
        if isinstance(val, float) and math.isnan(val):
            return ""
        if higher_better:
            if val >= threshold_good:
                return "good"
            if val >= threshold_bad:
                return "warn"
            return "bad"
        else:
            if val <= threshold_bad:
                return "good"
            if val <= threshold_good:
                return "warn"
            return "bad"

    cards_html = ""
    card_data = [
        ("Intra-class Sim", _fmt(intra["macro_avg"]), _card_class(intra["macro_avg"], True, 0.5, 0.2)),
        ("Inter-class Sim", _fmt(inter["global_avg"]), _card_class(inter["global_avg"], False, 0.3, 0.6)),
        ("Similarity Gap", _fmt(gap), _card_class(gap, True, 0.2, 0.05)),
    ]
    for k in ks:
        v = retrieval["recall_at_k"][k]
        card_data.append((f"Recall@{k}", _fmt(v), _card_class(v, True, 0.7, 0.3)))
    for k in ks:
        v = retrieval["precision_at_k"][k]
        card_data.append((f"Precision@{k}", _fmt(v), _card_class(v, True, 0.7, 0.3)))
    card_data.append(("1-NN Accuracy", _fmt(nn["nn_accuracy"]), _card_class(nn["nn_accuracy"], True, 0.7, 0.3)))

    for label, value, cls in card_data:
        cards_html += f'<div class="card"><div class="card-value {cls}">{value}</div><div class="card-label">{html.escape(label)}</div></div>\n'

    # --- Info bar ---
    info_items = [
        f"Input: <span>{html.escape(os.path.basename(input_path))}</span>",
        f"Samples: <span>{N}</span>",
        f"Dim: <span>{D}</span>",
        f"Classes: <span>{num_classes}</span>",
        f"Normalized: <span>{'Yes' if do_normalize else 'No'}</span>",
        f"Exclude negative: <span>{'Yes' if exclude_negative else 'No'}</span>",
    ]
    if exclude_negative:
        info_items.append(f"Negative label: <span>{negative_label}</span>")
    info_bar_html = "\n".join(f'<div class="info-item">{item}</div>' for item in info_items)

    # --- Per-class table ---
    pc_header = "<th>Class</th><th>#Samples</th><th>Intra Sim</th>"
    for k in ks:
        pc_header += f"<th>Recall@{k}</th>"
    for k in ks:
        pc_header += f"<th>Precision@{k}</th>"
    pc_header += "<th>NN Acc</th>"

    pc_rows = ""
    for row in per_class_rows:
        bg_style = ""
        pc_rows += f'<tr{bg_style}>'
        pc_rows += f'<td>{html.escape(str(row["class_name"]))}</td>'
        pc_rows += f'<td>{row["num_samples"]}</td>'
        pc_rows += f'<td>{_fmt(row["intra_class_similarity"])}</td>'
        for k in ks:
            pc_rows += f'<td>{_fmt(row[f"recall@{k}"])}</td>'
        for k in ks:
            pc_rows += f'<td>{_fmt(row[f"precision@{k}"])}</td>'
        pc_rows += f'<td>{_fmt(row["nn_accuracy"])}</td>'
        pc_rows += '</tr>\n'

    per_class_table = f"""<div class="table-wrap">
<table><tr>{pc_header}</tr>
{pc_rows}</table></div>"""

    # --- Top confusing pairs ---
    confusing_header = "<th>#</th><th>Class A</th><th>Class B</th><th>Similarity</th>"
    confusing_rows = ""
    for i, pair in enumerate(top_confusing):
        sim = pair["similarity"]
        cls = "pair-high" if sim > 0.6 else ("pair-mid" if sim > 0.4 else "")
        confusing_rows += f'<tr><td>{i + 1}</td><td>{html.escape(pair["name_i"])}</td><td>{html.escape(pair["name_j"])}</td><td class="{cls}">{_fmt(sim)}</td></tr>\n'

    if top_confusing:
        confusing_table = f"""<div class="table-wrap">
<table><tr>{confusing_header}</tr>
{confusing_rows}</table></div>"""
    else:
        confusing_table = '<div class="warning-banner">No confusing pairs found.</div>'

    # --- Images ---
    def _img_section(title, png_path):
        uri = _img_to_base64(png_path)
        if uri:
            return f"""<div class="img-section"><h3 style="margin-bottom:12px;font-size:1.05rem;">{html.escape(title)}</h3><img src="{uri}" alt="{html.escape(title)}"></div>"""
        return f"""<div class="img-section"><h3 style="margin-bottom:12px;font-size:1.05rem;">{html.escape(title)}</h3><div class="warning-banner">Image generation failed or file not found: {html.escape(os.path.basename(png_path))}</div></div>"""

    sim_img = _img_section("Inter-class Cosine Similarity Matrix", sim_png_path)
    cm_img = _img_section("1-NN Confusion Matrix", cm_png_path)
    tsne_img = _img_section("t-SNE Embedding Visualization", tsne_png_path)

    # --- Singleton warnings ---
    singleton_warnings = ""
    for lbl, cnt in zip(unique_labels, counts):
        if cnt < 2:
            name = class_names[lbl] if lbl < len(class_names) else str(lbl)
            singleton_warnings += f'<div class="warning-banner">Class {html.escape(str(lbl))} ({html.escape(str(name))}) has only {cnt} sample(s) — intra-class similarity and retrieval metrics may be unreliable.</div>\n'

    # --- Explanations ---
    explanations = """
<dl class="explanation">
  <dt>Intra-class Similarity</dt>
  <dd>Average cosine similarity between all pairs of samples within the same class.
      Higher values indicate tighter clusters. Macro average is the mean of per-class averages.</dd>

  <dt>Inter-class Similarity</dt>
  <dd>Average cosine similarity between samples of different classes.
      Lower values indicate better separation between classes.</dd>

  <dt>Similarity Gap</dt>
  <dd><code>intra_class_similarity - inter_class_similarity</code>.
      A larger gap means embeddings have stronger discriminative power.
      Gap &gt; 0.3 is generally considered good.</dd>

  <dt>Recall@K</dt>
  <dd>For each sample (query), retrieve the K nearest neighbors by cosine similarity.
      Recall@K = 1 if at least one neighbor shares the query's class.
      Reports the fraction of queries with successful retrieval.
      Note: samples whose class has only 1 member are excluded from this calculation.</dd>

  <dt>Precision@K</dt>
  <dd>For each query, the fraction of top-K neighbors that share the query's class.
      Averaged across all valid queries.</dd>

  <dt>1-NN Accuracy</dt>
  <dd>Uses the single nearest neighbor's class as the prediction.
      A simple but effective measure of embedding quality.
      High NN accuracy implies the embedding space is well-structured for classification.</dd>

  <dt>Top Confusing Pairs</dt>
  <dd>Class pairs with the highest average inter-class cosine similarity.
      These classes are most easily confused and may benefit from more training data or specialized augmentation.</dd>

  <dt>Negative Class</dt>
  <dd>When <code>--exclude-negative</code> is enabled, the negative class (e.g. "other") is excluded from
      metric computation because its samples are inherently diverse and would distort intra-class similarity.</dd>
</dl>
"""

    # --- Assemble ---
    report = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Embedding Evaluation Report</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
<h1>Embedding Evaluation Report</h1>
<p class="subtitle">Generated at {html.escape(timestamp)}</p>

<h2>Basic Information</h2>
<div class="info-bar">{info_bar_html}</div>

{singleton_warnings}

<h2>Core Metrics</h2>
<div class="cards">{cards_html}</div>

<h2>Per-class Metrics</h2>
{per_class_table}

<h2>Top Confusing Class Pairs</h2>
{confusing_table}

<h2>Visualizations</h2>
<div class="two-col">
{sim_img}
{cm_img}
</div>
{tsne_img}

<h2>Metrics Explanation</h2>
{explanations}

<footer>Embedding Evaluation Report &mdash; pointcloud_metric_learning</footer>
</div>
</body>
</html>"""

    return report
