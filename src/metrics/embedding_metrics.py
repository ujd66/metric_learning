"""Embedding evaluation metrics for metric learning.

All functions operate on numpy arrays (embeddings and labels) and do not
depend on any model code.
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity


def validate_embeddings(embeddings, labels):
    """Check for NaN / Inf and basic shape consistency."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {labels.shape}")
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(
            f"embeddings and labels length mismatch: {embeddings.shape[0]} vs {labels.shape[0]}"
        )

    nan_mask = np.isnan(embeddings)
    inf_mask = np.isinf(embeddings)
    if nan_mask.any() or inf_mask.any():
        bad_rows = np.where(nan_mask.any(axis=1) | inf_mask.any(axis=1))[0]
        raise ValueError(
            f"Found NaN/Inf in {len(bad_rows)} embedding rows (indices: {bad_rows[:10].tolist()})"
        )

    return embeddings, labels


def l2_normalize(embeddings):
    """L2 normalize each row."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms


def compute_cosine_similarity_matrix(embeddings):
    """Return [N, N] cosine similarity matrix."""
    normed = l2_normalize(embeddings)
    return cosine_similarity(normed)


# ---------------------------------------------------------------------------
# 1. Intra-class similarity
# ---------------------------------------------------------------------------

def compute_intra_class_similarity(sim_matrix, labels):
    """Per-class and macro-average intra-class cosine similarity.

    Returns dict with keys:
        per_class: {label: float}
        macro_avg: float
        global_avg: float  (average over all positive pairs)
    """
    unique_labels = np.unique(labels)
    per_class = {}
    all_sims = []

    for lbl in unique_labels:
        indices = np.where(labels == lbl)[0]
        n = len(indices)
        if n < 2:
            warnings.warn(f"Class {lbl} has only {n} sample(s), skipping intra-class similarity.")
            per_class[lbl] = float("nan")
            continue
        sub = sim_matrix[np.ix_(indices, indices)]
        # Exclude self-pairs (diagonal)
        mask = ~np.eye(n, dtype=bool)
        sims = sub[mask]
        per_class[lbl] = float(np.mean(sims))
        all_sims.extend(sims.tolist())

    macro_avg = float(np.nanmean(list(per_class.values()))) if per_class else float("nan")
    global_avg = float(np.mean(all_sims)) if all_sims else float("nan")

    return {
        "per_class": per_class,
        "macro_avg": macro_avg,
        "global_avg": global_avg,
    }


# ---------------------------------------------------------------------------
# 2. Inter-class similarity
# ---------------------------------------------------------------------------

def compute_inter_class_similarity(sim_matrix, labels):
    """Inter-class cosine similarity analysis.

    Returns dict:
        class_similarity_matrix: ndarray [C, C]
        unique_labels: ndarray
        global_avg: float
    """
    unique_labels = np.unique(labels)
    C = len(unique_labels)
    class_sim = np.zeros((C, C), dtype=np.float64)
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

    inter_sims = []

    for i, li in enumerate(unique_labels):
        idx_i = np.where(labels == li)[0]
        for j, lj in enumerate(unique_labels):
            idx_j = np.where(labels == lj)[0]
            sub = sim_matrix[np.ix_(idx_i, idx_j)]
            if i == j:
                class_sim[i, j] = float("nan")
            else:
                mean_sim = float(np.mean(sub))
                class_sim[i, j] = mean_sim
                inter_sims.extend(sub.ravel().tolist())

    global_avg = float(np.mean(inter_sims)) if inter_sims else float("nan")

    return {
        "class_similarity_matrix": class_sim,
        "unique_labels": unique_labels,
        "global_avg": global_avg,
    }


def compute_top_confusing_pairs(class_sim_matrix, unique_labels, class_names=None, top_k=10):
    """Return top-K most confusing class pairs (highest inter-class similarity)."""
    C = len(unique_labels)
    pairs = []
    for i in range(C):
        for j in range(i + 1, C):
            sim = class_sim_matrix[i, j]
            if np.isnan(sim):
                continue
            name_i = class_names[i] if class_names is not None else str(unique_labels[i])
            name_j = class_names[j] if class_names is not None else str(unique_labels[j])
            pairs.append({
                "class_i": int(unique_labels[i]),
                "class_j": int(unique_labels[j]),
                "name_i": name_i,
                "name_j": name_j,
                "similarity": float(sim),
            })
    pairs.sort(key=lambda x: x["similarity"], reverse=True)
    return pairs[:top_k]


# ---------------------------------------------------------------------------
# 3. Similarity gap
# ---------------------------------------------------------------------------

def compute_similarity_gap(intra_macro, inter_global):
    """similarity_gap = intra_class_similarity - inter_class_similarity."""
    return float(intra_macro - inter_global)


# ---------------------------------------------------------------------------
# 4 & 5. Recall@K and Precision@K
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(sim_matrix, labels, ks=(1, 3, 5, 10)):
    """Compute Recall@K and Precision@K for each sample.

    For each sample (query), find top-K nearest neighbors (excluding self).
    Recall@K = 1 if at least one neighbor shares the query's label.
    Precision@K = (number of same-label neighbors) / K.

    Samples whose class has only 1 member are skipped (with warning).

    Returns dict:
        recall_at_k: {k: float}  (macro average over valid queries)
        precision_at_k: {k: float}
        per_class_recall_at_k: {label: {k: float}}
        per_class_precision_at_k: {label: {k: float}}
        skipped_singletons: list[int]  (labels skipped)
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_count = dict(zip(unique_labels.tolist(), counts.tolist()))

    singleton_labels = [lbl for lbl, cnt in label_count.items() if cnt < 2]
    if singleton_labels:
        warnings.warn(
            f"Classes with <2 samples (skipped for retrieval metrics): {singleton_labels}"
        )

    N = len(labels)
    # Fill diagonal with -inf to exclude self
    np.fill_diagonal(sim_matrix, -np.inf)

    sorted_indices = np.argsort(-sim_matrix, axis=1)

    # Track per-class
    per_class_hits = {lbl: {k: [] for k in ks} for lbl in unique_labels if label_count[lbl] >= 2}
    per_class_prec = {lbl: {k: [] for k in ks} for lbl in unique_labels if label_count[lbl] >= 2}

    all_recall = {k: [] for k in ks}
    all_precision = {k: [] for k in ks}

    for i in range(N):
        lbl_i = int(labels[i])
        if label_count[lbl_i] < 2:
            continue

        topk_indices = sorted_indices[i]

        for k in ks:
            topk = topk_indices[:k]
            same_label_mask = labels[topk] == lbl_i
            num_hits = int(np.sum(same_label_mask))

            recall = 1.0 if num_hits > 0 else 0.0
            precision = num_hits / k

            all_recall[k].append(recall)
            all_precision[k].append(precision)
            per_class_hits[lbl_i][k].append(recall)
            per_class_prec[lbl_i][k].append(precision)

    recall_at_k = {k: float(np.mean(v)) if v else float("nan") for k, v in all_recall.items()}
    precision_at_k = {k: float(np.mean(v)) if v else float("nan") for k, v in all_precision.items()}

    per_class_recall_at_k = {}
    per_class_precision_at_k = {}
    for lbl in per_class_hits:
        per_class_recall_at_k[lbl] = {
            k: float(np.mean(v)) if v else float("nan") for k, v in per_class_hits[lbl].items()
        }
        per_class_precision_at_k[lbl] = {
            k: float(np.mean(v)) if v else float("nan") for k, v in per_class_prec[lbl].items()
        }

    return {
        "recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k,
        "per_class_recall_at_k": per_class_recall_at_k,
        "per_class_precision_at_k": per_class_precision_at_k,
        "skipped_singletons": singleton_labels,
    }


# ---------------------------------------------------------------------------
# 6. Nearest-neighbor classification accuracy
# ---------------------------------------------------------------------------

def compute_nn_accuracy(sim_matrix, labels):
    """1-NN classification: predict label of nearest neighbor (excluding self).

    Returns dict:
        nn_accuracy: float
        predictions: ndarray
        per_class_nn_accuracy: {label: float}
    """
    N = len(labels)
    np.fill_diagonal(sim_matrix, -np.inf)

    nn_indices = np.argmax(sim_matrix, axis=1)
    predictions = labels[nn_indices]
    correct = predictions == labels

    nn_accuracy = float(np.mean(correct))

    unique_labels = np.unique(labels)
    per_class = {}
    for lbl in unique_labels:
        mask = labels == lbl
        if mask.sum() == 0:
            per_class[lbl] = float("nan")
        else:
            per_class[lbl] = float(np.mean(correct[mask]))

    return {
        "nn_accuracy": nn_accuracy,
        "predictions": predictions,
        "per_class_nn_accuracy": per_class,
    }


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def plot_class_similarity_matrix(class_sim, unique_labels, class_names, save_path):
    """Heatmap of inter-class cosine similarity."""
    C = len(unique_labels)
    fig, ax = plt.subplots(figsize=(max(8, C * 0.6), max(6, C * 0.55)))
    display = class_sim.copy()
    # Set diagonal to nan for display
    np.fill_diagonal(display, np.nan)
    im = ax.imshow(display, cmap="RdYlBu_r", aspect="auto")
    ax.set_xticks(range(C))
    ax.set_yticks(range(C))
    tick_labels = [class_names[i] if class_names else str(unique_labels[i]) for i in range(C)]
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_title("Inter-class Cosine Similarity")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Confusion matrix for NN classification."""
    labels_sorted = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    names = [class_names[l] if l < len(class_names) else str(l) for l in labels_sorted]

    fig, ax = plt.subplots(figsize=(max(8, len(labels_sorted) * 0.6), max(6, len(labels_sorted) * 0.55)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks(range(len(labels_sorted)))
    ax.set_yticks(range(len(labels_sorted)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted (NN)")
    ax.set_title("1-NN Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_tsne(embeddings, labels, class_names, save_path):
    """t-SNE 2-D visualization. Warns and returns on failure."""
    n_samples = len(labels)
    unique_labels = np.unique(labels)
    if n_samples < 2 or len(unique_labels) < 2:
        warnings.warn("Too few samples or classes for t-SNE, skipping.")
        return

    perplexity = min(30, max(2, n_samples // 4))
    try:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
        emb_2d = tsne.fit_transform(embeddings)
    except Exception as e:
        warnings.warn(f"t-SNE failed: {e}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    for lbl in unique_labels:
        mask = labels == lbl
        name = class_names[lbl] if lbl < len(class_names) else str(lbl)
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], label=name, s=12, alpha=0.7)
    ax.legend(fontsize=6, loc="best", ncol=2)
    ax.set_title("t-SNE Embedding Visualization")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
