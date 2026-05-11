"""Standalone embedding evaluation script for metric learning.

Supports .npz, .npy (dict), and .csv input formats.
Does not depend on model code.

Usage:
    python scripts/evaluate_embeddings.py \
        --input outputs/embeddings/test_embeddings.npz \
        --output-dir outputs/reports/embedding_eval \
        --normalize true \
        --exclude-negative true \
        --negative-label 19 \
        --topk 1 3 5 10
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

# Allow running from project root without install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics.embedding_metrics import (
    compute_intra_class_similarity,
    compute_inter_class_similarity,
    compute_nn_accuracy,
    compute_retrieval_metrics,
    compute_similarity_gap,
    compute_top_confusing_pairs,
    l2_normalize,
    plot_class_similarity_matrix,
    plot_confusion_matrix,
    plot_tsne,
    validate_embeddings,
)


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]
    sample_ids = data["sample_ids"] if "sample_ids" in data else None
    class_names = data["class_names"] if "class_names" in data else None
    if class_names is not None:
        class_names = list(class_names)
    return embeddings, labels, sample_ids, class_names


def load_npy_dict(path):
    data = np.load(path, allow_pickle=True).item()
    embeddings = data["embeddings"]
    labels = data["labels"]
    sample_ids = data.get("sample_ids", None)
    class_names = data.get("class_names", None)
    return embeddings, labels, sample_ids, class_names


def load_csv(path):
    df = pd.read_csv(path)
    feat_cols = sorted([c for c in df.columns if c.startswith("feat_")])
    if not feat_cols:
        raise ValueError("CSV must contain feat_0, feat_1, ... columns")
    embeddings = df[feat_cols].values.astype(np.float64)

    label_col = df["label"]
    # If labels are strings, map to int
    if label_col.dtype == object:
        unique_names = sorted(label_col.unique())
        name_to_int = {n: i for i, n in enumerate(unique_names)}
        labels = label_col.map(name_to_int).values.astype(np.int64)
        class_names = unique_names
    else:
        labels = label_col.values.astype(np.int64)
        class_names = None

    sample_ids = df["sample_id"].values if "sample_id" in df.columns else None
    return embeddings, labels, sample_ids, class_names


def load_embeddings(path):
    """Dispatch to the correct loader based on file extension."""
    if path.endswith(".npz"):
        return load_npz(path)
    elif path.endswith(".npy"):
        return load_npy_dict(path)
    elif path.endswith(".csv"):
        return load_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path}. Use .npz, .npy, or .csv")


def load_class_names(path):
    """Load class_names.json mapping int -> name."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        mapping = json.load(f)
    # mapping is {"0": "name", "1": "name", ...}
    max_key = max(int(k) for k in mapping)
    names = [""] * (max_key + 1)
    for k, v in mapping.items():
        names[int(k)] = v
    return names


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate embedding quality for metric learning")
    parser.add_argument("--input", required=True, help="Path to embeddings file (.npz, .npy, .csv)")
    parser.add_argument("--output-dir", required=True, help="Directory to save evaluation results")
    parser.add_argument("--normalize", default="true", help="L2 normalize embeddings (true/false)")
    parser.add_argument("--exclude-negative", default="true", help="Exclude negative class from metrics")
    parser.add_argument("--negative-label", type=int, default=-1, help="Label id for negative class")
    parser.add_argument("--topk", nargs="+", type=int, default=[1, 3, 5, 10], help="K values for Recall@K / Precision@K")
    parser.add_argument("--class-names", default=None, help="Path to class_names.json")
    return parser.parse_args()


def str_to_bool(s):
    return s.lower() in ("true", "1", "yes")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    do_normalize = str_to_bool(args.normalize)
    exclude_negative = str_to_bool(args.exclude_negative)
    negative_label = args.negative_label
    ks = sorted(set(args.topk))

    # Load embeddings
    print(f"Loading embeddings from {args.input} ...")
    embeddings, labels, sample_ids, file_class_names = load_embeddings(args.input)
    print(f"  embeddings shape: {embeddings.shape}")
    print(f"  labels shape:     {labels.shape}")
    print(f"  unique labels:    {np.unique(labels).tolist()}")

    embeddings, labels = validate_embeddings(embeddings, labels)

    # Load class names
    class_names = None
    if args.class_names:
        class_names = load_class_names(args.class_names)
    if class_names is None and file_class_names is not None:
        # Use class names from the input file
        unique_labels = np.unique(labels)
        if len(file_class_names) >= len(unique_labels):
            class_names = file_class_names

    if class_names is None:
        unique_labels = np.unique(labels)
        class_names = [str(l) for l in unique_labels]

    # Exclude negative class if requested
    if exclude_negative:
        mask = labels != negative_label
        excluded_count = int((~mask).sum())
        if excluded_count > 0:
            print(f"  Excluding negative class (label={negative_label}): {excluded_count} samples removed")
            embeddings = embeddings[mask]
            labels = labels[mask]
            if sample_ids is not None:
                sample_ids = sample_ids[mask]

    print(f"  After filtering:  {len(labels)} samples, {len(np.unique(labels))} classes")

    # Check for empty / singleton classes
    unique_labels, counts = np.unique(labels, return_counts=True)
    for lbl, cnt in zip(unique_labels, counts):
        if cnt < 2:
            warnings.warn(f"Class {lbl} ({class_names[lbl] if lbl < len(class_names) else lbl}) has only {cnt} sample(s)")

    # Normalize
    if do_normalize:
        embeddings = l2_normalize(embeddings)
        print("  L2 normalization applied")

    # Compute cosine similarity matrix
    print("\nComputing cosine similarity matrix ...")
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)

    # 1. Intra-class similarity
    print("Computing intra-class similarity ...")
    intra = compute_intra_class_similarity(sim_matrix, labels)

    # 2. Inter-class similarity
    print("Computing inter-class similarity ...")
    inter = compute_inter_class_similarity(sim_matrix, labels)

    # 3. Similarity gap
    gap = compute_similarity_gap(intra["macro_avg"], inter["global_avg"])

    # Top confusing pairs
    top_confusing = compute_top_confusing_pairs(
        inter["class_similarity_matrix"],
        inter["unique_labels"],
        class_names=[class_names[l] if l < len(class_names) else str(l) for l in inter["unique_labels"]],
        top_k=10,
    )

    # 4 & 5. Recall@K / Precision@K
    print(f"Computing Recall@K and Precision@K (K={ks}) ...")
    retrieval = compute_retrieval_metrics(sim_matrix, labels, ks=ks)

    # 6. NN accuracy
    print("Computing 1-NN classification accuracy ...")
    nn = compute_nn_accuracy(sim_matrix, labels)

    # Build names list aligned to unique_labels used in inter-class matrix
    inter_class_names = [
        class_names[l] if l < len(class_names) else str(l) for l in inter["unique_labels"]
    ]

    # ---- Save outputs ----

    # metrics.json
    metrics_dict = {
        "intra_class_similarity": {
            "macro_avg": intra["macro_avg"],
            "global_avg": intra["global_avg"],
            "per_class": {str(k): v for k, v in intra["per_class"].items()},
        },
        "inter_class_similarity": {
            "global_avg": inter["global_avg"],
        },
        "similarity_gap": gap,
        "recall_at_k": {f"recall@{k}": v for k, v in retrieval["recall_at_k"].items()},
        "precision_at_k": {f"precision@{k}": v for k, v in retrieval["precision_at_k"].items()},
        "nn_accuracy": nn["nn_accuracy"],
        "per_class_nn_accuracy": {str(k): v for k, v in nn["per_class_nn_accuracy"].items()},
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {metrics_path}")

    # per_class_metrics.csv
    per_class_rows = []
    for lbl in unique_labels:
        row = {
            "label": int(lbl),
            "class_name": class_names[lbl] if lbl < len(class_names) else str(lbl),
            "num_samples": int(counts[np.where(unique_labels == lbl)[0][0]]),
            "intra_class_similarity": intra["per_class"].get(int(lbl), float("nan")),
        }
        for k in ks:
            per_class_recall = retrieval["per_class_recall_at_k"].get(int(lbl), {})
            per_class_prec = retrieval["per_class_precision_at_k"].get(int(lbl), {})
            row[f"recall@{k}"] = per_class_recall.get(k, float("nan"))
            row[f"precision@{k}"] = per_class_prec.get(k, float("nan"))
        row["nn_accuracy"] = nn["per_class_nn_accuracy"].get(int(lbl), float("nan"))
        per_class_rows.append(row)

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_path = os.path.join(args.output_dir, "per_class_metrics.csv")
    per_class_df.to_csv(per_class_path, index=False)
    print(f"Saved: {per_class_path}")

    # class_similarity_matrix.csv
    sim_df = pd.DataFrame(
        inter["class_similarity_matrix"],
        index=inter_class_names,
        columns=inter_class_names,
    )
    sim_csv_path = os.path.join(args.output_dir, "class_similarity_matrix.csv")
    sim_df.to_csv(sim_csv_path)
    print(f"Saved: {sim_csv_path}")

    # class_similarity_matrix.png
    sim_png_path = os.path.join(args.output_dir, "class_similarity_matrix.png")
    plot_class_similarity_matrix(inter["class_similarity_matrix"], inter["unique_labels"], inter_class_names, sim_png_path)
    print(f"Saved: {sim_png_path}")

    # confusion_matrix.png
    cm_png_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(labels, nn["predictions"], class_names, cm_png_path)
    print(f"Saved: {cm_png_path}")

    # top_confusing_pairs.csv
    confusing_df = pd.DataFrame(top_confusing)
    confusing_path = os.path.join(args.output_dir, "top_confusing_pairs.csv")
    confusing_df.to_csv(confusing_path, index=False)
    print(f"Saved: {confusing_path}")

    # embedding_tsne.png
    tsne_path = os.path.join(args.output_dir, "embedding_tsne.png")
    print("Generating t-SNE visualization ...")
    plot_tsne(embeddings, labels, class_names, tsne_path)
    if os.path.exists(tsne_path):
        print(f"Saved: {tsne_path}")

    # ---- Console summary ----
    print("\n" + "=" * 60)
    print("EMBEDDING EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Samples: {len(labels)}, Classes: {len(unique_labels)}")
    print(f"Intra-class similarity (macro): {intra['macro_avg']:.4f}")
    print(f"Intra-class similarity (global): {intra['global_avg']:.4f}")
    print(f"Inter-class similarity (global): {inter['global_avg']:.4f}")
    print(f"Similarity gap:                  {gap:.4f}")
    for k in ks:
        print(f"Recall@{k}:                       {retrieval['recall_at_k'][k]:.4f}")
    for k in ks:
        print(f"Precision@{k}:                    {retrieval['precision_at_k'][k]:.4f}")
    print(f"1-NN accuracy:                   {nn['nn_accuracy']:.4f}")
    if top_confusing:
        print(f"\nTop confusing pair: {top_confusing[0]['name_i']} <-> {top_confusing[0]['name_j']} (sim={top_confusing[0]['similarity']:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
