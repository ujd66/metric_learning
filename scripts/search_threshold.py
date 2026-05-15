"""Search optimal similarity threshold for OOD / unknown rejection.

Supports multiple threshold selection strategies:
  - known_quantile: use a quantile of val known similarity distribution
  - max_negative_rejection_under_known_accept_constraint: maximize neg reject with known accept >= target
  - best_balanced_score: maximize known_accept_rate + negative_reject_rate
  - manual: use config.ood.similarity_threshold directly

Usage:
    python scripts/search_threshold.py \
        --config configs/config.yaml \
        --checkpoint outputs/checkpoints/best.pt \
        --prototypes outputs/prototypes/baseline_prototypes.pt \
        --split val \
        --output outputs/prototypes/baseline_threshold.json
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.pointcloud_dataset import PointCloudDataset
from src.models.model_factory import build_model_from_checkpoint
from src.utils.config import load_config


def collate_fn(batch):
    points = torch.stack([b["points"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"points": points, "label": labels}


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    all_embeddings = []
    all_labels = []
    for batch in tqdm(loader, desc="Extract"):
        points = batch["points"].to(device)
        out = model(points)
        all_embeddings.append(out["embedding"].cpu().numpy())
        all_labels.extend(batch["label"].numpy())
    return np.concatenate(all_embeddings, axis=0), np.array(all_labels)


def compute_quantiles(data, quantiles=None):
    """Compute quantiles for a numpy array."""
    if quantiles is None:
        quantiles = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
    if len(data) == 0:
        return {f"p{int(q*100):02d}": None for q in quantiles}
    return {f"p{int(q*100):02d}": float(np.quantile(data, q)) for q in quantiles}


def search_with_strategy(
    strategy, threshold_curve, known_sims, negative_sims,
    target_known_accept_rate, known_quantile, manual_threshold,
):
    """Select threshold based on the given strategy.

    Returns (selected_threshold, selection_reason).
    """
    if strategy == "known_quantile":
        quantile_val = known_quantile
        threshold = float(np.quantile(known_sims, quantile_val))
        # Find closest threshold in curve for metrics
        closest = min(threshold_curve, key=lambda x: abs(x["threshold"] - threshold))
        reason = (
            f"Selected threshold={closest['threshold']:.2f} from known_quantile={quantile_val}: "
            f"known_accept={closest['known_accept_rate']:.4f}"
        )
        return closest, reason

    elif strategy == "max_negative_rejection_under_known_accept_constraint":
        candidates = [tc for tc in threshold_curve
                      if tc["known_accept_rate"] >= target_known_accept_rate]
        if candidates and negative_sims is not None and len(negative_sims) > 0:
            best = max(candidates, key=lambda x: x["negative_reject_rate"] or 0)
            reason = (
                f"Selected threshold={best['threshold']:.2f}: "
                f"known_accept={best['known_accept_rate']:.4f} >= {target_known_accept_rate}, "
                f"neg_reject={best['negative_reject_rate']:.4f}"
            )
        elif candidates:
            best = max(candidates, key=lambda x: x["threshold"])
            reason = (
                f"No negative samples. Selected highest threshold with "
                f"known_accept >= {target_known_accept_rate}: {best['threshold']:.2f}"
            )
        else:
            best = max(threshold_curve, key=lambda x: x["balanced_score"])
            reason = (
                f"No threshold meets known_accept >= {target_known_accept_rate}; "
                f"falling back to max balanced_score, threshold={best['threshold']:.2f}"
            )
        return best, reason

    elif strategy == "best_balanced_score":
        best = max(threshold_curve, key=lambda x: x["balanced_score"])
        reason = (
            f"Selected threshold={best['threshold']:.2f}: "
            f"balanced_score={best['balanced_score']:.4f} "
            f"(known_accept={best['known_accept_rate']:.4f}, "
            f"neg_reject={best.get('negative_reject_rate')})"
        )
        return best, reason

    elif strategy == "manual":
        t = manual_threshold
        closest = min(threshold_curve, key=lambda x: abs(x["threshold"] - t))
        reason = (
            f"Using manual threshold={closest['threshold']:.2f}: "
            f"known_accept={closest['known_accept_rate']:.4f}"
        )
        return closest, reason

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def main():
    parser = argparse.ArgumentParser(description="Search optimal similarity threshold for OOD rejection")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--prototypes", type=str, required=True, help="Path to prototypes.pt")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: outputs/prototypes/<proto_name>_threshold.json)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    num_known_classes = cfg["num_known_classes"]
    negative_label = cfg["negative_label"]

    # OOD config
    ood_cfg = cfg.get("ood", {})
    target_known_accept_rate = ood_cfg.get("target_known_accept_rate", 0.95)
    min_known_accept_rate = ood_cfg.get("min_known_accept_rate", 0.95)
    threshold_min = ood_cfg.get("threshold_min", 0.1)
    threshold_max = ood_cfg.get("threshold_max", 0.99)
    threshold_step = ood_cfg.get("threshold_step", 0.01)
    strategy = ood_cfg.get("threshold_selection", "known_quantile")
    known_quantile = ood_cfg.get("known_quantile", 0.05)
    manual_threshold = ood_cfg.get("similarity_threshold", 0.65)
    candidate_thresholds_cfg = ood_cfg.get("candidate_thresholds", [])

    # Determine output path
    if args.output is None:
        proto_base = os.path.splitext(os.path.basename(args.prototypes))[0]
        args.output = f"outputs/prototypes/{proto_base}_threshold.json"

    # Load prototypes
    print(f"Loading prototypes from {args.prototypes} ...")
    proto_data = torch.load(args.prototypes, map_location="cpu", weights_only=False)
    prototypes = proto_data["prototypes"].numpy()  # [num_known_classes, embedding_dim]
    proto_class_names = proto_data["class_names"]
    print(f"  Prototypes shape: {prototypes.shape}")

    # Load model
    model, _ = build_model_from_checkpoint(cfg, args.checkpoint)
    model = model.to(device)

    # Load dataset
    dataset = PointCloudDataset(
        root_dir=cfg["data"]["root"],
        split=args.split,
        num_points=cfg["num_points"],
        input_channels=cfg["input_channels"],
        augmentation_config={},
    )
    if len(dataset) == 0:
        print(f"[ERROR] No samples found in {args.split} split")
        sys.exit(1)

    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        num_workers=4, collate_fn=collate_fn)

    # Extract embeddings
    print(f"Extracting embeddings from {len(dataset)} samples ({args.split} split) ...")
    embeddings, labels = extract_embeddings(model, loader, device)

    # Compute nearest known similarity for each sample
    similarities = embeddings @ prototypes.T  # [N, C]
    nearest_sim = similarities.max(axis=1)  # [N]

    # Split into known and negative
    known_mask = labels < num_known_classes
    negative_mask = labels == negative_label

    n_known = int(known_mask.sum())
    n_negative = int(negative_mask.sum())

    print(f"  Known samples: {n_known}")
    print(f"  Negative samples: {n_negative}")

    known_sims = nearest_sim[known_mask]
    negative_sims = nearest_sim[negative_mask] if n_negative > 0 else None

    # --- Similarity quantiles ---
    known_quantiles = compute_quantiles(known_sims)
    negative_quantiles = compute_quantiles(negative_sims) if negative_sims is not None else {}

    print(f"\nKnown similarity quantiles:")
    for k, v in known_quantiles.items():
        print(f"  {k}: {v:.6f}" if v is not None else f"  {k}: N/A")
    if negative_sims is not None:
        print(f"Negative similarity quantiles:")
        for k, v in negative_quantiles.items():
            print(f"  {k}: {v:.6f}" if v is not None else f"  {k}: N/A")

    # --- Build threshold curve ---
    thresholds = np.arange(threshold_min, threshold_max + threshold_step / 2, threshold_step)
    threshold_curve = []

    for t in thresholds:
        t = float(t)
        known_accept = float((known_sims >= t).mean()) if n_known > 0 else 0.0
        known_reject = 1.0 - known_accept

        if n_negative > 0:
            negative_reject = float((negative_sims < t).mean())
            false_known = float((negative_sims >= t).mean())
        else:
            negative_reject = None
            false_known = None

        balanced = known_accept + (negative_reject if negative_reject is not None else 0.0)

        threshold_curve.append({
            "threshold": t,
            "known_accept_rate": known_accept,
            "known_reject_rate": known_reject,
            "negative_reject_rate": negative_reject,
            "false_known_rate": false_known,
            "balanced_score": balanced,
        })

    # --- Candidate threshold results ---
    candidate_results = []
    for ct in candidate_thresholds_cfg:
        closest = min(threshold_curve, key=lambda x: abs(x["threshold"] - ct))
        candidate_results.append({
            "requested_threshold": ct,
            "actual_threshold": closest["threshold"],
            "known_accept_rate": closest["known_accept_rate"],
            "known_reject_rate": closest["known_reject_rate"],
            "negative_reject_rate": closest.get("negative_reject_rate"),
            "false_known_rate": closest.get("false_known_rate"),
            "balanced_score": closest["balanced_score"],
        })

    # --- Select threshold ---
    best, reason = search_with_strategy(
        strategy, threshold_curve, known_sims, negative_sims,
        target_known_accept_rate, known_quantile, manual_threshold,
    )

    # --- Warnings ---
    warn_list = []
    if n_negative == 0:
        warn_list.append(
            "val split has no negative samples; threshold cannot be calibrated for negative rejection."
        )
    if 0 < n_negative < 10:
        warn_list.append(
            f"val split has only {n_negative} negative sample(s); threshold calibration may be unreliable. "
            "selected threshold mainly relies on known similarity quantile."
        )
    if strategy in ("max_negative_rejection_under_known_accept_constraint",) and n_negative < 10:
        warn_list.append(
            f"Strategy '{strategy}' selected but val negative samples are too few; "
            "selected threshold mainly relies on known similarity quantile."
        )

    # --- Build output ---
    result = {
        "selected_threshold": best["threshold"],
        "selection_strategy": strategy,
        "selection_reason": reason,
        "known_quantile": known_quantile,
        "target_known_accept_rate": target_known_accept_rate,
        "min_known_accept_rate": min_known_accept_rate,
        "known_accept_rate": best["known_accept_rate"],
        "known_reject_rate": best["known_reject_rate"],
        "negative_reject_rate": best.get("negative_reject_rate"),
        "false_known_rate": best.get("false_known_rate"),
        "known_similarity_quantiles": known_quantiles,
        "negative_similarity_quantiles": negative_quantiles if negative_quantiles else None,
        "candidate_threshold_results": candidate_results,
        "threshold_curve": threshold_curve,
        "num_known_samples": n_known,
        "num_negative_samples": n_negative,
        "warnings": warn_list,
    }

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nThreshold search results:")
    print(f"  Strategy: {strategy}")
    print(f"  Selected threshold: {best['threshold']:.2f}")
    print(f"  Known accept rate:  {best['known_accept_rate']:.4f}")
    print(f"  Known reject rate:  {best['known_reject_rate']:.4f}")
    if best.get("negative_reject_rate") is not None:
        print(f"  Negative reject rate: {best['negative_reject_rate']:.4f}")
        print(f"  False known rate:     {best['false_known_rate']:.4f}")
    print(f"  Reason: {reason}")
    if candidate_results:
        print(f"\n  Candidate threshold results:")
        for cr in candidate_results:
            print(f"    t={cr['requested_threshold']:.2f}: "
                  f"ka={cr['known_accept_rate']:.4f}, "
                  f"kr={cr['known_reject_rate']:.4f}, "
                  f"nr={cr['negative_reject_rate']}, "
                  f"bs={cr['balanced_score']:.4f}")
    for w in warn_list:
        print(f"  WARNING: {w}")
    print(f"\nSaved: {args.output}")

    # Also update prototypes file with threshold info
    proto_data["similarity_threshold"] = best["threshold"]
    torch.save(proto_data, args.prototypes)
    print(f"Updated prototypes file with threshold: {best['threshold']:.2f}")


if __name__ == "__main__":
    main()
