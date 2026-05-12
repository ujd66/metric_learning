"""Search optimal similarity threshold for OOD / unknown rejection.

Loads prototypes, extracts val split embeddings, and searches for the best
similarity threshold that maximizes negative rejection while maintaining
a target known acceptance rate.

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
from src.models.metric_model import MetricPointNet
from src.utils.checkpoint import load_checkpoint
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
    threshold_min = ood_cfg.get("threshold_min", 0.1)
    threshold_max = ood_cfg.get("threshold_max", 0.99)
    threshold_step = ood_cfg.get("threshold_step", 0.01)

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
    model = MetricPointNet(
        input_channels=cfg["input_channels"],
        num_classes=cfg["num_classes"],
        embedding_dim=cfg["embedding_dim"],
    ).to(device)
    load_checkpoint(args.checkpoint, model)

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
    # embeddings: [N, D] (already L2-normalized by model)
    # prototypes: [C, D] (already L2-normalized)
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

    # Search threshold
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

    # Select threshold
    if n_negative > 0:
        # Strategy: max negative rejection under known accept constraint
        candidates = [tc for tc in threshold_curve if tc["known_accept_rate"] >= target_known_accept_rate]
        if candidates:
            best = max(candidates, key=lambda x: x["negative_reject_rate"])
            reason = (f"Selected threshold={best['threshold']:.2f}: "
                      f"known_accept={best['known_accept_rate']:.4f} >= {target_known_accept_rate}, "
                      f"neg_reject={best['negative_reject_rate']:.4f}")
        else:
            best = max(threshold_curve, key=lambda x: x["balanced_score"])
            reason = (f"No threshold meets known_accept >= {target_known_accept_rate}; "
                      f"falling back to max balanced_score, threshold={best['threshold']:.2f}")
    else:
        # No negative samples: select threshold that guarantees target known acceptance
        warnings.warn(
            "val split has no negative samples; threshold cannot be calibrated for negative rejection."
        )
        candidates = [tc for tc in threshold_curve if tc["known_accept_rate"] >= target_known_accept_rate]
        if candidates:
            # Pick the highest threshold that still meets the target (most conservative)
            best = max(candidates, key=lambda x: x["threshold"])
            reason = (f"No negative samples. Selected highest threshold with "
                      f"known_accept >= {target_known_accept_rate}: {best['threshold']:.2f}")
        else:
            best = max(threshold_curve, key=lambda x: x["known_accept_rate"])
            reason = (f"No negative samples and no threshold meets known_accept >= {target_known_accept_rate}; "
                      f"selecting threshold with highest known_accept: {best['threshold']:.2f}")

    result = {
        "selected_threshold": best["threshold"],
        "selection_reason": reason,
        "target_known_accept_rate": target_known_accept_rate,
        "known_accept_rate": best["known_accept_rate"],
        "known_reject_rate": best["known_reject_rate"],
        "negative_reject_rate": best.get("negative_reject_rate"),
        "false_known_rate": best.get("false_known_rate"),
        "threshold_curve": threshold_curve,
        "num_known_samples": n_known,
        "num_negative_samples": n_negative,
        "warnings": [],
    }

    if n_negative == 0:
        result["warnings"].append(
            "val split has no negative samples; threshold cannot be calibrated for negative rejection."
        )
    if n_negative > 0 and n_negative < 10:
        result["warnings"].append(
            f"val split has only {n_negative} negative sample(s); threshold calibration may be unreliable."
        )

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nThreshold search results:")
    print(f"  Selected threshold: {best['threshold']:.2f}")
    print(f"  Known accept rate:  {best['known_accept_rate']:.4f}")
    print(f"  Known reject rate:  {best['known_reject_rate']:.4f}")
    if best.get("negative_reject_rate") is not None:
        print(f"  Negative reject rate: {best['negative_reject_rate']:.4f}")
        print(f"  False known rate:     {best['false_known_rate']:.4f}")
    print(f"  Reason: {reason}")
    print(f"\nSaved: {args.output}")

    # Also update prototypes file with threshold info
    proto_data["similarity_threshold"] = best["threshold"]
    torch.save(proto_data, args.prototypes)
    print(f"Updated prototypes file with threshold: {best['threshold']:.2f}")


if __name__ == "__main__":
    main()
