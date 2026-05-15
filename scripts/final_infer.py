"""Final unified inference with classifier → prototype → gallery pipeline.

Decision logic:
  1. Classifier negative → "negative"
  2. Prototype OOD rejection → "unknown"
  3. Prototype match → "known" (with gallery evidence)

Usage:
    python scripts/final_infer.py \
        --config configs/config.yaml \
        --checkpoint outputs/checkpoints/best.pt \
        --prototypes outputs/prototypes/baseline_prototypes.pt \
        --threshold-json outputs/prototypes/baseline_threshold_p05.json \
        --gallery outputs/gallery/baseline_train_gallery.pt \
        --input path/to/sample.pcd
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.model_factory import build_model_from_checkpoint
from src.utils.config import load_config
from src.utils.pointcloud_ops import load_pointcloud, normalize_points, sample_points


def main():
    parser = argparse.ArgumentParser(description="Final unified inference")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--prototypes", type=str, default=None,
                        help="Path to prototypes.pt (required for OOD rejection)")
    parser.add_argument("--threshold-json", type=str, default=None,
                        help="Path to threshold JSON")
    parser.add_argument("--gallery", type=str, default=None,
                        help="Path to gallery.pt for evidence retrieval")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to .npy or .pcd point cloud file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    num_known_classes = cfg["num_known_classes"]
    negative_label = cfg["negative_label"]

    # Load class names
    class_names_path = os.path.join(os.path.dirname(args.config), "class_names.json")
    class_names = {}
    if os.path.exists(class_names_path):
        with open(class_names_path) as f:
            class_names = json.load(f)

    def get_class_name(label):
        if label == negative_label:
            return "negative"
        return class_names.get(str(label), f"class_{label:03d}")

    # Supported / unsupported classes
    supported_cfg = cfg.get("supported_classes", {})
    supported_known_labels = set(supported_cfg.get("supported_known_labels", list(range(num_known_classes))))
    unsupported_known_labels = set(supported_cfg.get("unsupported_known_labels", []))

    # Also read from prototypes metadata if available
    proto_unsupported = set()
    proto_supported = set(range(num_known_classes))
    if args.prototypes:
        _tmp = torch.load(args.prototypes, map_location="cpu", weights_only=False)
        if "unsupported_known_labels" in _tmp:
            proto_unsupported = set(_tmp["unsupported_known_labels"])
            proto_supported = set(_tmp.get("supported_known_labels", list(range(num_known_classes))))
    # Merge: unsupported from config OR prototypes
    all_unsupported = unsupported_known_labels | proto_unsupported
    all_supported = (set(range(num_known_classes)) - all_unsupported) | (proto_supported - proto_unsupported)

    unsupported_class_names = [get_class_name(c) for c in sorted(all_unsupported)]

    # --- Load model ---
    model, _ = build_model_from_checkpoint(cfg, args.checkpoint)
    model = model.to(device)
    model.eval()

    # --- Load prototypes ---
    prototypes = None
    proto_class_names = None
    if args.prototypes:
        proto_data = torch.load(args.prototypes, map_location="cpu", weights_only=False)
        prototypes = proto_data["prototypes"].to(device)
        proto_class_names = proto_data["class_names"]

    # --- Load threshold ---
    similarity_threshold = None
    if args.threshold_json:
        with open(args.threshold_json) as f:
            threshold_data = json.load(f)
        similarity_threshold = threshold_data["selected_threshold"]

    # --- Load gallery ---
    gallery_embeddings = None
    gallery_labels = None
    gallery_class_names = None
    gallery_sample_ids = None
    gallery_source_paths = None
    if args.gallery:
        gal = torch.load(args.gallery, map_location="cpu", weights_only=False)
        gallery_embeddings = gal["embeddings"].to(device)  # [N, D]
        gallery_labels = gal["labels"].tolist()
        gallery_class_names = gal["class_names"]
        gallery_sample_ids = gal["sample_ids"]
        gallery_source_paths = gal.get("source_paths", [])

    # === Step 1: Model forward ===
    points = load_pointcloud(args.input)
    if points.shape[1] > cfg["input_channels"]:
        points = points[:, :cfg["input_channels"]]
    elif points.shape[1] < cfg["input_channels"]:
        pad = np.zeros((points.shape[0], cfg["input_channels"] - points.shape[1]), dtype=np.float64)
        points = np.concatenate([points, pad], axis=1)

    points = sample_points(points, cfg["num_points"])
    points = normalize_points(points)
    points_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(points_tensor)
        embedding = out["embedding"]  # [1, D]
        logits = out["logits"]
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_label = int(probs.argmax())
    confidence = float(probs[pred_label])

    # Classifier top-5
    top5_indices = probs.argsort()[::-1][:5]
    classifier_top5 = [
        {"label": int(i), "class_name": get_class_name(int(i)), "confidence": float(probs[i])}
        for i in top5_indices
    ]

    classifier_result = {
        "pred_label": pred_label,
        "pred_class_name": get_class_name(pred_label),
        "confidence": confidence,
        "top5": classifier_top5,
    }

    # === Step 2: Classifier negative check ===
    final_type = None
    final_label = None
    reason = None

    if pred_label == negative_label:
        final_type = "negative"
        final_label = "negative"
        reason = "classified_as_negative"

    # === Step 2.5: Unsupported known class check ===
    if final_type is None and pred_label in all_unsupported:
        final_type = "unsupported_known_class"
        final_label = get_class_name(pred_label)
        reason = "classifier_predicted_unsupported_known_class"

    # === Step 3: Prototype OOD rejection ===
    prototype_result = {
        "nearest_label": None,
        "nearest_class_name": None,
        "nearest_similarity": None,
        "threshold": similarity_threshold,
        "top5": [],
    }

    if final_type is None and prototypes is not None:
        sims = torch.nn.functional.cosine_similarity(embedding, prototypes, dim=1)
        sims_np = sims.cpu().numpy()
        top5_proto_idx = sims_np.argsort()[::-1][:5]

        prototype_result["top5"] = [
            {"label": int(idx), "class_name": proto_class_names[idx],
             "similarity": float(sims_np[idx])}
            for idx in top5_proto_idx
        ]

        nearest_label = int(sims_np.argmax())
        nearest_sim = float(sims_np.max())

        prototype_result["nearest_label"] = nearest_label
        prototype_result["nearest_class_name"] = proto_class_names[nearest_label]
        prototype_result["nearest_similarity"] = nearest_sim

        if similarity_threshold is not None and nearest_sim < similarity_threshold:
            final_type = "unknown"
            final_label = "unknown"
            reason = "far_from_known_prototypes"

        # === Step 4: Known class ===
        if final_type is None:
            final_type = "known"
            final_label = proto_class_names[nearest_label]
            reason = "matched_known_prototype"

    elif final_type is None:
        # No prototypes, fall back to classifier
        final_type = "known"
        final_label = get_class_name(pred_label)
        reason = "classifier_only"

    # === Step 5: Gallery evidence ===
    gallery_result = {
        "enabled": gallery_embeddings is not None,
        "top1_label": None,
        "top1_class_name": None,
        "top1_similarity": None,
        "top5": [],
    }

    if gallery_embeddings is not None:
        gal_sims = torch.nn.functional.cosine_similarity(embedding, gallery_embeddings, dim=1)
        gal_sims_np = gal_sims.cpu().numpy()
        top5_gal_idx = gal_sims_np.argsort()[::-1][:5]

        gallery_result["top1_label"] = gallery_labels[top5_gal_idx[0]]
        gallery_result["top1_class_name"] = gallery_class_names[top5_gal_idx[0]]
        gallery_result["top1_similarity"] = float(gal_sims_np[top5_gal_idx[0]])

        gallery_result["top5"] = []
        for idx in top5_gal_idx:
            entry = {
                "sample_id": gallery_sample_ids[idx],
                "label": gallery_labels[idx],
                "class_name": gallery_class_names[idx],
                "similarity": float(gal_sims_np[idx]),
            }
            if gallery_source_paths and idx < len(gallery_source_paths):
                entry["source_path"] = gallery_source_paths[idx] or None
            else:
                entry["source_path"] = None
            gallery_result["top5"].append(entry)

    # === Risk level ===
    risk_level = "safe"

    if final_type == "unsupported_known_class":
        risk_level = "unsupported_class_no_training_data"

    if gallery_result["enabled"] and final_type == "known" and prototype_result["nearest_label"] is not None:
        if gallery_result["top1_label"] != prototype_result["nearest_label"]:
            risk_level = "prototype_gallery_conflict"

    if risk_level == "safe" and prototype_result["nearest_similarity"] is not None and similarity_threshold is not None:
        margin = prototype_result["nearest_similarity"] - similarity_threshold
        if 0 <= margin < 0.03:
            risk_level = "near_threshold_boundary"

    # === Build output ===
    result = {
        "final_type": final_type,
        "final_label": final_label,
        "reason": reason,
        "risk_level": risk_level,
        "classifier": classifier_result,
        "prototype": prototype_result,
        "gallery": gallery_result,
        "coverage": {
            "supported_known_labels": sorted(all_supported),
            "unsupported_known_labels": sorted(all_unsupported),
            "unsupported_class_names": unsupported_class_names,
            "incomplete_known_class_coverage": len(all_unsupported) > 0,
        },
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
