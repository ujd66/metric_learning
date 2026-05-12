import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.metric_model import MetricPointNet
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.pointcloud_ops import load_pointcloud, normalize_points, sample_points


def main():
    parser = argparse.ArgumentParser(description="Single-sample inference with optional prototype rejection")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--input", type=str, required=True, help="Path to a single .npy or .pcd point cloud file")
    parser.add_argument("--prototypes", type=str, default=None,
                        help="Path to prototypes.pt for OOD rejection")
    parser.add_argument("--threshold-json", type=str, default=None,
                        help="Path to threshold JSON (from search_threshold.py)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    num_known_classes = cfg["num_known_classes"]
    negative_label = cfg["negative_label"]

    model = MetricPointNet(
        input_channels=cfg["input_channels"],
        num_classes=cfg["num_classes"],
        embedding_dim=cfg["embedding_dim"],
    ).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

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

    # Load prototypes and threshold if provided
    prototypes = None
    proto_class_names = None
    similarity_threshold = None

    if args.prototypes:
        proto_data = torch.load(args.prototypes, map_location="cpu", weights_only=False)
        prototypes = proto_data["prototypes"].to(device)  # [num_known, D]
        proto_class_names = proto_data["class_names"]

    if args.threshold_json:
        with open(args.threshold_json) as f:
            threshold_data = json.load(f)
        similarity_threshold = threshold_data["selected_threshold"]

    # Load point cloud
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
        probs = torch.softmax(out["logits"], dim=1).cpu().numpy()[0]

    pred_label = int(probs.argmax())
    confidence = float(probs[pred_label])
    top5_indices = probs.argsort()[::-1][:5]
    top5_classifier_probs = [{"label": int(i), "name": get_class_name(int(i)),
                               "confidence": float(probs[i])} for i in top5_indices]

    # Prototype-based analysis
    top5_prototype_sims = []
    nearest_known_label = None
    nearest_known_class = None
    nearest_similarity = None
    final_type = None
    final_label = None
    reason = None

    if prototypes is not None:
        # Cosine similarity with all known prototypes
        sims = torch.nn.functional.cosine_similarity(embedding, prototypes, dim=1)
        sims_np = sims.cpu().numpy()
        top5_proto_idx = sims_np.argsort()[::-1][:5]

        for idx in top5_proto_idx:
            top5_prototype_sims.append({
                "label": int(idx),
                "name": proto_class_names[idx],
                "similarity": float(sims_np[idx]),
            })

        nearest_known_label = int(sims_np.argmax())
        nearest_known_class = proto_class_names[nearest_known_label]
        nearest_similarity = float(sims_np.max())

    # Decision logic
    if pred_label == negative_label:
        final_type = "negative"
        final_label = "negative"
        reason = "classified_as_negative"
    elif prototypes is not None and similarity_threshold is not None:
        if nearest_similarity < similarity_threshold:
            final_type = "unknown"
            final_label = "unknown"
            reason = "far_from_all_known_prototypes"
        else:
            final_type = "known"
            final_label = nearest_known_class
            reason = "matched_known_prototype"
    else:
        final_type = "known"
        final_label = get_class_name(pred_label)
        reason = "classifier_only"

    result = {
        "final_type": final_type,
        "final_label": final_label,
        "reason": reason,
        "classifier_pred": pred_label,
        "classifier_pred_name": get_class_name(pred_label),
        "classifier_confidence": confidence,
    }

    if prototypes is not None:
        result["nearest_known_class"] = nearest_known_class
        result["nearest_known_label"] = nearest_known_label
        result["nearest_similarity"] = nearest_similarity
        result["similarity_threshold"] = similarity_threshold

    result["top5_classifier_probs"] = top5_classifier_probs
    result["top5_prototype_similarities"] = top5_prototype_sims

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
