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
    parser = argparse.ArgumentParser(description="Single-sample inference")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--input", type=str, required=True, help="Path to a single .npy or .pcd point cloud file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    model = MetricPointNet(
        input_channels=cfg["input_channels"],
        num_classes=cfg["num_classes"],
        embedding_dim=cfg["embedding_dim"],
    ).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

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
        probs = torch.softmax(out["logits"], dim=1).cpu().numpy()[0]

    pred_label = int(probs.argmax())
    confidence = float(probs[pred_label])
    top5_indices = probs.argsort()[::-1][:5]
    top5 = [{"label": int(i), "confidence": float(probs[i])} for i in top5_indices]

    class_names_path = os.path.join(os.path.dirname(args.config), "class_names.json")
    pred_class_name = f"class_{pred_label:03d}"
    if pred_label == cfg["negative_label"]:
        pred_class_name = "negative"
    elif os.path.exists(class_names_path):
        with open(class_names_path) as f:
            name_map = json.load(f)
        pred_class_name = name_map.get(str(pred_label), f"class_{pred_label:03d}")

    result = {
        "pred_label": pred_label,
        "pred_class_name": pred_class_name,
        "confidence": confidence,
        "top5": top5,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
