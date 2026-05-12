import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.pointcloud_ops import (
    load_pointcloud,
    normalize_points,
    random_dropout,
    random_jitter,
    random_z_rotation,
    sample_points,
)

_SUPPORTED_EXTS = (".npy", ".pcd")


def _parse_label_from_dirname(dirname, negative_label=19):
    if dirname == "negative":
        return negative_label
    m = re.match(r"class_(\d+)", dirname)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse label from directory name: {dirname}")


class PointCloudDataset(Dataset):
    def __init__(self, root_dir, split, num_points, input_channels, augmentation_config=None):
        self.num_points = num_points
        self.input_channels = input_channels
        self.augmentation = augmentation_config or {}
        self.is_train = split in ("train",)

        split_dir = os.path.join(root_dir, split)
        self.samples = []

        if not os.path.isdir(split_dir):
            return

        for class_dir in sorted(os.listdir(split_dir)):
            class_path = os.path.join(split_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
            label = _parse_label_from_dirname(class_dir)
            for fname in sorted(os.listdir(class_path)):
                if any(fname.endswith(ext) for ext in _SUPPORTED_EXTS):
                    self.samples.append({
                        "path": os.path.join(class_path, fname),
                        "label": label,
                        "class_name": class_dir,
                        "sample_id": os.path.splitext(fname)[0],
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        points = load_pointcloud(info["path"])

        if points.shape[1] > self.input_channels:
            points = points[:, :self.input_channels]
        elif points.shape[1] < self.input_channels:
            pad = np.zeros((points.shape[0], self.input_channels - points.shape[1]), dtype=np.float64)
            points = np.concatenate([points, pad], axis=1)

        points = sample_points(points, self.num_points)
        points = normalize_points(points)

        if self.is_train:
            if self.augmentation.get("use_random_z_rotation", False):
                points = random_z_rotation(points)
            if self.augmentation.get("use_jitter", False):
                points = random_jitter(
                    points,
                    self.augmentation.get("jitter_sigma", 0.01),
                    self.augmentation.get("jitter_clip", 0.05),
                )
            if self.augmentation.get("use_random_dropout", False):
                points = random_dropout(points, self.augmentation.get("dropout_ratio", 0.1))
                points = sample_points(points, self.num_points)

        return {
            "points": torch.tensor(points, dtype=torch.float32),
            "label": int(info["label"]),
            "class_name": info["class_name"],
            "sample_id": str(info["sample_id"]),
        }
