"""Model factory — build_model(config) returns a model with unified interface.

All models expose forward() -> {"embedding": Tensor, "logits": Tensor}.

Supported backbones:
  - "pointnet"   (default, always available)
  - "pointnet2"  (always available, built-in SSG implementation)
  - "pointnext"  (requires OpenPoints)
  - "openpoints" (requires OpenPoints, same adapter)
"""

import torch
import torch.nn as nn


def build_model(cfg) -> nn.Module:
    """Instantiate a model based on config.

    Reads cfg["model"]["backbone"] (default "pointnet").
    Returns an nn.Module whose forward() returns
    {"embedding": Tensor [B, D], "logits": Tensor [B, C]}.
    """
    model_cfg = cfg.get("model", {})
    backbone = model_cfg.get("backbone", "pointnet").lower()

    input_channels = cfg.get("input_channels", 3)
    num_classes = cfg.get("num_classes", 20)
    embedding_dim = cfg.get("embedding_dim", 256)

    if backbone == "pointnet":
        from src.models.metric_model import MetricPointNet
        return MetricPointNet(
            input_channels=input_channels,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
        )

    elif backbone in ("pointnet2", "pointnet++"):
        from src.models.pointnet2 import MetricPointNet2
        return MetricPointNet2(
            input_channels=input_channels,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
        )

    elif backbone in ("pointnext", "openpoints"):
        from src.models.openpoints_adapter import OpenPointsAdapter
        return OpenPointsAdapter(cfg)

    else:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Supported: pointnet, pointnet2, pointnext, openpoints"
        )


def build_model_from_checkpoint(cfg, checkpoint_path):
    """Build model and load checkpoint, recovering backbone from saved config.

    If the checkpoint contains a saved config (written by train.py), uses that
    to determine the backbone.  Otherwise falls back to the passed-in cfg.

    Returns:
        (model, extra_ckpt_data)
    """
    # Peek at checkpoint for embedded config
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_cfg = ckpt.get("config", None)

    # Merge: saved config takes priority for model architecture
    if saved_cfg is not None:
        effective_cfg = dict(cfg)
        effective_cfg.update(saved_cfg)
    else:
        effective_cfg = cfg

    model = build_model(effective_cfg)
    model.load_state_dict(ckpt["model_state_dict"])

    extra = {k: v for k, v in ckpt.items()
             if k not in ("model_state_dict", "optimizer_state_dict", "epoch", "config")}
    return model, extra
