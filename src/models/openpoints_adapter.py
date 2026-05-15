"""OpenPoints / PointNeXt optional adapter.

Provides a thin wrapper around OpenPoints models so they plug into the
unified build_model() factory.  OpenPoints is an *optional* dependency —
importing this module never crashes; only actually selecting
backbone="openpoints" or "pointnext" will trigger the import check.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


_OPENPOINTS_AVAILABLE = None  # cached check result


def is_openpoints_available():
    """Return True if the openpoints package is importable."""
    global _OPENPOINTS_AVAILABLE
    if _OPENPOINTS_AVAILABLE is None:
        try:
            import openpoints  # noqa: F401
            _OPENPOINTS_AVAILABLE = True
        except ImportError:
            _OPENPOINTS_AVAILABLE = False
    return _OPENPOINTS_AVAILABLE


def _build_openpoints_backbone(cfg):
    """Build an OpenPoints backbone from config.

    Supports backbone sub-types via cfg["model"]["openpoints_cfg"] dict.
    Falls back to a default PointNeXt-S configuration.
    """
    if not is_openpoints_available():
        raise RuntimeError(
            "OpenPoints is not installed. Use backbone=pointnet or backbone=pointnet2, "
            "or install OpenPoints:  pip install openpoints"
        )

    from openpoints.models import build_model_from_cfg
    from openpoints.config import get_cfg

    op_cfg = cfg.get("model", {}).get("openpoints_cfg", {})
    model_type = op_cfg.get("type", "pointnext_s")

    # Default PointNeXt-S config
    defaults = {
        "pointnext_s": dict(
            model_name="PointNeXt",
            encoder_args=dict(
                enc_name="PointNeXt-S",
                in_channels=cfg.get("input_channels", 3),
            ),
            cls_args=dict(
                num_classes=cfg.get("num_classes", 20),
            ),
        ),
        "pointnext_b": dict(
            model_name="PointNeXt",
            encoder_args=dict(
                enc_name="PointNeXt-B",
                in_channels=cfg.get("input_channels", 3),
            ),
            cls_args=dict(
                num_classes=cfg.get("num_classes", 20),
            ),
        ),
        "pointnext_l": dict(
            model_name="PointNeXt",
            encoder_args=dict(
                enc_name="PointNeXt-L",
                in_channels=cfg.get("input_channels", 3),
            ),
            cls_args=dict(
                num_classes=cfg.get("num_classes", 20),
            ),
        ),
    }

    model_cfg_dict = defaults.get(model_type, defaults["pointnext_s"])
    # Merge user overrides
    if op_cfg:
        import copy
        model_cfg_dict = copy.deepcopy(model_cfg_dict)
        for k, v in op_cfg.items():
            if k != "type":
                model_cfg_dict[k] = v

    backbone_model = build_model_from_cfg(model_cfg_dict)
    return backbone_model


class OpenPointsAdapter(nn.Module):
    """Adapter that wraps an OpenPoints model to produce unified output.

    Forward returns {"embedding": Tensor, "logits": Tensor}.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = _build_openpoints_backbone(cfg)
        embedding_dim = cfg.get("embedding_dim", 256)
        num_classes = cfg.get("num_classes", 20)

        # Detect the internal feature dim from the OpenPoints model
        # Most OpenPoints encoders output 256-dim features for -S variant
        internal_dim = getattr(self.backbone, "encoder_dim", 256)

        self.embedding_head = nn.Sequential(
            nn.Linear(internal_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
        )
        self.classifier_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, points):
        """
        Args:
            points: [B, N, C] — point cloud batch

        Returns:
            dict with "embedding" [B, embedding_dim] and "logits" [B, num_classes]
        """
        # OpenPoints expects [B, C, N] or dict depending on version
        # Try the most common interface first
        try:
            feat = self.backbone(points)  # [B, internal_dim]
        except Exception:
            # Fallback: transpose to [B, C, N]
            feat = self.backbone(points.transpose(1, 2))

        if isinstance(feat, dict):
            feat = feat.get("feat", feat.get("features", None))
            if feat is None:
                raise ValueError("OpenPoints model returned unexpected dict format")

        if feat.dim() == 3:
            feat = feat.mean(dim=2)  # pool if needed

        embedding = self.embedding_head(feat)
        embedding = F.normalize(embedding, p=2, dim=1)
        logits = self.classifier_head(embedding)
        return {
            "embedding": embedding,
            "logits": logits,
        }
