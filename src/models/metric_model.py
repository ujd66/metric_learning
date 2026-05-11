import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pointnet import PointNetBackbone


class MetricPointNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=20, embedding_dim=256):
        super().__init__()
        self.backbone = PointNetBackbone(input_channels)
        self.embedding_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
        )
        self.classifier_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, points):
        feat = self.backbone(points)  # [B, 1024]
        embedding = self.embedding_head(feat)  # [B, embedding_dim]
        embedding = F.normalize(embedding, p=2, dim=1)
        logits = self.classifier_head(embedding)  # [B, num_classes]
        return {
            "embedding": embedding,
            "logits": logits,
        }
