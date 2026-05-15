"""PointNet++ SSG (Single-Scale Grouping) for point cloud classification.

Simplified implementation using kNN grouping instead of ball query.
Input: [B, N, C]  ->  output: embedding [B, D] + logits [B, num_classes]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn_group(xyz, points, k):
    """kNN grouping.

    Args:
        xyz: [B, N, 3]  query positions (centroids)
        points: [B, Np, C]  all point features (or xyz if first layer)
        k: number of neighbors

    Returns:
        grouped: [B, N, k, C]  grouped features centered at each query point
    """
    # pairwise distance: [B, N, Np]
    diff = xyz.unsqueeze(2) - points[:, :, :3].unsqueeze(1)  # [B, N, 1, 3] - [B, 1, Np, 3]
    dist = (diff ** 2).sum(-1)  # [B, N, Np]

    # topk indices (smallest distance)
    _, idx = dist.topk(k, dim=-1, largest=False, sorted=False)  # [B, N, k]

    # gather
    B, N, C = points.shape
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, C)  # [B, N, k, C]
    points_expanded = points.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, Np, C]
    grouped = torch.gather(points_expanded, 2, idx_expanded)  # [B, N, k, C]

    return grouped, idx


def farthest_point_sample(xyz, npoint):
    """Farthest point sampling.

    Args:
        xyz: [B, N, 3]
        npoint: number of points to sample

    Returns:
        centroids: [B, npoint]  indices of sampled points
    """
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.ones(B, N, device=xyz.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(B, dtype=torch.long, device=xyz.device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(-1)
        distance = torch.min(distance, dist)
        farthest = distance.max(dim=-1)[1]

    return centroids


def index_points(points, idx):
    """Gather points by index.

    Args:
        points: [B, N, C]
        idx: [B, S]

    Returns:
        new_points: [B, S, C]
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


class _PointNetSetAbstraction(nn.Module):
    """Set Abstraction module for PointNet++.

    Uses kNN instead of ball query for simplicity.
    """

    def __init__(self, in_channels, mlp_channels, k=32, group_all=False):
        super().__init__()
        self.k = k
        self.group_all = group_all

        layers = []
        in_c = in_channels
        for out_c in mlp_channels:
            layers.append(nn.Conv2d(in_c, out_c, 1))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            in_c = out_c
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points):
        """
        Args:
            xyz: [B, N, 3]  point positions
            points: [B, N, C]  point features (C can be 3 for first layer)

        Returns:
            new_xyz: [B, S, 3]  sampled positions
            new_points: [B, S, D]  aggregated features
        """
        B, N, _ = xyz.shape

        if self.group_all:
            # Global abstraction: use all points, one centroid
            new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
            grouped = points.unsqueeze(1)  # [B, 1, N, C]
        else:
            # Sample centroids via FPS
            npoint = max(N // 4, 16)  # reduce by 4x
            centroid_idx = farthest_point_sample(xyz, npoint)
            new_xyz = index_points(xyz, centroid_idx)  # [B, npoint, 3]

            # kNN grouping around centroids
            # Compute distances from centroids to all points
            diff = new_xyz.unsqueeze(2) - xyz.unsqueeze(1)  # [B, npoint, N, 3]
            dist = (diff ** 2).sum(-1)  # [B, npoint, N]

            k = min(self.k, N)
            _, idx = dist.topk(k, dim=-1, largest=False, sorted=False)  # [B, npoint, k]

            # Gather grouped points
            idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, points.shape[-1])
            points_expanded = points.unsqueeze(1).expand(-1, npoint, -1, -1)
            grouped = torch.gather(points_expanded, 2, idx_expanded)  # [B, npoint, k, C]

            # Normalize positions relative to centroids
            grouped_xyz = grouped[:, :, :, :3] - new_xyz.unsqueeze(2)
            if points.shape[-1] > 3:
                grouped_feat = torch.cat([grouped_xyz, grouped[:, :, :, 3:]], dim=-1)
            else:
                grouped_feat = grouped_xyz

            grouped = grouped_feat

        # MLP on grouped points: [B, npoint, k, C] -> conv over k dim
        grouped = grouped.permute(0, 3, 1, 2)  # [B, C, npoint, k]
        new_points = self.mlp(grouped)  # [B, D, npoint, k]
        new_points = new_points.max(dim=-1)[0]  # [B, D, npoint]
        new_points = new_points.permute(0, 2, 1)  # [B, npoint, D]

        return new_xyz, new_points


class PointNet2Backbone(nn.Module):
    """PointNet++ SSG backbone.

    3 set abstraction levels + global feature.
    """

    def __init__(self, input_channels=3):
        super().__init__()
        # SA1: input (N, C) -> output (N/4, 128)
        # Input channels: 3 (xyz) + (input_channels - 3) extra features
        extra = max(input_channels - 3, 0)
        self.sa1 = _PointNetSetAbstraction(
            in_channels=3 + extra,
            mlp_channels=[64, 64, 128],
            k=32,
        )
        # SA2: input (N/4, 128+3) -> output (N/16, 256)
        self.sa2 = _PointNetSetAbstraction(
            in_channels=128 + 3,
            mlp_channels=[128, 128, 256],
            k=32,
        )
        # SA3: input (N/16, 256+3) -> output (1, 1024) — global
        self.sa3 = _PointNetSetAbstraction(
            in_channels=256 + 3,
            mlp_channels=[256, 512, 1024],
            k=32,
            group_all=True,
        )

    def forward(self, x):
        """
        Args:
            x: [B, N, C]

        Returns:
            global_feat: [B, 1024]
        """
        xyz = x[:, :, :3]  # [B, N, 3]
        points = x if x.shape[-1] > 3 else xyz

        xyz, points = self.sa1(xyz, points)
        points = torch.cat([xyz, points], dim=-1)

        xyz, points = self.sa2(xyz, points)
        points = torch.cat([xyz, points], dim=-1)

        _, global_feat = self.sa3(xyz, points)  # [B, 1, 1024]
        global_feat = global_feat.squeeze(1)  # [B, 1024]

        return global_feat


class MetricPointNet2(nn.Module):
    """PointNet++ SSG model with embedding head and classifier."""

    def __init__(self, input_channels=3, num_classes=20, embedding_dim=256):
        super().__init__()
        self.backbone = PointNet2Backbone(input_channels)
        self.embedding_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
        )
        self.classifier_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, points):
        """
        Args:
            points: [B, N, C]

        Returns:
            dict with "embedding" [B, embedding_dim] and "logits" [B, num_classes]
        """
        feat = self.backbone(points)  # [B, 1024]
        embedding = self.embedding_head(feat)  # [B, embedding_dim]
        embedding = F.normalize(embedding, p=2, dim=1)
        logits = self.classifier_head(embedding)  # [B, num_classes]
        return {
            "embedding": embedding,
            "logits": logits,
        }
