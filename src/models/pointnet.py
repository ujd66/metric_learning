import torch
import torch.nn as nn
import torch.nn.functional as F


class _TNet(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.size(0)
        device = x.device
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        identity = torch.eye(self.k, device=device).view(1, self.k * self.k).expand(B, -1)
        x = x + identity
        x = x.view(B, self.k, self.k)
        return x


class PointNetBackbone(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.input_transform = _TNet(input_channels)
        self.feature_transform = _TNet(64)

        self.conv1 = nn.Conv1d(input_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.conv5 = nn.Conv1d(512, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, x):
        # x: [B, N, C] -> [B, C, N]
        x = x.transpose(2, 1)

        # Input transform
        t_input = self.input_transform(x)
        x = torch.bmm(x.transpose(2, 1), t_input).transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))

        # Feature transform
        t_feat = self.feature_transform(x)
        x = torch.bmm(x.transpose(2, 1), t_feat).transpose(2, 1)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Max pooling -> global feature
        x = torch.max(x, 2)[0]  # [B, 1024]
        return x
