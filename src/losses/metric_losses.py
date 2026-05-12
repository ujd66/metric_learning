"""Metric learning losses for embedding training.

SupervisedContrastiveLoss: pushes same-class embeddings together and
different-class embeddings apart using cosine similarity in a supervised
contrastive learning framework (SupCon, Khosla et al. 2020).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss.

    Args:
        temperature: Scaling factor for cosine similarity. Lower values make
            the loss focus more on hard negatives. Default: 0.07.
        negative_label: Label id for the negative class. Samples with this
            label are excluded from the loss computation by default.
        include_negative: If True, negative-label samples participate in the
            loss. Default: False.
    """

    def __init__(self, temperature=0.07, negative_label=19, include_negative=False):
        super().__init__()
        self.temperature = temperature
        self.negative_label = negative_label
        self.include_negative = include_negative

    def forward(self, embeddings, labels):
        """Compute supervised contrastive loss.

        Args:
            embeddings: [B, D] raw embeddings (will be L2-normalized).
            labels: [B] integer class labels.

        Returns:
            Scalar loss. Returns 0.0 if no valid samples in the batch.
        """
        device = embeddings.device
        B = embeddings.size(0)
        if B < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # L2 normalize
        emb = F.normalize(embeddings, p=2, dim=1)

        # Exclude negative label samples if configured
        if not self.include_negative:
            valid_mask = labels != self.negative_label
        else:
            valid_mask = torch.ones(B, dtype=torch.bool, device=device)

        valid_indices = torch.where(valid_mask)[0]
        if len(valid_indices) < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        emb = emb[valid_indices]      # [B', D]
        lbl = labels[valid_indices]   # [B']
        Bv = emb.size(0)

        # Cosine similarity matrix (already normalized, so dot = cosine)
        sim = torch.mm(emb, emb.t())  # [B', B']

        # Mask: same label (positive pairs)
        label_eq = lbl.unsqueeze(0) == lbl.unsqueeze(1)  # [B', B']

        # Exclude self-pairs from positives
        self_mask = torch.eye(Bv, dtype=torch.bool, device=device)
        pos_mask = label_eq & ~self_mask  # [B', B']

        # Count positives per anchor
        num_positives = pos_mask.sum(dim=1)  # [B']

        # Anchors that have at least one positive
        valid_anchors = num_positives > 0
        if valid_anchors.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Scale by temperature
        sim_scaled = sim / self.temperature

        # For numerical stability, subtract max per row before exp
        logits_max, _ = sim_scaled.max(dim=1, keepdim=True)
        logits = sim_scaled - logits_max.detach()

        # Denominator: sum over all j != i (including negatives)
        # Use masked_fill to avoid in-place operation that breaks autograd
        exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
        log_denom = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Log probability of each positive pair
        log_prob = logits - log_denom  # [B', B']

        # Mean log probability over positives for each anchor
        mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / (num_positives.float() + 1e-12)

        # Loss: negative mean of log probabilities over valid anchors
        loss = -mean_log_prob_pos[valid_anchors].mean()

        # Guard against NaN
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=device, requires_grad=True)

        return loss
