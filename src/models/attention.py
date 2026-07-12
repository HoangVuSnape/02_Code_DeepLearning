# src/models/attention.py
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """SE-Block: v = GAP(s * F), s = sigmoid(W2 ReLU(W1 GAP(F)))."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid), nn.ReLU(),
            nn.Linear(mid, channels), nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        weights = self.fc(x.mean(dim=[2, 3])).view(b, c, 1, 1)
        return x * weights


class TemporalAttention(nn.Module):
    """v = sum_t alpha_t h_t, alpha = softmax(w.h_t) voi mask PAD."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.score_fn = nn.Linear(hidden_dim, 1)

    def forward(self, seq_outputs, pad_mask=None):
        scores = self.score_fn(seq_outputs).squeeze(-1)            # [B, T]
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        context = (seq_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return context, weights


class GatedAttention(nn.Module):
    """z' = z * sigmoid(Wz) — cong loc thong tin sau fusion."""

    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(self, x):
        return x * self.gate(x)
