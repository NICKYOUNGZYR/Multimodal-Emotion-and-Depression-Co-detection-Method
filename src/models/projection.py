from __future__ import annotations
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    Map modality feature -> shared contrastive space.
    """
    def __init__(self, in_dim: int, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
