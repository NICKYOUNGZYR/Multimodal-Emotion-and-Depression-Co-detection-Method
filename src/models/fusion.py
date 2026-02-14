from __future__ import annotations
import torch


def concat_fusion(*features: torch.Tensor) -> torch.Tensor:
    feats = [f for f in features if f is not None]
    return torch.cat(feats, dim=-1)
