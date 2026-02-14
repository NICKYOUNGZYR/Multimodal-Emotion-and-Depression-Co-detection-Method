from __future__ import annotations
import torch
import torch.nn.functional as F


def entropy_minimization(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=-1)
    ent = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=-1).mean()
    return ent


def most_likely_category_loss(
    logits: torch.Tensor,
    sharpen_temperature: float = 0.5,
    margin: float = 0.2,
) -> torch.Tensor:

    Ts = float(sharpen_temperature)
    with torch.no_grad():
        p_sharp = F.softmax(logits / Ts, dim=-1)

    log_p = F.log_softmax(logits, dim=-1)
    ce = -(p_sharp * log_p).sum(dim=-1).mean()

    top2 = torch.topk(logits, k=2, dim=-1).values
    gap = top2[:, 0] - top2[:, 1]
    margin_loss = F.relu(margin - gap).mean()

    return ce + margin_loss
