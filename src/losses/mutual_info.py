from __future__ import annotations
import torch
import torch.nn.functional as F


def symmetric_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    log_p = torch.log(p.clamp_min(1e-8))
    log_q = torch.log(q.clamp_min(1e-8))
    kl_pq = (p * (log_p - log_q)).sum(dim=-1).mean()
    kl_qp = (q * (log_q - log_p)).sum(dim=-1).mean()
    return 0.5 * (kl_pq + kl_qp)


def mutual_information_clustering(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=-1)
    p_mean = p.mean(dim=0)
    H_mean = -(p_mean * torch.log(p_mean.clamp_min(1e-8))).sum()
    H_each = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=-1).mean()
    mi = H_mean - H_each
    return -mi  # minimize negative MI
