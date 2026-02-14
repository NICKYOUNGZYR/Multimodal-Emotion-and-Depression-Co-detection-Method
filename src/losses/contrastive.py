from __future__ import annotations
from typing import Tuple
import torch
import torch.nn.functional as F


def info_nce_logits(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:

    return (z1 @ z2.t()) / temperature


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    B = z1.size(0)
    logits_12 = info_nce_logits(z1, z2, temperature)
    logits_21 = info_nce_logits(z2, z1, temperature)
    labels = torch.arange(B, device=z1.device)
    loss = (F.cross_entropy(logits_12, labels) + F.cross_entropy(logits_21, labels)) * 0.5
    return loss


def ceat_triplet_loss(z_eeg: torch.Tensor, z_audio: torch.Tensor, z_text: torch.Tensor, temperature: float) -> Tuple[torch.Tensor, dict]:
    le_a = contrastive_loss(z_eeg, z_audio, temperature)
    le_t = contrastive_loss(z_eeg, z_text, temperature)
    la_t = contrastive_loss(z_audio, z_text, temperature)
    total = le_a + le_t + la_t
    return total, {"c_ea": le_a.detach(), "c_et": le_t.detach(), "c_at": la_t.detach()}
