from __future__ import annotations
import torch
import torch.nn.functional as F


def kl_distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    KL || softmax * T^2
    """
    T = float(temperature)
    p_t = F.softmax(teacher_logits / T, dim=-1)
    log_p_s = F.log_softmax(student_logits / T, dim=-1)
    loss = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
    return loss
