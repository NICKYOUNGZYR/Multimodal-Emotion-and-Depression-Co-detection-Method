from __future__ import annotations
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F

from .contrastive import ceat_triplet_loss
from .distill import kl_distill_loss
from .mutual_info import symmetric_kl, mutual_information_clustering
from .most_likely import most_likely_category_loss, entropy_minimization


def source_total_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    proj: Dict[str, torch.Tensor],
    temperature: float,
    lambda_contrastive: float = 1.0,
) -> (torch.Tensor, Dict[str, float]):
    task = F.cross_entropy(logits, labels)
    c_total, c_parts = ceat_triplet_loss(proj["eeg"], proj["audio"], proj["text"], temperature)
    total = task + lambda_contrastive * c_total
    logs = {
        "loss": float(total.detach().cpu()),
        "task_ce": float(task.detach().cpu()),
        "contrastive": float(c_total.detach().cpu()),
        **{k: float(v.cpu()) for k, v in c_parts.items()},
    }
    return total, logs


def sfda_total_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    alpha_distill: float,
    beta_mi: float,
    gamma_most: float,
    distill_temperature: float,
    sharpen_temperature: float,
    margin: float,
    use_entropy_task: bool = True,
) -> (torch.Tensor, Dict[str, float]):
    dist = kl_distill_loss(student_logits, teacher_logits, temperature=distill_temperature)

    mi_agree = symmetric_kl(teacher_logits.detach(), student_logits)

    mi_cluster = mutual_information_clustering(student_logits)

    most = most_likely_category_loss(student_logits, sharpen_temperature=sharpen_temperature, margin=margin)

    task = entropy_minimization(student_logits) if use_entropy_task else torch.tensor(0.0, device=student_logits.device)

    total = task + alpha_distill * dist + beta_mi * (mi_agree + 0.5 * mi_cluster) + gamma_most * most

    logs = {
        "loss": float(total.detach().cpu()),
        "task_entropy": float(task.detach().cpu()),
        "distill_kl": float(dist.detach().cpu()),
        "mi_agree_symkl": float(mi_agree.detach().cpu()),
        "mi_cluster_neg": float(mi_cluster.detach().cpu()),
        "most_likely": float(most.detach().cpu()),
    }
    return total, logs
