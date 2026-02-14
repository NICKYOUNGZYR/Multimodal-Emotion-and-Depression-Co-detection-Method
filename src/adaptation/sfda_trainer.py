from __future__ import annotations
from typing import Dict, Any
import torch
from torch.optim import Optimizer

from src.losses.total_loss import sfda_total_loss
from .freeze_thaw import FreezeThawController


class SFDATrainer:
    def __init__(
        self,
        student,
        teacher,
        optimizer: Optimizer,
        device: torch.device,
        sfda_cfg: Dict[str, Any],
    ):
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.device = device

        self.alpha = float(sfda_cfg["sfda"]["alpha_distill"])
        self.beta = float(sfda_cfg["sfda"]["beta_mutual_info"])
        self.gamma = float(sfda_cfg["sfda"]["gamma_most_likely"])
        self.Td = float(sfda_cfg["sfda"]["distill_temperature"])

        most_cfg = sfda_cfg.get("most_likely", {})
        self.Ts = float(most_cfg.get("sharpen_temperature", 0.5))
        self.margin = float(most_cfg.get("margin", 0.2))
        self.use_entropy_task = bool(most_cfg.get("use_entropy_minimization_as_task", True))

        ft = sfda_cfg.get("freeze_thaw", {})
        self.freeze_thaw = FreezeThawController(
            self.student,
            stage0_epochs=int(ft.get("stage0_epochs", 2)),
            stage1_unfreeze_every=int(ft.get("stage1_unfreeze_every", 1)),
        )

        # teacher frozen
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def train_one_epoch(self, loader, epoch: int) -> Dict[str, float]:
        self.student.train()
        self.freeze_thaw.apply(epoch)

        meter = {}
        n = 0

        for batch in loader:
            batch = self._to_device(batch)
            with torch.no_grad():
                t_out = self.teacher(batch)
                teacher_logits = t_out.logits

            s_out = self.student(batch)
            student_logits = s_out.logits

            loss, logs = sfda_total_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                alpha_distill=self.alpha,
                beta_mi=self.beta,
                gamma_most=self.gamma,
                distill_temperature=self.Td,
                sharpen_temperature=self.Ts,
                margin=self.margin,
                use_entropy_task=self.use_entropy_task,
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 5.0)
            self.optimizer.step()

            # accumulate
            for k, v in logs.items():
                meter[k] = meter.get(k, 0.0) + float(v)
            n += 1

        for k in list(meter.keys()):
            meter[k] /= max(1, n)
        return meter

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch["eeg_grid"] = batch["eeg_grid"].to(self.device, non_blocking=True)
        batch["audio_spec"] = batch["audio_spec"].to(self.device, non_blocking=True)
        batch["audio_len"] = batch["audio_len"].to(self.device, non_blocking=True)
        batch["label"] = batch["label"].to(self.device, non_blocking=True)
        batch["is_labeled"] = batch["is_labeled"].to(self.device, non_blocking=True)
        batch["subject_id"] = batch["subject_id"].to(self.device, non_blocking=True)
        # text dict
        for k in batch["text"]:
            batch["text"][k] = batch["text"][k].to(self.device, non_blocking=True)
        return batch
