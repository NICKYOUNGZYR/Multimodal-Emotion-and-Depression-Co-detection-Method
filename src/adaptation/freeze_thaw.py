from __future__ import annotations
from typing import List
import torch.nn as nn


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


class FreezeThawController:
    """
    - stage0: freeze encoders, train proj + head
    - stage1: unfreeze some modules gradually
    """
    def __init__(self, model: nn.Module, stage0_epochs: int = 2, stage1_unfreeze_every: int = 1):
        self.model = model
        self.stage0_epochs = stage0_epochs
        self.stage1_unfreeze_every = stage1_unfreeze_every

    def apply(self, epoch: int) -> None:
        # stage0
        if epoch < self.stage0_epochs:
            for name, m in self.model.named_children():
                if name in ("eeg_encoder", "audio_encoder", "text_encoder"):
                    set_requires_grad(m, False)
                else:
                    set_requires_grad(m, True)
            return

        # stage1+: unfreeze all gradually (simple baseline)
        # every N epochs after stage0, unfreeze encoders
        for name, m in self.model.named_children():
            set_requires_grad(m, True)
