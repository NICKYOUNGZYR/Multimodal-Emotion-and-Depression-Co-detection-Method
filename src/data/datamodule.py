from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader

from .datasets import MultiModalEEGDataset, DatasetConfig
from .audio_preprocess import AudioPreprocessConfig
from .text_preprocess import TextPreprocessConfig

def multimodal_collate(batch):
    """
    Collate with variable-length audio_spec time dim.
    """
    eeg = torch.stack([b["eeg_grid"] for b in batch], dim=0)

    # audio pad
    specs = [b["audio_spec"] for b in batch]   # (F, T)
    F = specs[0].shape[0]
    T_max = max(s.shape[1] for s in specs)
    audio = torch.zeros((len(batch), F, T_max), dtype=torch.float32)
    audio_len = torch.zeros((len(batch),), dtype=torch.long)
    for i, s in enumerate(specs):
        audio[i, :, :s.shape[1]] = s
        audio_len[i] = s.shape[1]

    # text
    text_keys = batch[0]["text"].keys()
    text = {k: torch.stack([b["text"][k] for b in batch], dim=0) for k in text_keys}

    labels = torch.stack([b["label"] for b in batch], dim=0)
    is_labeled = torch.stack([b["is_labeled"] for b in batch], dim=0)
    subject_id = torch.stack([b["subject_id"] for b in batch], dim=0)

    return {
        "eeg_grid": eeg,
        "audio_spec": audio,
        "audio_len": audio_len,
        "text": text,
        "label": labels,
        "is_labeled": is_labeled,
        "subject_id": subject_id,
    }

@dataclass
class DataModuleConfig:
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 42

class MultiModalDataModule:
    def __init__(
        self,
        dcfg: DatasetConfig,
        acfg: AudioPreprocessConfig,
        tcfg: TextPreprocessConfig,
        mcfg: DataModuleConfig,
    ):
        self.dcfg = dcfg
        self.acfg = acfg
        self.tcfg = tcfg
        self.mcfg = mcfg

        self.ds_source_train = None
        self.ds_source_val = None
        self.ds_target_unlabeled = None
        self.ds_test = None

    def setup(self):
        self.ds_source_train = MultiModalEEGDataset(self.dcfg, self.acfg, self.tcfg, mode="source_train", seed=self.mcfg.seed)
        self.ds_source_val   = MultiModalEEGDataset(self.dcfg, self.acfg, self.tcfg, mode="source_val", seed=self.mcfg.seed + 1)
        self.ds_target_unlabeled = MultiModalEEGDataset(self.dcfg, self.acfg, self.tcfg, mode="target_unlabeled", seed=self.mcfg.seed + 2)
        self.ds_test         = MultiModalEEGDataset(self.dcfg, self.acfg, self.tcfg, mode="test", seed=self.mcfg.seed + 3)

    def train_loader(self):
        return DataLoader(
            self.ds_source_train,
            batch_size=self.mcfg.batch_size,
            shuffle=True,
            num_workers=self.mcfg.num_workers,
            collate_fn=multimodal_collate,
            pin_memory=True,
        )

    def val_loader(self):
        return DataLoader(
            self.ds_source_val,
            batch_size=self.mcfg.batch_size,
            shuffle=False,
            num_workers=self.mcfg.num_workers,
            collate_fn=multimodal_collate,
            pin_memory=True,
        )

    def target_loader(self):
        return DataLoader(
            self.ds_target_unlabeled,
            batch_size=self.mcfg.batch_size,
            shuffle=True,
            num_workers=self.mcfg.num_workers,
            collate_fn=multimodal_collate,
            pin_memory=True,
        )

    def test_loader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.mcfg.batch_size,
            shuffle=False,
            num_workers=self.mcfg.num_workers,
            collate_fn=multimodal_collate,
            pin_memory=True,
        )
