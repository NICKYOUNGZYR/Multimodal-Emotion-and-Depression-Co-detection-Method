from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .eeg_preprocess import EEGPreprocessConfig, compute_de_features, map_channels_to_grid
from .audio_preprocess import AudioPreprocessConfig, waveform_to_spectrogram
from .text_preprocess import TextTokenizer, TextPreprocessConfig

@dataclass
class DatasetConfig:
    name: str
    root: str
    split: str = "train_test"
    num_classes: int = 3
    held_out_subject: int = 1
    window_sec: float = 0.5
    sample_rate: int = 200
    bands: List[Tuple[float, float]] = None
    eeg_grid_h: int = 9
    eeg_grid_w: int = 9
    use_synthetic_if_missing: bool = True

class MultiModalSample:
    """
    Standardized sample container
    """
    pass

class MultiModalEEGDataset(Dataset):
    """
    dataset skeleton.
    """
    def __init__(
        self,
        dcfg: DatasetConfig,
        acfg: AudioPreprocessConfig,
        tcfg: TextPreprocessConfig,
        mode: str,  # "source_train" | "source_val" | "target_unlabeled" | "test"
        seed: int = 42,
    ):
        self.dcfg = dcfg
        self.acfg = acfg
        self.tcfg = tcfg
        self.mode = mode
        self.rng = np.random.RandomState(seed)

        self.eeg_cfg = EEGPreprocessConfig(
            sample_rate=dcfg.sample_rate,
            window_sec=dcfg.window_sec,
            bands=dcfg.bands,
            grid_h=dcfg.eeg_grid_h,
            grid_w=dcfg.eeg_grid_w,
        )
        self.tokenizer = TextTokenizer(tcfg)

        self.meta = self._load_or_build_meta()

    def _load_or_build_meta(self) -> List[Dict[str, Any]]:
        meta_path = os.path.join(self.dcfg.root, "meta.npy")
        if os.path.exists(meta_path):
            meta = np.load(meta_path, allow_pickle=True).tolist()
            return meta

        if not self.dcfg.use_synthetic_if_missing:
            raise FileNotFoundError(f"meta.npy not found at {meta_path}")

        # synthetic meta
        n = 200 if "train" in self.mode else 60
        meta = []
        for i in range(n):
            meta.append({
                "eeg_path": None,
                "audio_path": None,
                "text": f"synthetic sample {i}",
                "label": int(self.rng.randint(0, self.dcfg.num_classes)),
                "subject_id": int(self.rng.randint(1, 16)),
            })
        return meta

    def _apply_split_filter(self, item: Dict[str, Any]) -> bool:
        if self.dcfg.split != "loso":
            return True
        # LOSO: held_out_subject is test subject; others are train
        sid = int(item.get("subject_id", -1))
        if self.mode in ("test", "source_val"):
            return sid == int(self.dcfg.held_out_subject)
        return sid != int(self.dcfg.held_out_subject)

    def __len__(self) -> int:
        return sum(1 for m in self.meta if self._apply_split_filter(m))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # map idx over filtered meta
        filtered = [m for m in self.meta if self._apply_split_filter(m)]
        m = filtered[idx]

        eeg = self._load_eeg(m.get("eeg_path"))
        wav = self._load_audio(m.get("audio_path"))
        text = m.get("text", "")
        label = m.get("label", -1)

        # EEG -> DE -> grid
        de = compute_de_features(eeg, self.eeg_cfg)        # (W, B, C)
        grid = map_channels_to_grid(
            de, self.dcfg.eeg_grid_h, self.dcfg.eeg_grid_w
        )  # (W, B, H, W)

        # Audio -> spec
        spec = waveform_to_spectrogram(wav, self.acfg)     # (F, T)

        # Text -> ids/mask
        toks = self.tokenizer(text)

        out = {
            "eeg_grid": torch.tensor(grid, dtype=torch.float32),   # (Wn, B, H, W)
            "audio_spec": spec.to(torch.float32),                  # (F, T)
            "text": toks,                                          # dict
            "label": torch.tensor(label, dtype=torch.long),
            "subject_id": torch.tensor(int(m.get("subject_id", -1)), dtype=torch.long),
            "is_labeled": torch.tensor(label >= 0, dtype=torch.bool),
        }

        # Target-unlabeled mode: mask labels
        if self.mode == "target_unlabeled":
            out["label"] = torch.tensor(-1, dtype=torch.long)
            out["is_labeled"] = torch.tensor(False, dtype=torch.bool)

        return out

    def _load_eeg(self, path: Optional[str]) -> np.ndarray:
        if path and os.path.exists(path):
            eeg = np.load(path).astype(np.float32)  # (C,T)
            return eeg
        # synthetic: 62 channels, 10 seconds
        C = 62
        T = int(self.dcfg.sample_rate * 10)
        return self.rng.randn(C, T).astype(np.float32)

    def _load_audio(self, path: Optional[str]) -> np.ndarray:
        if path and os.path.exists(path):
            wav = np.load(path).astype(np.float32)
            return wav
        # synthetic: max_seconds waveform
        L = int(self.acfg.sample_rate * self.acfg.max_seconds)
        return self.rng.randn(L).astype(np.float32)
