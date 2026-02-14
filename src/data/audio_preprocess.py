from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

@dataclass
class AudioPreprocessConfig:
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 80
    max_seconds: int = 10
    frame_mask_ratio: float = 0.3

def waveform_to_spectrogram(
    wav: np.ndarray,
    cfg: AudioPreprocessConfig,
) -> torch.Tensor:
    wav_t = torch.tensor(wav, dtype=torch.float32)
    max_len = cfg.sample_rate * cfg.max_seconds
    if wav_t.numel() > max_len:
        wav_t = wav_t[:max_len]
    elif wav_t.numel() < max_len:
        pad = max_len - wav_t.numel()
        wav_t = torch.nn.functional.pad(wav_t, (0, pad))

    spec = torch.stft(
        wav_t,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.n_fft,
        return_complex=True,
    )  # (freq, time)
    mag = spec.abs()  # (freq, time)

    if mag.size(0) > cfg.n_mels:
        mag = mag[:cfg.n_mels, :]
    elif mag.size(0) < cfg.n_mels:
        mag = torch.nn.functional.pad(mag, (0, 0, 0, cfg.n_mels - mag.size(0)))

    return mag  # (n_mels, time)

def apply_frame_mask(
    spec: torch.Tensor,
    mask_ratio: float,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Frame-level random masking:
    spec: (F, T)
    return masked_spec, keep_mask (T,) where 1 means kept
    """
    F, T = spec.shape
    if T == 0:
        return spec, torch.ones((0,), dtype=torch.bool)

    num_mask = int(T * mask_ratio)
    keep = torch.ones((T,), dtype=torch.bool)
    if num_mask > 0:
        idx = torch.randperm(T, generator=generator)[:num_mask]
        keep[idx] = False
    masked = spec.clone()
    masked[:, ~keep] = 0.0
    return masked, keep
