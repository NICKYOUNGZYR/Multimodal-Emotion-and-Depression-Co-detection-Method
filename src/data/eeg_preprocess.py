from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

@dataclass
class EEGPreprocessConfig:
    sample_rate: int = 200
    window_sec: float = 0.5
    bands: List[Tuple[float, float]] = None
    grid_h: int = 9
    grid_w: int = 9

def _de_from_signal(x: np.ndarray, eps: float = 1e-8) -> float:
    """
    Differential Entropy (DE) for approx Gaussian:
        DE = 0.5 * ln(2*pi*e*sigma^2)
    x: (T,)
    """
    var = float(np.var(x) + eps)
    return 0.5 * math.log(2.0 * math.pi * math.e * var)

def compute_de_features(
    eeg: np.ndarray,
    cfg: EEGPreprocessConfig,
) -> np.ndarray:
    """
    Compute DE features per channel per band per window.
    eeg: (C, T)
    return: (num_windows, num_bands, C)
    """
    if cfg.bands is None:
        cfg.bands = [(1, 4), (4, 8), (8, 14), (14, 31)]

    C, T = eeg.shape
    win = int(cfg.sample_rate * cfg.window_sec)
    num_windows = max(1, T // win)

    out = np.zeros((num_windows, len(cfg.bands), C), dtype=np.float32)
    for w in range(num_windows):
        seg = eeg[:, w * win:(w + 1) * win]  # (C, win)
        for b in range(len(cfg.bands)):
            for c in range(C):
                out[w, b, c] = _de_from_signal(seg[c])
    return out

def map_channels_to_grid(
    de_band_chan: np.ndarray,
    grid_h: int,
    grid_w: int,
    channel_map: Optional[List[Tuple[int,int]]] = None,
) -> np.ndarray:
    """
    Map channels onto a 2D grid to enable 2D-CNN.
    de_band_chan: (..., C)
    return: (..., grid_h, grid_w)
    """
    *prefix, C = de_band_chan.shape
    grid = np.zeros((*prefix, grid_h, grid_w), dtype=np.float32)

    if channel_map is None:
        coords = []
        idx = 0
        for r in range(grid_h):
            for c in range(grid_w):
                coords.append((r, c))
        channel_map = coords[:C]

    for ch, (r, c) in enumerate(channel_map):
        if 0 <= r < grid_h and 0 <= c < grid_w:
            grid[..., r, c] = de_band_chan[..., ch]
    return grid
