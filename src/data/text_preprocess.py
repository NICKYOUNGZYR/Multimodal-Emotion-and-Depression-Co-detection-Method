from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class TextPreprocessConfig:
    max_length: int = 128
    backbone: str = "bert-base-uncased"

class TextTokenizer:

    def __init__(self, cfg: TextPreprocessConfig):
        self.cfg = cfg
        self._hf = None
        try:
            from transformers import AutoTokenizer
            self._hf = AutoTokenizer.from_pretrained(cfg.backbone)
        except Exception:
            self._hf = None

    def __call__(self, text: str) -> Dict[str, Any]:
        text = text if text is not None else ""
        if self._hf is not None:
            out = self._hf(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            )
            # squeeze batch dim
            return {k: v.squeeze(0) for k, v in out.items()}
        # fallback: whitespace token ids (very rough but reproducible)
        tokens = text.lower().split()[: self.cfg.max_length]
        ids = [min(30000, (hash(t) % 30000)) for t in tokens]
        # pad
        pad_len = self.cfg.max_length - len(ids)
        ids = ids + [0] * pad_len
        attn = [1] * (self.cfg.max_length - pad_len) + [0] * pad_len
        import torch
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
