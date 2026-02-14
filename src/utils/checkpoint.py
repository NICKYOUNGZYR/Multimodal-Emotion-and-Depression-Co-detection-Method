from __future__ import annotations
import os
from typing import Any, Dict, Optional
import torch


def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def save_model(path: str, model: torch.nn.Module, extra: Optional[Dict[str, Any]] = None) -> None:
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    save_checkpoint(path, payload)


def load_model_weights(model: torch.nn.Module, ckpt_path: str, map_location: str = "cpu") -> Dict[str, Any]:
    ckpt = load_checkpoint(ckpt_path, map_location=map_location)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    return ckpt
