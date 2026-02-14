from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .eeg_encoder import EEGEncoder
from .audio_encoder import AudioEncoder
from .text_encoder import TextEncoder
from .projection import ProjectionHead
from .fusion import concat_fusion
from .classifier import ClassifierHead


@dataclass
class ModelOutputs:
    logits: torch.Tensor
    fused_feat: torch.Tensor
    proj: Dict[str, torch.Tensor]   # projected embeddings for contrastive learning
    feats: Dict[str, torch.Tensor]  # raw modality features


class MultiModalModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        mc = cfg["model_cfg"]["model"] if "model_cfg" in cfg else cfg.get("model", cfg)
        # support both merged cfg styles
        model_cfg = cfg.get("model_cfg", {}).get("model", cfg.get("model", {}))
        enc_cfg = cfg.get("model_cfg", {}).get("encoders", cfg.get("encoders", {}))
        contrast_cfg = cfg.get("model_cfg", {}).get("contrastive", cfg.get("contrastive", {}))
        fusion_cfg = cfg.get("model_cfg", {}).get("fusion", cfg.get("fusion", {}))

        # dataset params (for bands/classes)
        dset = cfg.get("dataset", {})
        num_classes = int(dset.get("num_classes", cfg.get("num_classes", 3)))
        in_bands = len(dset.get("bands", [(1,4),(4,8),(8,14),(14,31)]))

        # Audio input bins
        audio_cfg = cfg.get("audio", {})
        n_mels = int(audio_cfg.get("n_mels", 80))

        # Text backbone
        text_cfg = enc_cfg.get("text", {})
        backbone = text_cfg.get("backbone", cfg.get("text", {}).get("backbone", "bert-base-uncased"))
        freeze_text = bool(text_cfg.get("freeze", True))

        # encoder hyperparams
        eeg_h = enc_cfg.get("eeg", {})
        aud_h = enc_cfg.get("audio", {})

        self.eeg_encoder = EEGEncoder(
            in_bands=in_bands,
            cnn_channels=tuple(eeg_h.get("cnn_channels", [32, 64, 128])),
            lstm_hidden=int(eeg_h.get("lstm_hidden", 128)),
            lstm_layers=int(eeg_h.get("lstm_layers", 1)),
            dropout=float(eeg_h.get("dropout", 0.2)),
        )
        self.audio_encoder = AudioEncoder(
            in_feats=n_mels,
            lstm_hidden=int(aud_h.get("lstm_hidden", 128)),
            lstm_layers=int(aud_h.get("lstm_layers", 1)),
            dropout=float(aud_h.get("dropout", 0.2)),
        )
        self.text_encoder = TextEncoder(backbone=backbone, freeze=freeze_text)

        shared_dim = int(cfg.get("model_cfg", {}).get("dims", {}).get("shared", cfg.get("dims", {}).get("shared", 256)))
        # Projection heads
        self.proj_eeg = ProjectionHead(self.eeg_encoder.out_dim, shared_dim)
        self.proj_audio = ProjectionHead(self.audio_encoder.out_dim, shared_dim)
        self.proj_text = ProjectionHead(self.text_encoder.out_dim, shared_dim)

        # Fusion + classifier
        fusion_type = fusion_cfg.get("type", "concat")
        assert fusion_type == "concat", "This skeleton supports concat fusion first."
        fused_dim = self.eeg_encoder.out_dim + self.audio_encoder.out_dim + self.text_encoder.out_dim
        self.classifier = ClassifierHead(fused_dim, num_classes=num_classes)

        # temperature for CEAT
        self.tau = float(contrast_cfg.get("temperature", 0.07))

    def forward_embeddings(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        eeg_feat = self.eeg_encoder(batch["eeg_grid"])
        aud_feat = self.audio_encoder(batch["audio_spec"], batch["audio_len"])
        txt = batch["text"]
        txt_feat = self.text_encoder(txt["input_ids"], txt["attention_mask"])

        z_e = F.normalize(self.proj_eeg(eeg_feat), dim=-1)
        z_a = F.normalize(self.proj_audio(aud_feat), dim=-1)
        z_t = F.normalize(self.proj_text(txt_feat), dim=-1)

        return {"eeg": z_e, "audio": z_a, "text": z_t}

    def forward_logits(self, batch: Dict[str, Any]) -> ModelOutputs:
        eeg_feat = self.eeg_encoder(batch["eeg_grid"])
        aud_feat = self.audio_encoder(batch["audio_spec"], batch["audio_len"])
        txt = batch["text"]
        txt_feat = self.text_encoder(txt["input_ids"], txt["attention_mask"])

        fused = concat_fusion(eeg_feat, aud_feat, txt_feat)
        logits = self.classifier(fused)

        proj = {
            "eeg": F.normalize(self.proj_eeg(eeg_feat), dim=-1),
            "audio": F.normalize(self.proj_audio(aud_feat), dim=-1),
            "text": F.normalize(self.proj_text(txt_feat), dim=-1),
        }
        feats = {"eeg": eeg_feat, "audio": aud_feat, "text": txt_feat}
        return ModelOutputs(logits=logits, fused_feat=fused, proj=proj, feats=feats)

    def forward(self, batch: Dict[str, Any]) -> ModelOutputs:
        return self.forward_logits(batch)
