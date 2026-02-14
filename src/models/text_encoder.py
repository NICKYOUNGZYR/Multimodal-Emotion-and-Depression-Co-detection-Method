from __future__ import annotations
import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    Input: input_ids (B,L), attention_mask (B,L)
    Output: text_feat (B, D)
    """
    def __init__(self, backbone: str = "bert-base-uncased", freeze: bool = True, vocab_size: int = 30522, hidden: int = 768):
        super().__init__()
        self.use_hf = False
        self.out_dim = hidden

        try:
            from transformers import AutoModel  # optional
            self.model = AutoModel.from_pretrained(backbone)
            self.use_hf = True
            self.out_dim = int(self.model.config.hidden_size)
            if freeze:
                for p in self.model.parameters():
                    p.requires_grad = False
        except Exception:
            # fallback lightweight text encoder
            self.model = nn.Embedding(vocab_size, hidden)
            self.use_hf = False
            self.out_dim = hidden

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(x.dtype).unsqueeze(-1)  # (B,L,1)
        summed = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.use_hf:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                return out.pooler_output
            return self.masked_mean(out.last_hidden_state, attention_mask)
        else:
            emb = self.model(input_ids)  # (B,L,D)
            return self.masked_mean(emb, attention_mask)
