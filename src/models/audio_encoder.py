from __future__ import annotations
import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    """
    Input: audio_spec (B, F, T), audio_len (B,)
    Output: audio_feat (B, D)
    """
    def __init__(self, in_feats: int, lstm_hidden=128, lstm_layers=1, dropout=0.2):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=in_feats,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.out_dim = lstm_hidden * 2

    @staticmethod
    def _masked_mean(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        lengths: (B,)
        """
        B, T, D = x.shape
        device = x.device
        idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        mask = idx < lengths.unsqueeze(1)  # (B,T) bool
        mask_f = mask.unsqueeze(-1).to(x.dtype)
        summed = (x * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def forward(self, audio_spec: torch.Tensor, audio_len: torch.Tensor) -> torch.Tensor:
        """
        audio_spec: (B, F, T)  -> transpose to (B,T,F)
        audio_len:  (B,)
        """
        x = audio_spec.transpose(1, 2)  # (B, T, F)

        # pack for efficiency (optional but robust)
        lengths = audio_len.clamp(min=1).to("cpu")
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bilstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (B, T, 2H)


        feat = self._masked_mean(out, audio_len.to(out.device))
        return feat
