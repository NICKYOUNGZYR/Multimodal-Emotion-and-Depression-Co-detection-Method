from __future__ import annotations
import torch
import torch.nn as nn


class EEGEncoder(nn.Module):
    def __init__(
        self,
        in_bands: int,
        cnn_channels=(32, 64, 128),
        lstm_hidden=128,
        lstm_layers=1,
        dropout=0.2,
    ):
        super().__init__()
        c1, c2, c3 = cnn_channels

        self.cnn = nn.Sequential(
            nn.Conv2d(in_bands, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.lstm = nn.LSTM(
            input_size=c3,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.out_dim = lstm_hidden

    def forward(self, eeg_grid: torch.Tensor) -> torch.Tensor:

        B, Wn, Band, H, W = eeg_grid.shape
        x = eeg_grid.reshape(B * Wn, Band, H, W)              # (B*Wn, Band, H, W)
        x = self.cnn(x).flatten(1)                            # (B*Wn, C3)
        x = x.reshape(B, Wn, -1)                              # (B, Wn, C3)

        out, (h, c) = self.lstm(x)                            # h: (layers, B, hidden)
        feat = h[-1]                                          # (B, hidden)
        return feat
