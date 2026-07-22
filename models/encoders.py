"""Shared image encoders for glioma models."""

import torch.nn as nn


class ROI2DEncoder(nn.Module):
    def __init__(self, in_ch: int = 7, feat_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(192, feat_dim),
            nn.LayerNorm(feat_dim),
        )

    def forward(self, x):
        return self.net(x)

    def forward_with_tokens(self, x):
        """Return the legacy pooled feature and pre-pooling spatial tokens."""
        value = x
        tokens = None
        for index, layer in enumerate(self.net):
            value = layer(value)
            if index == 14:
                tokens = value.flatten(start_dim=2).transpose(1, 2)
        return value, tokens


__all__ = ["ROI2DEncoder"]
