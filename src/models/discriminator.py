"""Spectral-normalized discriminator for SST image patches."""

from __future__ import annotations

import torch
from torch import nn


def _spectral_conv2d(
    in_channels: int,
    out_channels: int,
    stride: int,
) -> nn.Module:  # ネットワーク定義: spectral norm付きの5x5畳み込み層を返す
    return nn.utils.spectral_norm(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=5,
            stride=stride,
            padding=2,
        )
    )


class Discriminator(nn.Module):  # ネットワーク定義: SST画像パッチを本物か偽物か判定する
    """Patch discriminator that returns one logit per SST image."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        assert in_channels > 0, f"in_channels must be positive, got {in_channels}"
        self.in_channels = in_channels

        self.d1 = _spectral_conv2d(in_channels, 64, stride=2)
        self.d2 = _spectral_conv2d(64, 128, stride=2)
        self.d3 = _spectral_conv2d(128, 256, stride=2)
        self.d4 = _spectral_conv2d(256, 256, stride=2)
        self.d5 = _spectral_conv2d(256, 256, stride=2)
        self.d6 = _spectral_conv2d(256, 1, stride=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"x must be [B, C, H, W], got {tuple(x.shape)}"
        batch, channels, height, width = x.shape
        assert batch > 0, "x batch dimension must be positive"
        assert channels == self.in_channels, (
            f"x must have {self.in_channels} channels, got {channels}"
        )
        assert height == width, f"x must be square for this baseline, got {height}x{width}"
        assert height % 32 == 0, f"x height must be divisible by 32, got {height}"
        assert width % 32 == 0, f"x width must be divisible by 32, got {width}"

        x = self.activation(self.d1(x))
        assert x.shape == (batch, 64, height // 2, width // 2), tuple(x.shape)
        x = self.activation(self.d2(x))
        assert x.shape == (batch, 128, height // 4, width // 4), tuple(x.shape)
        x = self.activation(self.d3(x))
        assert x.shape == (batch, 256, height // 8, width // 8), tuple(x.shape)
        x = self.activation(self.d4(x))
        assert x.shape == (batch, 256, height // 16, width // 16), tuple(x.shape)
        x = self.activation(self.d5(x))
        assert x.shape == (batch, 256, height // 32, width // 32), tuple(x.shape)
        x = self.d6(x)
        assert x.shape == (batch, 1, height // 32, width // 32), tuple(x.shape)

        logits = x.mean(dim=(2, 3))
        assert logits.shape == (batch, 1), tuple(logits.shape)
        return logits
