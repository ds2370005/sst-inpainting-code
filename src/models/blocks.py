"""Reusable model blocks for the SST generators."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _pair(value: int | Sequence[int]) -> tuple[int, int]:  # ネットワーク部品: kernelやdilationを縦横2要素にそろえる
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError(f"Expected an int or a length-2 sequence, got {value!r}.")
    return int(value[0]), int(value[1])


def _reflection_padding(
    kernel_size: int | Sequence[int],
    dilation: int | Sequence[int],
) -> tuple[int, int, int, int]:  # ネットワーク部品: 畳み込みのための左右上下パディング量を計算する
    kernel_h, kernel_w = _pair(kernel_size)
    dilation_h, dilation_w = _pair(dilation)
    pad_h = dilation_h * (kernel_h - 1) // 2
    pad_w = dilation_w * (kernel_w - 1) // 2
    return pad_w, pad_w, pad_h, pad_h


class ReflectionConv2d(nn.Module):  # ネットワーク部品: 反射パディングのあとにConv2dをかける
    """ReflectionPad2d followed by Conv2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        dilation: int | Sequence[int] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.padding = _reflection_padding(kernel_size, dilation)
        self.pad = nn.ReflectionPad2d(self.padding)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"Expected [B, C, H, W], got {tuple(x.shape)}"
        return self.conv(self.pad(x))


class ConvELU(nn.Module):  # ネットワーク部品: 反射パディング付きConv2dのあとにELUを通す
    """Reflection-padded Conv2d followed by ELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        dilation: int | Sequence[int] = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ReflectionConv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
            ),
            nn.ELU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DilatedConvELU(ConvELU):  # ネットワーク部品: dilationを広げたConvELUを使う
    """Reflection-padded dilated Conv2d followed by ELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            dilation=dilation,
        )


class UpsampleConvELU(nn.Module):  # ネットワーク部品: 2倍アップサンプルしてからConvELUをかける
    """Bilinear upsampling followed by ConvELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvELU(in_channels, out_channels, kernel_size=kernel_size, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

