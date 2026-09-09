"""Generator architectures for SST average estimation and anomaly inpainting."""

from __future__ import annotations

import torch
from torch import nn

from src.models.blocks import ConvELU, DilatedConvELU, ReflectionConv2d


def _assert_4d_image_tensor(x: torch.Tensor, name: str) -> tuple[int, int, int, int]:  # ネットワーク定義: [B,C,H,W] の画像テンソルかを確認する
    assert x.ndim == 4, f"{name} must be [B, C, H, W], got {tuple(x.shape)}"
    batch, channels, height, width = x.shape
    assert batch > 0, f"{name} batch dimension must be positive"
    assert channels > 0, f"{name} channel dimension must be positive"
    assert height == width, f"{name} must be square for this baseline, got {height}x{width}"
    assert height % 8 == 0, f"{name} height must be divisible by 8, got {height}"
    assert width % 8 == 0, f"{name} width must be divisible by 8, got {width}"
    assert height // 8 > 16, (
        f"{name} is too small for dilation=16 with reflection padding; got {height}x{width}"
    )
    return batch, channels, height, width


def _assert_volume_tensor(
    x: torch.Tensor,
    name: str,
    expected_time_steps: int,
) -> tuple[int, int, int, int, int]:  # ネットワーク定義: [B,1,T,H,W] の時系列ボリュームかを確認する
    assert x.ndim == 5, f"{name} must be [B, C, T, H, W], got {tuple(x.shape)}"
    batch, channels, time_steps, height, width = x.shape
    assert batch > 0, f"{name} batch dimension must be positive"
    assert channels == 1, f"{name} must have one channel, got {channels}"
    assert time_steps == expected_time_steps, (
        f"{name} must have T={expected_time_steps}, got T={time_steps}"
    )
    assert height == width, f"{name} must be square for this baseline, got {height}x{width}"
    assert height % 8 == 0, f"{name} height must be divisible by 8, got {height}"
    assert width % 8 == 0, f"{name} width must be divisible by 8, got {width}"
    assert height // 8 > 16, (
        f"{name} is too small for dilation=16 with reflection padding; got {height}x{width}"
    )
    return batch, channels, time_steps, height, width


def _assert_single_time_volume(
    x: torch.Tensor,
    name: str,
    batch: int,
    height: int,
    width: int,
) -> None:  # ネットワーク定義: [B,1,1,H,W] の1時刻画像かを確認する
    expected_shape = (batch, 1, 1, height, width)
    assert x.shape == expected_shape, (
        f"{name} must have shape {expected_shape}, got {tuple(x.shape)}"
    )


class EncoderDecoder2D(nn.Module):  # ネットワーク定義: 時系列特徴を2D画像に戻す共通エンコーダデコーダ
    """Shared 2D encoder-decoder after temporal features become channels."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        assert in_channels > 0, f"in_channels must be positive, got {in_channels}"
        self.in_channels = in_channels

        self.e1 = ConvELU(in_channels, 32, kernel_size=5, stride=1)
        self.e2 = ConvELU(32, 64, kernel_size=3, stride=2)
        self.e3 = ConvELU(64, 64, kernel_size=3, stride=1)
        self.e4 = ConvELU(64, 128, kernel_size=3, stride=2)
        self.e5 = ConvELU(128, 128, kernel_size=3, stride=1)
        self.e6 = ConvELU(128, 256, kernel_size=3, stride=2)
        self.e7 = DilatedConvELU(256, 256, dilation=2)
        self.e8 = DilatedConvELU(256, 256, dilation=4)
        self.e9 = DilatedConvELU(256, 256, dilation=8)
        self.e10 = DilatedConvELU(256, 256, dilation=16)

        self.d1 = ConvELU(256, 128, kernel_size=3, stride=1)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d3 = ConvELU(128, 64, kernel_size=3, stride=1)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d5 = ConvELU(64, 32, kernel_size=3, stride=1)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d7 = ConvELU(32, 16, kernel_size=3, stride=1)
        self.d8 = ReflectionConv2d(16, 1, kernel_size=3, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = _assert_4d_image_tensor(x, "x")
        assert channels == self.in_channels, (
            f"x must have {self.in_channels} channels, got {channels}"
        )

        x = self.e1(x)
        assert x.shape == (batch, 32, height, width), tuple(x.shape)
        x = self.e2(x)
        assert x.shape == (batch, 64, height // 2, width // 2), tuple(x.shape)
        x = self.e3(x)
        assert x.shape == (batch, 64, height // 2, width // 2), tuple(x.shape)
        x = self.e4(x)
        assert x.shape == (batch, 128, height // 4, width // 4), tuple(x.shape)
        x = self.e5(x)
        assert x.shape == (batch, 128, height // 4, width // 4), tuple(x.shape)
        x = self.e6(x)
        assert x.shape == (batch, 256, height // 8, width // 8), tuple(x.shape)
        x = self.e7(x)
        assert x.shape == (batch, 256, height // 8, width // 8), tuple(x.shape)
        x = self.e8(x)
        assert x.shape == (batch, 256, height // 8, width // 8), tuple(x.shape)
        x = self.e9(x)
        assert x.shape == (batch, 256, height // 8, width // 8), tuple(x.shape)
        x = self.e10(x)
        assert x.shape == (batch, 256, height // 8, width // 8), tuple(x.shape)

        x = self.d1(x)
        assert x.shape == (batch, 128, height // 8, width // 8), tuple(x.shape)
        x = self.up1(x)
        assert x.shape == (batch, 128, height // 4, width // 4), tuple(x.shape)
        x = self.d3(x)
        assert x.shape == (batch, 64, height // 4, width // 4), tuple(x.shape)
        x = self.up2(x)
        assert x.shape == (batch, 64, height // 2, width // 2), tuple(x.shape)
        x = self.d5(x)
        assert x.shape == (batch, 32, height // 2, width // 2), tuple(x.shape)
        x = self.up3(x)
        assert x.shape == (batch, 32, height, width), tuple(x.shape)
        x = self.d7(x)
        assert x.shape == (batch, 16, height, width), tuple(x.shape)
        x = self.d8(x)
        assert x.shape == (batch, 1, height, width), tuple(x.shape)

        out = torch.tanh(x)
        assert out.shape == (batch, 1, height, width), tuple(out.shape)
        return out


class AverageEstimationGenerator(nn.Module):  # ネットワーク定義: 月平均と時系列SSTから週平均SSTを推定する
    """Estimate a cloud-free weekly average SST image."""

    def __init__(self, time_steps: int = 9) -> None:
        super().__init__()
        assert time_steps > 0, f"time_steps must be positive, got {time_steps}"
        self.time_steps = time_steps
        self.sst_projection = nn.Conv3d(2, 1, kernel_size=1)
        self.monthly_projection = nn.Conv3d(2, 1, kernel_size=1)
        self.fusion_projection = nn.Conv3d(2, 1, kernel_size=1)
        self.encoder_decoder = EncoderDecoder2D(in_channels=time_steps)

    def forward(
        self,
        sst_volume: torch.Tensor,
        mask_volume: torch.Tensor,
        monthly_average: torch.Tensor,
        monthly_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, time_steps, height, width = _assert_volume_tensor(
            sst_volume,
            "sst_volume",
            self.time_steps,
        )
        assert mask_volume.shape == sst_volume.shape, (
            f"mask_volume must match sst_volume, got {tuple(mask_volume.shape)} "
            f"and {tuple(sst_volume.shape)}"
        )
        _assert_volume_tensor(mask_volume, "mask_volume", self.time_steps)
        _assert_single_time_volume(monthly_average, "monthly_average", batch, height, width)
        _assert_single_time_volume(monthly_mask, "monthly_mask", batch, height, width)

        sst_branch_input = torch.cat([sst_volume, mask_volume], dim=1)
        assert sst_branch_input.shape == (batch, 2, time_steps, height, width)
        sst_feature = self.sst_projection(sst_branch_input)
        assert sst_feature.shape == (batch, 1, time_steps, height, width)

        monthly_branch_input = torch.cat([monthly_average, monthly_mask], dim=1)
        assert monthly_branch_input.shape == (batch, 2, 1, height, width)
        monthly_feature = self.monthly_projection(monthly_branch_input)
        assert monthly_feature.shape == (batch, 1, 1, height, width)
        monthly_feature = monthly_feature.repeat(1, 1, time_steps, 1, 1)
        assert monthly_feature.shape == (batch, 1, time_steps, height, width)

        x = torch.cat([sst_feature, monthly_feature], dim=1)
        assert x.shape == (batch, 2, time_steps, height, width)
        x = self.fusion_projection(x)
        assert x.shape == (batch, 1, time_steps, height, width)
        x = x.squeeze(1)
        assert x.shape == (batch, time_steps, height, width)

        pred_weekly = self.encoder_decoder(x)
        assert pred_weekly.shape == (batch, 1, height, width), tuple(pred_weekly.shape)
        return pred_weekly


class AnomalyInpaintingGenerator(nn.Module):  # ネットワーク定義: 週平均からの偏差としてSST異常を補完する
    """Estimate SST anomaly relative to the predicted weekly average."""

    def __init__(self, time_steps: int = 9) -> None:
        super().__init__()
        assert time_steps > 0, f"time_steps must be positive, got {time_steps}"
        self.time_steps = time_steps
        self.projection = nn.Conv3d(2, 1, kernel_size=1)
        self.encoder_decoder = EncoderDecoder2D(in_channels=time_steps)

    def forward(
        self,
        sst_volume: torch.Tensor,
        mask_volume: torch.Tensor,
        weekly_average: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, time_steps, height, width = _assert_volume_tensor(
            sst_volume,
            "sst_volume",
            self.time_steps,
        )
        assert mask_volume.shape == sst_volume.shape, (
            f"mask_volume must match sst_volume, got {tuple(mask_volume.shape)} "
            f"and {tuple(sst_volume.shape)}"
        )
        _assert_volume_tensor(mask_volume, "mask_volume", self.time_steps)
        _assert_single_time_volume(weekly_average, "weekly_average", batch, height, width)

        weekly_feature = weekly_average.repeat(1, 1, time_steps, 1, 1)
        assert weekly_feature.shape == (batch, 1, time_steps, height, width)
        anomaly_volume = (sst_volume - weekly_feature) * mask_volume
        assert anomaly_volume.shape == (batch, 1, time_steps, height, width)

        x = torch.cat([anomaly_volume, mask_volume], dim=1)
        assert x.shape == (batch, 2, time_steps, height, width)
        x = self.projection(x)
        assert x.shape == (batch, 1, time_steps, height, width)
        x = x.squeeze(1)
        assert x.shape == (batch, time_steps, height, width)

        pred_anomaly = self.encoder_decoder(x)
        assert pred_anomaly.shape == (batch, 1, height, width), tuple(pred_anomaly.shape)
        return pred_anomaly

