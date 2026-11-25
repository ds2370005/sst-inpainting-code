# 旧実装・旧テスト: 月平均とSST時系列から欠損のない週平均を推定する旧生成器を定義する。
"""旧方式の学習型週平均推定器。"""
# ── 使用するライブラリ・共通処理 ──────────────────
import torch
from torch import nn
from src.models.generator import EncoderDecoder2D, _assert_volume_tensor, _assert_single_time_volume

# ── 処理に使用する関数・クラス ──────────────────
class AverageEstimationGenerator(nn.Module):  # ネットワーク定義: 月平均と時系列SSTから週平均SSTを推定する
    """Estimate a cloud-free weekly average SST image."""

    def __init__(self, time_steps: int = 9) -> None:
        super().__init__()
        assert time_steps > 0, f"time_steps must be positive, got {time_steps}"
        self.time_steps = time_steps
        self.sst_projection = nn.Conv3d(2, 1, kernel_size=1)
        self.monthly_projection = nn.Conv3d(2, 1, kernel_size=1)
        # Fig. 8: concatenate nine temporal features and one monthly feature.
        self.encoder_decoder = EncoderDecoder2D(in_channels=time_steps + 1)

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
        x = torch.cat([sst_feature, monthly_feature], dim=2).squeeze(1)
        assert x.shape == (batch, time_steps + 1, height, width)

        pred_weekly = self.encoder_decoder(x)
        assert pred_weekly.shape == (batch, 1, height, width), tuple(pred_weekly.shape)
        return pred_weekly


