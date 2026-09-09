"""Average-estimation dataset interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.datasets.loaders import (
    ArrayFieldLoader,
    assert_shape,
    load_index_records,
    load_required_tensor,
)


AVERAGE_FIELDS = (
    "sst_volume",
    "mask_volume",
    "monthly_average",
    "monthly_mask",
    "weekly_average",
    "weekly_mask",
    "assimilation_weekly",
)


class AverageDataset(Dataset[dict[str, torch.Tensor]]):  # データローダー: indexファイルから平均推定用の学習サンプルを読む
    """Dataset backed by an index file of preprocessed tensor field paths."""

    def __init__(
        self,
        index_file: str,
        config: dict[str, Any],
        loader: ArrayFieldLoader | None = None,
    ) -> None:
        super().__init__()
        self.index_file = Path(index_file)
        self.config = config
        self.records = load_index_records(self.index_file)
        self.loader = loader or ArrayFieldLoader()
        self.base_dir = self.index_file.parent
        assert self.records, f"Index file has no samples: {self.index_file}"

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        sample = {
            field: load_required_tensor(
                record,
                field,
                base_dir=self.base_dir,
                loader=self.loader,
            )
            for field in AVERAGE_FIELDS
        }
        self._validate_sample(sample)
        return sample

    def _validate_sample(self, sample: dict[str, torch.Tensor]) -> None:
        data_config = self.config["data"]
        time_steps = int(data_config["time_steps"])
        patch_size = int(data_config["patch_size"])
        assert_shape(sample["sst_volume"], "sst_volume", (1, time_steps, patch_size, patch_size))
        assert_shape(sample["mask_volume"], "mask_volume", (1, time_steps, patch_size, patch_size))
        assert_shape(
            sample["monthly_average"],
            "monthly_average",
            (1, 1, patch_size, patch_size),
        )
        assert_shape(sample["monthly_mask"], "monthly_mask", (1, 1, patch_size, patch_size))
        assert_shape(sample["weekly_average"], "weekly_average", (1, patch_size, patch_size))
        assert_shape(sample["weekly_mask"], "weekly_mask", (1, patch_size, patch_size))
        assert_shape(
            sample["assimilation_weekly"],
            "assimilation_weekly",
            (1, patch_size, patch_size),
        )


class SyntheticAverageDataset(Dataset[dict[str, torch.Tensor]]):  # 学習用: 平均推定の形状確認用に疑似バッチを作る
    """Synthetic AverageDataset-compatible tensors for early pipeline tests."""

    def __init__(
        self,
        *,
        num_samples: int,
        time_steps: int,
        patch_size: int,
        observed_probability: float = 0.8,
    ) -> None:
        super().__init__()
        assert num_samples > 0, f"num_samples must be positive, got {num_samples}"
        assert time_steps > 0, f"time_steps must be positive, got {time_steps}"
        assert patch_size > 0, f"patch_size must be positive, got {patch_size}"
        assert 0.0 < observed_probability <= 1.0, (
            "observed_probability must be in (0, 1], "
            f"got {observed_probability}"
        )
        self.num_samples = num_samples
        self.time_steps = time_steps
        self.patch_size = patch_size
        self.observed_probability = observed_probability

    def __len__(self) -> int:
        return self.num_samples

    def _mask(self, *shape: int) -> torch.Tensor:
        return (torch.rand(*shape) < self.observed_probability).float()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        del idx
        height = width = self.patch_size
        weekly_average = torch.rand(1, height, width) * 2.0 - 1.0
        monthly_average = weekly_average.unsqueeze(1) + 0.05 * torch.randn(
            1,
            1,
            height,
            width,
        )
        sst_volume = weekly_average.unsqueeze(1).repeat(1, self.time_steps, 1, 1)
        sst_volume = (sst_volume + 0.1 * torch.randn_like(sst_volume)).clamp(-1.0, 1.0)
        return {
            "sst_volume": sst_volume,
            "mask_volume": self._mask(1, self.time_steps, height, width),
            "monthly_average": monthly_average.clamp(-1.0, 1.0),
            "monthly_mask": self._mask(1, 1, height, width),
            "weekly_average": weekly_average,
            "weekly_mask": self._mask(1, height, width),
            "assimilation_weekly": (
                weekly_average + 0.02 * torch.randn_like(weekly_average)
            ).clamp(-1.0, 1.0),
        }
