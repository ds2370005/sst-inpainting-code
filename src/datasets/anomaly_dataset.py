"""Anomaly-inpainting dataset interfaces."""

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


ANOMALY_FIELDS = (
    "sst_volume",
    "mask_volume",
    "target_sst",
    "target_mask",
    "weekly_average",
    "assimilation_sst",
)


class AnomalyDataset(Dataset[dict[str, torch.Tensor]]):  # データローダー: indexファイルから異常補完用の学習サンプルを読む
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
            for field in ANOMALY_FIELDS
        }
        if "random_cloud_mask" in record:
            sample["random_cloud_mask"] = load_required_tensor(
                record,
                "random_cloud_mask",
                base_dir=self.base_dir,
                loader=self.loader,
            )
        else:
            sample["random_cloud_mask"] = sample["target_mask"].clone()
        self._validate_sample(sample)
        return sample

    def _validate_sample(self, sample: dict[str, torch.Tensor]) -> None:
        data_config = self.config["data"]
        time_steps = int(data_config["time_steps"])
        patch_size = int(data_config["patch_size"])
        assert_shape(sample["sst_volume"], "sst_volume", (1, time_steps, patch_size, patch_size))
        assert_shape(sample["mask_volume"], "mask_volume", (1, time_steps, patch_size, patch_size))
        assert_shape(sample["target_sst"], "target_sst", (1, patch_size, patch_size))
        assert_shape(sample["target_mask"], "target_mask", (1, patch_size, patch_size))
        assert_shape(sample["weekly_average"], "weekly_average", (1, patch_size, patch_size))
        assert_shape(sample["assimilation_sst"], "assimilation_sst", (1, patch_size, patch_size))
        assert_shape(
            sample["random_cloud_mask"],
            "random_cloud_mask",
            (1, patch_size, patch_size),
        )


class SyntheticAnomalyDataset(Dataset[dict[str, torch.Tensor]]):  # 学習用: 異常補完の形状確認用に疑似バッチを作る
    """Synthetic AnomalyDataset-compatible tensors for early pipeline tests."""

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
        target_sst = (weekly_average + 0.25 * torch.randn_like(weekly_average)).clamp(
            -1.0,
            1.0,
        )
        sst_volume = target_sst.unsqueeze(1).repeat(1, self.time_steps, 1, 1)
        sst_volume = (sst_volume + 0.1 * torch.randn_like(sst_volume)).clamp(-1.0, 1.0)
        target_mask = self._mask(1, height, width)
        return {
            "sst_volume": sst_volume,
            "mask_volume": self._mask(1, self.time_steps, height, width),
            "target_sst": target_sst,
            "target_mask": target_mask,
            "weekly_average": weekly_average,
            "assimilation_sst": (target_sst + 0.02 * torch.randn_like(target_sst)).clamp(
                -1.0,
                1.0,
            ),
            "random_cloud_mask": target_mask.clone(),
        }
