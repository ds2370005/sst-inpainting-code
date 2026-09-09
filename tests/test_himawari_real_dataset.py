"""Tests for direct training from preprocessed Himawari NPZ samples."""

from pathlib import Path

import numpy as np
import torch

from src.datasets.himawari_patch_dataset import HimawariPatchDataset
from src.losses import masked_mse_loss


def _write_sample(path: Path) -> None:
    sst = np.full((1, 2, 2, 2), 17.5, dtype=np.float32)
    cloud = np.zeros_like(sst, dtype=np.uint8)
    cloud[0, 0, 0, 1] = 1
    sst[0, 0, 1, 0] = np.nan  # Land/missing but not cloud.
    weekly = np.full((1, 2, 2), 17.5, dtype=np.float32)
    weekly[0, 1, 1] = np.nan
    monthly = np.full((1, 1, 2, 2), 17.5, dtype=np.float32)
    np.savez_compressed(
        path,
        sst_volume=sst,
        cloud_mask_volume=cloud,
        weekly_average=weekly,
        monthly_average=monthly,
    )


def test_real_average_dataset_derives_masks_and_normalizes(tmp_path: Path) -> None:
    _write_sample(tmp_path / "sample.npz")
    dataset = HimawariPatchDataset(
        tmp_path,
        stage="average",
        time_steps=2,
        patch_size=2,
        min_temp=0.0,
        max_temp=35.0,
    )

    sample = dataset[0]

    assert sample["sst_volume"].shape == (1, 2, 2, 2)
    assert sample["sst_volume"][0, 0, 0, 0].item() == 0.0
    assert sample["mask_volume"][0, 0, 0, 1].item() == 0.0  # Cloud.
    assert sample["mask_volume"][0, 0, 1, 0].item() == 0.0  # Land/missing.
    assert sample["weekly_mask"][0, 1, 1].item() == 0.0
    assert torch.isfinite(sample["sst_volume"]).all()
    assert torch.isfinite(sample["weekly_average"]).all()


def test_real_anomaly_dataset_uses_newest_frame_as_target(tmp_path: Path) -> None:
    _write_sample(tmp_path / "sample.npz")
    dataset = HimawariPatchDataset(
        tmp_path,
        stage="anomaly",
        time_steps=2,
        patch_size=2,
        min_temp=0.0,
        max_temp=35.0,
    )

    sample = dataset[0]

    torch.testing.assert_close(sample["target_sst"], sample["sst_volume"][:, -1])
    torch.testing.assert_close(sample["target_mask"], sample["mask_volume"][:, -1])
    assert "assimilation_sst" not in sample


def test_masked_mse_is_safe_when_invalid_target_is_nan() -> None:
    pred = torch.tensor([[[[1.0, 2.0]]]])
    target = torch.tensor([[[[3.0, float("nan")]]]])
    mask = torch.tensor([[[[1.0, 0.0]]]])

    loss = masked_mse_loss(pred, target, mask)

    assert torch.isfinite(loss)
    assert loss.item() == 4.0
