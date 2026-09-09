"""Tests for dataset loading and field parsing helpers."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.datasets import AnomalyDataset, ArrayFieldLoader, AverageDataset


def _config(patch_size: int = 8, time_steps: int = 3) -> dict:
    return {
        "data": {
            "patch_size": patch_size,
            "time_steps": time_steps,
        }
    }


def _write_index(tmp_path: Path, record: dict) -> Path:
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps({"samples": [record]}), encoding="utf-8")
    return index_path


def test_average_dataset_loads_npz_record(tmp_path: Path) -> None:
    patch_size = 8
    time_steps = 3
    arrays = {
        "sst_volume": np.random.randn(1, time_steps, patch_size, patch_size).astype(np.float32),
        "mask_volume": np.ones((1, time_steps, patch_size, patch_size), dtype=np.float32),
        "monthly_average": np.random.randn(1, 1, patch_size, patch_size).astype(np.float32),
        "monthly_mask": np.ones((1, 1, patch_size, patch_size), dtype=np.float32),
        "weekly_average": np.random.randn(1, patch_size, patch_size).astype(np.float32),
        "weekly_mask": np.ones((1, patch_size, patch_size), dtype=np.float32),
        "assimilation_weekly": np.random.randn(1, patch_size, patch_size).astype(np.float32),
    }
    npz_path = tmp_path / "average_sample.npz"
    np.savez(npz_path, **arrays)
    index_path = _write_index(
        tmp_path,
        {key: {"path": npz_path.name, "key": key} for key in arrays},
    )

    dataset = AverageDataset(str(index_path), _config(patch_size, time_steps))
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["sst_volume"].shape == (1, time_steps, patch_size, patch_size)
    assert sample["monthly_average"].shape == (1, 1, patch_size, patch_size)
    assert sample["weekly_average"].shape == (1, patch_size, patch_size)
    assert sample["sst_volume"].dtype == torch.float32


def test_anomaly_dataset_loads_npz_record_with_default_random_cloud_mask(tmp_path: Path) -> None:
    patch_size = 8
    time_steps = 3
    arrays = {
        "sst_volume": np.random.randn(1, time_steps, patch_size, patch_size).astype(np.float32),
        "mask_volume": np.ones((1, time_steps, patch_size, patch_size), dtype=np.float32),
        "target_sst": np.random.randn(1, patch_size, patch_size).astype(np.float32),
        "target_mask": np.ones((1, patch_size, patch_size), dtype=np.float32),
        "weekly_average": np.random.randn(1, patch_size, patch_size).astype(np.float32),
        "assimilation_sst": np.random.randn(1, patch_size, patch_size).astype(np.float32),
    }
    npz_path = tmp_path / "anomaly_sample.npz"
    np.savez(npz_path, **arrays)
    index_path = _write_index(
        tmp_path,
        {key: {"path": npz_path.name, "key": key} for key in arrays},
    )

    dataset = AnomalyDataset(str(index_path), _config(patch_size, time_steps))
    sample = dataset[0]

    assert sample["sst_volume"].shape == (1, time_steps, patch_size, patch_size)
    assert sample["target_sst"].shape == (1, patch_size, patch_size)
    assert sample["random_cloud_mask"].shape == (1, patch_size, patch_size)
    assert torch.equal(sample["random_cloud_mask"], sample["target_mask"])


def test_array_field_loader_supports_npy_indexing(tmp_path: Path) -> None:
    array = np.random.randn(2, 1, 3, 8, 8).astype(np.float32)
    path = tmp_path / "sst_volume.npy"
    np.save(path, array)

    tensor = ArrayFieldLoader().load_tensor({"path": path.name, "index": 1}, tmp_path)

    assert tensor.shape == (1, 3, 8, 8)
    assert torch.allclose(tensor, torch.from_numpy(array[1]))


def test_array_field_loader_leaves_netcdf_and_geotiff_as_extension_hooks(tmp_path: Path) -> None:
    loader = ArrayFieldLoader()

    with pytest.raises(NotImplementedError, match="NetCDF"):
        loader.load_array(tmp_path / "sample.nc")
    with pytest.raises(NotImplementedError, match="GeoTIFF"):
        loader.load_array(tmp_path / "sample.tif")
