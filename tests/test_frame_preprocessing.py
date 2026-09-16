"""Single-observation storage tests using small real NetCDF fixtures."""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

from src.himawaridata.frame_preprocessing import create_observation_patches


def write_nc(root, when, *, shift=0):
    root.mkdir(exist_ok=True)
    path = root / f"{when:%Y%m%d%H%M%S}-sample.nc"
    with Dataset(path, "w") as ds:
        for name, size in [("time", 1), ("lat", 2), ("lon", 2)]:
            ds.createDimension(name, size)
        ds.createVariable("lat", "f4", ("lat",))[:] = [18, 19]
        ds.createVariable("lon", "f4", ("lon",))[:] = [118 + shift, 119 + shift]
        ds.createVariable("sea_surface_temperature", "f4", ("time", "lat", "lon"))[:] = 293.15
        quality = ds.createVariable("quality_level", "i1", ("time", "lat", "lon"))
        quality[:] = [[[5, 0], [0, 0]]]
        ds.createVariable("l2p_flags", "i2", ("time", "lat", "lon"))[:] = 0
    return path


def test_single_frames_keep_cloudy_observations_and_resume(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    start = datetime(2025, 4, 1)
    for hour in [0, 1]:
        write_nc(raw, start + timedelta(hours=hour))
    assert create_observation_patches(raw, out, patch_size=2, generate_temporal_averages=False) == 2
    files = sorted((out / "frames/himawari").glob("*.npz"))
    assert len(files) == 2
    with np.load(files[0]) as data:
        assert data["sst"].shape == (2, 2)
        assert data["sst"].dtype == np.float32
        assert data["cloud_mask"].dtype == np.uint8
        assert float(data["cloud_fraction"]) == .75
        assert float(data["valid_fraction"]) == .25
        assert str(data["timestamp"]) == start.isoformat()
        assert "sst_volume" not in data and "weekly_average" not in data
    before = files[0].read_bytes()
    assert create_observation_patches(raw, out, patch_size=2, generate_temporal_averages=False) == 0
    assert files[0].read_bytes() == before


def test_frame_limit_and_missing_history_do_not_block_storage(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    for hour in [0, 1]:
        write_nc(raw, datetime(2025, 4, 1, hour))
    assert create_observation_patches(raw, out, patch_size=2, max_frames=1) == 1
    assert len(list(out.rglob("*.npz"))) == 1


def test_complete_history_writes_separate_averages(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    for day in range(30):
        write_nc(raw, datetime(2025, 4, 1) + timedelta(days=day))
    create_observation_patches(raw, out, patch_size=2)
    assert len(list((out / "frames/himawari").glob("*.npz"))) == 30
    averages = list((out / "averages/himawari").glob("*.npz"))
    assert len(averages) == 1
    with np.load(averages[0]) as data:
        assert data["weekly_average"].shape == (1, 2, 2)
        assert data["monthly_average"].shape == (1, 1, 2, 2)
        np.testing.assert_allclose(data["weekly_average"][0, 0, 0], 20, atol=1e-4)
        assert np.isnan(data["monthly_average"][0, 0, 1, 1])
    with np.load(out / "frames/himawari" / averages[0].name) as data:
        assert "weekly_average" not in data


def test_coordinate_mismatch_is_rejected(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    write_nc(raw, datetime(2025, 4, 1))
    write_nc(raw, datetime(2025, 4, 2), shift=.1)
    with pytest.raises(ValueError, match="grid differs"):
        create_observation_patches(raw, out, patch_size=2, generate_temporal_averages=False)
