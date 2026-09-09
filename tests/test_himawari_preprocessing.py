"""Focused tests for bulk Himawari patch preprocessing helpers."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from src.himawaridata.crop_nc_to_1650 import (
    build_fixed_grid,
    build_time_sequences,
    cloud_mask_pool_path,
    create_cloud_mask,
    _add_valid_sst_to_accumulator,
    _average_from_accumulator,
    load_logged_cloud_fractions,
    open_processing_log,
    save_cloud_mask_to_pool,
    save_patch_sample,
    update_sample_with_temporal_averages,
    write_log_row,
)


def test_fixed_grid_is_centered_and_nonoverlapping() -> None:
    grid = build_fixed_grid(slice(100, 1750), slice(200, 1850), 256)

    assert len(grid) == 36
    assert grid[0][4:] == (57, 57)
    assert grid[-1][4:] == (1337, 1337)

    for index, (_, _, lat_a, lon_a, _, _) in enumerate(grid):
        for _, _, lat_b, lon_b, _, _ in grid[index + 1 :]:
            overlaps = (
                max(lat_a.start, lat_b.start) < min(lat_a.stop, lat_b.stop)
                and max(lon_a.start, lon_b.start) < min(lon_a.stop, lon_b.stop)
            )
            assert not overlaps


def test_time_sequences_are_oldest_to_newest_at_exact_intervals() -> None:
    start = datetime(2025, 4, 1)
    indexed = {
        start + timedelta(hours=hour): Path(f"{hour}.nc")
        for hour in range(0, 13)
    }

    sequences = list(build_time_sequences(indexed, time_steps=3, interval_hours=6))

    assert len(sequences) == 1
    target, paths = sequences[0]
    assert target == start + timedelta(hours=12)
    assert paths == [Path("0.nc"), Path("6.nc"), Path("12.nc")]


def test_atomic_save_and_rejected_cloud_resume_log(tmp_path: Path) -> None:
    timestamps = [datetime(2025, 4, 1) + timedelta(hours=6 * i) for i in range(2)]
    paths = [tmp_path / f"{timestamp:%Y%m%d%H%M%S}-sample.nc" for timestamp in timestamps]
    output = tmp_path / "sample.npz"
    save_patch_sample(
        output,
        paths,
        timestamps[-1],
        np.zeros((1, 2, 4, 4), dtype=np.float32),
        np.ones((1, 2, 4, 4), dtype=np.uint8),
        np.arange(4, dtype=np.float32),
        np.arange(4, dtype=np.float32),
        57,
        57,
        0,
        0,
    )

    assert output.exists()
    assert not list(tmp_path.glob("*.tmp"))
    with np.load(output) as archive:
        assert archive["sst_volume"].shape == (1, 2, 4, 4)
        assert archive["cloud_mask_volume"].shape == (1, 2, 4, 4)
        assert "mask_volume" not in archive
        assert str(archive["sst_units"]) == "degree_Celsius"

    handle, writer = open_processing_log(tmp_path)
    write_log_row(
        handle,
        writer,
        target_time=timestamps[-1],
        row=2,
        column=3,
        status="rejected_cloud",
        output_path=tmp_path / "rejected.npz",
        cloud_fraction=0.75,
    )
    handle.close()

    fractions = load_logged_cloud_fractions(tmp_path)
    assert fractions[(timestamps[-1].isoformat(), 2, 3)] == 0.75


def test_cloud_mask_pool_save_is_binary_atomic_and_uniquely_named(
    tmp_path: Path,
) -> None:
    timestamp = datetime(2025, 4, 1, 6)
    output = cloud_mask_pool_path(tmp_path, timestamp, 2, 3)
    mask = np.asarray([[1, 0], [0, 1]], dtype=np.uint8)

    save_cloud_mask_to_pool(output, mask)

    assert output.name == "20250401060000_r02_c03_cloud.npy"
    assert not list(tmp_path.glob("*.tmp"))
    saved = np.load(output)
    assert saved.shape == (1, 2, 2)
    assert saved.dtype == np.uint8
    np.testing.assert_array_equal(saved[0], mask)


def test_cloud_mask_excludes_land_and_marks_only_missing_ocean() -> None:
    sst = np.ma.asarray([[290.0, np.nan, np.nan, 291.0]])
    quality = np.ma.asarray([[5, 0, 0, 3]])
    # Second pixel is land; third is missing ocean; fourth is low-quality ocean.
    flags = np.asarray([[0, 2, 0, 0]], dtype=np.int16)

    cloud = create_cloud_mask(sst, quality, flags, min_quality_level=4)

    np.testing.assert_array_equal(cloud, [[0, 0, 1, 1]])


def test_temporal_average_ignores_clouds_and_enforces_observation_count() -> None:
    total = np.zeros((1, 3), dtype=np.float32)
    count = np.zeros((1, 3), dtype=np.uint16)
    frames = [
        (np.asarray([[10.0, 20.0, np.nan]]), np.asarray([[0, 0, 0]])),
        (np.asarray([[12.0, 22.0, 30.0]]), np.asarray([[0, 1, 0]])),
        (np.asarray([[14.0, 24.0, 32.0]]), np.asarray([[0, 0, 1]])),
    ]
    for sst, cloud in frames:
        _add_valid_sst_to_accumulator(total, count, sst, cloud)

    average = _average_from_accumulator(total, count, min_observations=2)

    np.testing.assert_allclose(average[0, :2], [12.0, 22.0])
    assert np.isnan(average[0, 2])


def test_temporal_averages_are_added_without_additional_masks(tmp_path: Path) -> None:
    timestamps = [datetime(2025, 4, 1)]
    paths = [tmp_path / "20250401000000-sample.nc"]
    output = tmp_path / "sample.npz"
    save_patch_sample(
        output,
        paths,
        timestamps[0],
        np.zeros((1, 1, 2, 2), dtype=np.float32),
        np.zeros((1, 1, 2, 2), dtype=np.uint8),
        np.arange(2, dtype=np.float32),
        np.arange(2, dtype=np.float32),
        0,
        0,
        0,
        0,
    )

    update_sample_with_temporal_averages(
        output,
        np.full((2, 2), 15.0, dtype=np.float32),
        np.full((2, 2), 14.0, dtype=np.float32),
    )

    with np.load(output) as archive:
        assert archive["weekly_average"].shape == (1, 2, 2)
        assert archive["monthly_average"].shape == (1, 1, 2, 2)
        assert "weekly_mask" not in archive
        assert "monthly_mask" not in archive
