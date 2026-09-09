"""Create aligned 9-time-step SST patches and cloud masks from Himawari data.

The geographic study area is first identified as 17--50 N, 117--150 E, but
the intermediate 1650 x 1650 arrays are not saved.  Its centered 1536 x 1536
region is divided into a fixed, nonoverlapping 6 x 6 grid of 256 x 256 patches.
The same grid is read from nine NetCDF files separated by six hours.

Every saved sample contains the nine aligned masks as ``cloud_mask_volume``. Each
two-dimensional mask is also stored once in a separate cloud-mask pool, keyed
by observation time and grid position, so training can draw a mask from a
different observation to create artificial missing regions.

The JAXA GHRSST L3C product already contains the result of cloud screening.
Its ``quality_level`` and valid SST pixels are converted to Hirahara et al.'s
binary convention: 1 = cloud-screened missing ocean; 0 = clear ocean or land.
Land is explicitly removed with the GHRSST ``l2p_flags`` land bit, so land
shapes never enter the training cloud-mask pool. Hirahara et al. cite
Merchant et al. (2005), "Probabilistic
physically based cloud screening of satellite infrared imagery for operational
sea surface temperature retrieval", for the cloud-screening method.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import deque
from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


DEFAULT_LAT_MIN = 17.0
DEFAULT_LAT_MAX = 50.0
DEFAULT_LON_MIN = 117.0
DEFAULT_LON_MAX = 150.0
DEFAULT_PATCH_SIZE = 256
DEFAULT_TIME_STEPS = 9
DEFAULT_TIME_INTERVAL_HOURS = 6
DEFAULT_MIN_QUALITY_LEVEL = 4
DEFAULT_WEEKLY_DAYS = 7
DEFAULT_MONTHLY_DAYS = 30
DEFAULT_WEEKLY_MIN_OBSERVATIONS = 3
DEFAULT_MONTHLY_MIN_OBSERVATIONS = 5
GHRSST_LAND_FLAG = 2
KELVIN_TO_CELSIUS_OFFSET = 273.15
TIMESTAMP_PATTERN = re.compile(r"^(\d{14})-")


def find_contiguous_slice(
    values: np.ndarray,
    lower: float,
    upper: float,
    name: str,
) -> slice:
    mask = (values >= lower) & (values < upper)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError(f"No {name} values found in range [{lower}, {upper}).")
    if indices[-1] - indices[0] + 1 != indices.size:
        raise ValueError(f"{name} range [{lower}, {upper}) is not contiguous.")
    return slice(int(indices[0]), int(indices[-1] + 1))


def create_cloud_mask(
    sea_surface_temperature: np.ndarray,
    quality_level: np.ndarray,
    l2p_flags: np.ndarray,
    min_quality_level: int = DEFAULT_MIN_QUALITY_LEVEL,
) -> np.ndarray:
    """Return a binary mask containing only cloudy/missing ocean pixels.

    The L3C product stores retrieved SST after cloud screening rather than an
    independent categorical cloud field. Therefore a non-land ocean pixel at
    which no acceptable SST remains is treated as cloud-screened missing.
    Convention: 1 = cloud/missing ocean, 0 = clear ocean or land.
    """
    if not 0 <= min_quality_level <= 5:
        raise ValueError("min_quality_level must be between 0 and 5.")
    if not (
        sea_surface_temperature.shape == quality_level.shape == l2p_flags.shape
    ):
        raise ValueError("SST, quality_level, and l2p_flags must have the same shape.")

    sst_mask = np.ma.getmaskarray(sea_surface_temperature)
    quality_mask = np.ma.getmaskarray(quality_level)
    sst = np.asarray(np.ma.filled(sea_surface_temperature, np.nan))
    quality = np.asarray(np.ma.filled(quality_level, -1))
    flags = np.asarray(np.ma.filled(l2p_flags, 0), dtype=np.int64)
    land = (flags & GHRSST_LAND_FLAG) != 0
    clear = (
        ~sst_mask
        & np.isfinite(sst)
        & ~quality_mask
        & (quality >= min_quality_level)
    )
    return ((~land) & (~clear)).astype(np.uint8)


def timestamp_from_path(path: Path) -> datetime:
    match = TIMESTAMP_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Filename does not start with YYYYmmddHHMMSS: {path.name}")
    return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")


def index_netcdf_files(input_root: Path) -> dict[datetime, Path]:
    """Index files by observation time and reject ambiguous duplicates."""
    indexed: dict[datetime, Path] = {}
    for path in sorted(input_root.rglob("*.nc")):
        timestamp = timestamp_from_path(path)
        if timestamp in indexed:
            raise ValueError(
                f"Duplicate NetCDF timestamp {timestamp}: {indexed[timestamp]} and {path}"
            )
        indexed[timestamp] = path
    return indexed


def build_time_sequences(
    indexed_files: dict[datetime, Path],
    time_steps: int = DEFAULT_TIME_STEPS,
    interval_hours: int = DEFAULT_TIME_INTERVAL_HOURS,
) -> Iterator[tuple[datetime, list[Path]]]:
    """Yield oldest-to-newest sequences with exact temporal spacing."""
    if time_steps <= 0 or interval_hours <= 0:
        raise ValueError("time_steps and interval_hours must be positive.")
    interval = timedelta(hours=interval_hours)
    for target_time in sorted(indexed_files):
        times = [
            target_time - interval * offset
            for offset in range(time_steps - 1, -1, -1)
        ]
        if all(timestamp in indexed_files for timestamp in times):
            yield target_time, [indexed_files[timestamp] for timestamp in times]


def study_area_slices(
    path: Path,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> tuple[slice, slice, np.ndarray, np.ndarray]:
    with Dataset(path) as dataset:
        lat = np.asarray(dataset.variables["lat"][:])
        lon = np.asarray(dataset.variables["lon"][:])
    lat_slice = find_contiguous_slice(lat, lat_min, lat_max, "latitude")
    lon_slice = find_contiguous_slice(lon, lon_min, lon_max, "longitude")
    return lat_slice, lon_slice, lat[lat_slice], lon[lon_slice]


def build_fixed_grid(
    area_lat_slice: slice,
    area_lon_slice: slice,
    patch_size: int,
) -> list[tuple[int, int, slice, slice, int, int]]:
    """Return a centered, nonoverlapping grid covering complete patches only."""
    area_height = area_lat_slice.stop - area_lat_slice.start
    area_width = area_lon_slice.stop - area_lon_slice.start
    if patch_size <= 0 or patch_size > min(area_height, area_width):
        raise ValueError(
            f"patch_size={patch_size} does not fit in {area_height}x{area_width} area."
        )
    rows = area_height // patch_size
    columns = area_width // patch_size
    top_margin = (area_height - rows * patch_size) // 2
    left_margin = (area_width - columns * patch_size) // 2
    grid = []
    for row in range(rows):
        y = top_margin + row * patch_size
        lat_slice = slice(
            area_lat_slice.start + y,
            area_lat_slice.start + y + patch_size,
        )
        for column in range(columns):
            x = left_margin + column * patch_size
            lon_slice = slice(
                area_lon_slice.start + x,
                area_lon_slice.start + x + patch_size,
            )
            grid.append((row, column, lat_slice, lon_slice, y, x))
    return grid


def read_sst_and_mask_patch(
    path: Path,
    lat_slice: slice,
    lon_slice: slice,
    min_quality_level: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read one patch as degrees Celsius; missing SST values remain NaN."""
    with Dataset(path) as dataset:
        missing = {"sea_surface_temperature", "quality_level", "l2p_flags"}.difference(
            dataset.variables
        )
        if missing:
            raise KeyError(f"{path} is missing variables: {sorted(missing)}")
        sst_raw = dataset.variables["sea_surface_temperature"][
            0, lat_slice, lon_slice
        ]
        quality = dataset.variables["quality_level"][0, lat_slice, lon_slice]
        l2p_flags = dataset.variables["l2p_flags"][0, lat_slice, lon_slice]
        lat = np.asarray(dataset.variables["lat"][lat_slice])
        lon = np.asarray(dataset.variables["lon"][lon_slice])
        # netCDF4 automatically applies the GHRSST scale_factor=0.01 and
        # add_offset=273.15, so sst_raw is already decoded in kelvin.
        sst_kelvin = np.asarray(np.ma.filled(sst_raw, np.nan), dtype=np.float32)
        sst = sst_kelvin - np.float32(KELVIN_TO_CELSIUS_OFFSET)
        cloud_mask = create_cloud_mask(
            sst_raw,
            quality,
            l2p_flags,
            min_quality_level=min_quality_level,
        )
    return sst, cloud_mask, lat, lon


def extract_aligned_patch_sequence(
    paths: Sequence[Path],
    lat_slice: slice,
    lon_slice: slice,
    min_quality_level: int,
) -> tuple[np.ndarray, np.ndarray]:
    sst_frames: list[np.ndarray] = []
    mask_frames: list[np.ndarray] = []
    expected_shape = (lat_slice.stop - lat_slice.start, lon_slice.stop - lon_slice.start)
    reference_lat: np.ndarray | None = None
    reference_lon: np.ndarray | None = None
    for path in paths:
        sst, mask, lat, lon = read_sst_and_mask_patch(
            path, lat_slice, lon_slice, min_quality_level
        )
        if sst.shape != expected_shape:
            raise ValueError(f"Unexpected patch shape {sst.shape} in {path}.")
        if reference_lat is None:
            reference_lat, reference_lon = lat, lon
        elif not np.array_equal(lat, reference_lat) or not np.array_equal(
            lon, reference_lon
        ):
            raise ValueError(f"Latitude/longitude grid differs within sequence: {path}")
        sst_frames.append(sst)
        mask_frames.append(mask)
    # Dataset interface expects [C=1, T, H, W].
    return np.stack(sst_frames)[None, ...], np.stack(mask_frames)[None, ...]


def extract_aligned_grid_sequence(
    paths: Sequence[Path],
    grid: Sequence[tuple[int, int, slice, slice, int, int]],
    min_quality_level: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, slice, slice]:
    """Read the complete grid once per time step instead of once per patch."""
    if not grid:
        raise ValueError("grid must contain at least one patch.")
    grid_lat_slice = slice(
        min(item[2].start for item in grid),
        max(item[2].stop for item in grid),
    )
    grid_lon_slice = slice(
        min(item[3].start for item in grid),
        max(item[3].stop for item in grid),
    )
    expected_shape = (
        grid_lat_slice.stop - grid_lat_slice.start,
        grid_lon_slice.stop - grid_lon_slice.start,
    )
    sst_frames: list[np.ndarray] = []
    mask_frames: list[np.ndarray] = []
    reference_lat: np.ndarray | None = None
    reference_lon: np.ndarray | None = None

    for path in paths:
        sst, mask, lat, lon = read_sst_and_mask_patch(
            path,
            grid_lat_slice,
            grid_lon_slice,
            min_quality_level,
        )
        if sst.shape != expected_shape:
            raise ValueError(f"Unexpected grid shape {sst.shape} in {path}.")
        if reference_lat is None:
            reference_lat, reference_lon = lat, lon
        elif not np.array_equal(lat, reference_lat) or not np.array_equal(
            lon, reference_lon
        ):
            raise ValueError(f"Latitude/longitude grid differs within sequence: {path}")
        sst_frames.append(sst)
        mask_frames.append(mask)

    assert reference_lat is not None and reference_lon is not None
    return (
        np.stack(sst_frames),
        np.stack(mask_frames),
        reference_lat,
        reference_lon,
        grid_lat_slice,
        grid_lon_slice,
    )


def save_patch_sample(
    output_path: Path,
    paths: Sequence[Path],
    target_time: datetime,
    sst_volume: np.ndarray,
    cloud_mask_volume: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    area_y: int,
    area_x: int,
    grid_row: int,
    grid_column: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                sst_volume=sst_volume,
                cloud_mask_volume=cloud_mask_volume,
                lat=lat,
                lon=lon,
                timestamps=np.asarray(
                    [timestamp_from_path(path).isoformat() for path in paths]
                ),
                source_files=np.asarray([str(path) for path in paths]),
                target_timestamp=np.asarray(target_time.isoformat()),
                area_patch_origin_yx=np.asarray([area_y, area_x], dtype=np.int32),
                grid_row_column=np.asarray([grid_row, grid_column], dtype=np.int16),
                sst_units=np.asarray("degree_Celsius"),
            )
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def cloud_mask_pool_path(
    pool_root: Path,
    timestamp: datetime,
    grid_row: int,
    grid_column: int,
) -> Path:
    """Return the unique path for one observation-time/grid cloud mask."""
    return pool_root / (
        f"{timestamp:%Y%m%d%H%M%S}_r{grid_row:02d}_c{grid_column:02d}_cloud.npy"
    )


def save_cloud_mask_to_pool(output_path: Path, cloud_mask: np.ndarray) -> None:
    """Atomically save one [1, H, W] uint8 mask for random-mask training."""
    mask = np.asarray(cloud_mask, dtype=np.uint8)
    if mask.ndim == 2:
        mask = mask[None, ...]
    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError(
            f"Cloud-pool mask must have shape [1, H, W], got {mask.shape}."
        )
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("Cloud-pool mask must contain only binary values 0 and 1.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            np.save(handle, mask, allow_pickle=False)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def update_sample_with_temporal_averages(
    sample_path: Path,
    weekly_average: np.ndarray,
    monthly_average: np.ndarray,
) -> None:
    """Atomically add physical-unit weekly/monthly averages to one sample."""
    weekly = np.asarray(weekly_average, dtype=np.float32)
    monthly = np.asarray(monthly_average, dtype=np.float32)
    if weekly.ndim == 2:
        weekly = weekly[None, ...]
    if monthly.ndim == 2:
        monthly = monthly[None, None, ...]
    if weekly.ndim != 3 or weekly.shape[0] != 1:
        raise ValueError(f"weekly_average must have shape [1, H, W], got {weekly.shape}.")
    if monthly.ndim != 4 or monthly.shape[:2] != (1, 1):
        raise ValueError(
            f"monthly_average must have shape [1, 1, H, W], got {monthly.shape}."
        )
    if weekly.shape[-2:] != monthly.shape[-2:]:
        raise ValueError("Weekly and monthly average spatial shapes must match.")

    with np.load(sample_path, allow_pickle=False) as archive:
        fields = {key: np.asarray(archive[key]) for key in archive.files}
    fields["weekly_average"] = weekly
    fields["monthly_average"] = monthly
    fields["average_units"] = np.asarray("degree_Celsius")
    fields["average_missing_value"] = np.asarray("NaN")

    temporary_path = sample_path.with_name(f".{sample_path.name}.averages.tmp")
    try:
        with temporary_path.open("wb") as handle:
            np.savez_compressed(handle, **fields)
        temporary_path.replace(sample_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _add_valid_sst_to_accumulator(
    total: np.ndarray,
    count: np.ndarray,
    sst: np.ndarray,
    cloud_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(sst) & (cloud_mask == 0)
    values = np.where(valid, sst, 0.0).astype(np.float32, copy=False)
    total += values
    count += valid.astype(np.uint16)
    return values, valid


def _average_from_accumulator(
    total: np.ndarray,
    count: np.ndarray,
    min_observations: int,
) -> np.ndarray:
    average = np.full(total.shape, np.nan, dtype=np.float32)
    accepted = count >= min_observations
    np.divide(total, count, out=average, where=accepted)
    return average


def add_temporal_averages_to_samples(
    indexed_files: dict[datetime, Path],
    target_times: Sequence[datetime],
    output_root: Path,
    grid: Sequence[tuple[int, int, slice, slice, int, int]],
    min_quality_level: int,
    *,
    weekly_days: int = DEFAULT_WEEKLY_DAYS,
    monthly_days: int = DEFAULT_MONTHLY_DAYS,
    weekly_min_observations: int = DEFAULT_WEEKLY_MIN_OBSERVATIONS,
    monthly_min_observations: int = DEFAULT_MONTHLY_MIN_OBSERVATIONS,
    overwrite: bool = False,
) -> tuple[int, int, int]:
    """Attach Hirahara-style same-hour 7/30-day averages to saved samples.

    Only targets with a complete 30-calendar-day source window are processed.
    Pixel averages ignore cloudy/missing SSTs; pixels below the paper's minimum
    observation counts remain NaN. No weekly/monthly validity masks are saved.
    """
    if not (0 < weekly_days <= monthly_days):
        raise ValueError("Require 0 < weekly_days <= monthly_days.")
    if not 0 < weekly_min_observations <= weekly_days:
        raise ValueError("weekly_min_observations must be in [1, weekly_days].")
    if not 0 < monthly_min_observations <= monthly_days:
        raise ValueError("monthly_min_observations must be in [1, monthly_days].")
    if not grid:
        raise ValueError("grid must contain at least one patch.")

    targets = set(target_times)
    targets_by_hour = {
        hour: sorted(timestamp for timestamp in targets if timestamp.hour == hour)
        for hour in range(24)
    }
    grid_lat_slice = slice(
        min(item[2].start for item in grid), max(item[2].stop for item in grid)
    )
    grid_lon_slice = slice(
        min(item[3].start for item in grid), max(item[3].stop for item in grid)
    )
    grid_shape = (
        grid_lat_slice.stop - grid_lat_slice.start,
        grid_lon_slice.stop - grid_lon_slice.start,
    )
    updated = 0
    skipped_existing = 0
    skipped_incomplete_history = 0

    for hour, hour_targets in targets_by_hour.items():
        if not hour_targets:
            continue
        last_target = hour_targets[-1]
        timestamps = sorted(
            timestamp
            for timestamp in indexed_files
            if timestamp.hour == hour and timestamp <= last_target
        )
        monthly_queue: deque[tuple[datetime, np.ndarray, np.ndarray]] = deque()
        weekly_queue: deque[tuple[datetime, np.ndarray, np.ndarray]] = deque()
        monthly_sum = np.zeros(grid_shape, dtype=np.float32)
        weekly_sum = np.zeros(grid_shape, dtype=np.float32)
        monthly_count = np.zeros(grid_shape, dtype=np.uint16)
        weekly_count = np.zeros(grid_shape, dtype=np.uint16)

        for timestamp in timestamps:
            sst, cloud_mask, _, _ = read_sst_and_mask_patch(
                indexed_files[timestamp],
                grid_lat_slice,
                grid_lon_slice,
                min_quality_level,
            )
            values, valid = _add_valid_sst_to_accumulator(
                monthly_sum, monthly_count, sst, cloud_mask
            )
            weekly_sum += values
            weekly_count += valid.astype(np.uint16)
            monthly_queue.append((timestamp, values, valid))
            weekly_queue.append((timestamp, values, valid))

            monthly_start = timestamp - timedelta(days=monthly_days - 1)
            while monthly_queue and monthly_queue[0][0] < monthly_start:
                _, old_values, old_valid = monthly_queue.popleft()
                monthly_sum -= old_values
                monthly_count -= old_valid.astype(np.uint16)
            weekly_start = timestamp - timedelta(days=weekly_days - 1)
            while weekly_queue and weekly_queue[0][0] < weekly_start:
                _, old_values, old_valid = weekly_queue.popleft()
                weekly_sum -= old_values
                weekly_count -= old_valid.astype(np.uint16)

            if timestamp not in targets:
                continue
            required_monthly_times = (
                timestamp - timedelta(days=offset)
                for offset in range(monthly_days)
            )
            if not all(required in indexed_files for required in required_monthly_times):
                skipped_incomplete_history += 1
                continue

            weekly_average_grid = _average_from_accumulator(
                weekly_sum, weekly_count, weekly_min_observations
            )
            monthly_average_grid = _average_from_accumulator(
                monthly_sum, monthly_count, monthly_min_observations
            )
            for row, column, lat_slice, lon_slice, _, _ in grid:
                sample_path = output_root / (
                    f"{timestamp:%Y%m%d%H%M%S}_r{row:02d}_c{column:02d}.npz"
                )
                if not sample_path.exists():
                    continue
                if not overwrite:
                    with np.load(sample_path, allow_pickle=False) as archive:
                        if "weekly_average" in archive and "monthly_average" in archive:
                            skipped_existing += 1
                            continue
                local_y = lat_slice.start - grid_lat_slice.start
                local_x = lon_slice.start - grid_lon_slice.start
                patch_y = slice(local_y, local_y + (lat_slice.stop - lat_slice.start))
                patch_x = slice(local_x, local_x + (lon_slice.stop - lon_slice.start))
                update_sample_with_temporal_averages(
                    sample_path,
                    weekly_average_grid[patch_y, patch_x],
                    monthly_average_grid[patch_y, patch_x],
                )
                updated += 1

    return updated, skipped_existing, skipped_incomplete_history


LOG_FIELDS = (
    "target_timestamp",
    "grid_row",
    "grid_column",
    "status",
    "cloud_fraction",
    "output_path",
    "message",
)


def open_processing_log(output_root: Path) -> tuple[object, csv.DictWriter]:
    """Open an append-only CSV log suitable for interrupted/resumed runs."""
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "processing_log.csv"
    needs_header = not log_path.exists() or log_path.stat().st_size == 0
    handle = log_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
    if needs_header:
        writer.writeheader()
        handle.flush()
    return handle, writer


def load_logged_cloud_fractions(output_root: Path) -> dict[tuple[str, int, int], float]:
    """Load previously measured rejected-patch cloud fractions for fast resume."""
    log_path = output_root / "processing_log.csv"
    if not log_path.exists():
        return {}
    fractions: dict[tuple[str, int, int], float] = {}
    with log_path.open("r", encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            if record.get("status") != "rejected_cloud":
                continue
            try:
                key = (
                    record["target_timestamp"],
                    int(record["grid_row"]),
                    int(record["grid_column"]),
                )
                fractions[key] = float(record["cloud_fraction"])
            except (KeyError, TypeError, ValueError):
                continue
    return fractions


def write_log_row(
    handle: object,
    writer: csv.DictWriter,
    *,
    target_time: datetime,
    row: int,
    column: int,
    status: str,
    output_path: Path,
    cloud_fraction: float | None = None,
    message: str = "",
) -> None:
    writer.writerow(
        {
            "target_timestamp": target_time.isoformat(),
            "grid_row": row,
            "grid_column": column,
            "status": status,
            "cloud_fraction": (
                "" if cloud_fraction is None else f"{cloud_fraction:.6f}"
            ),
            "output_path": str(output_path),
            "message": message,
        }
    )
    handle.flush()


def create_training_patches(
    input_root: Path,
    output_root: Path,
    *,
    lat_min: float = DEFAULT_LAT_MIN,
    lat_max: float = DEFAULT_LAT_MAX,
    lon_min: float = DEFAULT_LON_MIN,
    lon_max: float = DEFAULT_LON_MAX,
    patch_size: int = DEFAULT_PATCH_SIZE,
    time_steps: int = DEFAULT_TIME_STEPS,
    interval_hours: int = DEFAULT_TIME_INTERVAL_HOURS,
    min_quality_level: int = DEFAULT_MIN_QUALITY_LEVEL,
    max_cloud_fraction: float = 0.5,
    max_sequences: int | None = None,
    overwrite: bool = False,
    cloud_mask_pool_root: Path | None = None,
    generate_temporal_averages: bool = True,
) -> int:
    """Create samples plus a deduplicated pool of their per-time cloud masks."""
    if not 0.0 <= max_cloud_fraction <= 1.0:
        raise ValueError("max_cloud_fraction must be in [0, 1].")
    if cloud_mask_pool_root is None:
        cloud_mask_pool_root = output_root / "cloud_only_mask_pool"

    indexed = index_netcdf_files(input_root)
    if not indexed:
        raise FileNotFoundError(f"No NetCDF files found under {input_root}")
    sequences = list(build_time_sequences(indexed, time_steps, interval_hours))
    if max_sequences is not None:
        sequences = sequences[:max_sequences]
    if not sequences:
        raise ValueError(
            f"No complete {time_steps}-step sequences at {interval_hours}-hour intervals."
        )

    reference_path = sequences[0][1][-1]
    area_lat, area_lon, _, _ = study_area_slices(
        reference_path, lat_min, lat_max, lon_min, lon_max
    )
    grid = build_fixed_grid(area_lat, area_lon, patch_size)
    grid_rows = (area_lat.stop - area_lat.start) // patch_size
    grid_columns = (area_lon.stop - area_lon.start) // patch_size
    print(
        f"Fixed grid: {grid_rows}x{grid_columns}={len(grid)} patches per sequence; "
        f"study area={(area_lat.stop - area_lat.start)}x"
        f"{(area_lon.stop - area_lon.start)}."
    )
    saved = 0
    rejected = 0
    skipped_existing = 0
    skipped_rejected = 0
    failed = 0
    pooled_masks = 0
    skipped_pool_masks = 0

    logged_cloud_fractions = load_logged_cloud_fractions(output_root)
    log_handle, log_writer = open_processing_log(output_root)
    try:
        for sequence_index, (target_time, paths) in enumerate(sequences, start=1):
            outputs = {
                (row, column): output_root
                / f"{target_time:%Y%m%d%H%M%S}_r{row:02d}_c{column:02d}.npz"
                for row, column, *_ in grid
            }
            pool_outputs = {
                (time_index, row, column): cloud_mask_pool_path(
                    cloud_mask_pool_root,
                    timestamp_from_path(path),
                    row,
                    column,
                )
                for time_index, path in enumerate(paths)
                for row, column, *_ in grid
            }
            pending_grid = []
            for grid_item in grid:
                row, column, *_ = grid_item
                output_path = outputs[(row, column)]
                if not overwrite and output_path.exists():
                    try:
                        with np.load(output_path) as archive:
                            has_cloud_mask = "cloud_mask_volume" in archive
                    except (OSError, ValueError):
                        has_cloud_mask = False
                    if has_cloud_mask:
                        skipped_existing += 1
                        write_log_row(
                            log_handle,
                            log_writer,
                            target_time=target_time,
                            row=row,
                            column=column,
                            status="skipped_existing",
                            output_path=output_path,
                        )
                        continue
                rejection_key = (target_time.isoformat(), row, column)
                prior_cloud_fraction = logged_cloud_fractions.get(rejection_key)
                if (
                    not overwrite
                    and prior_cloud_fraction is not None
                    and prior_cloud_fraction > max_cloud_fraction
                ):
                    skipped_rejected += 1
                    write_log_row(
                        log_handle,
                        log_writer,
                        target_time=target_time,
                        row=row,
                        column=column,
                        status="skipped_rejected",
                        output_path=output_path,
                        cloud_fraction=prior_cloud_fraction,
                    )
                    continue
                pending_grid.append(grid_item)

            missing_pool_masks = [
                path
                for path in pool_outputs.values()
                if overwrite or not path.exists()
            ]
            if not pending_grid and not missing_pool_masks:
                print(
                    f"[{sequence_index}/{len(sequences)}] "
                    f"{target_time:%Y%m%d%H%M%S}: all patches and masks complete."
                )
                continue

            try:
                (
                    grid_sst,
                    grid_mask,
                    grid_lat,
                    grid_lon,
                    grid_lat_slice,
                    grid_lon_slice,
                ) = extract_aligned_grid_sequence(
                    paths,
                    grid,
                    min_quality_level,
                )
            except Exception as exc:
                failed += len(pending_grid)
                for row, column, *_ in pending_grid:
                    write_log_row(
                        log_handle,
                        log_writer,
                        target_time=target_time,
                        row=row,
                        column=column,
                        status="error",
                        output_path=outputs[(row, column)],
                        message=f"{type(exc).__name__}: {exc}",
                    )
                print(
                    f"[{sequence_index}/{len(sequences)}] "
                    f"{target_time:%Y%m%d%H%M%S}: failed: {exc}"
                )
                continue

            sequence_pooled = 0
            for row, column, lat_slice, lon_slice, _, _ in grid:
                local_y = lat_slice.start - grid_lat_slice.start
                local_x = lon_slice.start - grid_lon_slice.start
                patch_y = slice(local_y, local_y + patch_size)
                patch_x = slice(local_x, local_x + patch_size)
                for time_index in range(len(paths)):
                    pool_path = pool_outputs[(time_index, row, column)]
                    if not overwrite and pool_path.exists():
                        skipped_pool_masks += 1
                        continue
                    try:
                        save_cloud_mask_to_pool(
                            pool_path,
                            grid_mask[time_index, patch_y, patch_x],
                        )
                    except Exception as exc:
                        failed += 1
                        print(f"Failed to save cloud mask {pool_path}: {exc}")
                        continue
                    pooled_masks += 1
                    sequence_pooled += 1

            if not pending_grid:
                print(
                    f"[{sequence_index}/{len(sequences)}] "
                    f"{target_time:%Y%m%d%H%M%S}: samples complete, "
                    f"pooled_masks={sequence_pooled}."
                )
                continue

            sequence_saved = 0
            sequence_rejected = 0
            for row, column, lat_slice, lon_slice, y, x in pending_grid:
                local_y = lat_slice.start - grid_lat_slice.start
                local_x = lon_slice.start - grid_lon_slice.start
                patch_y = slice(local_y, local_y + patch_size)
                patch_x = slice(local_x, local_x + patch_size)
                target_cloud_mask = grid_mask[-1, patch_y, patch_x]
                cloud_fraction = float(target_cloud_mask.mean())
                output_path = outputs[(row, column)]
                if cloud_fraction > max_cloud_fraction:
                    rejected += 1
                    sequence_rejected += 1
                    write_log_row(
                        log_handle,
                        log_writer,
                        target_time=target_time,
                        row=row,
                        column=column,
                        status="rejected_cloud",
                        output_path=output_path,
                        cloud_fraction=cloud_fraction,
                    )
                    continue

                try:
                    save_patch_sample(
                        output_path,
                        paths,
                        target_time,
                        grid_sst[:, patch_y, patch_x][None, ...],
                        grid_mask[:, patch_y, patch_x][None, ...],
                        grid_lat[patch_y],
                        grid_lon[patch_x],
                        y,
                        x,
                        row,
                        column,
                    )
                except Exception as exc:
                    failed += 1
                    write_log_row(
                        log_handle,
                        log_writer,
                        target_time=target_time,
                        row=row,
                        column=column,
                        status="error",
                        output_path=output_path,
                        cloud_fraction=cloud_fraction,
                        message=f"{type(exc).__name__}: {exc}",
                    )
                    continue

                saved += 1
                sequence_saved += 1
                write_log_row(
                    log_handle,
                    log_writer,
                    target_time=target_time,
                    row=row,
                    column=column,
                    status="saved",
                    output_path=output_path,
                    cloud_fraction=cloud_fraction,
                )

            print(
                f"[{sequence_index}/{len(sequences)}] "
                f"{target_time:%Y%m%d%H%M%S}: saved={sequence_saved}, "
                f"rejected={sequence_rejected}, pooled_masks={sequence_pooled}, "
                f"pending={len(pending_grid)}."
            )
    finally:
        log_handle.close()

    if generate_temporal_averages:
        print("Generating same-hour 7-day and 30-day SST averages...")
        averages_updated, averages_existing, incomplete_history = (
            add_temporal_averages_to_samples(
                indexed,
                [target_time for target_time, _ in sequences],
                output_root,
                grid,
                min_quality_level,
                overwrite=overwrite,
            )
        )
        print(
            f"Temporal averages: updated={averages_updated}, "
            f"skipped_existing={averages_existing}, "
            f"targets_without_30day_history={incomplete_history}."
        )

    print(
        f"Summary: saved={saved}, rejected_cloud={rejected}, "
        f"skipped_existing={skipped_existing}, "
        f"skipped_rejected={skipped_rejected}, pooled_masks={pooled_masks}, "
        f"skipped_pool_masks={skipped_pool_masks}, failed={failed}."
    )
    return saved


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=base / "himawari_sst_data")
    parser.add_argument(
        "output", nargs="?", type=Path, default=base / "himawari_sst_patches256"
    )
    parser.add_argument("--lat-min", type=float, default=DEFAULT_LAT_MIN)
    parser.add_argument("--lat-max", type=float, default=DEFAULT_LAT_MAX)
    parser.add_argument("--lon-min", type=float, default=DEFAULT_LON_MIN)
    parser.add_argument("--lon-max", type=float, default=DEFAULT_LON_MAX)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--time-steps", type=int, default=DEFAULT_TIME_STEPS)
    parser.add_argument("--interval-hours", type=int, default=DEFAULT_TIME_INTERVAL_HOURS)
    parser.add_argument("--min-quality-level", type=int, choices=range(6), default=4)
    parser.add_argument("--max-cloud-fraction", type=float, default=0.5)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument(
        "--cloud-mask-pool",
        type=Path,
        help="Cloud-only mask-pool directory (default: OUTPUT/cloud_only_mask_pool).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing NPZ outputs instead of resuming around them.",
    )
    parser.add_argument(
        "--skip-temporal-averages",
        action="store_true",
        help="Create SST/cloud patches only, without 7/30-day averages.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_dir():
        raise ValueError("Patch extraction input must be a directory of NetCDF files.")
    count = create_training_patches(
        args.input,
        args.output,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        patch_size=args.patch_size,
        time_steps=args.time_steps,
        interval_hours=args.interval_hours,
        min_quality_level=args.min_quality_level,
        max_cloud_fraction=args.max_cloud_fraction,
        max_sequences=args.max_sequences,
        overwrite=args.overwrite,
        cloud_mask_pool_root=args.cloud_mask_pool,
        generate_temporal_averages=not args.skip_temporal_averages,
    )
    print(f"Created {count} aligned training patches in {args.output}")


if __name__ == "__main__":
    main()
