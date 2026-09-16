"""Index single-observation products without duplicating temporal arrays."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re

import numpy as np

NAME = re.compile(r"^(\d{14})_r(\d+)_c(\d+)\.npz$")


def frame_key(path: Path) -> tuple[datetime, int, int]:
    match = NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Invalid observation filename: {path}")
    return datetime.strptime(match[1], "%Y%m%d%H%M%S"), int(match[2]), int(match[3])


@dataclass(frozen=True)
class TemporalSample:
    frames: tuple[Path, ...]
    target: Path
    averages: Path


def read_product(path: Path, satellite: str, *, averages: bool = False) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    timestamp, row, column = frame_key(path)
    time_key = "target_timestamp" if averages else "timestamp"
    if (int(data["format_version"]) != 2 or str(data["satellite"]) != satellite
            or datetime.fromisoformat(str(data[time_key])) != timestamp
            or not np.array_equal(data["grid_row_column"], [row, column])):
        raise ValueError(f"Metadata does not match observation identity: {path}")
    return data


def read_frame(path: Path, satellite: str, patch_size: int) -> dict:
    data = read_product(path, satellite)
    shape = (patch_size, patch_size)
    if data["sst"].shape != shape or data["cloud_mask"].shape != shape:
        raise ValueError(f"Invalid frame shape: {path}")
    if not np.isin(data["cloud_mask"], [0, 1]).all():
        raise ValueError(f"Nonbinary cloud mask: {path}")
    validate_coordinates(data, path, patch_size)
    return data


def validate_coordinates(data: dict, path: Path, patch_size: int) -> None:
    for name in ("lat", "lon"):
        if data[name].shape != (patch_size,) or not np.isfinite(data[name]).all():
            raise ValueError(f"Invalid {name} coordinates: {path}")


def build_frame_index(root: Path, satellite: str, time_steps: int,
                      interval_hours: int, patch_size: int,
                      max_cloud_fraction: float) -> tuple[list[TemporalSample], dict]:
    frames = {}
    for path in sorted((root / "frames" / satellite).glob("*.npz")):
        key = frame_key(path)
        if key in frames:
            raise ValueError(f"Duplicate observation identity: {path}")
        frames[key] = path
    counts = Counter()
    samples = []
    for (timestamp, row, column), target in sorted(frames.items()):
        average_path = root / "averages" / satellite / target.name
        keys = [(timestamp - timedelta(hours=interval_hours * offset), row, column)
                for offset in range(time_steps - 1, -1, -1)]
        if any(key not in frames for key in keys):
            counts["missing_history"] += 1
            continue
        if not average_path.is_file():
            counts["missing_averages"] += 1
            continue
        data = read_frame(target, satellite, patch_size)
        cloud = data["cloud_mask"]
        if float(cloud.mean()) > max_cloud_fraction:
            counts["cloudy_target"] += 1
            continue
        if not (np.isfinite(data["sst"]) & (cloud == 0)).any():
            counts["invalid_target"] += 1
            continue
        samples.append(TemporalSample(tuple(frames[key] for key in keys), target, average_path))
    counts["accepted"] = len(samples)
    return samples, dict(counts)


def load_temporal_sample(sample: TemporalSample, satellite: str, patch_size: int):
    # Keep the target explicit rather than treating array position as its identity.
    products = [read_frame(path, satellite, patch_size) for path in sample.frames]
    target = products[sample.frames.index(sample.target)]
    averages = read_product(sample.averages, satellite, averages=True)
    validate_coordinates(averages, sample.averages, patch_size)
    for product, path in list(zip(products, sample.frames)) + [(averages, sample.averages)]:
        if any(not np.array_equal(product[key], target[key]) for key in ("lat", "lon")):
            raise ValueError(f"Latitude/longitude grid differs from target: {path}")
    sst = np.stack([p["sst"] for p in products]).astype(np.float32)[None]
    cloud = np.stack([p["cloud_mask"] for p in products])[None]
    return (sst, cloud, np.asarray(averages["weekly_average"], dtype=np.float32),
            np.asarray(averages["monthly_average"], dtype=np.float32),
            np.asarray(target["sst"], dtype=np.float32)[None], target["cloud_mask"][None])
