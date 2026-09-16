"""Store each observation/grid patch once; build temporal volumes in the loader."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.himawaridata.crop_nc_to_1650 import (
    add_temporal_averages_to_samples,
    build_fixed_grid,
    index_netcdf_files,
    read_sst_and_mask_patch,
    study_area_slices,
)


def atomic_save_npz(path: Path, **fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **fields)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def create_observation_patches(
    input_root: Path, output_root: Path, *,
    patch_size: int = 256,
    lat_min: float = 17.0, lat_max: float = 50.0,
    lon_min: float = 117.0, lon_max: float = 150.0,
    min_quality_level: int = 4,
    max_frames: int | None = None,
    overwrite: bool = False,
    generate_temporal_averages: bool = True,
) -> int:
    """Save all patches, including cloudy ones needed as temporal context.

    SST stays in Celsius with missing values as NaN. Cloud mask retains the
    legacy convention: 1=missing/cloudy ocean, 0=clear ocean or land.
    A frame is [H,W], without channel or time dimensions. Averages are optional
    separate products; their absence never prevents saving an observation.
    """
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    indexed = index_netcdf_files(input_root)
    if not indexed:
        raise FileNotFoundError(f"No NetCDF files found under {input_root}")
    times = sorted(indexed)
    if max_frames is not None:
        times = times[:max_frames]
    area_y, area_x, _, _ = study_area_slices(
        indexed[times[0]], lat_min, lat_max, lon_min, lon_max
    )
    grid = build_fixed_grid(area_y, area_x, patch_size)
    ys = slice(min(g[2].start for g in grid), max(g[2].stop for g in grid))
    xs = slice(min(g[3].start for g in grid), max(g[3].stop for g in grid))
    frame_root = output_root / "frames" / "himawari"
    average_root = output_root / "averages" / "himawari"
    reference = None
    saved = skipped = 0
    for index, timestamp in enumerate(times, 1):
        source = indexed[timestamp]
        sst, cloud, lat, lon = read_sst_and_mask_patch(source, ys, xs, min_quality_level)
        if sst.shape != (ys.stop - ys.start, xs.stop - xs.start):
            raise ValueError(f"Unexpected grid shape in {source}: {sst.shape}")
        if reference is None:
            reference = (lat.copy(), lon.copy())
        elif not np.array_equal(lat, reference[0]) or not np.array_equal(lon, reference[1]):
            raise ValueError(f"Latitude/longitude grid differs: {source}")
        for row, column, py, px, y, x in grid:
            cy = slice(py.start - ys.start, py.stop - ys.start)
            cx = slice(px.start - xs.start, px.stop - xs.start)
            output = frame_root / f"{timestamp:%Y%m%d%H%M%S}_r{row:02d}_c{column:02d}.npz"
            if output.exists() and not overwrite:
                with np.load(output, allow_pickle=False) as old:
                    compatible = (
                        int(old["format_version"]) == 2
                        and str(old["timestamp"]) == timestamp.isoformat()
                        and str(old["satellite"]) == "himawari"
                        and int(old["min_quality_level"]) == min_quality_level
                        and old["sst"].shape == (patch_size, patch_size)
                        and old["cloud_mask"].shape == (patch_size, patch_size)
                        and np.array_equal(old["lat"], lat[cy])
                        and np.array_equal(old["lon"], lon[cx])
                    )
                if not compatible:
                    raise ValueError(f"Incompatible existing frame: {output}; use a new output directory")
                skipped += 1
                continue
            patch, mask = sst[cy, cx], cloud[cy, cx]
            atomic_save_npz(
                output,
                format_version=np.asarray(2), satellite=np.asarray("himawari"),
                timestamp=np.asarray(timestamp.isoformat()),
                sst=np.asarray(patch, dtype=np.float32),
                cloud_mask=np.asarray(mask, dtype=np.uint8),
                lat=lat[cy], lon=lon[cx],
                source_file=np.asarray(str(source)),
                grid_row_column=np.asarray([row, column], dtype=np.int16),
                area_patch_origin_yx=np.asarray([y, x], dtype=np.int32),
                min_quality_level=np.asarray(min_quality_level),
                cloud_fraction=np.asarray(mask.mean(), dtype=np.float32),
                valid_fraction=np.asarray((np.isfinite(patch) & (mask == 0)).mean(), dtype=np.float32),
                sst_units=np.asarray("degree_Celsius"),
            )
            saved += 1
        print(f"[{index}/{len(times)}] {timestamp:%Y%m%d%H%M%S}: saved={saved}, skipped={skipped}", flush=True)
    if generate_temporal_averages:
        updated, existing, incomplete = add_temporal_averages_to_samples(
            indexed, times, frame_root, grid, min_quality_level,
            overwrite=overwrite, averages_root=average_root,
        )
        print(f"Separate averages: updated={updated}, skipped={existing}, incomplete_history={incomplete}")
    print(f"Observation patches: saved={saved}, skipped={skipped}; {frame_root}")
    return saved
