#!/usr/bin/env python3
"""Extract JMA Psbs FD01 1m daily forecasts and trailing seven-day means.

Only numpy and eccodes are required. No network access. See README_GPV.md.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import re
import shutil
import tarfile
import tempfile

import numpy as np

LOG = logging.getLogger("gpv")
ARCHIVE = re.compile(r"Z__C_RJTD_(\d{14})_OCN_GPV_Rjp_Gll2km_grib2.tar.gz$")
SUFFIX = "_Lz1-1000_Psbs_FD01_grib2.bin"


def atomic_npz(path, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
        try:
            np.savez_compressed(f, **fields)
            f.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def read_field(path, ec, initial):
    """Multi-field GRIB support: search actual parameter and depth, not order."""
    ec.codes_grib_multi_support_on()
    with path.open("rb") as f:
        while True:
            gid = ec.codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                get = lambda key: ec.codes_get(gid, key)
                if tuple(int(get(k)) for k in (
                    "discipline", "parameterCategory", "parameterNumber"
                )) != (10, 4, 15):
                    continue
                depth = float(get("scaledValueOfFirstFixedSurface")) * 10 ** (
                    -int(get("scaleFactorOfFirstFixedSurface")))
                if int(get("typeOfFirstFixedSurface")) != 160 or abs(depth - 1) > 1e-6:
                    continue
                stamp = datetime.strptime(
                    f"{int(get('dataDate')):08d}{int(get('dataTime')):04d}",
                    "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
                # Request hours explicitly; ecCodes step keys otherwise depend on units.
                ec.codes_set(gid, "stepUnits", 1)
                if (stamp != initial or get("stepType") != "avg"
                        or float(get("startStep")) != 0 or float(get("endStep")) != 24):
                    raise ValueError("Expected FD01 average over initial time + [0,24] hours")
                end = datetime.strptime(
                    f"{int(get('validityDate')):08d}{int(get('validityTime')):04d}",
                    "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
                if end != stamp + timedelta(days=1):
                    raise ValueError("Unexpected averaging end time")
                if get("gridType") != "regular_ll":
                    raise ValueError("Only regular latitude/longitude grids are supported")
                ny, nx = int(get("Nj")), int(get("Ni"))
                # Validate scan layout using coordinates rather than assuming reshape order.
                lat2 = np.asarray(ec.codes_get_array(gid, "latitudes")).reshape(ny, nx)
                lon2 = np.asarray(ec.codes_get_array(gid, "longitudes")).reshape(ny, nx)
                if not (np.allclose(lat2, lat2[:, :1]) and np.allclose(lon2, lon2[:1, :])):
                    raise ValueError("Unsupported GRIB scan layout")
                values = np.asarray(ec.codes_get_values(gid)).reshape(ny, nx)
                valid = np.isfinite(values) & (values != float(get("missingValue")))
                if int(get("bitmapPresent")):
                    valid &= np.asarray(ec.codes_get_array(gid, "bitmap")).reshape(ny, nx).astype(bool)
                sst = np.where(valid, values - 273.15, np.nan).astype(np.float32)
                if not valid.any():
                    raise ValueError("No valid water temperatures")
                return sst, lat2[:, 0], lon2[0], {
                    "period_start": stamp.isoformat(), "period_end": end.isoformat(),
                    "initial_time": stamp.isoformat(), "depth_m": depth,
                    "product": "JMA_Rjp_Psbs_FD01_forecast_daily_mean",
                    "sst_units": "degree_Celsius",
                }
            finally:
                ec.codes_release(gid)
    raise ValueError("No Psbs depth=1m field found")


def extract(archive, scratch, ec, initial):
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(SUFFIX):
                # Copy only the required member; never use its name as a local path.
                with tar.extractfile(member) as source, scratch.open("wb") as out:
                    shutil.copyfileobj(source, out, length=1024 * 1024)
                sst, lat, lon, meta = read_field(scratch, ec, initial)
                meta.update(source_archive=str(archive.resolve()), source_member=member.name)
                return sst, lat, lon, meta
    raise ValueError("Psbs FD01 upper-layer member not found")


def axis_check(axis):
    delta = np.diff(axis)
    if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all():
        raise ValueError("Coordinates must be finite 1D arrays of length >=2")
    if not ((delta > 0).all() or (delta < 0).all()):
        raise ValueError("Coordinates must be strictly monotonic")


def nearest_axis(source, target):
    axis_check(source)
    axis_check(target)
    order = np.argsort(source)
    asc = source[order]
    upper = np.clip(np.searchsorted(asc, target), 1, len(asc) - 1)
    lower = upper - 1
    chosen = np.where(abs(target - asc[lower]) <= abs(target - asc[upper]), lower, upper)
    inside = (target >= asc[0]) & (target <= asc[-1])
    return order[chosen], inside


def regrid(sst, lat, lon, target_lat, target_lon):
    iy, inside_y = nearest_axis(lat, target_lat)
    ix, inside_x = nearest_axis(lon, target_lon)
    result = sst[np.ix_(iy, ix)].copy()
    result[~(inside_y[:, None] & inside_x[None, :])] = np.nan
    return result


def weekly_mean(arrays):
    if len(arrays) != 7:
        raise ValueError("Exactly seven consecutive daily fields are required")
    stack = np.stack(arrays)
    count = np.isfinite(stack).sum(axis=0).astype(np.uint8)
    mean = np.where(count == 7, np.nansum(stack, axis=0) / 7, np.nan).astype(np.float32)
    return mean, count


def save_product(root, kind, day, sst, lat, lon, meta, args, counts=None):
    valid = np.isfinite(sst)
    fields = dict(sst=sst, valid_mask=valid.astype(np.uint8), lat=lat, lon=lon,
                  metadata_json=json.dumps(meta, sort_keys=True))
    if counts is not None:
        fields["valid_day_count"] = counts
    atomic_npz(root / kind / f"{day}.npz", **fields)
    key = "assimilation_sst" if kind == "daily" else "assimilation_weekly"
    records = []
    size = args.patch_size
    for y in range(0, len(lat) - size + 1, size):
        for x in range(0, len(lon) - size + 1, size):
            patch = sst[y:y+size, x:x+size]
            mask = np.isfinite(patch)
            if not mask.any():
                continue
            normalized = 2 * (patch - args.min_temp) / (args.max_temp - args.min_temp) - 1
            normalized = np.where(mask, normalized, 0).astype(np.float32)
            relative = Path("patches") / kind / f"{day}_r{y//size:02d}_c{x//size:02d}.npz"
            atomic_npz(root / relative, **{
                key: normalized[None], "valid_mask": mask[None].astype(np.uint8),
                "sst": patch, "lat": lat[y:y+size], "lon": lon[x:x+size],
                "min_temp": args.min_temp, "max_temp": args.max_temp,
                "metadata_json": json.dumps(meta, sort_keys=True),
            })
            records.append({"path": str(relative), "key": key,
                            "valid_fraction": float(mask.mean()),
                            "adversarial_eligible": bool(mask.all()),
                            "outside_normalization_range": int((mask & (
                                (patch < args.min_temp) | (patch > args.max_temp))).sum())})
    return records


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "gpv_processed")
    p.add_argument("--reference-grid", type=Path, help="NPZ with 1D lat/lon, e.g. a Himawari frame")
    p.add_argument("--bounds", nargs=4, type=float, default=(19.98, 50, 117, 150),
                   metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--min-temp", type=float, default=0)
    p.add_argument("--max-temp", type=float, default=35)
    p.add_argument("--max-archives", type=int)
    args = p.parse_args(argv)
    if args.patch_size <= 0 or not np.isfinite([args.min_temp, args.max_temp]).all() or args.max_temp <= args.min_temp:
        p.error("Invalid patch size or temperature range")
    if args.max_archives is not None and args.max_archives <= 0:
        p.error("--max-archives must be positive")
    south, north, west, east = args.bounds
    if not np.isfinite(args.bounds).all() or not (-90 <= south < north <= 90 and -180 <= west < east <= 360):
        p.error("Invalid bounds")
    indexed = {}
    for path in sorted(args.input.glob("*.tar.gz")):
        match = ARCHIVE.fullmatch(path.name)
        if match:
            stamp = datetime.strptime(match[1], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            if stamp.hour or stamp.minute or stamp.second:
                raise ValueError(f"Expected 00 UTC initial time: {path}")
            if stamp in indexed:
                raise ValueError(f"Duplicate initial time: {path}")
            indexed[stamp] = path
    if not indexed:
        p.error("No matching JMA Rjp archives found")
    times = sorted(indexed)
    if args.max_archives:
        times = times[:args.max_archives]
    missing = []
    cursor = times[0]
    while cursor <= times[-1]:
        if cursor not in times:
            missing.append(cursor.date().isoformat())
        cursor += timedelta(days=1)
    target = None
    if args.reference_grid:
        with np.load(args.reference_grid, allow_pickle=False) as f:
            target = (f["lat"].copy(), f["lon"].copy())
        for axis in target:
            axis_check(axis)
    try:
        import eccodes as ec
    except ImportError:
        p.error("Install dependencies: python3 -m pip install numpy eccodes")
    # A fresh output directory prevents stale products and mixed preprocessing settings.
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Output directory must be empty; choose a new directory")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"format_version": 1, "settings": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
              "missing_archive_days": missing, "processed": [], "errors": [], "patches": [],
              "weekly_unavailable": [], "grid_method": "nearest_neighbor" if target else "native_crop"}
    history = deque(maxlen=7)
    reference = None
    with tempfile.TemporaryDirectory(prefix="gpv_work_", dir=args.output) as work:
        for i, stamp in enumerate(times, 1):
            day = stamp.strftime("%Y%m%d")
            try:
                sst, lat, lon, meta = extract(indexed[stamp], Path(work) / "field.bin", ec, stamp)
                if reference is None:
                    reference = (lat.copy(), lon.copy())
                elif not all(np.array_equal(a, b) for a,b in zip(reference, (lat,lon))):
                    raise ValueError("Source grid changed between archives")
                if target is not None:
                    sst = regrid(sst, lat, lon, *target)
                    lat, lon = target
                else:
                    iy = np.flatnonzero((lat >= south) & (lat <= north))
                    ix = np.flatnonzero((lon >= west) & (lon <= east))
                    sst, lat, lon = sst[np.ix_(iy, ix)], lat[iy], lon[ix]
                if min(sst.shape) < args.patch_size or not np.isfinite(sst).any():
                    raise ValueError("Selected region has no valid water or is smaller than patch size")
                meta["grid_method"] = report["grid_method"]
                meta["normalization"] = {"min_temp": args.min_temp, "max_temp": args.max_temp, "clipped": False}
                report["patches"].extend(save_product(args.output, "daily", day, sst, lat, lon, meta, args))
                if history and stamp - history[-1][0] != timedelta(days=1):
                    history.clear()
                history.append((stamp, sst, meta["source_archive"]))
                if len(history) == 7:
                    weekly, counts = weekly_mean([h[1] for h in history])
                    weekly_meta = dict(meta, period_start=history[0][0].isoformat(),
                                       aggregation="trailing_7_daily_FD01_means",
                                       source_archives=[h[2] for h in history])
                    report["patches"].extend(save_product(args.output, "weekly", day, weekly, lat, lon, weekly_meta, args, counts))
                else:
                    report["weekly_unavailable"].append(day)
                report["processed"].append(day)
                LOG.info("[%d/%d] %s valid=%.2f%%", i, len(times), day, 100*np.isfinite(sst).mean())
            except Exception as exc:
                history.clear()
                LOG.exception("Failed %s", day)
                report["errors"].append({"day": day, "error": str(exc)})
            temporary = args.output / "report.json.tmp"
            temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
            temporary.replace(args.output / "report.json")
    LOG.info("Complete: %d days, %d failures, %d missing dates", len(report["processed"]), len(report["errors"]), len(missing))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
