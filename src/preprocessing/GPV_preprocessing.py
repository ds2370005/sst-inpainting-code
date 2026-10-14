"""JMA GPV extraction and alignment; independent of satellite preprocessing."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import deque
import re
import shutil
import tarfile
import tempfile
import numpy as np
from .common import save_frame, read_array
SUFFIX = "_Lz1-1000_Psbs_FD01_grib2.bin"
ARCHIVE = re.compile(r"Z__C_RJTD_(\d{14})_OCN_GPV_Rjp_Gll2km_grib2.tar.gz$")

def read_field(path, ec, initial):
    """Multi-field GRIB support: search actual parameter and depth, not order."""
    ec.codes_grib_multi_support_on()
    result = None
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
                if result is not None:
                    raise ValueError("Duplicate Psbs 1m field")
                result = sst, lat2[:, 0], lon2[0], {
                    "period_start": stamp.isoformat(), "period_end": end.isoformat(),
                    "initial_time": stamp.isoformat(), "depth_m": depth,
                    "product": "JMA_Rjp_Psbs_FD01_forecast_daily_mean",
                    "sst_units": "degree_Celsius",
                }
            finally:
                ec.codes_release(gid)
    if result is None:
        raise ValueError("No Psbs depth=1m field found")
    return result


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



def bilinear(sst, lat, lon, target_lat, target_lon):
    """Coordinate-based bilinear interpolation; no extrapolation or coast filling."""
    for axis in (lat, lon, target_lat, target_lon):
        delta = np.diff(axis)
        if len(axis) < 2 or not np.isfinite(axis).all() or not ((delta>0).all() or (delta<0).all()):
            raise ValueError('Expected finite monotonic 1D coordinates')
    yi,xi = np.argsort(lat),np.argsort(lon)
    lat,lon,sst = lat[yi],lon[xi],sst[np.ix_(yi,xi)]
    y = np.clip(np.searchsorted(lat,target_lat),1,len(lat)-1)
    x = np.clip(np.searchsorted(lon,target_lon),1,len(lon)-1)
    wy = (target_lat-lat[y-1])/(lat[y]-lat[y-1])
    wx = (target_lon-lon[x-1])/(lon[x]-lon[x-1])
    valid = ((target_lat>=lat[0]) & (target_lat<=lat[-1]))[:,None] & ((target_lon>=lon[0]) & (target_lon<=lon[-1]))[None,:]
    result = np.zeros((len(target_lat),len(target_lon)),np.float64)
    for ry,cy in ((y-1,1-wy),(y,wy)):
        for rx,cx in ((x-1,1-wx),(x,wx)):
            weights = cy[:,None]*cx[None,:]
            field = sst[np.ix_(ry,rx)]
            needed = weights > 1e-12
            valid &= ~needed | np.isfinite(field)
            result += np.where(np.isfinite(field),field,0)*weights
    return np.where(valid,result,np.nan).astype(np.float32)


def preprocess(input_root, output_root, target_grid, max_archives=None):
    import eccodes as ec
    indexed = {}
    for path in sorted(Path(input_root).glob('*.tar.gz')):
        match = ARCHIVE.fullmatch(path.name)
        if not match:
            continue
        stamp = datetime.strptime(match[1],'%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
        if stamp.hour or stamp.minute or stamp.second or stamp in indexed:
            raise ValueError(f'Invalid or duplicate GPV initial time: {path}')
        indexed[stamp] = path
    if not indexed:
        raise ValueError('No matching GPV archives')
    times = sorted(indexed)
    if max_archives:
        times = times[:max_archives]
    missing=[]; cursor=times[0]
    while cursor <= times[-1]:
        if cursor not in indexed:
            missing.append(cursor.date().isoformat())
        cursor += timedelta(days=1)
    daily=[]; weekly=[]; history=deque(maxlen=7)
    reference=None
    with tempfile.TemporaryDirectory(prefix='gpv_work_',dir=output_root) as work:
        for i,stamp in enumerate(times,1):
            field,lat,lon,meta = extract(indexed[stamp],Path(work)/'field.bin',ec,stamp)
            if reference is None:
                reference=(lat.copy(),lon.copy())
            elif not all(np.array_equal(a,b) for a,b in zip(reference,(lat,lon))):
                raise ValueError('GPV source grid changed')
            field=bilinear(field,lat,lon,*target_grid)
            if not np.isfinite(field).any():
                raise ValueError('GPV has no valid data on satellite grid')
            day=stamp.strftime('%Y%m%d')
            relative=str(Path('gpv/daily')/day)
            meta['grid_method']='bilinear_on_himawari_coordinates'
            save_frame(output_root,relative,dict(sst=field,valid_mask=np.isfinite(field).astype(np.uint8)),meta)
            daily.append(dict(time=stamp.replace(tzinfo=None).isoformat(),path=relative))
            if history and stamp-history[-1][0]!=timedelta(days=1):
                history.clear()
            history.append((stamp,relative,meta['source_archive']))
            if len(history)==7:
                total=np.zeros(field.shape,np.float64); count=np.zeros(field.shape,np.uint8)
                for _,path,_ in history:
                    values=read_array(output_root,path)
                    valid=np.isfinite(values)
                    total+=np.where(valid,values,0);count+=valid
                mean=np.where(count==7,total/7,np.nan).astype(np.float32)
                average_path=str(Path('gpv/weekly')/day)
                weekly_meta=dict(meta,period_start=history[0][0].isoformat(),source_archives=[h[2] for h in history],aggregation='seven_consecutive_FD01_means')
                save_frame(output_root,average_path,dict(sst=mean,valid_mask=(count==7).astype(np.uint8),valid_day_count=count),weekly_meta)
                weekly.append(dict(time=stamp.replace(tzinfo=None).isoformat(),path=average_path))
            print(f'GPV [{i}/{len(times)}] {day}',flush=True)
    return dict(daily=daily,weekly=weekly,missing_days=missing)
