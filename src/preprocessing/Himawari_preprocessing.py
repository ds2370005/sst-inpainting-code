"""Himawari NetCDF -> full-region Celsius, cloud/land/valid masks."""
from datetime import datetime
from pathlib import Path
import re
import numpy as np
from netCDF4 import Dataset
from .common import save_frame


def decode_masks(sst, quality, flags, min_quality):
    """Cloud-screened missing ocean, not an independent cloud classification."""
    values = np.asarray(np.ma.filled(sst, np.nan), dtype=np.float32)
    quality_values = np.asarray(np.ma.filled(quality.astype(np.int16), -1))
    flag_values = np.asarray(np.ma.filled(flags, 0), dtype=np.int64)
    known = ~np.ma.getmaskarray(flags)
    land = known & ((flag_values & 2) != 0)
    ocean = known & ~land
    valid = ocean & np.isfinite(values) & ~np.ma.getmaskarray(sst)
    valid &= ~np.ma.getmaskarray(quality) & (quality_values >= min_quality)
    cloud = ocean & ~valid
    return np.where(valid, values - np.float32(273.15), np.nan), cloud, land, valid


def preprocess(input_root, output_root, bounds, min_quality=4, max_frames=None):
    south,north,west,east = bounds
    indexed = {}
    for path in sorted(Path(input_root).rglob('*.nc')):
        match = re.match(r'^(\d{14})-', path.name)
        if not match:
            raise ValueError(f'Invalid Himawari filename: {path}')
        stamp = datetime.strptime(match[1], '%Y%m%d%H%M%S')
        if stamp in indexed:
            raise ValueError(f'Duplicate observation: {stamp}')
        indexed[stamp] = path
    if not indexed:
        raise ValueError('No Himawari NetCDF files')
    times = sorted(indexed)
    if max_frames:
        times = times[:max_frames]
    records = []
    reference = None
    for i, stamp in enumerate(times, 1):
        path = indexed[stamp]
        with Dataset(path) as ds:
            lat = np.asarray(ds['lat'][:]); lon = np.asarray(ds['lon'][:])
            iy = np.flatnonzero((lat >= south) & (lat < north))
            ix = np.flatnonzero((lon >= west) & (lon < east))
            if not len(iy) or not len(ix):
                raise ValueError(f'Empty study area: {path}')
            if np.any(np.diff(iy)!=1) or np.any(np.diff(ix)!=1):
                raise ValueError('Noncontiguous study area')
            ys = slice(iy[0],iy[-1]+1); xs = slice(ix[0],ix[-1]+1)
            lat,lon = lat[ys],lon[xs]
            if reference is None:
                reference = (lat.copy(),lon.copy())
            elif not all(np.array_equal(a,b) for a,b in zip(reference,(lat,lon))):
                raise ValueError(f'Grid changed: {path}')
            variable = ds['sea_surface_temperature']
            units = str(getattr(variable,'units','kelvin')).lower()
            if units not in ('k','kelvin','kelvins'):
                raise ValueError(f'Expected decoded kelvin, got {units}')
            sst,cloud,land,valid = decode_masks(variable[0,ys,xs],ds['quality_level'][0,ys,xs],ds['l2p_flags'][0,ys,xs],min_quality)
        relative = Path('himawari') / stamp.strftime('%Y%m%d%H%M%S')
        save_frame(output_root, relative, dict(sst=sst.astype(np.float32),cloud_mask=cloud.astype(np.uint8),land_mask=land.astype(np.uint8),valid_mask=valid.astype(np.uint8)),
                   dict(timestamp=stamp.isoformat(),timezone='UTC',source=str(path.resolve()),min_quality=min_quality,units='degree_Celsius'))
        records.append(dict(time=stamp.isoformat(),path=str(relative)))
        print(f'Himawari [{i}/{len(times)}] {stamp}',flush=True)
    for key,axis in zip(('lat','lon'), reference):
        np.save(Path(output_root)/f'{key}.npy', axis)
    return records, reference
