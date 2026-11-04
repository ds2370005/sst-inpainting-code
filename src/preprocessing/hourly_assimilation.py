"""Build independent hourly/weekly physical-model pools from aligned Celsius NPY frames.

Input simulation.json: units=degC, cadence_hours=1, time_basis=UTC,
frames=[{time: ISO UTC, path: directory containing sst.npy}]. lat/lon.npy
must already match the satellite grid. Native product decoding is separate.
"""
import argparse
from collections import deque
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import numpy as np
from src.preprocessing.common import read_array, save_frame, write_json, window_sum, choose_origins


def parse_time(value):
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(timezone.utc).replace(tzinfo=None)
    if stamp.minute or stamp.second or stamp.microsecond:
        raise ValueError('Hourly frames must be aligned to whole UTC hours')
    return stamp


def prepare(source, output, satellite_root, patch_size=256, patches_per_frame=8, seed=42):
    source, output, satellite_root = map(Path, (source, output, satellite_root))
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory')
    metadata = json.loads((source/'simulation.json').read_text())
    for key, expected in [('units','degC'), ('cadence_hours',1), ('time_basis','UTC')]:
        if metadata.get(key) != expected:
            raise ValueError(f'simulation.json requires {key}={expected!r}; daily means are not hourly data')
    if patch_size <= 0 or patches_per_frame <= 0:
        raise ValueError('Patch size and count must be positive')
    coordinates = {}
    for key in ('lat','lon'):
        coordinates[key] = np.load(source/f'{key}.npy', allow_pickle=False)
        reference = np.load(satellite_root/f'{key}.npy', allow_pickle=False)
        if coordinates[key].shape != reference.shape or not np.allclose(coordinates[key], reference, rtol=0, atol=1e-6):
            raise ValueError(f'{key} must match the satellite grid; regrid first')
    rows = sorted(metadata['frames'], key=lambda row: parse_time(row['time']))
    times = [parse_time(row['time']) for row in rows]
    if len(set(times)) != len(times):
        raise ValueError('Duplicate simulation timestamps')
    if not rows:
        raise ValueError('No simulation frames')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'run.json', {'status':'running'})
    for key, value in coordinates.items(): np.save(output/f'{key}.npy', value)
    shape = (len(coordinates['lat']), len(coordinates['lon']))
    if min(shape) < patch_size: raise ValueError('Grid smaller than patch size')
    sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.int32)
    queue = deque()
    pools = {'hourly':[], 'weekly':[]}
    rng = np.random.default_rng(seed)
    previous = None
    for row, stamp in zip(rows, times):
        # Never bridge a missing hour with a fictitious weekly mean.
        if previous is not None and stamp-previous != timedelta(hours=1):
            queue.clear(); sums.fill(0); counts.fill(0)
        values = read_array(source, row['path'])
        if values.shape != shape: raise ValueError('Simulation frame shape differs from coordinates')
        valid = np.isfinite(values)
        def record(kind, data, valid_mask, extra):
            origins = choose_origins(window_sum(valid_mask, patch_size)==patch_size**2, patches_per_frame, rng)
            if not origins: return
            path = save_frame(output, f'{kind}/{stamp:%Y%m%dT%H%M%S}', {'sst':np.asarray(data, dtype=np.float32)},
                              dict(time=stamp.isoformat(), units='degC', **extra))
            pools[kind].extend(dict(time=stamp.isoformat(), path=path, origin=origin) for origin in origins)
        record('hourly', values, valid, {'source':row['path']})
        sums += np.where(valid, values, 0); counts += valid
        queue.append(row)
        if len(queue)>168:
            old = read_array(source, queue.popleft()['path']); ok=np.isfinite(old)
            sums -= np.where(ok,old,0); counts -= ok
        if len(queue)==168:
            full=counts==168
            mean=np.where(full,sums/168,np.nan)
            record('weekly', mean, full, {'aggregation':'168_consecutive_hourly_frames',
                   'first_sample':(stamp-timedelta(hours=167)).isoformat(), 'last_sample':stamp.isoformat()})
        previous=stamp
    if not pools['hourly'] or not pools['weekly']:
        raise ValueError('No complete hourly/weekly patches; need 168 consecutive hours and valid ocean pixels')
    result=dict(format_version=1, units='degC', cadence_hours=1, time_basis='UTC',
                patch_size=patch_size, weekly_aggregation='168_consecutive_hourly_frames', pools=pools,
                source=str(source.resolve()))
    write_json(output/'physical_pool.json',result)
    write_json(output/'run.json',{'status':'complete'})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--satellite-root',type=Path,required=True)
    parser.add_argument('--patch-size',type=int,default=256)
    parser.add_argument('--patches-per-frame',type=int,default=8)
    args=parser.parse_args()
    result=prepare(args.source,args.output,args.satellite_root,args.patch_size,args.patches_per_frame)
    print({key:len(value) for key,value in result['pools'].items()})

if __name__=='__main__': main()
