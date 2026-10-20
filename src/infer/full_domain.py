"""Iterative full-domain weekly reconstruction (Hirahara IV-C), then anomaly.

The paper gives no numerical inference cloud threshold, scan lattice, overlap
rule or deadlock rule. Our explicit choices: <=50% missing ocean in a weekly
patch, random selection on an overlapping lattice, last-write-wins, fail closed
when no advancing patch exists. Monthly and satellite inputs remain original.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.models import AverageEstimationGenerator, AnomalyInpaintingGenerator
from src.infer.reconstruct import resolve_device
from src.preprocessing.common import FORMAT_VERSION, read_array, window_sum, write_json
from src.utils.config import load_config
from src.utils.normalization import anomaly_to_sst_scale


def parse_time(value):
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(timezone.utc).replace(tzinfo=None)
    return stamp


def load_domain(root, target_time, config):
    root = Path(root)
    if json.loads((root/'run.json').read_text()).get('status') != 'complete':
        raise ValueError('Preprocessing is incomplete')
    manifest = json.loads((root/'dataset.json').read_text())
    if manifest['format_version'] != FORMAT_VERSION:
        raise ValueError('Unsupported preprocessing format')
    settings = manifest['settings']
    for key in ('time_steps','time_interval_hours','min_temp','max_temp','satellite'):
        if settings.get(key) != config['data'].get(key):
            raise ValueError(f'Data/config mismatch: {key}')
    if config['data'].get('sample_transforms'):
        raise ValueError('Full-domain inference does not support sample_transforms')
    stamp = parse_time(target_time)
    index = {parse_time(r['time']):r['path'] for r in manifest['himawari_frames']}
    times = [stamp-timedelta(hours=settings['time_interval_hours']*i)
             for i in reversed(range(settings['time_steps']))]
    days = [stamp-timedelta(days=i) for i in range(30)]
    missing = sorted(set(times+days)-index.keys())
    if missing:
        raise ValueError(f'Missing required SST history: {missing[0].isoformat()} ({len(missing)} timestamps)')
    lat = np.load(root/'lat.npy',allow_pickle=False)
    lon = np.load(root/'lon.npy',allow_pickle=False)
    shape = (len(lat),len(lon))
    def read(stamp, key='sst'):
        a = np.asarray(read_array(root,index[stamp],key))
        if a.shape != shape:
            raise ValueError(f'Frame/grid shape mismatch: {index[stamp]} / {key}')
        return a
    # Unknown coast/flags are excluded, rather than turning all non-land into sea.
    ocean = (read(stamp,'valid_mask')==1) | (read(stamp,'cloud_mask')==1)
    if not ocean.any():
        raise ValueError('No known ocean pixels in target frame')
    sst = np.stack([read(t) for t in times]).astype(np.float32)
    sst[:,~ocean] = np.nan
    monthly_sum = np.zeros(shape,np.float64); monthly_count = np.zeros(shape,np.int16)
    weekly_sum = np.zeros(shape,np.float64); weekly_count = np.zeros(shape,np.int16)
    for i,t in enumerate(days):
        a = read(t); valid = np.isfinite(a) & ocean
        monthly_sum += np.where(valid,a,0); monthly_count += valid
        if i<7:
            weekly_sum += np.where(valid,a,0); weekly_count += valid
    def mean(total,count,minimum):
        out = np.full(shape,np.nan,np.float32)
        np.divide(total,count,out=out,where=(count>=minimum)&ocean)
        return out
    return dict(sst=sst, ocean=ocean, lat=lat, lon=lon,
                monthly=mean(monthly_sum,monthly_count,settings['monthly_min_observations']),
                weekly=mean(weekly_sum,weekly_count,settings['weekly_min_observations']),
                target_time=stamp.isoformat(), input_times=[t.isoformat() for t in times])


def origins(length, patch, stride):
    return np.unique(np.r_[np.arange(0,length-patch+1,stride),length-patch]).astype(int)


def reconstruct_weekly(weekly, ocean, predict_patch, *, patch=256, stride=128,
                       max_missing=.5, max_iterations=10000, seed=42, progress=None):
    """Each prediction overwrites the whole ocean part of the selected patch.

    A separate coverage mask tracks network-reconstructed pixels, including
    initially observed pixels. Each iteration must advance coverage.
    """
    weekly = np.asarray(weekly,dtype=np.float32)
    ocean = np.asarray(ocean,dtype=bool)
    if weekly.ndim != 2 or weekly.shape != ocean.shape or min(weekly.shape)<patch:
        raise ValueError('Weekly/ocean shapes must match and contain a full patch')
    if not 1<=stride<=patch or not 0<=max_missing<=1 or max_iterations<1 or not ocean.any():
        raise ValueError('Invalid iteration settings or empty ocean')
    ys=origins(weekly.shape[0],patch,stride); xs=origins(weekly.shape[1],patch,stride)
    grid=np.ix_(ys,xs)
    sea_counts=window_sum(ocean,patch)[grid]
    canvas=np.where(ocean,weekly,np.nan).copy()
    coverage=np.zeros_like(ocean)
    rng=np.random.default_rng(seed); records=[]; reason='max_iterations'
    for iteration in range(max_iterations):
        if np.all(coverage[ocean]):
            reason='complete';break
        missing=window_sum(ocean & ~np.isfinite(canvas),patch)[grid]
        new=window_sum(ocean & ~coverage,patch)[grid]
        eligible=(sea_counts>0)&(new>0)&(missing<=max_missing*sea_counts)
        candidates=np.argwhere(eligible)
        if not len(candidates):
            reason='no_eligible_patch';break
        iy,ix=candidates[int(rng.integers(len(candidates)))]
        y,x=int(ys[iy]),int(xs[ix]); sl=np.s_[y:y+patch,x:x+patch]
        predicted=np.asarray(predict_patch(y,x),dtype=np.float32)
        sea=ocean[sl]
        if predicted.shape!=(patch,patch) or not np.isfinite(predicted[sea]).all():
            raise ValueError('Weekly generator returned invalid shape or nonfinite ocean values')
        canvas[sl]=np.where(sea,predicted,np.nan)
        coverage[sl] |= sea
        record=dict(iteration=iteration+1,origin=[y,x],new_pixels=int(new[iy,ix]),
                    missing_fraction_before=float(missing[iy,ix]/sea_counts[iy,ix]))
        records.append(record)
        if progress:progress(record,int((ocean & ~coverage).sum()))
    if np.all(coverage[ocean]):reason='complete'
    return canvas,coverage,dict(status=reason,iterations=len(records),patches=records,
        remaining_ocean_pixels=int((ocean & ~coverage).sum()),
        patch_size=patch,stride=stride,max_missing_fraction=max_missing,seed=seed)


def normalized(values, data):
    valid=np.isfinite(values)
    array=np.where(valid,2*(values-data['min_temp'])/(data['max_temp']-data['min_temp'])-1,0)
    return torch.from_numpy(array.astype(np.float32)),torch.from_numpy(valid.astype(np.float32))


def full_anomaly(model, sst, weekly, ocean, data, device):
    """One full-field forward, with edge padding for the three downsamplings."""
    h,w=weekly.shape
    # Reflection dilation=16 needs bottleneck dimensions >16.
    ph=max(136,math.ceil(h/8)*8)-h; pw=max(136,math.ceil(w/8)*8)-w
    v,m=normalized(sst,data); base,_=normalized(weekly,data)
    def pad(a):return F.pad(a,(0,pw,0,ph,0,0),mode='replicate')
    with torch.no_grad():
        result=model(pad(v[None,None]).to(device),pad(m[None,None]).to(device),
                     pad(base[None,None,None]).to(device))
    if result.shape!=(1,1,h+ph,w+pw):
        raise ValueError('Unexpected full-domain anomaly output shape')
    anomaly=result[0,0,:h,:w].cpu().numpy()*float(data['anomaly_range'])
    if not np.isfinite(anomaly[ocean]).all():
        raise ValueError('Nonfinite full-domain anomaly')
    return np.where(ocean,anomaly,np.nan).astype(np.float32)


def run(args):
    started=time.perf_counter()
    config=load_config(args.config); data=config['data']
    if data['patch_size']!=256:
        raise ValueError('Paper iterative weekly reconstruction requires 256x256 patches')
    anomaly_to_sst_scale(data)  # validates normalization
    out=Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty --output-dir')
    domain=load_domain(args.data_root,args.target_time,config)
    device=resolve_device(args.device or config['training']['device'])
    average=AverageEstimationGenerator(time_steps=data['time_steps']).to(device).eval()
    anomaly=AnomalyInpaintingGenerator(time_steps=data['time_steps'],
              anomaly_scale=anomaly_to_sst_scale(data)).to(device).eval()
    for model,path in [(average,args.average_checkpoint),(anomaly,args.anomaly_checkpoint)]:
        ckpt=torch.load(path,map_location=device,weights_only=True)
        if 'config' not in ckpt:
            raise ValueError('Full-domain checkpoint requires training config metadata')
        for key in ('time_steps','min_temp','max_temp','anomaly_range'):
            if ckpt['config']['data'].get(key)!=data.get(key):
                raise ValueError(f'Checkpoint/config mismatch: {key}')
        from src.utils.checkpoint import load_generator_state
        load_generator_state(model,ckpt)
    out.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',target_time=domain['target_time'],input_times=domain['input_times'],
                data_root=str(Path(args.data_root).resolve()),
                average_checkpoint=str(Path(args.average_checkpoint).resolve()),
                anomaly_checkpoint=str(Path(args.anomaly_checkpoint).resolve()),
                settings=dict(patch_size=256,stride=args.stride,max_missing=args.max_missing,
                              max_iterations=args.max_iterations,seed=args.seed),
                interpretation='weekly missing-ocean fraction; random lattice; overwrite overlaps; full-field anomaly')
    write_json(out/'run.json',report)
    def predict(y,x):
        sl=np.s_[y:y+256,x:x+256]
        volume,mask=normalized(domain['sst'][:,sl[0],sl[1]],data)
        monthly,monthly_mask=normalized(domain['monthly'][sl],data)
        with torch.no_grad():
            pred=average(volume[None,None].to(device),mask[None,None].to(device),
                         monthly[None,None,None].to(device),monthly_mask[None,None,None].to(device))
        return (pred[0,0].cpu().numpy()+1)*(data['max_temp']-data['min_temp'])/2+data['min_temp']
    try:
        canvas,coverage,iteration=reconstruct_weekly(domain['weekly'],domain['ocean'],predict,
            stride=args.stride,max_missing=args.max_missing,max_iterations=args.max_iterations,seed=args.seed,
            progress=lambda r,n:print(f'weekly iteration={r["iteration"]} origin={r["origin"]} remaining={n}',flush=True))
        report['weekly_iteration']=iteration
        np.savez_compressed(out/'weekly_reconstruction.npz',weekly_canvas_celsius=canvas,
            predicted_weekly_celsius=np.where(coverage,canvas,np.nan),reconstructed_mask=coverage,
            observed_weekly_celsius=domain['weekly'],observed_monthly_celsius=domain['monthly'],
            ocean_mask=domain['ocean'],latitude=domain['lat'],longitude=domain['lon'])
        if iteration['status']!='complete':
            raise RuntimeError(f'Weekly reconstruction incomplete: {iteration["status"]}; remaining={iteration["remaining_ocean_pixels"]}')
        delta=full_anomaly(anomaly,domain['sst'],canvas,domain['ocean'],data,device)
        np.savez_compressed(out/'full_reconstruction.npz',predicted_weekly_celsius=canvas,
            predicted_anomaly_celsius=delta,reconstructed_sst_celsius=canvas+delta,
            observed_sst_celsius=domain['sst'][-1],ocean_mask=domain['ocean'],
            latitude=domain['lat'],longitude=domain['lon'])
        report['status']='complete'
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc));raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-started
        write_json(out/'run.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('config.yaml'))
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--target-time',required=True,help='UTC ISO timestamp, e.g. 2025-05-30T00:00:00')
    p.add_argument('--average-checkpoint',type=Path,required=True)
    p.add_argument('--anomaly-checkpoint',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device')
    p.add_argument('--stride',type=int,default=128)
    p.add_argument('--max-missing',type=float,default=.5)
    p.add_argument('--max-iterations',type=int,default=10000)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    try:run(args)
    except (ValueError,RuntimeError,OSError,KeyError) as exc:p.exit(1,f'Error: {exc}\n')


if __name__=='__main__':main()
