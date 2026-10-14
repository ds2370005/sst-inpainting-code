"""One command prepares Himawari masks, aligned GPV, and random crop indices."""
import argparse
from pathlib import Path
import numpy as np
from src.utils.config import load_config
from . import Himawari_preprocessing, GPV_preprocessing
from .common import FORMAT_VERSION, write_json
from .sampling import build_indices


def run(args):
    # Fail before expensive satellite conversion if the GRIB runtime is missing.
    import eccodes  # noqa: F401
    config=load_config(args.config)
    settings=dict(config['data'], seed=int(config.get('seed',42)), patches_per_frame=args.patches_per_frame,
                  weekly_min_observations=3,monthly_min_observations=5,
                  bounds=list(args.bounds),min_quality=args.min_quality,
                  cloud_policy='quality_screened_missing_ocean',crop_policy='random_full_ocean',
                  gpv_policy='FD01_depth1m_daily_forecast',gpv_interpolation='bilinear')
    if settings.get('satellite','himawari')!='himawari':
        raise ValueError('Only Himawari is supported')
    for key in ('patch_size','time_steps','time_interval_hours','patches_per_frame'):
        if int(settings[key])!=settings[key] or settings[key]<=0:raise ValueError(f'Invalid {key}')
    if not 0<=settings['max_cloud_fraction']<=1 or not settings['max_temp']>settings['min_temp']:
        raise ValueError('Invalid cloud fraction or normalization')
    if not np.isfinite([settings['min_temp'],settings['max_temp']]).all():raise ValueError('Nonfinite temperatures')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use an empty output directory (old products are not overwritten)')
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'run.json',dict(status='running',settings=settings))
    try:
        frames,grid=Himawari_preprocessing.preprocess(args.himawari_input,args.output,args.bounds,args.min_quality,args.max_frames)
        gpv=GPV_preprocessing.preprocess(args.gpv_input,args.output,grid,args.max_archives)
        index=build_indices(args.output,frames,gpv,settings)
        result=dict(format_version=FORMAT_VERSION,settings=settings,himawari_frames=frames,gpv=gpv,**index)
        write_json(args.output/'dataset.json',result)
        ready=bool(index['samples'] and all(index['gpv_pools'].values()))
        write_json(args.output/'run.json',dict(status='complete',training_ready=ready,
                    samples=len(index['samples']),gpv_daily=len(index['gpv_pools']['daily']),gpv_weekly=len(index['gpv_pools']['weekly']),settings=settings))
        print(f"Preprocessing complete. training_ready={ready}; {args.output / 'dataset.json'}")
        if not ready:print('Insufficient histories or valid crops. Inspect run.json and dataset.json; training will reject empty pools.')
        return result
    except BaseException as exc:
        write_json(args.output/'run.json',dict(status='failed',error=str(exc),settings=settings))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--himawari-input',type=Path,required=True)
    p.add_argument('--gpv-input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',default='config.yaml')
    p.add_argument('--bounds',type=float,nargs=4,default=[17,50,117,150],metavar=('SOUTH','NORTH','WEST','EAST'))
    p.add_argument('--min-quality',type=int,choices=range(6),default=4)
    p.add_argument('--patches-per-frame',type=int,default=8)
    p.add_argument('--max-frames',type=int)
    p.add_argument('--max-archives',type=int)
    args=p.parse_args()
    south,north,west,east=args.bounds
    if not np.isfinite(args.bounds).all() or not -90<=south<north<=90 or not -180<=west<east<=360:
        p.error('Invalid bounds')
    if any(v is not None and v<=0 for v in (args.max_frames,args.max_archives,args.patches_per_frame)):
        p.error('Limits must be positive')
    run(args)


if __name__=='__main__':main()
