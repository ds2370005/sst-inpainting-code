"""Shared visualization of single-observation format_version=2 NPZ products."""
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

from src.datasets.frame_index import frame_key, read_frame
from src.utils.config import load_config


def load_sequence(root, *, target=None, satellite='himawari', time_steps=9,
                  interval_hours=6, patch_size=256):
    """Select exact times, oldest first, without applying training selection rules."""
    if time_steps <= 0 or interval_hours <= 0 or patch_size <= 0:
        raise ValueError('時刻数・時間間隔・パッチサイズは正の値が必要です')
    root = Path(root)
    folder = root/'frames'/satellite if (root/'frames').is_dir() else root
    paths = sorted(folder.glob('*.npz'))
    if not paths:
        raise ValueError(f'観測NPZがありません: {folder}')
    index = {frame_key(p): p for p in paths}

    def keys_for(key):
        time, row, col = key
        return [(time-timedelta(hours=interval_hours*i),row,col)
                for i in range(time_steps-1,-1,-1)]

    if target is not None:
        requested = Path(target)
        path = folder/requested if requested.parent == Path('.') else requested
        key = frame_key(path)
        if key not in index or index[key].resolve() != path.resolve():
            raise ValueError(f'対象NPZが入力フォルダにありません: {path}')
        keys = keys_for(key)
        missing = [f'{t:%Y%m%d%H%M%S}_r{r:02d}_c{c:02d}.npz' for t,r,c in keys if (t,r,c) not in index]
        if missing:
            raise ValueError('必要な観測NPZが不足しています:\n'+'\n'.join(missing))
    else:
        key = next((k for k in sorted(index) if all(t in index for t in keys_for(k))), None)
        if key is None:
            raise ValueError(f'同一区画で{interval_hours}時間間隔の{time_steps}枚が揃う対象がありません')
        keys = keys_for(key)
    selected = [index[k] for k in keys]
    products = [read_frame(p,satellite,patch_size) for p in selected]
    reference = products[-1]
    for product,path in zip(products,selected):
        if any(not np.array_equal(product[k],reference[k]) for k in ('lat','lon')):
            raise ValueError(f'対象と緯度経度が一致しません: {path}')
    return selected, products


def main(kind, default_data_root, default_config):
    parser = argparse.ArgumentParser(description='1時刻1NPZの観測を同一区画・古い順に並べて可視化します。')
    parser.add_argument('--data-root',type=Path,default=default_data_root,
                        help='frames/を含む親、またはframes/himawari/を指定')
    parser.add_argument('--target',help='最後の時刻のNPZ名またはパス。省略時は履歴が揃う最初の対象')
    parser.add_argument('--config',type=Path,default=default_config)
    parser.add_argument('--time-steps',type=int,help='枚数。1なら単一観測を表示')
    parser.add_argument('--interval-hours',type=int)
    parser.add_argument('--output',type=Path,help='出力画像のパス')
    if kind == 'sst':
        parser.add_argument('--vmin',type=float)
        parser.add_argument('--vmax',type=float)
    args = parser.parse_args()
    try:
        config = load_config(args.config)['data']
        steps = args.time_steps if args.time_steps is not None else int(config.get('time_steps',9))
        interval = args.interval_hours if args.interval_hours is not None else int(config.get('time_interval_hours',6))
        paths,products = load_sequence(args.data_root,target=args.target,
            satellite=str(config.get('satellite','himawari')),time_steps=steps,
            interval_hours=interval,patch_size=int(config.get('patch_size',256)))
        cloud = np.stack([p['cloud_mask'] for p in products])
        sst = np.stack([p['sst'] for p in products])
        valid = np.isfinite(sst) & (cloud==0)
        values = cloud if kind == 'mask' else np.ma.array(sst,mask=~valid)
        if kind == 'mask':
            cmap = ListedColormap(['white','black'])
            color_options = dict(cmap=cmap,norm=BoundaryNorm([-.5,.5,1.5],cmap.N))
            legend = 'Black (1): cloud / missing ocean | White (0): clear ocean or land'
        else:
            limits = np.percentile(sst[valid],[2,98]) if valid.any() else [0.,35.]
            lower = float(limits[0]) if args.vmin is None else args.vmin
            upper = float(limits[1]) if args.vmax is None else args.vmax
            if lower==upper and args.vmin is None and args.vmax is None:
                lower,upper = lower-.5,upper+.5
            if not np.isfinite([lower,upper]).all() or lower>=upper:
                raise ValueError('色範囲は有限値でvmin < vmaxにしてください')
            cmap = plt.get_cmap('turbo').copy()
            cmap.set_bad('lightgray')
            color_options = dict(cmap=cmap,vmin=lower,vmax=upper)
            legend = 'Gray = missing / invalid (including land)'
        columns = min(3,steps)
        rows = (steps+columns-1)//columns
        fig,axes = plt.subplots(rows,columns,figsize=(4*columns+1,3.5*rows+1),squeeze=False,constrained_layout=True)
        active=[]
        try:
            for i,ax in enumerate(axes.flat):
                if i>=steps:
                    ax.set_visible(False)
                    continue
                active.append(ax)
                product=products[i]
                im=ax.pcolormesh(product['lon'],product['lat'],values[i],shading='nearest',**color_options)
                timestamp=frame_key(paths[i])[0]
                ratio = f'Cloud / missing: {cloud[i].mean():.1%}' if kind=='mask' else f'Valid: {valid[i].mean():.1%}'
                ax.set_title(f'{timestamp:%Y-%m-%d %H:%M:%S}\n{(i-steps+1)*interval:+d} h | {ratio}',fontsize=10)
                ax.set_xlabel('Longitude (deg E)')
                ax.set_ylabel('Latitude (deg N)')
                ax.ticklabel_format(useOffset=False,style='plain')
            if kind=='sst':
                fig.colorbar(im,ax=active,label='SST (deg C)',extend='both')
            fig.suptitle(f'{paths[-1].stem}\n{legend}',fontsize=12)
            suffix = f'{steps}masks' if kind=='mask' else f'{steps}frames'
            output = args.output or Path('outputs')/f'{paths[-1].stem}_{suffix}.png'
            output.parent.mkdir(parents=True,exist_ok=True)
            fig.savefig(output,dpi=150)
        finally:
            plt.close(fig)
        for p in paths:
            print('入力:',p)
        print(f'形状: {values.shape}; 有効SST率: {valid.mean():.1%}')
        print('保存先:',output.resolve())
    except (OSError,ValueError,KeyError) as exc:
        parser.exit(1,f'エラー: {exc}\n')
