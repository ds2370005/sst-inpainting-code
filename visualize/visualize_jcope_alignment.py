#!/usr/bin/env python3
"""補間後JCOPEの水温・欠損マスク・統計を確認する。学習結果の評価ではない。"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def edges(axis):
    """格子中心から描画用の境界を求める。昇順・降順の両方に対応。"""
    axis=np.asarray(axis,dtype=float)
    if axis.ndim!=1 or len(axis)<2 or not np.isfinite(axis).all():
        raise ValueError('座標は有限の1次元配列が必要です')
    delta=np.diff(axis)
    if not ((delta>0).all() or (delta<0).all()):
        raise ValueError('座標は単調である必要があります')
    return np.r_[axis[0]-delta[0]/2,(axis[:-1]+axis[1:])/2,axis[-1]+delta[-1]/2]


def visualize(root,output=None,max_frames=2,start_index=0,vmin=0.,vmax=35.,cmap='jet',dpi=150):
    root=Path(root)
    output=Path(output) if output else root/'visualizations'
    if max_frames<1 or start_index<0 or dpi<1 or not np.isfinite([vmin,vmax]).all() or vmax<=vmin:
        raise ValueError('表示枚数・開始位置・描画範囲を確認してください')
    state=json.loads((root/'run.json').read_text())
    if state.get('status')!='complete':raise ValueError('補間処理が完了していません')
    manifest=json.loads((root/'simulation.json').read_text())
    if manifest.get('units')!='degC':raise ValueError('℃のデータが必要です')
    lat=np.load(root/'lat.npy',allow_pickle=False)
    lon=np.load(root/'lon.npy',allow_pickle=False)
    ye,xe=edges(lat),edges(lon)
    rows=manifest['frames'][start_index:start_index+max_frames]
    if not rows:raise ValueError('指定されたフレームがありません')
    output.mkdir(parents=True,exist_ok=True)
    colors=plt.get_cmap(cmap).copy();colors.set_bad('#d3d3d3')
    stats=[]
    for number,row in enumerate(rows,start_index):
        field=np.load(root/row['path']/'sst.npy',mmap_mode='r',allow_pickle=False)
        if field.shape!=(len(lat),len(lon)):raise ValueError('座標と水温の形状が一致しません')
        valid=np.isfinite(field); values=field[valid]
        record=dict(time=row['time'],source=row['path'],shape=list(field.shape),
                    total_pixels=int(field.size),valid_pixels=int(valid.sum()),
                    missing_pixels=int((~valid).sum()),valid_fraction=float(valid.mean()),
                    min_c=float(values.min()) if values.size else None,
                    max_c=float(values.max()) if values.size else None,
                    mean_c=float(values.mean(dtype=np.float64)) if values.size else None,
                    std_c=float(values.std(dtype=np.float64)) if values.size else None,
                    below_scale_pixels=int((values<vmin).sum()),above_scale_pixels=int((values>vmax).sum()))
        fig,axes=plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
        artist=axes[0].pcolormesh(xe,ye,np.ma.masked_invalid(field),cmap=colors,vmin=vmin,vmax=vmax,shading='flat')
        fig.colorbar(artist,ax=axes[0],label='SST (deg C)',extend='both')
        axes[0].set_title('Aligned surface SST (gray = missing)')
        mask=axes[1].pcolormesh(xe,ye,valid.astype(np.uint8),cmap=ListedColormap(['#d3d3d3','#2166ac']),vmin=-.5,vmax=1.5,shading='flat')
        bar=fig.colorbar(mask,ax=axes[1],ticks=[0,1]);bar.ax.set_yticklabels(['Missing','Valid'])
        axes[1].set_title(f'Valid pixels: {valid.mean():.2%}')
        for ax in axes:
            ax.set_xlabel('Longitude (deg E)');ax.set_ylabel('Latitude (deg N)')
            ax.set_xlim(min(xe),max(xe));ax.set_ylim(min(ye),max(ye))
            ax.set_aspect('equal');ax.grid(alpha=.2)
        fig.suptitle(f'JCOPE bilinear alignment check | {row["time"]}\nMissing includes land, source gaps and outside coverage')
        filename=f'jcope_alignment_{number:05d}.png'
        fig.savefig(output/filename,dpi=dpi);plt.close(fig)
        record['image']=filename;stats.append(record)
        print(f'Saved: {output/filename}',flush=True)
    report=dict(input_root=str(root.resolve()),vmin=vmin,vmax=vmax,cmap=cmap,frames=stats)
    (output/'statistics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(f'Saved: {output/"statistics.json"}',flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='JCOPE補間処理の出力フォルダ')
    p.add_argument('--output',type=Path,help='省略時は入力フォルダ内のvisualizations')
    p.add_argument('--max-frames',type=int,default=2)
    p.add_argument('--start-index',type=int,default=0)
    p.add_argument('--vmin',type=float,default=0)
    p.add_argument('--vmax',type=float,default=35)
    p.add_argument('--cmap',default='jet')
    p.add_argument('--dpi',type=int,default=150)
    a=p.parse_args();visualize(a.input,a.output,a.max_frames,a.start_index,a.vmin,a.vmax,a.cmap,a.dpi)

if __name__=='__main__':main()
