#!/usr/bin/env python3
"""Plot Celsius full-domain products, including incomplete weekly coverage."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def visualize(directory, output=None):
    directory=Path(directory)
    report=json.loads((directory/'run.json').read_text())
    if report['status']=='complete':
        with np.load(directory/'full_reconstruction.npz',allow_pickle=False) as f:
            lat,lon=f['latitude'],f['longitude']
            panels=[('Observed SST',f['observed_sst_celsius'],False),
                    ('Reconstructed weekly',f['predicted_weekly_celsius'],False),
                    ('Predicted anomaly',f['predicted_anomaly_celsius'],True),
                    ('Reconstructed SST',f['reconstructed_sst_celsius'],False)]
        title='Full-domain reconstruction: not held-out accuracy evaluation'
    else:
        with np.load(directory/'weekly_reconstruction.npz',allow_pickle=False) as f:
            lat,lon=f['latitude'],f['longitude']
            panels=[('Observed weekly',f['observed_weekly_celsius'],False),
                    ('Network-reconstructed weekly only',f['predicted_weekly_celsius'],False)]
        title='INCOMPLETE weekly reconstruction; gray includes unreconstructed ocean'
    output=Path(output) if output else directory/'full_domain_preview.png'
    fig,axes=plt.subplots(1,len(panels),figsize=(6*len(panels),6),constrained_layout=True)
    try:
        for ax,(name,values,anomaly) in zip(axes,panels):
            cmap=plt.get_cmap('coolwarm' if anomaly else 'jet').copy();cmap.set_bad('lightgray')
            limits=(-5,5) if anomaly else (0,35)
            ax.set_facecolor('lightgray')
            im=ax.pcolormesh(lon,lat,np.ma.masked_invalid(values),shading='auto',cmap=cmap,vmin=limits[0],vmax=limits[1])
            ax.set(title=name,xlabel='Longitude (E)',ylabel='Latitude (N)')
            fig.colorbar(im,ax=ax,label='deg C')
        fig.suptitle(title+'\n'+report['target_time']+' UTC')
        output.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(output,dpi=150)
    finally:plt.close(fig)
    print(f'Saved: {output}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,required=True,help='Directory containing full-domain run.json and NPZs')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    visualize(args.input_dir,args.output)
