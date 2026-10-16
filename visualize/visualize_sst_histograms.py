#!/usr/bin/env python3
"""Compare saved reconstruction distributions; no training or downloads."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml


def volume(array):
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 4:
        array = array[None]
    if array.ndim != 5 or array.shape[:2] != (1, 1) or array.shape[2] < 1:
        raise ValueError(f'Expected one input sample [1,1,T,H,W] or [1,T,H,W]: {array.shape}')
    return array[0, 0]


def field(array):
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 4 or array.shape[:2] != (1, 1):
        raise ValueError(f'Expected one prediction [1,1,H,W]: {array.shape}')
    return array[0, 0]


def statistics(values):
    finite = values[np.isfinite(values)]
    result = dict(count=int(finite.size), nonfinite=int(values.size-finite.size))
    if not finite.size:
        return dict(result, min=None, max=None, mean=None, std=None, span=None,
                    p01=None, p50=None, p99=None)
    return dict(result, min=float(finite.min()), max=float(finite.max()),
                mean=float(finite.mean()), std=float(finite.std()),
                span=float(np.ptp(finite)),
                **dict(zip(('p01','p50','p99'), map(float, np.percentile(finite,[1,50,99])))))


def observed_weekly(root, input_path, observed):
    """Recover the same crop's training weekly target, checking source identity."""
    meta = json.loads(input_path.with_suffix('.json').read_text())
    manifest = json.loads((root/'dataset.json').read_text())
    record = manifest['samples'][int(meta['sample_index'])]
    if record['time'] != meta['target_time'] or record['origin'] != meta['origin']:
        raise ValueError('Preview provenance does not match dataset')
    y, x = record['origin']
    h, w = observed.shape
    def crop(path):
        array = np.array(np.load(root/path/'sst.npy', mmap_mode='r', allow_pickle=False)[y:y+h,x:x+w])
        if array.shape != observed.shape:
            raise ValueError('Source crop shape mismatch')
        return array
    if not np.allclose(crop(Path(record['frames'][-1])), observed, atol=1e-4, equal_nan=True):
        raise ValueError('Source SST crop does not match reconstruction input')
    history = np.stack([crop(Path(path)) for path in record['history'][:7]])
    count = np.isfinite(history).sum(axis=0)
    result = np.full(observed.shape, np.nan)
    np.divide(np.nansum(history, axis=0), count, out=result,
              where=count >= manifest['settings']['weekly_min_observations'])
    return result


def draw_histograms(path, groups, bins, dpi, records, histogram_rows):
    fig, axes = plt.subplots(1, len(groups), figsize=(6*len(groups), 4.8), squeeze=False,
                             constrained_layout=True)
    try:
        for ax, (group, series, unit) in zip(axes[0], groups):
            finite = [values[np.isfinite(values)] for values in series.values()]
            available = [v for v in finite if v.size]
            if not available:
                ax.text(.5,.5,'No valid pixels',ha='center',transform=ax.transAxes)
                continue
            # Shared bins per comparison, spanning all finite values (no clipping).
            edges = np.histogram_bin_edges(np.concatenate(available), bins=bins)
            for (label, values), values_finite in zip(series.items(), finite):
                stats = statistics(values)
                records.append(dict(group=group, series=label, unit=unit, **stats))
                if not values_finite.size:
                    continue
                counts, _ = np.histogram(values_finite, bins=edges)
                percent = counts / values_finite.size * 100
                ax.stairs(percent, edges, label=f'{label}\nn={stats["count"]}, std={stats["std"]:.6g}', linewidth=1.7)
                for left, right, count, pct in zip(edges[:-1],edges[1:],counts,percent):
                    histogram_rows.append((group,label,float(left),float(right),int(count),float(pct)))
            ax.set(title=group, xlabel=unit, ylabel='Pixels per bin (%)')
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        fig.suptitle('Saved reconstruction diagnostics: not held-out or cloud-gap accuracy')
        fig.savefig(path, dpi=dpi)
    finally:
        plt.close(fig)
    print(f'Saved: {path}')


def visualize(run_dir, output_dir=None, input_path=None, data_root=None, bins=80, dpi=150):
    run_dir = Path(run_dir).expanduser().resolve()
    input_path = Path(input_path).expanduser().resolve() if input_path else run_dir/'outputs/preview_input.npz'
    output_dir = Path(output_dir).expanduser() if output_dir else run_dir/'histograms'
    if bins < 2 or dpi < 1:
        raise ValueError('bins must be >= 2 and dpi must be positive')
    config = yaml.safe_load((run_dir/'config.yaml').read_text())['data']
    low, high, scale = map(float, (config['min_temp'],config['max_temp'],config['anomaly_range']))
    if not np.isfinite([low,high,scale]).all() or high <= low or scale <= 0:
        raise ValueError('Invalid normalization settings')
    def celsius(a):
        return (a+1)*(high-low)/2+low
    with np.load(input_path, allow_pickle=False) as data:
        sst, masks = volume(data['sst_volume']), volume(data['mask_volume'])
        monthly, monthly_mask = volume(data['monthly_average']), volume(data['monthly_mask'])
    if sst.shape != masks.shape or monthly.shape != monthly_mask.shape or monthly.shape != (1,*sst.shape[1:]):
        raise ValueError('Input and mask shapes do not match')
    if not np.isin(masks,[0,1]).all() or not np.isin(monthly_mask,[0,1]).all():
        raise ValueError('Masks must be binary: 1=valid, 0=missing')
    observed = np.where(masks[-1]==1,celsius(sst[-1]),np.nan)
    monthly = np.where(monthly_mask[0]==1,celsius(monthly[0]),np.nan)
    with np.load(run_dir/'outputs/reconstruction.npz',allow_pickle=False) as data:
        weekly = celsius(field(data['pred_weekly']))
        anomaly = field(data['pred_anomaly'])*scale
        restored = celsius(field(data['pred_sst']))
    predictions = dict(predicted_weekly=weekly, predicted_anomaly=anomaly, reconstructed_sst=restored)
    for label, values in predictions.items():
        if values.shape != observed.shape:
            raise ValueError(f'{label}: prediction/input spatial shape mismatch')
    valid = np.isfinite(observed)
    if not valid.any():
        raise ValueError('No valid observed SST pixels')
    if any(not np.isfinite(v[valid]).all() for v in predictions.values()):
        raise ValueError('Nonfinite prediction on observed valid pixels')
    # Every curve in each comparison uses exactly the same spatial support.
    groups = [
        ('Latest SST / predicted mean / reconstruction',
         {'Observed SST':observed[valid], 'Predicted weekly':weekly[valid], 'Reconstructed SST':restored[valid]}, 'SST (deg C)'),
        ('Anomaly relative to predicted weekly mean',
         {'Observed SST - predicted weekly':(observed-weekly)[valid], 'Predicted anomaly':anomaly[valid]}, 'SST difference (deg C)'),
    ]
    monthly_valid = np.isfinite(monthly)
    if monthly_valid.any():
        if not np.isfinite(weekly[monthly_valid]).all():
            raise ValueError('Nonfinite weekly prediction on monthly valid pixels')
        groups.append(('Monthly vs predicted weekly (different periods)',
                       {'Observed monthly':monthly[monthly_valid], 'Predicted weekly':weekly[monthly_valid]}, 'SST (deg C)'))
    if data_root:
        target = observed_weekly(Path(data_root).expanduser().resolve(), input_path, observed)
        mask = np.isfinite(target)
        if not np.isfinite(weekly[mask]).all():
            raise ValueError('Nonfinite weekly prediction on weekly reference pixels')
        groups.append(('Observed vs predicted weekly mean',
                       {'Observed weekly':target[mask], 'Predicted weekly':weekly[mask]}, 'SST (deg C)'))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Output directory is not empty; use a new --output-dir')
    output_dir.mkdir(parents=True,exist_ok=True)
    records, histogram_rows = [], []
    draw_histograms(output_dir/'comparison.png', groups[:2], bins, dpi, records, histogram_rows)
    if len(groups)>2:
        draw_histograms(output_dir/'average_comparison.png',groups[2:],bins,dpi,records,histogram_rows)
    # Each predicted output gets its own narrow axis to expose tiny variations.
    zoom = [(label+' (observed pixels; auto range)',{label:values[valid]},
             'SST difference (deg C)' if label=='predicted_anomaly' else 'SST (deg C)')
            for label,values in predictions.items()]
    draw_histograms(output_dir/'prediction_zoom.png',zoom,bins,dpi,records,histogram_rows)
    summary = dict(purpose='saved_reconstruction_distribution_diagnostics_not_validation',
                   input_path=str(input_path), run_dir=str(run_dir),
                   actual_training_batch_verified=False, bins=bins,
                   observed_valid_pixels=int(valid.sum()), total_pixels=int(valid.size),
                   normalization=dict(min_temp=low,max_temp=high,anomaly_range=scale),
                   all_grid_predictions={k:statistics(v) for k,v in predictions.items()},
                   comparisons=records,
                   notes=['Comparisons use matching reference-valid pixels within each panel.',
                          'All-grid statistics include missing/land positions; they are not accuracy metrics.',
                          'Observed SST minus predicted weekly is a residual, not an independent true anomaly.',
                          'Latest SST and weekly/monthly means describe different averaging periods.'])
    (output_dir/'statistics.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    with (output_dir/'statistics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    with (output_dir/'histograms.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['group','series','bin_left_C','bin_right_C','count','percent']);writer.writerows(histogram_rows)
    for label,values in predictions.items():
        s=statistics(values[valid])
        print(f'{label}: min={s["min"]:.9g}, max={s["max"]:.9g}, std={s["std"]:.9g}, span={s["span"]:.9g} C')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--input',type=Path,help='Actual normalized reconstruction input NPZ')
    parser.add_argument('--data-root',type=Path,help='Unified dataset; enables matching observed weekly reference')
    parser.add_argument('--bins',type=int,default=80)
    parser.add_argument('--dpi',type=int,default=150)
    args=parser.parse_args()
    try:
        visualize(args.run_dir,args.output_dir,args.input,args.data_root,args.bins,args.dpi)
    except (OSError,ValueError,KeyError,IndexError) as exc:
        parser.exit(1,f'Error: {exc}\n')


if __name__=='__main__':
    main()
