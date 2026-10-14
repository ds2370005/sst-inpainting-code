#!/usr/bin/env python3
"""Regression diagnostics; optional temperature-threshold ROC/PR classification.

By default compare reconstruction with observed input pixels, NOT hidden truth.
External truth NPZ: sst_celsius and valid_mask on exactly the prediction grid.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml


def regression_metrics(truth, prediction):
    error = prediction - truth
    denominator = np.sum((truth-truth.mean())**2)
    correlation = None
    if truth.size > 1 and np.std(truth) > 0 and np.std(prediction) > 0:
        correlation = float(np.corrcoef(truth, prediction)[0,1])
    return dict(n_pixels=int(truth.size), mae_C=float(np.abs(error).mean()),
                rmse_C=float(np.sqrt(np.mean(error**2))), bias_C=float(error.mean()),
                r2=float(1-np.sum(error**2)/denominator) if denominator>0 else None,
                pearson_r=correlation)


def binary_curves(labels, scores):
    """Group tied scores before ROC/PR integration; positive class is high SST."""
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum()); negatives = len(labels)-positives
    if not positives or not negatives:
        return None
    order = np.argsort(-scores, kind='stable')
    ordered = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered)), len(ordered)-1]
    tp = np.cumsum(labels[order])[ends]
    fp = (ends+1)-tp
    tpr = np.r_[0, tp/positives]; fpr = np.r_[0, fp/negatives]
    precision = np.r_[1, tp/(tp+fp)]
    # trapezoidal ROC; stepwise AP, not trapezoidal PR area
    auc = float(np.sum(np.diff(fpr)*(tpr[:-1]+tpr[1:])/2))
    ap = float(np.sum(np.diff(tpr)*precision[1:]))
    return dict(fpr=fpr, tpr=tpr, precision=precision, auc=auc, average_precision=ap)


def as_field(array):
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None,None]
    if array.ndim != 4 or array.shape[1] != 1 or array.shape[0] < 1:
        raise ValueError(f'Expected [B,1,H,W] or [H,W], got {array.shape}')
    return array


def save_figure(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_logs(run_dir, out):
    rows=[]
    pattern=re.compile(r'epoch=(\d+)/(\d+)\s+loss_d=(\S+)\s+loss_g=(\S+)\s+loss_rec=(\S+)\s+loss_adv=(\S+)')
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    any_found=False
    for ax,stage in zip(axes,('average','anomaly')):
        path=run_dir/'logs'/f'{stage}.log'
        records=[]
        if path.exists():
            for match in pattern.finditer(path.read_text()):
                values=[float(match[i]) for i in range(3,7)]
                if not np.isfinite(values).all():
                    continue
                records.append((int(match[1]),*values))
        if records:
            any_found=True
            data=np.asarray(records)
            for i,key in enumerate(('loss_d','loss_g','loss_rec','loss_adv'),1):
                ax.plot(data[:,0],data[:,i],marker='o',label=key)
            rows.extend((stage,*record) for record in records)
            ax.legend()
        else:
            ax.text(.5,.5,'No finite epoch logs',ha='center',transform=ax.transAxes)
        ax.set(title=f'{stage}: training losses (not validation)',xlabel='Epoch',ylabel='Loss')
    if any_found:
        save_figure(fig,out/'training_losses.png')
    else:
        plt.close(fig)
    with (out/'training_losses.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['stage','epoch','loss_d','loss_g','loss_rec','loss_adv']);writer.writerows(rows)
    state_path=run_dir/'run.json'
    if state_path.exists():
        state=json.loads(state_path.read_text())
        times={stage:info['elapsed_seconds'] for stage,info in state.get('stages',{}).items() if 'elapsed_seconds' in info}
        if times:
            fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
            ax.bar(list(times),list(times.values()))
            ax.set(ylabel='Wall-clock seconds',title='Stage runtime (includes I/O and process startup)')
            save_figure(fig,out/'runtimes.png')


def evaluate(args):
    run_dir=args.run_dir.expanduser().resolve()
    out=args.output_dir or run_dir/'evaluation'
    with (run_dir/'config.yaml').open() as f:
        config=yaml.safe_load(f)['data']
    low,high=float(config['min_temp']),float(config['max_temp'])
    if not np.isfinite([low,high]).all() or high<=low:
        raise ValueError('Invalid normalization range')
    with np.load(run_dir/'outputs/reconstruction.npz',allow_pickle=False) as f:
        prediction=as_field(f['pred_sst']).astype(np.float64)
    prediction=(prediction+1)*(high-low)/2+low
    if args.truth:
        with np.load(args.truth,allow_pickle=False) as f:
            truth=as_field(f['sst_celsius']).astype(np.float64)
            mask=as_field(f['valid_mask'])
        scope='external_reference; independence and coordinate alignment must be verified by user'
        source=str(args.truth)
    else:
        source=args.input or run_dir/'outputs/preview_input.npz'
        with np.load(source,allow_pickle=False) as f:
            volume=f['sst_volume'];masks=f['mask_volume']
            if volume.ndim==4:
                volume=volume[None];masks=masks[None]
            if volume.ndim!=5 or volume.shape[1]!=1 or volume.shape!=masks.shape:
                raise ValueError('Invalid inference input dimensions')
            truth=volume[:,:,-1].astype(np.float64)*(high-low)/2+(high+low)/2
            mask=masks[:,:,-1]
        scope='observed_input_pixels_only; not held-out accuracy or cloud-gap accuracy'
    if truth.shape!=prediction.shape or mask.shape!=truth.shape:
        raise ValueError('Reference, mask and prediction shapes must match exactly; no broadcasting/resizing')
    if not np.isin(mask,[0,1]).all():
        raise ValueError('Mask must be binary')
    valid=(mask==1)&np.isfinite(truth)
    if not valid.any():
        raise ValueError('No valid reference pixels')
    if not np.isfinite(prediction[valid]).all():
        raise ValueError('Nonfinite predictions at valid reference pixels')
    y,p=truth[valid],prediction[valid]
    metrics=dict(scope=scope,reference_source=str(source),regression=regression_metrics(y,p),
                 excluded_reference_pixels=int(valid.size-valid.sum()),
                 classification=None)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose a new/empty --output-dir to avoid mixing reports')
    out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(16,4.8),constrained_layout=True)
    scatter=axes[0].hexbin(y,p,gridsize=65,mincnt=1,bins='log',cmap='viridis')
    limits=[float(min(y.min(),p.min())),float(max(y.max(),p.max()))]
    if limits[0]==limits[1]:limits=[limits[0]-1,limits[1]+1]
    axes[0].plot(limits,limits,'r--');axes[0].set(xlabel='Reference SST (C)',ylabel='Predicted SST (C)',title='Pixel density and identity line')
    fig.colorbar(scatter,ax=axes[0],label='Pixel count (log scale)')
    axes[1].hist(p-y,bins=60);axes[1].axvline(0,color='r',linestyle='--')
    axes[1].set(xlabel='Prediction - reference (C)',ylabel='Pixels',title='Error distribution')
    axes[2].hexbin(y,p-y,gridsize=65,mincnt=1,bins='log',cmap='viridis');axes[2].axhline(0,color='r',linestyle='--')
    axes[2].set(xlabel='Reference SST (C)',ylabel='Error (C)',title='Error vs temperature')
    fig.suptitle('External reference diagnostics' if args.truth else 'Observed input pixels only: NOT held-out/cloud-gap evaluation')
    save_figure(fig,out/'regression.png')
    if args.temperature_threshold is not None:
        threshold=args.temperature_threshold
        labels=y>=threshold;decisions=p>=threshold
        tn=int((~labels&~decisions).sum());fp=int((~labels&decisions).sum())
        fn=int((labels&~decisions).sum());tp=int((labels&decisions).sum())
        curves=binary_curves(labels,p)
        metrics['classification']=dict(positive_definition=f'reference SST >= {threshold} C',
            score='predicted SST in C (not a probability)',decision_threshold_C=threshold,
            tn=tn,fp=fp,fn=fn,tp=tp,accuracy=float((tp+tn)/len(y)),
            precision=float(tp/(tp+fp)) if tp+fp else None,
            recall=float(tp/(tp+fn)) if tp+fn else None,
            f1=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else None,
            roc_auc=curves['auc'] if curves else None,
            average_precision=curves['average_precision'] if curves else None,
            curve_status='ok' if curves else 'undefined: reference contains only one class')
        fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
        axes[0].imshow([[tn,fp],[fn,tp]],cmap='Blues')
        for (i,j),v in np.ndenumerate(np.array([[tn,fp],[fn,tp]])):
            axes[0].text(j,i,str(v),ha='center',va='center',color='darkorange')
        axes[0].set(xticks=[0,1],yticks=[0,1],xticklabels=['below','high'],yticklabels=['below','high'],xlabel='Predicted class',ylabel='Reference class',title=f'Confusion matrix at {threshold} C')
        if curves:
            axes[1].plot(curves['fpr'],curves['tpr'],label=f"AUC={curves['auc']:.3f}");axes[1].plot([0,1],[0,1],'k--');axes[1].legend()
            axes[2].step(curves['tpr'],curves['precision'],where='pre',label=f"AP={curves['average_precision']:.3f}");axes[2].axhline(labels.mean(),color='k',linestyle='--');axes[2].legend()
            with (out/'roc_pr.csv').open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['fpr','tpr_recall','precision']);writer.writerows(zip(curves['fpr'],curves['tpr'],curves['precision']))
        else:
            for ax in axes[1:]:ax.text(.5,.5,'Undefined: only one reference class',ha='center',transform=ax.transAxes)
        axes[1].set(xlabel='False positive rate',ylabel='True positive rate',title='ROC',xlim=(0,1),ylim=(0,1.02))
        axes[2].set(xlabel='Recall',ylabel='Precision',title='Precision-recall',xlim=(0,1),ylim=(0,1.02))
        scope_title = 'External reference' if args.truth else 'Observed input only; NOT held-out/cloud-gap evaluation'
        fig.suptitle(f'{scope_title}\nHigh-temperature classification, reference SST >= {threshold} C')
        save_figure(fig,out/'classification.png')
    plot_logs(run_dir,out)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(metrics,indent=2,allow_nan=False))
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--input',type=Path,help='Normalized inference NPZ, default outputs/preview_input.npz')
    parser.add_argument('--truth',type=Path,help='Aligned reference NPZ with sst_celsius and valid_mask')
    parser.add_argument('--temperature-threshold',type=float,help='Enable ROC/PR for reference SST >= this Celsius threshold')
    args=parser.parse_args()
    if args.temperature_threshold is not None and not np.isfinite(args.temperature_threshold):
        parser.error('Temperature threshold must be finite')
    try:evaluate(args)
    except (OSError,ValueError,KeyError) as exc:parser.exit(1,f'Error: {exc}\n')


if __name__=='__main__':main()
