"""9時刻全域平均＋最近傍基準を使い、アノマリー生成器で全域を一括復元。"""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from src.preprocessing.full_domain_input import load_full_frames, paths_for_time
from src.preprocessing.common import save_frame
from src.models.temporal_baseline import temporal_mean_nearest
from src.models import AnomalyInpaintingGenerator
from src.utils.checkpoint import load_generator_state
from src.utils.config import load_config
from src.utils.normalization import anomaly_to_sst_scale
from src.infer.reconstruct import resolve_device


def predict_full(model, sst, mask, baseline):
    """8の倍数へ右・下だけ反射拡張し、推論後に元の座標範囲へ戻す。"""
    h,w=sst.shape[-2:]
    if min(h,w)<137:
        raise ValueError('Domain is too small for dilation 16')
    pad=(0,(-w)%8,0,(-h)%8)
    def padded(x):
        shape=x.shape
        return F.pad(x.reshape(-1,1,h,w),pad,mode='reflect').reshape(*shape[:-2],h+pad[3],w+pad[1])
    with torch.inference_mode():
        anomaly=model(padded(sst),padded(mask),padded(baseline).unsqueeze(2))
    return anomaly[...,:h,:w]


def run(config, root, target_time, output, checkpoint=None, device='cpu', baseline_only=False):
    root,output=Path(root),Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new or empty output directory')
    data=config['data']
    if data.get('baseline_method')!='temporal_mean_nearest' or data.get('baseline_scope')!='full_domain':
        raise ValueError('Use temporal_mean_nearest with baseline_scope: full_domain')
    if int(data['time_steps'])!=9:
        raise ValueError('Nine time steps required')
    paths,times=paths_for_time(root,target_time,int(data['time_interval_hours']))
    sst,mask,ocean=load_full_frames(root,paths,data)
    diagnostics=temporal_mean_nearest(sst,mask,ocean)
    h,w=sst.shape[-2:]
    lat=np.load(root/'lat.npy',allow_pickle=False);lon=np.load(root/'lon.npy',allow_pickle=False)
    if (len(lat),len(lon))!=(h,w):
        raise ValueError('Coordinates differ from full-frame shape')
    scale=(float(data['max_temp'])-float(data['min_temp']))/2
    def celsius(value):
        return np.where(ocean[0,0].numpy(),(value[0,0].cpu().numpy()+1)*scale+data['min_temp'],np.nan)
    arrays={
        'baseline_sst':celsius(diagnostics['baseline']),
        'observed_sst':np.where(mask[0,0,-1].numpy()>0,(sst[0,0,-1].numpy()+1)*scale+data['min_temp'],np.nan),
        'ocean_mask':ocean[0,0].numpy(),
        'observation_count':diagnostics['observation_count'][0,0].numpy(),
        'interpolated_mask':diagnostics['interpolated_mask'][0,0].numpy(),
        'distance_pixels':diagnostics['distance_pixels'][0,0].numpy(),
        'lat':lat,'lon':lon,
    }
    if not baseline_only:
        if checkpoint is None:
            raise ValueError('Reconstruction requires --anomaly-checkpoint')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        trained=saved.get('config',{}).get('data',{})
        for key in ('baseline_method','baseline_scope','time_steps','time_interval_hours','min_temp','max_temp','anomaly_range'):
            if trained.get(key)!=data.get(key):
                raise ValueError(f'Checkpoint setting differs: {key}; retrain for this baseline')
        dev=resolve_device(device)
        model=AnomalyInpaintingGenerator(time_steps=9,anomaly_scale=anomaly_to_sst_scale(data)).to(dev)
        load_generator_state(model,saved);model.eval()
        anomaly=predict_full(model,sst.to(dev),mask.to(dev),diagnostics['baseline'].to(dev)).cpu()
        arrays['pred_anomaly']=np.where(ocean[0,0].numpy(),anomaly[0,0].numpy()*data['anomaly_range'],np.nan)
        arrays['reconstructed_sst']=celsius(diagnostics['baseline']+anomaly*anomaly_to_sst_scale(data))
    metadata=dict(status='complete',target_time=times[-1],input_times=times,source_frames=paths,
                  baseline_method='temporal_mean_nearest',baseline_scope='full_domain',
                  units='degC',shape=[h,w],baseline_only=baseline_only,
                  checkpoint=str(checkpoint) if checkpoint else None,
                  padding='reflect to multiple of 8, cropped after inference',
                  note='9時刻の基準SST。週平均ではない。距離は画素単位、最近傍は陸を越える場合がある。')
    save_frame(output,'.',arrays,metadata)
    print(f'Saved full domain {h}x{w}: {output}',flush=True)
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='config_temporal_mean.yaml')
    p.add_argument('--data-root',required=True)
    p.add_argument('--target-time',required=True,help='UTC ISO datetime')
    p.add_argument('--output',required=True)
    p.add_argument('--anomaly-checkpoint')
    p.add_argument('--baseline-only',action='store_true')
    p.add_argument('--device')
    a=p.parse_args();config=load_config(a.config)
    run(config,a.data_root,a.target_time,a.output,a.anomaly_checkpoint,a.device or config['training']['device'],a.baseline_only)


if __name__=='__main__':main()
