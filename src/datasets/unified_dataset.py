"""Unified lazy loader. Sample transforms extend features without changing readers."""
from pathlib import Path
import importlib
import json
import re
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from src.preprocessing.common import FORMAT_VERSION,read_array


def observed_mean(fields, minimum):
    values=np.stack(fields)
    valid=np.isfinite(values)
    count=valid.sum(0)
    result=np.full(count.shape,np.nan,np.float32)
    np.divide(np.where(valid,values,0).sum(0),count,out=result,where=count>=minimum)
    return result


def load_transforms(names):
    """Each config entry is 'package.module:ClassName'; classes take no arguments."""
    transforms=[]
    for name in names:
        module,attribute=name.split(':',1)
        transforms.append(getattr(importlib.import_module(module),attribute)())
    return transforms


class UnifiedSSTDataset(Dataset):
    def __init__(self,root,config,*,stage,transforms=()):
        if stage not in ('average','anomaly'):raise ValueError('Invalid stage')
        self.root=Path(root);self.stage=stage
        status=json.loads((self.root/'run.json').read_text())
        if status.get('status')!='complete':raise ValueError('Preprocessing has not completed')
        manifest=json.loads((self.root/'dataset.json').read_text())
        if manifest['format_version']!=FORMAT_VERSION:raise ValueError('Unsupported dataset format')
        self.settings=manifest['settings']
        for key in ('patch_size','time_steps','time_interval_hours','min_temp','max_temp','satellite','max_cloud_fraction'):
            if config['data'].get(key)!=self.settings.get(key):
                raise ValueError(f'Training/preprocessing setting differs: {key}')
        self.samples=manifest['samples']
        self.selection_counts=dict(manifest['selection_counts'])
        months=config['data'].get('training_months')
        if months is not None:
            if not isinstance(months,list) or not months or any(
                not isinstance(m,str) or re.fullmatch(r'\d{4}-(0[1-9]|1[0-2])',m) is None for m in months
            ):
                raise ValueError('data.training_months must be a nonempty list of YYYY-MM strings')
            before=len(self.samples)
            self.samples=[row for row in self.samples if row['time'][:7] in months]
            if not self.samples:
                raise ValueError('No satellite samples in data.training_months')
            self.selection_counts.update(accepted_before_month_filter=before,
                                         excluded_by_month=before-len(self.samples),accepted=len(self.samples))
            print('Satellite target months:',sorted(set(months)),flush=True)
            print('Satellite samples per month:',dict(sorted(Counter(r['time'][:7] for r in self.samples).items())),flush=True)
        # 月の選択は対象時刻だけ。入力履歴・月平均のための過去フレームは維持する。
        # One eligible crop per source timestamp; never use the target image as donor.
        donors = {}
        for row in self.samples:
            for path in row['frames']:
                donors.setdefault(path, row['origin'])
        self.cloud_donors = list(donors.items())
        self.cloud_donor_index = {path:i for i,(path,_) in enumerate(self.cloud_donors)}
        if stage == 'anomaly' and len(self.cloud_donors) < 2:
            raise ValueError('Anomaly augmentation needs at least two distinct training images')
        self.physical_root = self.root
        mode = config['data'].get('assimilation_mode', 'legacy_gpv')
        if mode == 'hourly':
            location = config['data'].get('assimilation_root')
            if not location:
                raise ValueError('Hourly simulation training requires --assimilation-root (prepared physical pool)')
            self.physical_root = Path(location)
            state = json.loads((self.physical_root/'run.json').read_text())
            physical = json.loads((self.physical_root/'physical_pool.json').read_text())
            if state.get('status') != 'complete': raise ValueError('Hourly preprocessing incomplete')
            for key, expected in [('format_version',1), ('units','degC'), ('cadence_hours',1),
                                  ('time_basis','UTC'), ('patch_size',self.settings['patch_size']),
                                  ('weekly_aggregation','168_consecutive_hourly_frames')]:
                if physical.get(key) != expected: raise ValueError(f'Invalid physical pool: {key}')
            for key in ('lat','lon'):
                a=np.load(self.root/f'{key}.npy',allow_pickle=False)
                b=np.load(self.physical_root/f'{key}.npy',allow_pickle=False)
                if a.shape != b.shape or not np.allclose(a,b,rtol=0,atol=1e-6):
                    raise ValueError(f'Physical/satellite grid mismatch: {key}')
            self.pool=physical['pools']['weekly' if stage=='average' else 'hourly']
        elif mode == 'legacy_gpv':
            self.pool=manifest['gpv_pools']['weekly' if stage=='average' else 'daily']
        else:
            raise ValueError(f'Unknown assimilation_mode: {mode}')
        if not self.samples or not self.pool:raise ValueError('No eligible satellite samples or GPV pool; inspect dataset.json')
        self.transforms=tuple(transforms)
        self.seed=int(config.get('seed',42))
        self.size=int(self.settings['patch_size'])
        self.lat=np.load(self.root/'lat.npy',allow_pickle=False)
        self.lon=np.load(self.root/'lon.npy',allow_pickle=False)

    def __len__(self):return len(self.samples)

    def crop(self,path,key,origin):
        y,x=origin
        return np.array(read_array(self.root,path,key)[y:y+self.size,x:x+self.size],copy=True)

    def normalize(self,values):
        valid=np.isfinite(values)
        normalized=2*(values-self.settings['min_temp'])/(self.settings['max_temp']-self.settings['min_temp'])-1
        return torch.from_numpy(np.where(valid,normalized,0).astype(np.float32)),torch.from_numpy(valid.astype(np.float32))

    def __getitem__(self,index):
        row=self.samples[index];origin=row['origin']
        volume=np.stack([self.crop(path,'sst',origin) for path in row['frames']])[None]
        history=[self.crop(path,'sst',origin) for path in row['history']]
        weekly=observed_mean(history[:7],self.settings['weekly_min_observations'])
        monthly=observed_mean(history,self.settings['monthly_min_observations'])
        sst,mask=self.normalize(volume)
        weekly,weekly_mask=self.normalize(weekly[None])
        monthly,monthly_mask=self.normalize(monthly[None,None])
        sample=dict(sst_volume=sst,mask_volume=mask,weekly_average=weekly,weekly_mask=weekly_mask,
                    monthly_average=monthly,monthly_mask=monthly_mask,
                    target_sst=sst[:,-1].clone(),target_mask=mask[:,-1].clone())
        # Fixed, reproducible independent sampling; no satellite/GPV time-location pairing.
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,index]))
        physical=self.pool[int(rng.integers(len(self.pool)))]
        y,x=physical['origin']
        values=np.array(read_array(self.physical_root,physical['path'])[y:y+self.size,x:x+self.size],copy=True)
        if values.shape != (self.size,self.size): raise ValueError('Physical patch outside grid')
        if not np.isfinite(values).all():raise ValueError('GPV index contains missing pixels; regenerate index')
        key='assimilation_weekly' if self.stage=='average' else 'assimilation_sst'
        sample[key]=self.normalize(values[None])[0]
        cloud=self.crop(row['frames'][-1],'cloud_mask',origin)
        sample['cloud_mask']=torch.from_numpy(cloud[None].astype(np.float32))
        if self.stage == 'anomaly':
            own = self.cloud_donor_index[row['frames'][-1]]
            donor = int(torch.randint(len(self.cloud_donors)-1, ()).item())
            donor += donor >= own
            path, donor_origin = self.cloud_donors[donor]
            donor_cloud = self.crop(path, 'cloud_mask', donor_origin)
            if not np.isin(donor_cloud, [0,1]).all():
                raise ValueError('Donor cloud mask must be binary (1=cloud)')
            sample['random_cloud_mask'] = torch.from_numpy((1-donor_cloud)[None].astype(np.float32))
        # Metadata is provided to hooks; only tensor dict is returned for existing training loops.
        context=dict(target_time=row['time'],origin=origin,
                     lat=self.lat[origin[0]:origin[0]+self.size],lon=self.lon[origin[1]:origin[1]+self.size],
                     gpv_time=physical['time'],gpv_origin=physical['origin'],stage=self.stage)
        for transform in self.transforms:
            sample=transform(sample,context)
        if not isinstance(sample,dict) or not all(isinstance(v,torch.Tensor) for v in sample.values()):
            raise TypeError('Sample transforms must return a dictionary of tensors')
        return sample


def create_unified_loader(root,config,*,stage,shuffle=True,transforms=None):
    hooks=load_transforms(config['data'].get('sample_transforms',[])) if transforms is None else transforms
    dataset=UnifiedSSTDataset(root,config,stage=stage,transforms=hooks)
    training=config['training']
    generator=torch.Generator().manual_seed(int(config.get('seed',42)))
    return DataLoader(dataset,batch_size=int(training['batch_size']),shuffle=shuffle,
                      num_workers=int(training.get('num_workers',0)),generator=generator,
                      pin_memory=training.get('device')=='cuda' and torch.cuda.is_available())


def create_training_loader(root,config,*,stage,shuffle=True):
    if (Path(root)/'dataset.json').exists():
        return create_unified_loader(root,config,stage=stage,shuffle=shuffle)
    if (Path(root)/'run.json').exists():
        raise ValueError('Unified preprocessing is incomplete; dataset.json is missing')
    if config['data'].get('assimilation_mode') == 'hourly':
        raise ValueError('Hourly simulation mode requires a unified satellite dataset.json')
    from olds.datasets.himawari_patch_dataset import create_himawari_loader
    return create_himawari_loader(root,config,stage=stage,shuffle=shuffle)
