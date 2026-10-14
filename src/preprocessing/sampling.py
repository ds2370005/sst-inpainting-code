"""Seeded random spatial indices shared by temporal fields; independent GPV pool."""
from datetime import datetime,timedelta
from collections import Counter
import numpy as np
from .common import read_array, window_sum, choose_origins


def build_indices(root, frames, gpv, settings):
    rng=np.random.default_rng(settings['seed'])
    size=settings['patch_size']; count=settings['patches_per_frame']
    indexed={datetime.fromisoformat(r['time']):r['path'] for r in frames}
    samples=[]; counts=Counter()
    for stamp,path in sorted(indexed.items()):
        times=[stamp-timedelta(hours=settings['time_interval_hours']*i) for i in reversed(range(settings['time_steps']))]
        days=[stamp-timedelta(days=i) for i in range(30)]
        if any(t not in indexed for t in times+days):
            counts['missing_history']+=1
            continue
        cloud=read_array(root,path,'cloud_mask')
        valid=read_array(root,path,'valid_mask')
        if min(valid.shape)<size:
            raise ValueError('Patch size exceeds study area')
        eligible=(window_sum(cloud,size)<=settings['max_cloud_fraction']*size*size)&(window_sum(valid,size)>0)
        # Do not present land/unknown coast as generated ocean to the discriminator.
        ocean=(np.asarray(valid,dtype=bool)|np.asarray(cloud,dtype=bool))
        eligible &= window_sum(ocean,size)==size*size
        origins=choose_origins(eligible,count,rng)
        if not origins:counts['no_eligible_crop']+=1
        for y,x in origins:
            history_valid=[np.isfinite(read_array(root,indexed[t])[y:y+size,x:x+size]) for t in days]
            if not ((np.sum(history_valid[:7],axis=0)>=settings['weekly_min_observations']).any()
                    and (np.sum(history_valid,axis=0)>=settings['monthly_min_observations']).any()):
                counts['insufficient_average_pixels']+=1
                continue
            samples.append(dict(time=stamp.isoformat(),frames=[indexed[t] for t in times],history=[indexed[t] for t in days],origin=[y,x]))
    pools={}
    for kind in ('daily','weekly'):
        pool=[]
        for record in gpv[kind]:
            sst=read_array(root,record['path'])
            if min(sst.shape)<size:
                raise ValueError('GPV region too small')
            eligible=window_sum(np.isfinite(sst),size)==size*size
            for origin in choose_origins(eligible,count,rng):
                pool.append(dict(path=record['path'],time=record['time'],origin=origin))
        pools[kind]=pool
    counts['accepted']=len(samples)
    return dict(samples=samples,gpv_pools=pools,selection_counts=dict(counts))
