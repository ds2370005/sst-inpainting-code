import numpy as np
import pytest
import torch

from src.infer.full_domain import reconstruct_weekly, full_anomaly, origins, load_domain


def test_iteration_updates_selection_until_ocean_covered():
    weekly=np.full((4,8),np.nan,np.float32); weekly[:,:2]=10
    ocean=np.ones_like(weekly,bool)
    seen=[]
    def predict(y,x):
        seen.append((y,x));return np.full((4,4),20+x)
    canvas,coverage,report=reconstruct_weekly(weekly,ocean,predict,patch=4,stride=2,max_missing=.5)
    assert report['status']=='complete'
    assert seen==[(0,0),(0,2),(0,4)]
    assert coverage.all()
    assert (canvas[:,:2]==20).all()  # replace observed too
    assert (canvas[:,2:4]==22).all() # overlapping last prediction wins
    assert (canvas[:,4:]==24).all()


def test_no_seed_returns_incomplete_not_fake_success():
    a=np.full((4,6),np.nan)
    _,mask,r=reconstruct_weekly(a,np.ones_like(a,bool),lambda *_:pytest.fail('must not predict'),patch=4,stride=2)
    assert r['status']=='no_eligible_patch' and not mask.any()


def test_boundaries_limit_and_excluded_land():
    a=np.ones((5,7));ocean=np.ones_like(a,bool);ocean[0,:]=False
    canvas,mask,r=reconstruct_weekly(a,ocean,lambda *_:np.ones((4,4)),patch=4,stride=3)
    assert r['status']=='complete' and mask[ocean].all()
    assert np.isnan(canvas[~ocean]).all() and not mask[~ocean].any()
    assert origins(7,4,2).tolist()==[0,2,3]
    _,_,r=reconstruct_weekly(a,ocean,lambda *_:np.ones((4,4)),patch=4,stride=2,max_iterations=1)
    assert r['status']=='max_iterations' and r['remaining_ocean_pixels']>0


def test_reject_nonfinite_prediction_and_deterministic_seed():
    a=np.ones((4,6));sea=np.ones_like(a,bool)
    with pytest.raises(ValueError,match='nonfinite'):
        reconstruct_weekly(a,sea,lambda *_:np.full((4,4),np.nan),patch=4,stride=2)
    r1=reconstruct_weekly(a,sea,lambda *_:np.ones((4,4)),patch=4,stride=2,seed=5)[2]
    r2=reconstruct_weekly(a,sea,lambda *_:np.ones((4,4)),patch=4,stride=2,seed=5)[2]
    assert r1==r2


def test_full_anomaly_rectangular_padding_units_and_land():
    class Fake(torch.nn.Module):
        def forward(self,v,m,b):
            assert v.shape==(1,1,9,144,152)
            assert b.shape==(1,1,1,144,152)
            return torch.full_like(v[:,:,0],.2)
    sea=np.ones((139,149),bool);sea[0,0]=False
    result=full_anomaly(Fake(),np.ones((9,139,149))*15,np.ones((139,149))*14,sea,
                       dict(min_temp=0,max_temp=35,anomaly_range=5),torch.device('cpu'))
    assert result.shape==(139,149) and np.isnan(result[0,0])
    np.testing.assert_allclose(result[sea],1.)


def test_real_generator_rectangular_forward():
    from src.models import AnomalyInpaintingGenerator
    model=AnomalyInpaintingGenerator(time_steps=2).eval()
    with torch.no_grad():
        x=torch.zeros(1,1,2,144,160)
        y=model(x,torch.ones_like(x),x[:,:,:1])
    assert y.shape==(1,1,144,160) and torch.isfinite(y).all()


def test_load_domain_uses_same_hour_histories_without_training_crops(tmp_path):
    from datetime import datetime,timedelta
    import json
    from src.preprocessing.common import FORMAT_VERSION
    settings=dict(time_steps=9,time_interval_hours=6,min_temp=0,max_temp=35,satellite='himawari',
                  weekly_min_observations=3,monthly_min_observations=5)
    target=datetime(2025,5,30)
    times={target-timedelta(days=i) for i in range(30)}
    times.update(target-timedelta(hours=6*i) for i in range(9))
    frames=[]
    for t in sorted(times):
        folder=tmp_path/t.strftime('%Y%m%d%H%M%S');folder.mkdir()
        values=np.full((2,3),t.day,np.float32); values[0,0]=np.nan
        np.save(folder/'sst.npy',values)
        np.save(folder/'valid_mask.npy',np.isfinite(values))
        np.save(folder/'cloud_mask.npy',np.zeros((2,3),bool))
        frames.append(dict(time=t.isoformat(),path=folder.name))
    np.save(tmp_path/'lat.npy',np.arange(2));np.save(tmp_path/'lon.npy',np.arange(3))
    (tmp_path/'run.json').write_text('{"status":"complete"}')
    manifest=dict(format_version=FORMAT_VERSION,settings=settings,himawari_frames=frames,samples=[])
    (tmp_path/'dataset.json').write_text(json.dumps(manifest))
    domain=load_domain(tmp_path,'2025-05-30T09:00:00+09:00',dict(data=settings))
    assert domain['monthly'][1,1]==15.5 and domain['weekly'][1,1]==27
    assert domain['sst'][-1,1,1]==30 and domain['sst'][0,1,1]==28
    assert not domain['ocean'][0,0]
    manifest['himawari_frames']=frames[1:]
    (tmp_path/'dataset.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Missing required'):
        load_domain(tmp_path,target.isoformat(),dict(data=settings))


@pytest.mark.parametrize('blocked',[False,True])
def test_run_writes_celsius_and_does_not_run_anomaly_on_incomplete(tmp_path,monkeypatch,blocked):
    import json,yaml
    from argparse import Namespace
    import src.infer.full_domain as module
    import src.utils.checkpoint as checkpoint
    config=dict(data=dict(patch_size=256,time_steps=9,min_temp=0,max_temp=35,anomaly_range=5),training=dict(device='cpu'))
    cfg=tmp_path/'config.yaml';cfg.write_text(yaml.safe_dump(config))
    ck=tmp_path/'weights.pth';torch.save(dict(config=config),ck)
    shape=(256,264)
    domain=dict(sst=np.full((9,*shape),15,np.float32),monthly=np.full(shape,14,np.float32),
                weekly=np.full(shape,np.nan if blocked else 14,np.float32),ocean=np.ones(shape,bool),
                lat=np.arange(shape[0]),lon=np.arange(shape[1]),target_time='2025-05-30T00:00:00',input_times=[])
    monkeypatch.setattr(module,'load_domain',lambda *args:domain)
    class Avg(torch.nn.Module):
        def forward(self,v,m,monthly,mm):return v[:,:,0]
    class Ano(torch.nn.Module):
        def forward(self,v,m,w):
            assert not blocked
            return torch.full_like(v[:,:,0],.2)
    monkeypatch.setattr(module,'AverageEstimationGenerator',lambda **kw:Avg())
    monkeypatch.setattr(module,'AnomalyInpaintingGenerator',lambda **kw:Ano())
    monkeypatch.setattr(checkpoint,'load_generator_state',lambda *args:None)
    args=Namespace(config=cfg,data_root=tmp_path,target_time=domain['target_time'],device='cpu',
        average_checkpoint=ck,anomaly_checkpoint=ck,output_dir=tmp_path/'out',stride=128,
        max_missing=.5,max_iterations=20,seed=42)
    if blocked:
        with pytest.raises(RuntimeError,match='incomplete'):module.run(args)
        assert not (args.output_dir/'full_reconstruction.npz').exists()
    else:
        assert module.run(args)['status']=='complete'
        with np.load(args.output_dir/'full_reconstruction.npz') as f:
            np.testing.assert_allclose(f['reconstructed_sst_celsius'],16,atol=1e-5)
            assert f['reconstructed_sst_celsius'].shape==shape
    report=json.loads((args.output_dir/'run.json').read_text())
    assert report['status']==('failed' if blocked else 'complete')
    assert (args.output_dir/'weekly_reconstruction.npz').exists()
