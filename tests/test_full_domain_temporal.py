# テスト: 全域9時刻入力と最近傍基準による復元が期待どおりに動くかを自動検査する。
# ── 使用するライブラリ・共通処理 ──────────────────
import json
import numpy as np
import pytest
import torch
from src.preprocessing.full_domain_input import load_full_frames, paths_for_time
from src.infer.full_domain import run, predict_full


# ── 動作確認用のデータ・テスト ──────────────────
def fixture_domain(tmp_path):
    from datetime import datetime, timedelta
    rows=[]
    for i in range(9):
        path=tmp_path/str(i);path.mkdir()
        a=np.full((141,150),np.nan,np.float32);a[0,0]=10+i
        land=np.zeros(a.shape,np.uint8);land[-1,-1]=1
        np.save(path/'sst.npy',a);np.save(path/'land_mask.npy',land)
        rows.append(dict(time=(datetime(2025,1,1)+timedelta(hours=6*i)).isoformat(),path=str(i)))
    (tmp_path/'dataset.json').write_text(json.dumps(dict(himawari_frames=rows)))
    np.save(tmp_path/'lat.npy',np.arange(141));np.save(tmp_path/'lon.npy',np.arange(150))
    return {'data':dict(baseline_method='temporal_mean_nearest',baseline_scope='full_domain',time_steps=9,time_interval_hours=6,min_temp=0,max_temp=35,anomaly_range=5)}


def test_full_domain_baseline_keeps_shape_land_and_coordinates(tmp_path):
    config=fixture_domain(tmp_path)
    out=tmp_path/'out'
    run(config,tmp_path,'2025-01-03T00:00:00Z',out,baseline_only=True)
    a=np.load(out/'baseline_sst.npy')
    assert a.shape==(141,150)
    assert np.isnan(a[-1,-1])
    np.testing.assert_allclose(a[:-1],14,atol=1e-5)
    assert np.load(out/'observation_count.npy')[0,0]==9
    with pytest.raises(ValueError,match='Missing required'):
        paths_for_time(tmp_path,'2025-01-03T06:00:00')


def test_full_mask_removes_values_before_averaging(tmp_path):
    config=fixture_domain(tmp_path)
    paths,_=paths_for_time(tmp_path,'2025-01-03T00:00:00')
    keep=np.ones((141,150),bool);keep[0,0]=False
    s,m,o=load_full_frames(tmp_path,paths,config['data'],keep)
    assert m.sum()==0 and s.sum()==0


def test_full_prediction_pads_without_resize_or_shift():
    class Identity(torch.nn.Module):
        def forward(self,sst,mask,baseline):
            assert sst.shape[-2:]==(144,152)
            return sst[:,:,0]
    s=torch.arange(141*150).reshape(1,1,1,141,150).expand(1,1,9,141,150).float()
    out=predict_full(Identity(),s,torch.ones_like(s),s[:,:,0])
    torch.testing.assert_close(out,s[:,:,0])
