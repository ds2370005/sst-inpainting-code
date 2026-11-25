# テスト: 9時刻平均・最近傍補間・基準SSTの学習と推論での扱いが期待どおりに動くかを自動検査する。
# ── 使用するライブラリ・共通処理 ──────────────────
import pytest
import torch
from src.models.temporal_baseline import temporal_mean_nearest


# ── 動作確認用のデータ・テスト ──────────────────
def test_mean_and_nearest_preserve_observations():
    s = torch.full((1,1,9,3,3), float('nan'))
    m = torch.zeros_like(s)
    s[0,0,0,0,0]=2; s[0,0,1,0,0]=4
    m[0,0,:2,0,0]=1
    result=temporal_mean_nearest(s,m)
    assert torch.all(result['baseline']==3)
    assert result['observation_count'][0,0,0,0]==2
    assert result['interpolated_mask'].sum()==8
    assert result['distance_pixels'][0,0,0,1]==1


def test_masked_values_do_not_leak_and_land_is_not_filled():
    s=torch.full((1,1,9,3,3),999.)
    m=torch.zeros_like(s); m[:,:,:,0,0]=1; s[:,:,:,0,0]=2
    ocean=torch.ones(1,1,3,3,dtype=torch.bool); ocean[:,:,2,2]=False
    result=temporal_mean_nearest(s,m,ocean)
    assert result['baseline'][0,0,1,1]==2
    assert result['baseline'][0,0,2,2]==0
    assert not result['interpolated_mask'][0,0,2,2]


def test_all_missing_is_rejected():
    with pytest.raises(ValueError,match='No valid'):
        temporal_mean_nearest(torch.zeros(1,1,9,3,3),torch.zeros(1,1,9,3,3))


def test_nearest_inference_uses_no_average_checkpoint(tmp_path, monkeypatch):
    import numpy as np
    import src.infer.reconstruct as module
    class Zero(torch.nn.Module):
        def forward(self, sst, *args):
            return torch.zeros_like(sst[:,:,0])
    monkeypatch.setattr(module, 'AnomalyInpaintingGenerator', lambda **kw: Zero())
    monkeypatch.setattr(module, 'load_generator_checkpoint', lambda *args: None)
    monkeypatch.setattr(module.torch, 'load', lambda *a,**k: {'config':{'data':{'baseline_method':'temporal_mean_nearest'}}})
    config={'data':{'time_steps':9,'min_temp':0.,'max_temp':35.,'anomaly_range':5.,'baseline_method':'temporal_mean_nearest'}}
    path=tmp_path/'input.npz'
    np.savez(path,sst_volume=np.full((1,9,3,3),.2),mask_volume=np.ones((1,9,3,3)),monthly_average=np.zeros((1,1,3,3)),monthly_mask=np.ones((1,1,3,3)))
    out=module.run_reconstruction(config=config,average_checkpoint=None,anomaly_checkpoint='unused',input_path=path,output_path=tmp_path/'out.npz',device=torch.device('cpu'))
    np.testing.assert_allclose(out['pred_sst'],.2)
    assert (out['observation_count']==9).all()


def test_time_varying_clouds_are_combined_without_common_mask():
    s=torch.zeros(1,1,9,2,2)
    m=torch.zeros_like(s)
    s[0,0,0,0,0]=10; m[0,0,0,0,0]=1
    s[0,0,1,0,1]=20; m[0,0,1,0,1]=1
    s[0,0,2,1,0]=30; m[0,0,2,1,0]=1
    result=temporal_mean_nearest(s,m)
    assert result['interpolated_mask'].sum()==1
    torch.testing.assert_close(result['baseline'][0,0,0],torch.tensor([10.,20.]))
    assert result['baseline'][0,0,1,0]==30


def test_progress_logging_does_not_change_result(capsys):
    s=torch.ones(1,1,9,3,3)
    m=torch.ones_like(s);m[:,:,:,1,1]=0
    expected=temporal_mean_nearest(s,m)
    assert capsys.readouterr().out==''
    actual=temporal_mean_nearest(s,m,log_prefix='[batch=1]')
    output=capsys.readouterr().out
    for label in ('平均開始','平均完了','最近傍補間開始','最近傍補間完了','補間画素=1'):
        assert label in output
    for key in expected:
        torch.testing.assert_close(actual[key],expected[key])
