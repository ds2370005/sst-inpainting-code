# 旧実装・旧テスト: 旧時系列可視化の時刻選択と欠落検出が期待どおりに動くかを自動検査する。
"""学習専用の選別を適用せず、正確な観測時刻で可視化することを確認する。"""
# ── 使用するライブラリ・共通処理 ──────────────────
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from old_network.old_visualize.frame_visualization import load_sequence


# ── 動作確認用のデータ・テスト ──────────────────
def frames(root):
    folder=root/'frames/himawari'
    folder.mkdir(parents=True)
    paths=[]
    for i in range(9):
        time=datetime(2025,4,1)+timedelta(hours=6*i)
        path=folder/f'{time:%Y%m%d%H%M%S}_r00_c00.npz'
        np.savez_compressed(path,format_version=2,satellite='himawari',timestamp=time.isoformat(),
                            grid_row_column=[0,0],sst=np.full((2,2),10.+i),
                            cloud_mask=np.ones((2,2),dtype=np.uint8),lat=[35.,34.],lon=[140.,141.])
        paths.append(path)
    return paths


def test_exact_order_without_averages_and_all_cloudy_target(tmp_path):
    expected=frames(tmp_path)
    paths,products=load_sequence(tmp_path,target=expected[-1],patch_size=2)
    assert paths==expected
    assert [p['sst'][0,0] for p in products]==list(range(10,19))
    direct,_=load_sequence(expected[0].parent,target=expected[-1].name,patch_size=2)
    assert direct==expected


def test_missing_time_is_reported_for_explicit_target(tmp_path):
    paths=frames(tmp_path)
    paths[4].unlink()
    with pytest.raises(ValueError,match=paths[4].name):
        load_sequence(tmp_path,target=paths[-1],patch_size=2)
    # 単一観測の読み込みに履歴データは必要ない。
    selected,_=load_sequence(tmp_path,target=paths[-1],patch_size=2,time_steps=1)
    assert selected==[paths[-1]]


@pytest.mark.parametrize('changes,reason',[
    ({'lat':np.array([35.1,34.])},'緯度経度'),
    ({'cloud_mask':np.full((2,2),2)},'Nonbinary'),
    ({'timestamp':'2024-01-01T00:00:00'},'identity'),
])
def test_misaligned_or_malformed_observations_rejected(tmp_path,changes,reason):
    paths=frames(tmp_path)
    with np.load(paths[3]) as archive:
        data=dict(archive)
    data.update(changes)
    np.savez_compressed(paths[3],**data)
    with pytest.raises(ValueError,match=reason):
        load_sequence(tmp_path,target=paths[-1],patch_size=2)


def test_auto_selection_skips_empty_sequence(tmp_path):
    paths=frames(tmp_path)
    # r00は有限の水温でも全画素がマスク済み。r01には有効な海上画素がある。
    for path in paths:
        with np.load(path) as archive:
            data=dict(archive)
        data.update(grid_row_column=[0,1],cloud_mask=np.zeros((2,2),dtype=np.uint8))
        np.savez_compressed(path.with_name(path.name.replace('c00','c01')),**data)
    selected,products=load_sequence(tmp_path,patch_size=2)
    assert all(p.name.endswith('c01.npz') for p in selected)
    assert len(products)==9


def test_auto_selection_reports_no_valid_data(tmp_path):
    frames(tmp_path)
    with pytest.raises(ValueError,match='すべてで有効SST画素が0'):
        load_sequence(tmp_path,patch_size=2)
