import importlib.util
from pathlib import Path
import json

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('histograms',Path(__file__).resolve().parents[1]/'visualize/visualize_sst_histograms.py')
hist=importlib.util.module_from_spec(spec)
spec.loader.exec_module(hist)


@pytest.fixture
def run(tmp_path):
    (tmp_path/'outputs').mkdir()
    (tmp_path/'config.yaml').write_text('data:\n  min_temp: 0\n  max_temp: 20\n  anomaly_range: 5\n')
    np.savez(tmp_path/'outputs/preview_input.npz',
             sst_volume=np.array([[[[-1.,0],[1.,999.]]]]),
             mask_volume=np.array([[[[1,1],[1,0]]]]),
             monthly_average=np.zeros((1,1,2,2)),monthly_mask=np.ones((1,1,2,2)))
    np.savez(tmp_path/'outputs/reconstruction.npz',pred_weekly=np.zeros((1,1,2,2)),
             pred_anomaly=np.full((1,1,2,2),.2), pred_sst=np.full((1,1,2,2),.1))
    return tmp_path


def test_units_matching_masks_and_constant_predictions(run):
    report=hist.visualize(run,dpi=40,bins=10)
    rows=report['comparisons']
    assert rows[0]['count']==3 and rows[0]['mean']==10
    assert rows[1]['std']==0 and rows[1]['mean']==10
    assert rows[2]['mean']==11
    residual=next(r for r in rows if r['series']=='Observed SST - predicted weekly')
    assert residual['min']==-10 and residual['max']==10
    anomaly=next(r for r in rows if r['series']=='Predicted anomaly')
    assert anomaly['mean']==1
    for name in ['comparison.png','average_comparison.png','prediction_zoom.png','histograms.csv','statistics.csv','statistics.json']:
        assert (run/'histograms'/name).stat().st_size>0
    with pytest.raises(ValueError,match='not empty'):
        hist.visualize(run)


def test_missing_predictions_are_reported_in_full_grid_but_not_reference(run):
    np.savez(run/'outputs/reconstruction.npz',pred_weekly=np.zeros((1,1,2,2)),
             pred_anomaly=np.array([[[[0,0],[0,np.nan]]]]),pred_sst=np.zeros((1,1,2,2)))
    report=hist.visualize(run,dpi=40,bins=5)
    assert report['all_grid_predictions']['predicted_anomaly']['nonfinite']==1


def test_nonfinite_on_reference_rejected(run):
    np.savez(run/'outputs/reconstruction.npz',pred_weekly=np.full((1,1,2,2),np.nan),
             pred_anomaly=np.zeros((1,1,2,2)),pred_sst=np.zeros((1,1,2,2)))
    with pytest.raises(ValueError,match='Nonfinite prediction'):
        hist.visualize(run)


def test_weekly_reference_checks_source_and_minimum_observations(run):
    root=run/'data';root.mkdir()
    observed=np.array([[0.,10.],[20.,np.nan]])
    for i in range(3):
        (root/str(i)).mkdir()
        np.save(root/str(i)/'sst.npy', observed+i)
    meta=dict(sample_index=0,target_time='2025-05-01',origin=[0,0])
    input_path=run/'outputs/preview_input.npz'
    input_path.with_suffix('.json').write_text(json.dumps(meta))
    manifest=dict(samples=[dict(time=meta['target_time'],origin=[0,0],frames=['0'],history=['0','1','2'])],
                  settings=dict(weekly_min_observations=3))
    (root/'dataset.json').write_text(json.dumps(manifest))
    weekly=hist.observed_weekly(root,input_path,observed)
    np.testing.assert_allclose(weekly,observed+1,equal_nan=True)
    with pytest.raises(ValueError,match='does not match'):
        hist.observed_weekly(root,input_path,observed+5)
