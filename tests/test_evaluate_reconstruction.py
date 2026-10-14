"""Evaluate synthetic predictions without training or accessing server data."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'evaluate_reconstruction', Path(__file__).resolve().parents[1]/'visualize/evaluate_reconstruction.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_regression_metrics():
    result = evaluation.regression_metrics(np.array([1., 2., 3.]), np.array([2., 2., 2.]))
    assert result['mae_C'] == pytest.approx(2/3)
    assert result['rmse_C'] == pytest.approx(np.sqrt(2/3))
    assert result['bias_C'] == 0
    assert result['r2'] == 0
    assert result['pearson_r'] is None
    assert evaluation.regression_metrics(np.ones(2), np.ones(2))['r2'] is None


@pytest.mark.parametrize('scores,auc,ap', [([0., 1.], 1., 1.), ([1., 0.], 0., .5), ([1., 1.], .5, .5)])
def test_roc_and_average_precision(scores, auc, ap):
    result = evaluation.binary_curves(np.array([False, True]), np.array(scores))
    assert result['auc'] == pytest.approx(auc)
    assert result['average_precision'] == pytest.approx(ap)
    assert result['fpr'][0] == result['tpr'][0] == 0
    assert result['fpr'][-1] == result['tpr'][-1] == 1


@pytest.fixture
def report(tmp_path):
    (tmp_path/'outputs').mkdir()
    (tmp_path/'logs').mkdir()
    (tmp_path/'config.yaml').write_text('data:\n  min_temp: 0\n  max_temp: 20\n')
    # Reference [0,10,20] C; prediction [1,10,19] C. Last pixel excluded.
    np.savez(tmp_path/'outputs/reconstruction.npz', pred_sst=np.array([[-.9,0],[.9,0]]))
    np.savez(tmp_path/'outputs/preview_input.npz',
             sst_volume=np.array([[[[-1.,0],[1.,0]]]]),
             mask_volume=np.array([[[[1,1],[1,0]]]]))
    (tmp_path/'logs/average.log').write_text('epoch=1/1 loss_d=1.4 loss_g=2.5 loss_rec=0.2 loss_adv=0.7\n')
    (tmp_path/'run.json').write_text(json.dumps({'stages': {'average': {'elapsed_seconds': 2.}}}))
    return argparse.Namespace(run_dir=tmp_path, output_dir=None, input=None, truth=None, temperature_threshold=10.)


def test_report_masks_units_and_outputs(report):
    metrics = evaluation.evaluate(report)
    assert metrics['regression']['n_pixels'] == 3
    assert metrics['regression']['mae_C'] == pytest.approx(2/3)
    assert metrics['excluded_reference_pixels'] == 1
    assert metrics['classification']['roc_auc'] == 1
    assert metrics['classification']['tp'] == 2
    out = report.run_dir/'evaluation'
    for name in ('regression.png','classification.png','training_losses.png','runtimes.png',
                 'metrics.json','training_losses.csv','roc_pr.csv'):
        assert (out/name).stat().st_size > 0
    assert 'not held-out' in json.loads((out/'metrics.json').read_text())['scope']
    with pytest.raises(ValueError, match='new/empty'):
        evaluation.evaluate(report)


def test_one_class_is_explicitly_undefined(report):
    report.temperature_threshold = 30.
    metrics = evaluation.evaluate(report)
    assert metrics['classification']['roc_auc'] is None
    assert 'only one class' in metrics['classification']['curve_status']
    assert not (report.run_dir/'evaluation/roc_pr.csv').exists()


def test_external_reference_shape_mismatch(report):
    report.truth = report.run_dir/'truth.npz'
    np.savez(report.truth, sst_celsius=np.zeros((3,3)), valid_mask=np.ones((3,3)))
    with pytest.raises(ValueError, match='shapes must match'):
        evaluation.evaluate(report)


def test_nonfinite_prediction_cannot_disappear_from_metrics(report):
    np.savez(report.run_dir/'outputs/reconstruction.npz', pred_sst=np.array([[np.nan,0],[0,0]]))
    with pytest.raises(ValueError, match='Nonfinite predictions'):
        evaluation.evaluate(report)
