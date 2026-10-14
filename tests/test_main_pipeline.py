"""Orchestrator tests with fake children: no downloads or real training."""
import importlib.util
import csv
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import yaml

spec = importlib.util.spec_from_file_location('experiment_main', Path(__file__).resolve().parents[1]/'main.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    config = dict(seed=42, data=dict(patch_size=2,time_steps=3,min_temp=0,max_temp=35),
                  training=dict(epochs=30,batch_size=32,device='cpu'), paths={})
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config))
    root = tmp_path/'data';root.mkdir()
    monkeypatch.setattr(runner,'validate_data',lambda *args:None)
    monkeypatch.setattr(runner,'validate_checkpoint',lambda path,config:runner.require_file(path))
    return tmp_path,path,root


def input_npz(path):
    np.savez(path,sst_volume=np.zeros((1,3,2,2)),mask_volume=np.ones((1,3,2,2)),
             monthly_average=np.zeros((1,1,2,2)),monthly_mask=np.ones((1,1,2,2)))


def test_stage_order_and_checkpoint_handoff(setup):
    tmp,config,data = setup
    source=tmp/'input.npz';input_npz(source)
    args=runner.parse_args(['--config',str(config),'--data-root',str(data),
                           '--run-dir',str(tmp/'run'),'--inference-input',str(source),'--smoke'])
    calls=[]
    def execute(command,log):
        calls.append(command)
        cfg=yaml.safe_load(Path(command[command.index('--config')+1]).read_text())
        assert cfg['training']['epochs']==1 and cfg['training']['batch_size']==1
        module=command[3]
        if 'train_' in module:
            stage=module.rsplit('_',1)[-1]
            target=Path(cfg['paths']['checkpoint_dir'])/stage/'latest.pth'
            target.parent.mkdir(parents=True);target.write_bytes(b'checkpoint')
        else:
            out=Path(command[command.index('--output')+1])
            np.savez(out,pred_weekly=[0],pred_anomaly=[0],pred_sst=[0])
        log.write_text('ok')
    result=runner.run(args,execute)
    assert result['status']=='complete'
    assert result['elapsed_seconds'] >= sum(s['elapsed_seconds'] for s in result['stages'].values())
    assert all(s['elapsed_seconds'] >= 0 and s['finished_at'] for s in result['stages'].values())
    with (tmp/'run/timings.csv').open() as handle:
        timings = list(csv.DictReader(handle))
    assert [r['stage'] for r in timings] == ['average', 'anomaly', 'reconstruct', 'total']
    assert all(r['status'] == 'complete' for r in timings)
    assert [c[3] for c in calls]==['src.train.train_average','src.train.train_anomaly','src.infer.reconstruct']
    assert calls[1][calls[1].index('--average-checkpoint')+1]==str(tmp/'run/checkpoints/average/latest.pth')
    assert '--max-batches' in calls[0]
    with pytest.raises(ValueError,match='empty --run-dir'):runner.run(args,execute)


def test_failure_stops_later_stages(setup):
    tmp,config,data=setup
    args=runner.parse_args(['--config',str(config),'--data-root',str(data),'--run-dir',str(tmp/'fail')])
    calls=[]
    def execute(command,log):
        calls.append(command)
        raise subprocess.CalledProcessError(2,command)
    with pytest.raises(subprocess.CalledProcessError):runner.run(args,execute)
    assert len(calls)==1
    status=json.loads((tmp/'fail/run.json').read_text())
    assert status['status']=='failed' and status['stages']['average']['status']=='failed'
    assert status['elapsed_seconds'] >= status['stages']['average']['elapsed_seconds'] >= 0
    with (tmp/'fail/timings.csv').open() as handle:
        timings = list(csv.DictReader(handle))
    assert [r['status'] for r in timings] == ['failed', 'failed']


def test_preprocess_only(setup):
    tmp,config,data=setup
    args=runner.parse_args(['--config',str(config),'--run-dir',str(tmp/'prep'),
                           '--himawari-input',str(data),'--gpv-input',str(data),'--stop-after','preprocess'])
    def execute(command,log):
        assert command[3]=='src.preprocessing.pipeline'
        out=Path(command[command.index('--output')+1]);out.mkdir()
        (out/'run.json').write_text('{"status":"complete"}')
    assert runner.run(args,execute)['status']=='complete'


def test_external_checkpoint_required():
    with pytest.raises(SystemExit):
        runner.parse_args(['--run-dir','unused','--data-root','data','--start-at','anomaly'])
    with pytest.raises(SystemExit):
        runner.parse_args(['--run-dir','unused','--start-at','reconstruct','--stop-after','average'])


def test_input_masks_and_finiteness(tmp_path):
    config={'data':{'patch_size':2,'time_steps':3}}
    path=tmp_path/'in.npz';input_npz(path)
    runner.validate_input(path,config)
    with np.load(path) as archive:
        fields={key:archive[key] for key in archive.files}
    fields['sst_volume'][0,0,0,0]=np.nan
    np.savez(path,**fields)
    with pytest.raises(ValueError,match='NaN/Inf'):runner.validate_input(path,config)


def test_execute_streams_log_and_reports_failure(tmp_path):
    import sys
    log=tmp_path/'child.log'
    runner.execute([sys.executable,'-c','print("child output")'],log)
    assert 'child output' in log.read_text()
    with pytest.raises(subprocess.CalledProcessError):
        runner.execute([sys.executable,'-c','raise SystemExit(3)'],log)


def test_preview_export_and_provenance(tmp_path,monkeypatch):
    import torch
    import src.datasets.unified_dataset as datasets
    class FakeDataset:
        def __init__(self,*args,**kw):
            self.size=2
            self.lat=np.array([30.,31.]);self.lon=np.array([140.,141.])
            self.samples=[dict(time='2025-05-30T00:00:00',origin=[0,0])]
        def __len__(self):return 1
        def __getitem__(self,index):
            return dict(sst_volume=torch.zeros(1,3,2,2),mask_volume=torch.ones(1,3,2,2),
                        monthly_average=torch.zeros(1,1,2,2),monthly_mask=torch.ones(1,1,2,2))
    monkeypatch.setattr(datasets,'UnifiedSSTDataset',FakeDataset)
    config={'data':{'time_steps':3,'patch_size':2}}
    out=tmp_path/'preview.npz'
    runner.prepare_preview(tmp_path,config,0,out)
    runner.validate_input(out,config)
    provenance=json.loads(out.with_suffix('.json').read_text())
    assert provenance['purpose']=='training_sample_preview_not_evaluation'
    assert provenance['lat']==[30.,31.]
    with pytest.raises(ValueError,match='sample-index'):
        runner.prepare_preview(tmp_path,config,1,out)


def test_checkpoint_config_mismatch(tmp_path):
    import torch
    checkpoint=tmp_path/'checkpoint.pth'
    config={'data':{'patch_size':256,'min_temp':0}}
    torch.save({'config':config,'generator_state_dict':{}},checkpoint)
    runner.validate_checkpoint(checkpoint,config)
    with pytest.raises(ValueError,match='min_temp'):
        runner.validate_checkpoint(checkpoint,{'data':{'patch_size':256,'min_temp':-2}})
