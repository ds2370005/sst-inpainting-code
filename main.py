#!/usr/bin/env python3
"""Manage preprocessing, two-stage training, and patch reconstruction in one run."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from src.preprocessing.common import write_json
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parent
STAGES = ('preprocess', 'average', 'anomaly', 'reconstruct')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.yaml')
    parser.add_argument('--run-dir', type=Path, required=True, help='New or empty experiment directory')
    parser.add_argument('--data-root', type=Path, help='Completed unified preprocessing output')
    parser.add_argument('--himawari-input', type=Path)
    parser.add_argument('--gpv-input', type=Path)
    parser.add_argument('--start-at', choices=STAGES,
                        help='Default: average if --data-root is given, otherwise preprocess')
    parser.add_argument('--stop-after', choices=STAGES, default='reconstruct')
    parser.add_argument('--average-checkpoint', type=Path)
    parser.add_argument('--anomaly-checkpoint', type=Path)
    parser.add_argument('--inference-input', type=Path, help='Optional normalized inference NPZ')
    parser.add_argument('--sample-index', type=int, default=0,
                        help='Without --inference-input, reconstruct this training sample as a preview')
    parser.add_argument('--device')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--smoke', action='store_true', help='One batch per training stage, one epoch, batch size 1')
    parser.add_argument('--bounds', nargs=4, type=float, default=[17, 50, 117, 150])
    parser.add_argument('--patches-per-frame', type=int, default=8)
    parser.add_argument('--min-quality', type=int, choices=range(6), default=4)
    args = parser.parse_args(argv)
    args.start_at = args.start_at or ('average' if args.data_root else 'preprocess')
    start, stop = STAGES.index(args.start_at), STAGES.index(args.stop_after)
    if start > stop:
        parser.error('--start-at must not follow --stop-after')
    args.stages = STAGES[start:stop + 1]
    for name in ('config', 'run_dir', 'data_root', 'himawari_input', 'gpv_input',
                 'average_checkpoint', 'anomaly_checkpoint', 'inference_input'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    for name in ('epochs', 'batch_size'):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if args.sample_index < 0 or args.patches_per_frame <= 0 or (args.num_workers is not None and args.num_workers < 0):
        parser.error('Invalid sample index, worker count or patch count')
    if 'preprocess' in args.stages:
        if args.data_root:
            parser.error('Do not use --data-root when running preprocessing')
        if not args.himawari_input or not args.gpv_input:
            parser.error('Preprocessing requires --himawari-input and --gpv-input')
    elif ('average' in args.stages or 'anomaly' in args.stages or not args.inference_input) and not args.data_root:
        parser.error('This run requires --data-root')
    if 'average' in args.stages and args.average_checkpoint:
        parser.error('--average-checkpoint is for skipping average training, not resuming it')
    if 'anomaly' in args.stages and args.anomaly_checkpoint:
        parser.error('--anomaly-checkpoint is for skipping anomaly training, not resuming it')
    if start >= STAGES.index('anomaly') and not args.average_checkpoint:
        parser.error('Starting here requires --average-checkpoint')
    if args.start_at == 'reconstruct' and not args.anomaly_checkpoint:
        parser.error('Reconstruction requires --anomaly-checkpoint')
    return args


def execute(command, log_path):
    """Stream each child process to the terminal and an experiment log."""
    environment = dict(os.environ, PYTHONUNBUFFERED='1')
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace')
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            process.stdout.close()


def require_file(path):
    if path is None or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f'Missing or empty artifact: {path}')


def validate_data(root, config, stages, preview):
    from src.datasets.unified_dataset import UnifiedSSTDataset
    # Check only pools needed by the selected stages.
    for stage in ('average', 'anomaly'):
        if stage in stages or (stage == 'average' and preview):
            UnifiedSSTDataset(root, config, stage=stage)


def validate_checkpoint(path, config):
    import torch
    require_file(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if 'config' not in checkpoint:
        raise ValueError(f'Checkpoint has no configuration to validate: {path}')
    for key in ('time_steps', 'patch_size', 'time_interval_hours', 'min_temp',
                'max_temp', 'anomaly_range', 'satellite', 'sample_transforms'):
        default = [] if key == 'sample_transforms' else None
        if checkpoint['config']['data'].get(key, default) != config['data'].get(key, default):
            raise ValueError(f'Checkpoint configuration differs for {key}: {path}')


def prepare_preview(data_root, config, index, destination):
    """Export a registered training crop; this is explicitly not held-out evaluation."""
    import numpy as np
    from src.datasets.unified_dataset import UnifiedSSTDataset
    from src.infer.reconstruct import REQUIRED_INPUT_KEYS
    if config['data'].get('sample_transforms'):
        raise ValueError('Automatic preview with feature transforms is not supported; supply --inference-input')
    dataset = UnifiedSSTDataset(data_root, config, stage='average')
    if not 0 <= index < len(dataset):
        raise ValueError(f'--sample-index {index} outside 0..{len(dataset)-1}')
    sample = dataset[index]
    with destination.open('wb') as handle:
        np.savez_compressed(handle, **{key: sample[key].numpy() for key in REQUIRED_INPUT_KEYS})
    row = dataset.samples[index]
    y, x = row['origin']
    write_json(destination.with_suffix('.json'), dict(
        purpose='training_sample_preview_not_evaluation', sample_index=index,
        target_time=row['time'], origin=row['origin'],
        lat=dataset.lat[y:y+dataset.size].tolist(), lon=dataset.lon[x:x+dataset.size].tolist(),
        data_root=str(data_root), settings=config['data']))


def validate_input(path, config):
    import numpy as np
    size, steps = int(config['data']['patch_size']), int(config['data']['time_steps'])
    shapes = {'sst_volume': (1, steps, size, size), 'mask_volume': (1, steps, size, size),
              'monthly_average': (1, 1, size, size), 'monthly_mask': (1, 1, size, size)}
    require_file(path)
    with np.load(path, allow_pickle=False) as data:
        for key, shape in shapes.items():
            array = data[key]
            if not (array.shape == shape or (array.ndim == 5 and array.shape[0] > 0 and array.shape[1:] == shape)):
                raise ValueError(f'Invalid inference shape for {key}: {array.shape}')
            if not np.isfinite(array).all():
                raise ValueError(f'Inference input contains NaN/Inf: {key}')
        for field, mask in [('sst_volume', 'mask_volume'), ('monthly_average', 'monthly_mask')]:
            if data[field].shape != data[mask].shape or not np.isin(data[mask], [0, 1]).all():
                raise ValueError(f'Invalid mask: {mask}')
            if np.any(data[field][data[mask] == 0] != 0):
                raise ValueError(f'Missing pixels must be zero filled: {field}')
        if data['sst_volume'].ndim != data['monthly_average'].ndim:
            raise ValueError('Inference arrays must use consistent batch dimensions')
        if data['sst_volume'].ndim == 5 and data['sst_volume'].shape[0] != data['monthly_average'].shape[0]:
            raise ValueError('Inference batch sizes differ')


def run(args, executor=execute):
    config = load_config(args.config)
    for name in ('device', 'epochs', 'batch_size', 'num_workers'):
        value = getattr(args, name)
        if value is not None:
            config['training'][name] = value
    if args.smoke:
        config['training'].update(epochs=1, batch_size=1, num_workers=0)
    if int(config['training']['epochs']) <= 0 or int(config['training']['batch_size']) <= 0:
        raise ValueError('Epochs and batch size must be positive')
    for path in (args.average_checkpoint, args.anomaly_checkpoint):
        if path:
            validate_checkpoint(path, config)
    if args.inference_input:
        validate_input(args.inference_input, config)
    preview = 'reconstruct' in args.stages and not args.inference_input
    if preview and config['data'].get('sample_transforms'):
        raise ValueError('Automatic preview does not support feature transforms; supply --inference-input or stop after anomaly')
    if args.data_root:
        validate_data(args.data_root, config, args.stages, preview)
    for path in (args.himawari_input, args.gpv_input):
        if path is not None and not path.is_dir():
            raise ValueError(f'Input directory does not exist: {path}')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise ValueError('Use a new or empty --run-dir; existing experiments are not overwritten')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / 'logs').mkdir()
    config['paths']['checkpoint_dir'] = str(args.run_dir / 'checkpoints')
    config['paths']['output_dir'] = str(args.run_dir / 'outputs')
    config_path = args.run_dir / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    data_root = args.data_root or args.run_dir / 'data'
    average = args.average_checkpoint or args.run_dir / 'checkpoints/average/latest.pth'
    anomaly = args.anomaly_checkpoint or args.run_dir / 'checkpoints/anomaly/latest.pth'
    state = dict(status='running', started_at=datetime.now().astimezone().isoformat(),
                 stages={}, data_root=str(data_root), smoke=args.smoke,
                 request={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    state_path = args.run_dir / 'run.json'
    write_json(state_path, state)
    current = None
    try:
        for stage in args.stages:
            current = stage
            state['stages'][stage] = dict(status='running')
            write_json(state_path, state)
            print(f'=== {stage} ===', flush=True)
            prefix = [sys.executable, '-u', '-m']
            if stage == 'preprocess':
                command = prefix + ['src.preprocessing.pipeline', '--config', str(config_path),
                    '--himawari-input', str(args.himawari_input), '--gpv-input', str(args.gpv_input),
                    '--output', str(data_root), '--bounds', *map(str, args.bounds),
                    '--patches-per-frame', str(args.patches_per_frame), '--min-quality', str(args.min_quality)]
            elif stage in ('average', 'anomaly'):
                validate_data(data_root, config, [stage], False)
                command = prefix + [f'src.train.train_{stage}', '--config', str(config_path), '--data-root', str(data_root)]
                if stage == 'anomaly':
                    validate_checkpoint(average, config)
                    command += ['--average-checkpoint', str(average)]
                if args.smoke:
                    command += ['--max-batches', '1']
            else:
                validate_checkpoint(average, config)
                validate_checkpoint(anomaly, config)
                destination = args.run_dir / 'outputs'
                destination.mkdir(exist_ok=True)
                input_path = args.inference_input or destination / 'preview_input.npz'
                if preview:
                    print('Reconstructing one TRAINING crop as a preview, not a held-out evaluation.', flush=True)
                    prepare_preview(data_root, config, args.sample_index, input_path)
                validate_input(input_path, config)
                command = prefix + ['src.infer.reconstruct', '--config', str(config_path),
                    '--average-checkpoint', str(average), '--anomaly-checkpoint', str(anomaly),
                    '--input', str(input_path), '--output', str(destination / 'reconstruction.npz')]
            state['stages'][stage]['command'] = command
            write_json(state_path, state)
            executor(command, args.run_dir / 'logs' / f'{stage}.log')
            if stage in ('average', 'anomaly'):
                require_file(average if stage == 'average' else anomaly)
            elif stage == 'preprocess':
                status = json.loads((data_root / 'run.json').read_text())
                if status.get('status') != 'complete':
                    raise ValueError('Preprocessing did not finish successfully')
                validate_data(data_root, config, args.stages, preview)
            else:
                import numpy as np
                require_file(destination / 'reconstruction.npz')
                with np.load(destination / 'reconstruction.npz', allow_pickle=False) as f:
                    for key in ('pred_weekly', 'pred_anomaly', 'pred_sst'):
                        if not np.isfinite(f[key]).all():
                            raise ValueError(f'Nonfinite reconstruction: {key}')
            state['stages'][stage]['status'] = 'complete'
            write_json(state_path, state)
        state['status'] = 'complete'
    except BaseException as exc:
        state['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        if current:
            state['stages'][current].update(status=state['status'], error=str(exc))
        raise
    finally:
        state['finished_at'] = datetime.now().astimezone().isoformat()
        write_json(state_path, state)
    print(f'Completed. Results: {args.run_dir}', flush=True)
    return state


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print('Interrupted; see experiment run.json.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'Failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
