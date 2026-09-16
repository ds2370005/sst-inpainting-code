"""Temporal assembly, alignment and sample selection for frame-format datasets."""
from datetime import datetime, timedelta

import numpy as np
import pytest
import torch

from src.datasets.himawari_patch_dataset import HimawariPatchDataset, create_himawari_loader


def products(root, *, steps=9, interval=6):
    frames = root / 'frames/himawari'
    averages = root / 'averages/himawari'
    frames.mkdir(parents=True)
    averages.mkdir(parents=True)
    paths = []
    for i in range(steps):
        time = datetime(2025, 4, 28) + timedelta(hours=i * interval)
        path = frames / f'{time:%Y%m%d%H%M%S}_r00_c00.npz'
        fields = dict(format_version=2, satellite='himawari', timestamp=time.isoformat(),
                      grid_row_column=[0, 0], lat=[18., 19.], lon=[118., 119.],
                      sst=np.full((2, 2), i + 10., dtype=np.float32),
                      cloud_mask=np.zeros((2, 2), dtype=np.uint8))
        if i == 0:
            fields['cloud_mask'][:] = 1  # Cloudy context must not discard target.
        np.savez_compressed(path, **fields)
        paths.append(path)
    average = averages / paths[-1].name
    np.savez_compressed(average, format_version=2, satellite='himawari',
                        target_timestamp=time.isoformat(), grid_row_column=[0, 0],
                        lat=[18., 19.], lon=[118., 119.],
                        weekly_average=np.full((1, 2, 2), 15., dtype=np.float32),
                        monthly_average=np.full((1, 1, 2, 2), 14., dtype=np.float32))
    return paths, average


def dataset(root, **kwargs):
    return HimawariPatchDataset(root, stage='anomaly', time_steps=9,
                               patch_size=2, min_temp=0, max_temp=35, **kwargs)


def rewrite(path, **changes):
    with np.load(path) as archive:
        fields = dict(archive)
    fields.update(changes)
    np.savez_compressed(path, **fields)


def test_temporal_order_masks_target_and_batch(tmp_path):
    products(tmp_path)
    ds = dataset(tmp_path)
    assert len(ds) == 1
    sample = ds[0]
    assert sample['sst_volume'].shape == (1, 9, 2, 2)
    torch.testing.assert_close(sample['sst_volume'][0, 1:, 0, 0],
                               torch.tensor(2 * np.arange(11, 19) / 35 - 1, dtype=torch.float32))
    assert sample['mask_volume'][0, 0].sum() == 0
    assert sample['sst_volume'][0, 0].sum() == 0
    torch.testing.assert_close(sample['target_sst'], sample['sst_volume'][:, -1])
    config = dict(data=dict(time_steps=9, patch_size=2, min_temp=0, max_temp=35),
                  training=dict(batch_size=1, num_workers=0, device='cpu'))
    batch = next(iter(create_himawari_loader(tmp_path, config, stage='average')))
    assert batch['sst_volume'].shape == (1, 1, 9, 2, 2)
    assert batch['monthly_average'].shape == (1, 1, 1, 2, 2)
    assert batch['weekly_average'].shape == (1, 1, 2, 2)


@pytest.mark.parametrize('missing', ['frame', 'average'])
def test_missing_required_product_excludes_sample(tmp_path, missing):
    frames, average = products(tmp_path)
    (frames[3] if missing == 'frame' else average).unlink()
    with pytest.raises(ValueError, match='No eligible'):
        dataset(tmp_path)


@pytest.mark.parametrize('product', ['frame', 'average'])
def test_coordinate_mismatch_fails_loudly(tmp_path, product):
    frames, average = products(tmp_path)
    rewrite(frames[2] if product == 'frame' else average, lat=[18.1, 19.1])
    with pytest.raises(ValueError, match='grid differs'):
        dataset(tmp_path)[0]


@pytest.mark.parametrize('changes,reason', [
    ({'cloud_mask': np.ones((2, 2))}, 'cloudy_target'),
    ({'sst': np.full((2, 2), np.nan)}, 'invalid_target'),
])
def test_target_filter_uses_actual_arrays(tmp_path, changes, reason):
    frames, _ = products(tmp_path)
    rewrite(frames[-1], **changes)
    with pytest.raises(ValueError, match=reason):
        dataset(tmp_path)


def test_interval_configuration_and_identity(tmp_path):
    frames, average = products(tmp_path, interval=3)
    assert len(dataset(tmp_path, time_interval_hours=3)) == 1
    with pytest.raises(ValueError, match='No eligible'):
        dataset(tmp_path)
    rewrite(average, target_timestamp='2025-01-01T00:00:00')
    with pytest.raises(ValueError, match='identity'):
        dataset(tmp_path, time_interval_hours=3)[0]
