"""観測時刻平均と最近傍補間。入力と同じ水温尺度で基準画像を返す。"""
import time
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def temporal_mean_nearest(sst, mask, ocean_mask=None, *, log_prefix=None):
    """[B,1,9,H,W] -> 基準SSTと診断情報。全欠損は明示的に拒否する。

    ocean_mask は [B,1,H,W]。省略時は全画素を補間対象とするため、
    海陸を区別したい場合は必ず指定する。距離は格子の画素単位。
    """
    if sst.ndim != 5 or sst.shape[1:3] != (1, 9) or mask.shape != sst.shape:
        raise ValueError('Expected matching [B,1,9,H,W] SST and mask')
    if log_prefix is not None:
        if sst.is_cuda:
            torch.cuda.synchronize(sst.device)
        print(f"{log_prefix} 平均開始: samples={sst.shape[0]}, frames=9, size={sst.shape[-2]}x{sst.shape[-1]}", flush=True)
    started = time.perf_counter()
    valid = (mask > 0) & torch.isfinite(sst)
    shape = (sst.shape[0], 1, *sst.shape[-2:])
    ocean = torch.ones(shape, dtype=torch.bool, device=sst.device) if ocean_mask is None else ocean_mask.to(sst.device).bool()
    if ocean.shape != shape:
        raise ValueError('ocean_mask must be [B,1,H,W]')
    valid = valid & ocean.unsqueeze(2)
    count = valid.sum(dim=2)
    mean = torch.where(valid, sst, 0.).sum(dim=2) / count.clamp_min(1)
    mean_np = mean.detach().float().cpu().numpy()
    count_np = count.cpu().numpy()
    ocean_np = ocean.cpu().numpy()
    mean_seconds = time.perf_counter() - started
    missing_count = int(((count_np == 0) & ocean_np).sum())
    if log_prefix is not None:
        print(f"{log_prefix} 平均完了: {mean_seconds:.3f}s（CPU転送を含む）, 有効={int(((count_np > 0) & ocean_np).sum())}, 補間対象={missing_count}/{int(ocean_np.sum())}", flush=True)
        print(f"{log_prefix} 最近傍補間開始", flush=True)
    started = time.perf_counter()
    filled = mean_np.copy()
    distances = np.zeros(shape, dtype=np.float32)
    for b in range(sst.shape[0]):
        known = count_np[b, 0] > 0
        if not known.any():
            raise ValueError(f'No valid observations in nine frames (batch item {b})')
        distance, indices = distance_transform_edt(~known, return_indices=True)
        missing = ocean_np[b, 0] & ~known
        filled[b, 0][missing] = mean_np[b, 0][tuple(indices[:, missing])]
        filled[b, 0][~ocean_np[b, 0]] = 0.
        distances[b, 0][missing] = distance[missing]
    if log_prefix is not None:
        print(f"{log_prefix} 最近傍補間完了: {time.perf_counter()-started:.3f}s, 補間画素={missing_count}, 最大距離={float(distances.max()):.2f}px", flush=True)
    return {
        'baseline': torch.as_tensor(filled, device=sst.device, dtype=sst.dtype),
        'observation_count': count,
        'interpolated_mask': ocean & (count == 0),
        'distance_pixels': torch.as_tensor(distances, device=sst.device),
    }


def main():
    """保存済み入力から基準画像と診断配列を作成（学習は行わない）。"""
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--input', required=True, help='NPZ: sst_volume, mask_volume; optional ocean_mask')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        sst = torch.from_numpy(archive['sst_volume']).float()
        mask = torch.from_numpy(archive['mask_volume']).float()
        if sst.ndim == 4:
            sst, mask = sst.unsqueeze(0), mask.unsqueeze(0)
        ocean = torch.from_numpy(archive['ocean_mask']) if 'ocean_mask' in archive else None
        result = temporal_mean_nearest(sst, mask, ocean)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as handle:
        np.savez_compressed(handle, **{k:v.numpy() for k,v in result.items()})
    print(f'Saved: {output}; interpolated={int(result["interpolated_mask"].sum())}')


if __name__ == '__main__':
    main()
