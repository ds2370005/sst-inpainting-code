# 役割: 保存済み全域データから指定時刻に対応する9時刻のSSTとマスクを読み込む。
"""保存済み全域フレームから9時刻の入力を読む。生データの再変換は不要。"""
# ── 使用するライブラリ・共通処理 ──────────────────
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import numpy as np
import torch
from .common import read_array


# ── 処理に使用する関数・クラス ──────────────────
def load_full_frames(root, paths, settings, keep=None):
    volume = np.stack([np.asarray(read_array(root, p), dtype=np.float32) for p in paths])
    if len(paths) != 9:
        raise ValueError('Exactly nine frames are required')
    land = np.logical_or.reduce([np.asarray(read_array(root,p,'land_mask'), dtype=bool) for p in paths])
    valid = np.isfinite(volume) & ~land[None]
    if keep is not None:
        if keep.shape != land.shape:
            raise ValueError('Cloud mask grid differs from SST grid')
        valid &= np.asarray(keep, dtype=bool)[None]
    lo, hi = float(settings['min_temp']), float(settings['max_temp'])
    normalized = np.where(valid, 2*(volume-lo)/(hi-lo)-1, 0).astype(np.float32)
    return (torch.from_numpy(normalized[None,None]),
            torch.from_numpy(valid[None,None].astype(np.float32)),
            torch.from_numpy((~land)[None,None]))


def paths_for_time(root, target_time, interval=6):
    manifest = json.loads((Path(root)/'dataset.json').read_text())
    def stamp(s):
        t=datetime.fromisoformat(s.replace('Z','+00:00'))
        return t.astimezone(timezone.utc).replace(tzinfo=None) if t.tzinfo else t
    target=stamp(target_time)
    records={}
    for row in manifest['himawari_frames']:
        t=stamp(row['time'])
        if t in records:
            raise ValueError(f'Duplicate frame time: {t}')
        records[t]=row['path']
    times=[target-timedelta(hours=interval*i) for i in reversed(range(9))]
    missing=[t.isoformat() for t in times if t not in records]
    if missing:
        raise ValueError(f'Missing required hourly frames: {missing}')
    return [records[t] for t in times], [t.isoformat() for t in times]
