"""Shared disk format: full-region memory-mapped arrays, not fixed patches."""
from pathlib import Path
import json
import os
import tempfile
import numpy as np

FORMAT_VERSION = 3


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        try:
            json.dump(value, f, indent=2)
            f.close()
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


def save_frame(root, relative, arrays, metadata):
    directory = Path(root) / relative
    directory.mkdir(parents=True, exist_ok=True)
    for key, value in arrays.items():
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as f:
            tmp = Path(f.name)
            try:
                np.save(f, value, allow_pickle=False)
                f.close()
                os.replace(tmp, directory / f'{key}.npy')
            finally:
                tmp.unlink(missing_ok=True)
    write_json(directory / 'metadata.json', metadata)
    return str(relative)


def read_array(root, relative, key='sst'):
    return np.load(Path(root) / relative / f'{key}.npy', mmap_mode='r', allow_pickle=False)


def window_sum(mask, size):
    """Sum every possible square window without allocating sliding patches."""
    integral = np.pad(np.asarray(mask, dtype=np.int64), ((1,0),(1,0))).cumsum(0).cumsum(1)
    return integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]


def choose_origins(eligible, count, rng):
    indices = np.flatnonzero(eligible)
    if len(indices) == 0:
        return []
    selected = rng.choice(indices, size=min(count, len(indices)), replace=False)
    return [[int(i // eligible.shape[1]), int(i % eligible.shape[1])] for i in selected]
