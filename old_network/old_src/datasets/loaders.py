"""Modular array loading helpers for dataset interfaces."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


NETCDF_EXTENSIONS = {".nc", ".netcdf", ".cdf"}
GEOTIFF_EXTENSIONS = {".tif", ".tiff"}


def load_index_records(index_file: str | Path) -> list[dict[str, Any]]:  # データローダー: JSON/JSONL/CSVのindexをサンプル一覧として読む
    """Load sample records from JSON, JSONL, or CSV index files."""
    path = Path(index_file)
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, Mapping) and "samples" in data:
            data = data["samples"]
        assert isinstance(data, list), "JSON index must be a list or contain a samples list."
        return [dict(record) for record in data]
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(dict(json.loads(line)))
        return records
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(f"Unsupported index format: {path.suffix}")


class ArrayFieldLoader:  # データローダー: .npy/.npzの配列をTensorに変換する
    """Load tensor fields from path specs.

    Supported now:
    - .npy files
    - .npz files with an explicit key, or a single array inside

    Extension hooks intentionally raise for NetCDF and GeoTIFF so those loaders
    can be added without changing Dataset classes.
    """

    def load_tensor(self, spec: str | Mapping[str, Any], base_dir: Path) -> torch.Tensor:
        path, key, index = self._parse_spec(spec, base_dir)
        array = self.load_array(path, key=key)
        if index is not None:
            array = array[index]
        return torch.as_tensor(array, dtype=torch.float32)

    def load_array(self, path: Path, key: str | None = None) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            if key is not None:
                raise ValueError(f".npy field specs must not include key={key!r}: {path}")
            return np.load(path)
        if suffix == ".npz":
            with np.load(path) as archive:
                if key is None:
                    keys = list(archive.keys())
                    if len(keys) != 1:
                        raise ValueError(
                            f"NPZ field spec for {path} needs a key; found keys {keys}."
                        )
                    key = keys[0]
                return np.asarray(archive[key])
        if suffix in NETCDF_EXTENSIONS:
            raise NotImplementedError(
                "NetCDF loading is not implemented yet. Add a loader adapter here."
            )
        if suffix in GEOTIFF_EXTENSIONS:
            raise NotImplementedError(
                "GeoTIFF loading is not implemented yet. Add a loader adapter here."
            )
        raise ValueError(f"Unsupported array format: {path.suffix}")

    def _parse_spec(
        self,
        spec: str | Mapping[str, Any],
        base_dir: Path,
    ) -> tuple[Path, str | None, int | Sequence[int] | None]:
        if isinstance(spec, str):
            path = Path(spec)
            key = None
            index = None
        else:
            path = Path(str(spec["path"]))
            key = spec.get("key")
            index = spec.get("index")
        if not path.is_absolute():
            path = base_dir / path
        return path, key, index


def load_required_tensor(
    record: Mapping[str, Any],
    field: str,
    *,
    base_dir: Path,
    loader: ArrayFieldLoader,
) -> torch.Tensor:  # データローダー: indexレコードの必須フィールドをTensorとして読む
    assert field in record, f"Index record is missing required field {field!r}."
    return loader.load_tensor(record[field], base_dir)


def assert_shape(tensor: torch.Tensor, field: str, expected: tuple[int, ...]) -> None:  # 共通: 読み込んだTensorの形状を期待値と照合する
    assert tuple(tensor.shape) == expected, (
        f"{field} must have shape {expected}, got {tuple(tensor.shape)}"
    )
