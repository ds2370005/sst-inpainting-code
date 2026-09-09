"""Datasets backed directly by NPZ files from Himawari preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_sst(array: np.ndarray, min_temp: float, max_temp: float) -> np.ndarray:
    if not max_temp > min_temp:
        raise ValueError("max_temp must be greater than min_temp.")
    return 2.0 * (array - min_temp) / (max_temp - min_temp) - 1.0


class HimawariPatchDataset(Dataset[dict[str, torch.Tensor]]):
    """Read current preprocessor outputs for average or anomaly training.

    Stored values remain Celsius/NaN. This adapter derives valid masks and
    converts valid SST values to [-1, 1]; invalid values become zero only in
    the tensors passed to the network.
    """

    REQUIRED_KEYS = {
        "sst_volume",
        "cloud_mask_volume",
        "weekly_average",
        "monthly_average",
    }

    def __init__(
        self,
        data_root: str | Path,
        *,
        stage: Literal["average", "anomaly"],
        time_steps: int,
        patch_size: int,
        min_temp: float,
        max_temp: float,
    ) -> None:
        self.data_root = Path(data_root)
        self.stage = stage
        self.time_steps = time_steps
        self.patch_size = patch_size
        self.min_temp = min_temp
        self.max_temp = max_temp
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"Himawari patch directory not found: {self.data_root}")

        candidates = sorted(self.data_root.glob("*.npz"))
        self.paths: list[Path] = []
        for path in candidates:
            try:
                with np.load(path, allow_pickle=False) as archive:
                    if self.REQUIRED_KEYS.issubset(archive.files):
                        self.paths.append(path)
            except (OSError, ValueError):
                continue
        if not self.paths:
            raise ValueError(
                f"No complete Himawari NPZ samples found in {self.data_root}. "
                f"Required keys: {sorted(self.REQUIRED_KEYS)}"
            )

    def __len__(self) -> int:
        return len(self.paths)

    def _normalized_field(
        self,
        values: np.ndarray,
        valid: np.ndarray,
    ) -> torch.Tensor:
        normalized = normalize_sst(values, self.min_temp, self.max_temp)
        safe = np.where(valid, normalized, 0.0).astype(np.float32, copy=False)
        return torch.from_numpy(safe.copy())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as archive:
            sst = np.asarray(archive["sst_volume"], dtype=np.float32)
            cloud = np.asarray(archive["cloud_mask_volume"], dtype=np.uint8)
            weekly = np.asarray(archive["weekly_average"], dtype=np.float32)
            monthly = np.asarray(archive["monthly_average"], dtype=np.float32)

        expected_volume = (1, self.time_steps, self.patch_size, self.patch_size)
        if sst.shape != expected_volume or cloud.shape != expected_volume:
            raise ValueError(
                f"Unexpected volume shapes in {path}: SST={sst.shape}, cloud={cloud.shape}, "
                f"expected={expected_volume}."
            )
        if weekly.shape != (1, self.patch_size, self.patch_size):
            raise ValueError(f"Unexpected weekly_average shape {weekly.shape} in {path}.")
        if monthly.shape != (1, 1, self.patch_size, self.patch_size):
            raise ValueError(f"Unexpected monthly_average shape {monthly.shape} in {path}.")

        # Cloud masks contain only clouds. The finite-SST condition separately
        # excludes land and all other unavailable pixels.
        volume_valid = np.isfinite(sst) & (cloud == 0)
        weekly_valid = np.isfinite(weekly)
        monthly_valid = np.isfinite(monthly)
        common = {
            "sst_volume": self._normalized_field(sst, volume_valid),
            "mask_volume": torch.from_numpy(volume_valid.astype(np.float32)),
            "weekly_average": self._normalized_field(weekly, weekly_valid),
        }

        if self.stage == "average":
            return {
                **common,
                "monthly_average": self._normalized_field(monthly, monthly_valid),
                "monthly_mask": torch.from_numpy(monthly_valid.astype(np.float32)),
                "weekly_mask": torch.from_numpy(weekly_valid.astype(np.float32)),
            }

        target_valid = volume_valid[:, -1]
        return {
            **common,
            "target_sst": self._normalized_field(sst[:, -1], target_valid),
            "target_mask": torch.from_numpy(target_valid.astype(np.float32)),
        }


def create_himawari_loader(
    data_root: str | Path,
    config: dict,
    *,
    stage: Literal["average", "anomaly"],
    shuffle: bool = True,
) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:
    data = config["data"]
    training = config["training"]
    dataset = HimawariPatchDataset(
        data_root,
        stage=stage,
        time_steps=int(data["time_steps"]),
        patch_size=int(data["patch_size"]),
        min_temp=float(data["min_temp"]),
        max_temp=float(data["max_temp"]),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=shuffle,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=bool(
            training.get("device") == "cuda" and torch.cuda.is_available()
        ),
    )
