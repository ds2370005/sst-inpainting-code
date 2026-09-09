"""Tests for the reconstruction inference pipeline."""

from pathlib import Path

import numpy as np
import torch

from src.infer.reconstruct import run_reconstruction
from src.models import AnomalyInpaintingGenerator, AverageEstimationGenerator


def _config(patch_size: int = 160, time_steps: int = 9) -> dict:
    return {
        "data": {
            "patch_size": patch_size,
            "time_steps": time_steps,
        },
        "training": {
            "device": "cpu",
        },
    }


def _save_generator_checkpoint(path: Path, model: torch.nn.Module) -> None:
    torch.save({"generator_state_dict": model.state_dict()}, path)


def test_run_reconstruction_writes_npz_outputs(tmp_path: Path) -> None:
    patch_size = 160
    time_steps = 9
    config = _config(patch_size, time_steps)
    average_checkpoint = tmp_path / "average.pth"
    anomaly_checkpoint = tmp_path / "anomaly.pth"
    _save_generator_checkpoint(
        average_checkpoint,
        AverageEstimationGenerator(time_steps=time_steps),
    )
    _save_generator_checkpoint(
        anomaly_checkpoint,
        AnomalyInpaintingGenerator(time_steps=time_steps),
    )

    input_path = tmp_path / "sample_input.npz"
    np.savez(
        input_path,
        sst_volume=np.random.randn(1, time_steps, patch_size, patch_size).astype(np.float32),
        mask_volume=np.ones((1, time_steps, patch_size, patch_size), dtype=np.float32),
        monthly_average=np.random.randn(1, 1, patch_size, patch_size).astype(np.float32),
        monthly_mask=np.ones((1, 1, patch_size, patch_size), dtype=np.float32),
    )
    output_path = tmp_path / "sample_output.npz"

    output = run_reconstruction(
        config=config,
        average_checkpoint=average_checkpoint,
        anomaly_checkpoint=anomaly_checkpoint,
        input_path=input_path,
        output_path=output_path,
        device=torch.device("cpu"),
    )

    assert output_path.exists()
    assert output["pred_weekly"].shape == (1, 1, patch_size, patch_size)
    assert output["pred_anomaly"].shape == (1, 1, patch_size, patch_size)
    assert output["pred_sst"].shape == (1, 1, patch_size, patch_size)
    with np.load(output_path) as archive:
        assert sorted(archive.keys()) == ["pred_anomaly", "pred_sst", "pred_weekly"]
        assert archive["pred_sst"].shape == (1, 1, patch_size, patch_size)
