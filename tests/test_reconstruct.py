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
            "min_temp": 0.0,
            "max_temp": 35.0,
            "anomaly_range": 5.0,
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


def test_reconstruction_converts_anomaly_units(tmp_path, monkeypatch):
    import src.infer.reconstruct as module

    class Constant(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value

        def forward(self, sst, *args):
            return torch.full_like(sst[:, :, 0], self.value)

    monkeypatch.setattr(module, "AverageEstimationGenerator", lambda **kw: Constant(0.0))
    monkeypatch.setattr(module, "AnomalyInpaintingGenerator", lambda **kw: Constant(1.0))
    monkeypatch.setattr(module, "load_generator_checkpoint", lambda *args: None)
    path = tmp_path / "input.npz"
    np.savez(path, sst_volume=np.zeros((1, 9, 2, 2)),
             mask_volume=np.ones((1, 9, 2, 2)),
             monthly_average=np.zeros((1, 1, 2, 2)),
             monthly_mask=np.ones((1, 1, 2, 2)))
    result = run_reconstruction(config=_config(), average_checkpoint=path,
                                anomaly_checkpoint=path, input_path=path,
                                output_path=tmp_path / "output.npz", device=torch.device("cpu"))
    # Weekly normalized 0 is 17.5 C; unit anomaly +1 adds 5 C, not 17.5 C.
    celsius = (result["pred_sst"] + 1) * 35 / 2
    np.testing.assert_allclose(celsius, 22.5, atol=1e-5)
