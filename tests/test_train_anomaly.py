"""Tests for the anomaly-stage synthetic training loop."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.train.train_anomaly import (
    SyntheticAnomalyDataset,
    build_anomaly_training_components,
    create_synthetic_anomaly_loader,
    save_checkpoint,
    train_one_epoch,
)


def _synthetic_config(tmp_path: Path) -> dict:
    return {
        "seed": 42,
        "data": {
            "patch_size": 160,
            "time_steps": 9,
            "time_interval_hours": 6,
            "min_temp": 0.0,
            "max_temp": 35.0,
            "anomaly_range": 5.0,
        },
        "training": {
            "epochs": 1,
            "batch_size": 1,
            "num_workers": 0,
            "device": "cpu",
            "mixed_precision": False,
        },
        "optimizer": {
            "lr_generator": 1.0e-4,
            "lr_discriminator": 1.0e-8,
            "beta1": 0.5,
            "beta2": 0.99,
        },
        "loss": {
            "lambda_ave_rec": 10.0,
            "lambda_ave_adv": 0.1,
            "lambda_ano_rec": 10.0,
            "lambda_ano_adv": 0.1,
        },
        "paths": {
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "output_dir": str(tmp_path / "outputs"),
        },
    }


def test_synthetic_anomaly_dataset_shapes() -> None:
    dataset = SyntheticAnomalyDataset(num_samples=2, time_steps=9, patch_size=160)

    sample = dataset[0]

    assert sample["sst_volume"].shape == (1, 9, 160, 160)
    assert sample["mask_volume"].shape == (1, 9, 160, 160)
    assert sample["target_sst"].shape == (1, 160, 160)
    assert sample["target_mask"].shape == (1, 160, 160)
    assert sample["weekly_average"].shape == (1, 160, 160)
    assert sample["assimilation_sst"].shape == (1, 160, 160)


def test_train_anomaly_one_synthetic_batch(tmp_path: Path) -> None:
    config = _synthetic_config(tmp_path)
    device = torch.device("cpu")
    loader = create_synthetic_anomaly_loader(config, num_samples=1)
    generator, discriminator, optimizer_g, optimizer_d = build_anomaly_training_components(
        config,
        device,
    )

    metrics = train_one_epoch(
        generator=generator,
        discriminator=discriminator,
        loader=loader,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        config=config,
        device=device,
        max_batches=1,
    )
    checkpoint_path = Path(config["paths"]["checkpoint_dir"]) / "anomaly" / "latest.pth"
    save_checkpoint(
        path=checkpoint_path,
        epoch=1,
        generator=generator,
        discriminator=discriminator,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        config=config,
    )

    assert checkpoint_path.exists()
    for value in metrics.values():
        assert torch.isfinite(torch.tensor(value))


def test_train_anomaly_without_assimilation_uses_reconstruction_only(
    tmp_path: Path,
) -> None:
    config = _synthetic_config(tmp_path)
    device = torch.device("cpu")
    sample = SyntheticAnomalyDataset(num_samples=1, time_steps=9, patch_size=160)[0]
    sample.pop("assimilation_sst")
    loader = DataLoader([sample], batch_size=1)
    generator, discriminator, optimizer_g, optimizer_d = build_anomaly_training_components(
        config, device
    )

    metrics = train_one_epoch(
        generator=generator,
        discriminator=discriminator,
        loader=loader,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        config=config,
        device=device,
        max_batches=1,
    )

    assert metrics["loss_d"] == 0.0
    assert metrics["loss_adv"] == 0.0
    assert metrics["loss_rec"] > 0.0
