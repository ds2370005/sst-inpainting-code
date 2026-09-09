"""Tests for the generator forward passes."""

import torch

from src.models.generator import (
    AnomalyInpaintingGenerator,
    AverageEstimationGenerator,
    EncoderDecoder2D,
)


def test_encoder_decoder_random_forward() -> None:
    batch = 2
    time_steps = 9
    height = width = 256
    model = EncoderDecoder2D(in_channels=time_steps).eval()
    x = torch.randn(batch, time_steps, height, width)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (batch, 1, height, width)
    assert torch.isfinite(y).all()
    assert y.min().item() >= -1.0
    assert y.max().item() <= 1.0


def test_generators_random_forward() -> None:
    batch = 2
    time_steps = 9
    height = width = 256

    sst_volume = torch.randn(batch, 1, time_steps, height, width)
    mask_volume = torch.ones(batch, 1, time_steps, height, width)
    monthly_average = torch.randn(batch, 1, 1, height, width)
    monthly_mask = torch.ones(batch, 1, 1, height, width)

    average_generator = AverageEstimationGenerator(time_steps=time_steps).eval()
    anomaly_generator = AnomalyInpaintingGenerator(time_steps=time_steps).eval()

    with torch.no_grad():
        pred_weekly = average_generator(
            sst_volume,
            mask_volume,
            monthly_average,
            monthly_mask,
        )
        pred_anomaly = anomaly_generator(
            sst_volume,
            mask_volume,
            pred_weekly.unsqueeze(2),
        )

    assert pred_weekly.shape == (batch, 1, height, width)
    assert pred_anomaly.shape == (batch, 1, height, width)
    assert torch.isfinite(pred_weekly).all()
    assert torch.isfinite(pred_anomaly).all()
    assert pred_weekly.min().item() >= -1.0
    assert pred_weekly.max().item() <= 1.0
    assert pred_anomaly.min().item() >= -1.0
    assert pred_anomaly.max().item() <= 1.0

