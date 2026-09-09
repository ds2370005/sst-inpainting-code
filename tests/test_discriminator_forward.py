"""Tests for the discriminator forward pass."""

import torch

from src.models.discriminator import Discriminator


def test_discriminator_random_forward() -> None:
    batch = 2
    height = width = 256
    model = Discriminator().eval()
    x = torch.randn(batch, 1, height, width)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (batch, 1)
    assert torch.isfinite(logits).all()
