"""Tests for adversarial and reconstruction losses."""

import torch
import torch.nn.functional as F

from src.losses import (
    discriminator_loss,
    generator_adversarial_loss,
    masked_mse_loss,
)


def test_masked_mse_loss_matches_manual_value() -> None:
    pred = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]])
    target = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
    mask = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])

    loss = masked_mse_loss(pred, target, mask)

    expected = torch.tensor((1.0 + 9.0 + 16.0) / 3.0)
    assert torch.allclose(loss, expected)


def test_masked_mse_loss_supports_backward() -> None:
    pred = torch.randn(2, 1, 8, 8, requires_grad=True)
    target = torch.randn(2, 1, 8, 8)
    mask = torch.randint(0, 2, (2, 1, 8, 8), dtype=torch.float32)

    loss = masked_mse_loss(pred, target, mask)
    loss.backward()

    assert loss.ndim == 0
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_discriminator_loss_matches_bce_with_logits() -> None:
    real_logits = torch.tensor([[2.0], [0.5]])
    fake_logits = torch.tensor([[-1.0], [0.25]])

    loss = discriminator_loss(real_logits, fake_logits)

    expected = F.binary_cross_entropy_with_logits(
        real_logits,
        torch.ones_like(real_logits),
    ) + F.binary_cross_entropy_with_logits(
        fake_logits,
        torch.zeros_like(fake_logits),
    )
    assert torch.allclose(loss, expected)


def test_generator_adversarial_loss_matches_bce_with_logits() -> None:
    fake_logits = torch.tensor([[-1.0], [0.25]], requires_grad=True)

    loss = generator_adversarial_loss(fake_logits)
    expected = F.binary_cross_entropy_with_logits(
        fake_logits,
        torch.ones_like(fake_logits),
    )
    loss.backward()

    assert torch.allclose(loss.detach(), expected.detach())
    assert fake_logits.grad is not None
    assert torch.isfinite(fake_logits.grad).all()
