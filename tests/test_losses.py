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


def test_generator_adversarial_loss_matches_paper_minimax() -> None:
    fake_logits = torch.tensor([[-1.0], [0.25]], requires_grad=True)

    loss = generator_adversarial_loss(fake_logits)
    expected = torch.log1p(-torch.sigmoid(fake_logits)).mean()
    loss.backward()

    assert torch.allclose(loss.detach(), expected.detach())
    assert fake_logits.grad is not None
    assert torch.isfinite(fake_logits.grad).all()


def test_minimax_extreme_logits_and_gradient_direction():
    logits = torch.tensor([[-1000.], [0.], [1000.]], requires_grad=True)
    loss = generator_adversarial_loss(logits)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert (logits.grad <= 0).all()
    assert logits.grad[-1].item() < 0
    zero = torch.zeros(1, 1)
    assert torch.allclose(generator_adversarial_loss(zero), -torch.log(torch.tensor(2.)))
    assert torch.allclose(discriminator_loss(zero, zero), 2*torch.log(torch.tensor(2.)))
