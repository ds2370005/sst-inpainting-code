"""Adversarial physical-model losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _assert_logits(logits: torch.Tensor, name: str) -> None:  # 学習損失: Dの出力が[B,1]かを確認する
    assert logits.ndim == 2, f"{name} must be [B, 1], got {tuple(logits.shape)}"
    assert logits.shape[0] > 0, f"{name} batch dimension must be positive"
    assert logits.shape[1] == 1, f"{name} must have one logit channel"


def discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
) -> torch.Tensor:  # 学習損失: 本物を1・偽物を0として識別器を学習する
    """BCE loss that classifies real logits as 1 and fake logits as 0."""
    _assert_logits(real_logits, "real_logits")
    _assert_logits(fake_logits, "fake_logits")

    real_targets = torch.ones_like(real_logits)
    fake_targets = torch.zeros_like(fake_logits)
    real_loss = F.binary_cross_entropy_with_logits(real_logits, real_targets)
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_targets)
    return real_loss + fake_loss


def generator_adversarial_loss(fake_logits: torch.Tensor) -> torch.Tensor:  # 学習損失: 生成結果を本物と判定させる方向に学習する
    """BCE loss that encourages generated samples to be classified as real."""
    _assert_logits(fake_logits, "fake_logits")
    targets = torch.ones_like(fake_logits)
    return F.binary_cross_entropy_with_logits(fake_logits, targets)
