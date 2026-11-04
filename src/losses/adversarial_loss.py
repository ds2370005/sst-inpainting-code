"""ACPR Eqs. (3)-(5): global/local adversarial physical model loss.

D minimizes -log D(real) - log(1-D(fake)); G minimizes log(1-D(fake)).
The real term is constant with respect to G and omitted from its update.
D returns logits; logsigmoid avoids explicitly forming/logging probabilities.
This differs from the previous non-saturating G objective -log D(fake).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _assert_logits(logits: torch.Tensor, name: str) -> None:  # 学習損失: Dの出力が[B,1]かを確認する
    assert logits.ndim in (2, 4), f"{name} must be [B,1] or [B,1,H,W], got {tuple(logits.shape)}"
    assert logits.shape[0] > 0, f"{name} batch dimension must be positive"
    assert logits.shape[1] == 1, f"{name} must have one logit channel"


def _single_generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:  # 学習損失: 生成結果を本物と判定させる方向に学習する
    """Minimize log(1-D(fake)), the generator-dependent term of Eq. (2)."""
    _assert_logits(fake_logits, "fake_logits")
    return F.logsigmoid(-fake_logits).mean()


def _weighted(outputs, function, beta):
    if not 0 <= beta <= 1:
        raise ValueError("physical_beta must be in [0,1]")
    if isinstance(outputs, dict):
        if set(outputs) != {"global", "local"}:
            raise ValueError("Expected global/local logits")
        for name, value in outputs.items(): _assert_logits(value, name)
        return beta * function(outputs["global"]) + (1-beta) * function(outputs["local"])
    _assert_logits(outputs, "logits")
    return function(outputs)


def generator_adversarial_loss(fake_logits, beta=0.5):
    """Eq. (3)-(5): batch/patch mean, then global/local weighting."""
    return _weighted(fake_logits, _single_generator_loss, beta)


def discriminator_loss(real_logits, fake_logits, beta=0.5):
    """Negative of Eq. (3)-(5), since the optimizer minimizes."""
    if isinstance(real_logits, dict) != isinstance(fake_logits, dict):
        raise ValueError("Real and fake discriminator outputs must have the same structure")
    return (_weighted(real_logits, lambda z: -F.logsigmoid(z).mean(), beta)
            + _weighted(fake_logits, lambda z: -F.logsigmoid(-z).mean(), beta))
