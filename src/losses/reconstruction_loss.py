"""Masked reconstruction losses."""

from __future__ import annotations

import torch


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:  # 学習損失: マスクで有効画素だけを使ってMSEを計算する
    """Mean squared error over valid pixels only."""
    assert pred.ndim == 4, f"pred must be [B, C, H, W], got {tuple(pred.shape)}"
    assert target.shape == pred.shape, (
        f"target must match pred, got {tuple(target.shape)} and {tuple(pred.shape)}"
    )
    assert mask.shape == pred.shape, (
        f"mask must match pred, got {tuple(mask.shape)} and {tuple(pred.shape)}"
    )
    assert eps > 0.0, f"eps must be positive, got {eps}"

    valid = mask > 0
    safe_target = torch.where(valid, target, pred.detach())
    squared_error = torch.where(valid, (pred - safe_target) ** 2, 0.0)
    return squared_error.sum() / (valid.sum() + eps)
