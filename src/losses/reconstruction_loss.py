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


def masked_l2_loss(pred, target, mask):
    """ACPR Eq. (6): mean of per-image L2 norms, NOT pixel MSE/RMSE."""
    if pred.ndim != 4 or pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError("pred, target, mask must have identical [B,C,H,W] shapes")
    valid = mask > 0
    safe_target = torch.where(valid, target, pred.detach())
    error = torch.where(valid, pred - safe_target, 0.0)
    return torch.linalg.vector_norm(error.float().flatten(1), dim=1).mean()


def reconstruction_loss(pred, target, mask, config):
    kind = config.get("reconstruction", "masked_mse")
    if kind == "masked_l2":
        return masked_l2_loss(pred, target, mask)
    if kind == "masked_mse":
        return masked_mse_loss(pred, target, mask)
    raise ValueError(f"Unknown reconstruction loss: {kind}")


def loss_weights(config, stage):
    """Eq. (9); retain explicit legacy lambda settings for old configurations."""
    if "physical_alpha" in config:
        alpha = float(config["physical_alpha"])
        if not 0 < alpha < 1:
            raise ValueError("physical_alpha must be strictly between 0 and 1")
        return alpha, 1-alpha
    return float(config[f"lambda_{stage}_rec"]), float(config[f"lambda_{stage}_adv"])
