# 役割: 実画像と生成画像の識別結果から、生成器と識別器の敵対的損失を計算する。
"""平原論文の式(2)：物理モデルに基づく敵対的損失。

テンソル形式のロジットには論文の単一全体識別器を使用する。
辞書形式のロジットは、旧設定の全体・局所識別器を使う実験用に残している。
識別器Dは-log D(real)-log(1-D(fake))、生成器Gはlog(1-D(fake))を最小化する。
実画像の項はGに対して定数のため、Gの更新から省略する。
Dはロジットを返し、logsigmoidで確率や対数の直接計算を避ける。
これは以前の非飽和型の生成器目的関数-log D(fake)とは異なる。"""

from __future__ import annotations

# ── 使用するライブラリ・共通処理 ──────────────────
import torch
import torch.nn.functional as F


# ── 処理に使用する関数・クラス ──────────────────
def _assert_logits(logits: torch.Tensor, name: str) -> None:  # 学習損失: Dの出力が[B,1]かを確認する
    assert logits.ndim in (2, 4), f"{name} must be [B,1] or [B,1,H,W], got {tuple(logits.shape)}"
    assert logits.shape[0] > 0, f"{name} batch dimension must be positive"
    assert logits.shape[1] == 1, f"{name} must have one logit channel"


def _single_generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:  # 学習損失: 生成結果を本物と判定させる方向に学習する
    """式(2)の生成器に依存する項log(1-D(fake))を最小化する。"""
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
    """式(3)～(5)：バッチ・パッチ平均を取り、全体・局所の重みを適用する。"""
    return _weighted(fake_logits, _single_generator_loss, beta)


def discriminator_loss(real_logits, fake_logits, beta=0.5):
    """最適化器が最小化するため、式(3)～(5)の符号を反転する。"""
    if isinstance(real_logits, dict) != isinstance(fake_logits, dict):
        raise ValueError("Real and fake discriminator outputs must have the same structure")
    return (_weighted(real_logits, lambda z: -F.logsigmoid(z).mean(), beta)
            + _weighted(fake_logits, lambda z: -F.logsigmoid(-z).mean(), beta))
