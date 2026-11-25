# テスト: 全体識別器の入力制約と出力形状が期待どおりに動くかを自動検査する。
"""Tests for the discriminator forward pass."""

# ── 使用するライブラリ・共通処理 ──────────────────
import torch

from src.models.discriminator import Discriminator


# ── 動作確認用のデータ・テスト ──────────────────
def test_discriminator_random_forward() -> None:
    batch = 2
    height = width = 256
    model = Discriminator().eval()
    x = torch.randn(batch, 1, height, width)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (batch, 1)
    assert torch.isfinite(logits).all()
