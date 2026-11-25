# 旧実装・旧テスト: 旧global/local識別器の損失計算と勾配伝播が期待どおりに動くかを自動検査する。
"""通信や実学習データを使わず、数式・勾配・毎時データの時系列を検査する。"""
# ── 使用するライブラリ・共通処理 ──────────────────
from datetime import datetime, timedelta
import json
import numpy as np
import pytest
import torch
from src.losses import discriminator_loss, generator_adversarial_loss, masked_l2_loss
from src.models.discriminator import PhysicalModelDiscriminator
from src.preprocessing.hourly_assimilation import prepare


# ── 動作確認用のデータ・テスト ──────────────────
def test_weighted_minimax_matches_probabilities_and_gradients():
    fake={'global':torch.tensor([[.2],[-.3]],requires_grad=True),
          'local':torch.tensor([[[[.1,-.4]]],[[[.8,.3]]]],requires_grad=True)}
    real={key:value.detach()+.5 for key,value in fake.items()}
    beta=.3
    expected=sum(w*torch.log1p(-torch.sigmoid(fake[k])).mean() for k,w in [('global',beta),('local',1-beta)])
    actual=generator_adversarial_loss(fake,beta)
    torch.testing.assert_close(actual,expected)
    expected_d=-expected-sum(w*torch.log(torch.sigmoid(real[k])).mean() for k,w in [('global',beta),('local',1-beta)])
    torch.testing.assert_close(discriminator_loss(real,fake,beta),expected_d)
    actual.backward()
    assert all(torch.isfinite(v.grad).all() and (v.grad<0).all() for v in fake.values())
    for beta in (0,1): assert torch.isfinite(generator_adversarial_loss(fake,beta))
    with pytest.raises(ValueError): generator_adversarial_loss(fake,1.1)


def test_both_discriminators_receive_gradients():
    torch.set_num_threads(1)
    model=PhysicalModelDiscriminator()
    x=torch.randn(1,1,256,256,requires_grad=True)
    logits=model(x)
    assert logits['global'].shape==(1,1)
    assert logits['local'].shape==(1,1,32,32)
    generator_adversarial_loss(logits).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for branch in (model.global_discriminator,model.local_discriminator):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in branch.parameters())

