from pathlib import Path

import torch
import yaml

from src.losses import loss_weights, reconstruction_loss, generator_adversarial_loss
from src.models.discriminator import build_discriminator, Discriminator


def test_default_uses_single_discriminator_and_paper_losses():
    config = yaml.safe_load((Path(__file__).parents[1] / 'config_temporal_mean_patch.yaml').read_text())
    loss = config['loss']
    assert 'physical_alpha' not in loss
    assert 'physical_beta' not in loss
    model = build_discriminator(loss)
    assert type(model) is Discriminator
    for stage in ('ave', 'ano'):
        assert loss_weights(loss, stage) == (10., .1)
    pred = torch.tensor([[[[3., 4.]]]], requires_grad=True)
    rec = reconstruction_loss(pred, torch.zeros_like(pred), torch.ones_like(pred), loss)
    torch.testing.assert_close(rec, torch.tensor(5.))
    logits = torch.tensor([[0.]], requires_grad=True)
    adv = generator_adversarial_loss(logits)
    total = 10 * rec + .1 * adv
    total.backward()
    torch.testing.assert_close(pred.grad, torch.tensor([[[[6., 8.]]]]))
    torch.testing.assert_close(logits.grad, torch.tensor([[-.05]]))
