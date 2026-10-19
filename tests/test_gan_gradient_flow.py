"""Adversarial-only backward must reach the first generator layer."""
import copy

import pytest
import torch

from src.losses import generator_adversarial_loss
from src.train import train_average, train_anomaly
from src.utils.config import load_config
from src.utils.normalization import anomaly_to_sst_scale


@pytest.mark.parametrize('stage', ['average', 'anomaly'])
def test_adversarial_gradient_and_optimizer_updates(stage):
    torch.manual_seed(42)
    config = copy.deepcopy(load_config('config.yaml'))
    config['training'].update(epochs=1, batch_size=1, num_workers=0, device='cpu')
    module = train_average if stage == 'average' else train_anomaly
    build = getattr(module, f'build_{stage}_training_components')
    g, d, opt_g, opt_d = build(config, torch.device('cpu'))
    # Smooth spatial signal, rather than a spatially constant toy image.
    axis = torch.linspace(-.5, .5, 256)
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    sst = (xx + yy)[None, None, None].repeat(1, 1, 9, 1, 1)
    mask = torch.ones_like(sst)
    monthly = sst[:, :, :1] * .9
    weekly = sst[:, :, -1] * .95
    batch = dict(sst_volume=sst, mask_volume=mask, monthly_average=monthly,
                 monthly_mask=torch.ones_like(monthly), weekly_average=weekly,
                 weekly_mask=torch.ones_like(weekly), target_sst=sst[:, :, -1],
                 target_mask=mask[:, :, -1], assimilation_sst=weekly,
                 assimilation_weekly=weekly, random_cloud_mask=torch.ones_like(weekly))
    old_g = {k: p.detach().clone() for k, p in g.named_parameters()}
    old_d = {k: p.detach().clone() for k, p in d.named_parameters()}
    kwargs = dict(generator=g, discriminator=d, loader=[batch], optimizer_g=opt_g,
                  optimizer_d=opt_d, config=config, device=torch.device('cpu'), max_batches=1)
    if stage == 'average':
        kwargs['epoch_index'] = 0
    module.train_one_epoch(**kwargs)
    for model, old in ((g, old_g), (d, old_d)):
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.norm() > 0, name
            assert not torch.equal(parameter.detach(), old[name]), name
    g.zero_grad(set_to_none=True)
    if stage == 'average':
        prediction = g(sst, mask, monthly, torch.ones_like(monthly))
    else:
        prediction = weekly + g(sst, mask, weekly.unsqueeze(2)) * anomaly_to_sst_scale(config['data'])
    # No reconstruction loss here: any gradient must come through D.
    generator_adversarial_loss(d(prediction)).backward()
    first_gradient = next(g.parameters()).grad
    assert first_gradient is not None and torch.isfinite(first_gradient).all()
    assert first_gradient.norm() > 0
