"""Formula, gradients and hourly chronology; no network or real training data."""
from datetime import datetime, timedelta
import json
import numpy as np
import pytest
import torch
from src.losses import discriminator_loss, generator_adversarial_loss, masked_l2_loss
from src.models.discriminator import PhysicalModelDiscriminator
from src.preprocessing.hourly_assimilation import prepare


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


def test_l2_is_per_image_norm_and_ignores_missing():
    pred=torch.tensor([[[[3.,4.,1.]]],[[[0.,0.,0.]]]],requires_grad=True)
    target=torch.tensor([[[[0.,0.,float('nan')]]],[[[0.,0.,0.]]]])
    mask=torch.tensor([[[[1.,1.,0.]]],[[[1.,1.,1.]]]])
    loss=masked_l2_loss(pred,target,mask)
    assert loss.item()==2.5
    loss.backward()
    assert torch.isfinite(pred.grad).all() and pred.grad[0,0,0,2]==0


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


def make_source(tmp_path, count=169, gap=None):
    source=tmp_path/'source'; source.mkdir()
    np.save(source/'lat.npy',[20.,21.]);np.save(source/'lon.npy',[130.,131.])
    rows=[]
    for i in range(count):
        if i==gap: continue
        path=source/str(i);path.mkdir();np.save(path/'sst.npy',np.full((2,2),i/10,np.float32))
        rows.append(dict(time=(datetime(2025,1,1)+timedelta(hours=i)).isoformat(),path=str(i)))
    (source/'simulation.json').write_text(json.dumps(dict(units='degC',cadence_hours=1,time_basis='UTC',frames=rows)))
    return source


def test_hourly_pool_and_rolling_weekly(tmp_path):
    source=make_source(tmp_path)
    out=tmp_path/'out'
    result=prepare(source,out,source,2,1)
    assert len(result['pools']['hourly'])==169
    assert len(result['pools']['weekly'])==2
    for row,expected in zip(result['pools']['weekly'],[8.35,8.45]):
        np.testing.assert_allclose(np.load(out/row['path']/'sst.npy'),expected,rtol=1e-6)


def test_missing_hour_is_not_interpolated(tmp_path):
    source=make_source(tmp_path,gap=85)
    with pytest.raises(ValueError,match='168 consecutive'):
        prepare(source,tmp_path/'out',source,2,1)


def test_daily_source_rejected(tmp_path):
    source=make_source(tmp_path,count=1)
    path=source/'simulation.json';m=json.loads(path.read_text());m['cadence_hours']=24;path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='daily means'):
        prepare(source,tmp_path/'out',source,2,1)
