"""Paper architecture, unit conversion, augmentation and checkpoint contracts."""
import pytest
import torch
from torch import nn

from src.models import AverageEstimationGenerator, AnomalyInpaintingGenerator, Discriminator
from src.models.blocks import ReflectionConv2d
from src.train.train_anomaly import apply_random_cloud_mask
from src.utils.checkpoint import MODEL_FORMAT, load_generator_state, require_model_format


def test_monthly_feature_concatenated_without_temporal_repeat():
    model=AverageEstimationGenerator(time_steps=2)
    captured={}
    model.encoder_decoder.register_forward_pre_hook(lambda m,a:captured.update(x=a[0].detach().clone()))
    with torch.no_grad():
        model.sst_projection.weight.zero_();model.sst_projection.bias.fill_(2)
        model.monthly_projection.weight.zero_();model.monthly_projection.bias.fill_(3)
        x=torch.zeros(1,1,2,160,160)
        model(x,torch.ones_like(x),x[:,:,:1],torch.ones_like(x[:,:,:1]))
    assert captured['x'].shape==(1,3,160,160)
    assert (captured['x'][:,:2]==2).all() and (captured['x'][:,2:]==3).all()


def test_mask_hides_input_but_preserves_supervision():
    sst=torch.ones(1,1,3,2,2)
    original_mask=torch.ones_like(sst);original_mask[:,:,: ,1,1]=0
    batch=dict(sst_volume=sst,mask_volume=original_mask,
               random_cloud_mask=torch.tensor([[[[0.,1.],[1.,1.]]]]),
               target_sst=sst[:,:,-1].clone(),target_mask=original_mask[:,:,-1].clone())
    before={k:v.clone() for k,v in batch.items()}
    masked,valid=apply_random_cloud_mask(batch)
    assert (masked[:,:,:,0,0]==0).all() and (valid[:,:,:,0,0]==0).all()
    assert (valid[:,:,:,1,1]==0).all()  # original missing stays missing
    assert batch['target_mask'][0,0,0,0]==1  # artificially hidden remains supervised
    for k in before:assert torch.equal(before[k],batch[k])
    del batch['random_cloud_mask']
    with pytest.raises(ValueError,match='requires'):apply_random_cloud_mask(batch)

