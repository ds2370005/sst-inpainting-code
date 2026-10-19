"""Paper architecture, unit conversion, augmentation and checkpoint contracts."""
import pytest
import torch
from torch import nn

from src.models import AverageEstimationGenerator, AnomalyInpaintingGenerator, Discriminator
from src.models.blocks import ReflectionConv2d
from src.train.train_anomaly import apply_random_cloud_mask
from src.utils.checkpoint import MODEL_FORMAT, load_generator_state, require_model_format


def test_tables_and_scalar_discriminator():
    average = AverageEstimationGenerator()
    encoder = average.encoder_decoder
    assert isinstance(encoder.e10, ReflectionConv2d)  # no final ELU
    decoder_convs = [m for name,module in encoder.named_children() if name.startswith('d')
                     for m in module.modules() if isinstance(m,nn.Conv2d)]
    assert [m.out_channels for m in decoder_convs] == [128,128,64,64,32,32,16,16,1]
    assert not hasattr(average,'fusion_projection')
    assert encoder.e1.block[0].conv.in_channels == 10
    d = Discriminator()
    assert [getattr(d,f'd{i}').out_channels for i in range(1,8)] == [64,128,256,256,256,512,1]
    assert d.d7.kernel_size == (4,4) and d.d7.padding == (0,0)
    assert all(hasattr(getattr(d,f'd{i}'),'weight_orig') for i in range(1,8))
    with torch.no_grad():
        assert d(torch.randn(1,1,256,256)).shape == (1,1)
    with pytest.raises(ValueError,match='256'):
        d(torch.randn(1,1,160,160))


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


def test_anomaly_input_uses_difference_units_and_masks():
    model=AnomalyInpaintingGenerator(time_steps=2,anomaly_scale=.25)
    captured={}
    model.projection.register_forward_pre_hook(lambda m,a:captured.update(x=a[0].detach().clone()))
    x=torch.full((1,1,2,160,160),.125)
    mask=torch.ones_like(x);mask[:,:,:,:,0]=0
    with torch.no_grad():model(x,mask,torch.zeros_like(x[:,:,:1]))
    assert (captured['x'][:,0,:,:,1:]==.5).all()
    assert (captured['x'][:,0,:,:,0]==0).all()
    assert torch.equal(captured['x'][:,1:2],mask)


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


def test_reject_legacy_and_mismatched_scale():
    with pytest.raises(ValueError,match='retraining'):require_model_format({})
    model=AnomalyInpaintingGenerator(anomaly_scale=.5)
    other=AnomalyInpaintingGenerator(anomaly_scale=.25)
    checkpoint=dict(model_format=MODEL_FORMAT,generator_state_dict=other.state_dict())
    with pytest.raises(ValueError,match='normalization'):
        load_generator_state(model,checkpoint)


def test_decoder_clips_instead_of_tanh():
    model=AverageEstimationGenerator(time_steps=1).eval()
    with torch.no_grad():
        model.encoder_decoder.d8.conv.weight.zero_()
        model.encoder_decoder.d8.conv.bias.fill_(.5)
        x=torch.zeros(1,1,1,160,160)
        result=model(x,torch.ones_like(x),x,torch.ones_like(x))
    torch.testing.assert_close(result,torch.full_like(result,.5))
