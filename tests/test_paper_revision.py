# テスト: 生成器構造・偏差の扱い・チェックポイント互換性が期待どおりに動くかを自動検査する。
"""論文の構造・単位変換・データ拡張・チェックポイントの整合性を検査する。"""
# ── 使用するライブラリ・共通処理 ──────────────────
import pytest
import torch
from torch import nn

from src.models import AverageEstimationGenerator, AnomalyInpaintingGenerator, Discriminator
from src.models.blocks import ReflectionConv2d
from src.train.train_anomaly import apply_random_cloud_mask
from src.utils.checkpoint import MODEL_FORMAT, load_generator_state, require_model_format


# ── 動作確認用のデータ・テスト ──────────────────
def test_tables_and_scalar_discriminator():
    average = AnomalyInpaintingGenerator()
    encoder = average.encoder_decoder
    assert isinstance(encoder.e10, ReflectionConv2d)  # 最終層にはELUを置かない。
    decoder_convs = [m for name,module in encoder.named_children() if name.startswith('d')
                     for m in module.modules() if isinstance(m,nn.Conv2d)]
    assert [m.out_channels for m in decoder_convs] == [128,128,64,64,32,32,16,16,1]
    assert not hasattr(average,'fusion_projection')
    assert encoder.e1.block[0].conv.in_channels == 9
    d = Discriminator()
    assert [getattr(d,f'd{i}').out_channels for i in range(1,8)] == [64,128,256,256,256,512,1]
    assert d.d7.kernel_size == (4,4) and d.d7.padding == (0,0)
    assert all(hasattr(getattr(d,f'd{i}'),'weight_orig') for i in range(1,8))
    with torch.no_grad():
        assert d(torch.randn(1,1,256,256)).shape == (1,1)
    with pytest.raises(ValueError,match='256'):
        d(torch.randn(1,1,160,160))




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




def test_reject_legacy_and_mismatched_scale():
    with pytest.raises(ValueError,match='retraining'):require_model_format({})
    model=AnomalyInpaintingGenerator(anomaly_scale=.5)
    other=AnomalyInpaintingGenerator(anomaly_scale=.25)
    checkpoint=dict(model_format=MODEL_FORMAT,generator_state_dict=other.state_dict())
    with pytest.raises(ValueError,match='normalization'):
        load_generator_state(model,checkpoint)


def test_decoder_clips_instead_of_tanh():
    model=AnomalyInpaintingGenerator(time_steps=1).eval()
    with torch.no_grad():
        model.encoder_decoder.d8.conv.weight.zero_()
        model.encoder_decoder.d8.conv.bias.fill_(.5)
        x=torch.zeros(1,1,1,160,160)
        result=model(x,torch.ones_like(x),x)
    torch.testing.assert_close(result,torch.full_like(result,.5))
