"""Small NetCDF + mocked GRIB extraction tests; never contact external servers."""
from datetime import datetime,timedelta
from types import SimpleNamespace
import json
import sys
import numpy as np
import pytest
import torch
import yaml
from netCDF4 import Dataset
from src.preprocessing import GPV_preprocessing as gpv
from src.preprocessing.Himawari_preprocessing import decode_masks
from src.preprocessing.common import window_sum
from src.preprocessing.pipeline import run
from src.datasets.unified_dataset import UnifiedSSTDataset,create_unified_loader


def test_bilinear_coordinates_and_coast():
    values=np.array([[20.,22.],[10.,12.]])
    lat=np.array([2.,1.]);lon=np.array([10.,12.])
    actual=gpv.bilinear(values,lat,lon,np.array([2.,1.5,1.,0.]),np.array([10.,11.,12.]))
    np.testing.assert_allclose(actual[:3],[[20,21,22],[15,16,17],[10,11,12]])
    assert np.isnan(actual[3]).all()
    values[0,1]=np.nan
    actual=gpv.bilinear(values,lat,lon,np.array([2.,1.]),np.array([10.,11.,12.]))
    assert actual[0,0]==20 and np.isnan(actual[0,1])
    np.testing.assert_allclose(actual[1],[10,11,12])


def test_mask_land_unknown_and_quality():
    sst=np.ma.array([[293.15]*4],mask=[[0,0,0,0]])
    quality=np.ma.array([[5,5,1,5]])
    flags=np.ma.array([[0,2,0,0]],mask=[[0,0,0,1]])
    values,cloud,land,valid=decode_masks(sst,quality,flags,4)
    assert valid.tolist()==[[True,False,False,False]]
    assert cloud.tolist()==[[False,False,True,False]]
    assert land.tolist()==[[False,True,False,False]]
    assert np.isnan(values[0,1:]).all()


def test_window_sum():
    mask=np.arange(25).reshape(5,5)%2
    counts=window_sum(mask,2)
    for y in range(4):
        for x in range(4):assert counts[y,x]==mask[y:y+2,x:x+2].sum()


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    raw=tmp_path/'sat';raw.mkdir()
    gp=tmp_path/'gpv';gp.mkdir()
    start=datetime(2025,4,1)
    target=start+timedelta(days=29)
    times={start+timedelta(days=i) for i in range(30)}
    times.update(target-timedelta(hours=6*i) for i in range(9))
    for stamp in sorted(times):
        path=raw/f'{stamp:%Y%m%d%H%M%S}-test.nc'
        with Dataset(path,'w') as ds:
            for dim,size in [('time',1),('lat',6),('lon',6)]:ds.createDimension(dim,size)
            ds.createVariable('lat','f8',('lat',))[:]=np.arange(6.)+20
            ds.createVariable('lon','f8',('lon',))[:]=np.arange(6.)+120
            v=ds.createVariable('sea_surface_temperature','f4',('time','lat','lon'))
            v.units='kelvin';v[:]=273.15+stamp.day+stamp.hour/24
            ds.createVariable('quality_level','i1',('time','lat','lon'))[:]=5
            ds.createVariable('l2p_flags','i2',('time','lat','lon'))[:]=0
    for day in range(1,9):
        (gp/f'Z__C_RJTD_202504{day:02d}000000_OCN_GPV_Rjp_Gll2km_grib2.tar.gz').touch()
    def fake_extract(path,scratch,ec,stamp):
        return np.full((6,6),stamp.day,np.float32),np.arange(6.)+20,np.arange(6.)+120,dict(
            source_archive=str(path),period_start=stamp.isoformat(),period_end=(stamp+timedelta(days=1)).isoformat())
    monkeypatch.setattr(gpv,'extract',fake_extract)
    monkeypatch.setitem(sys.modules,'eccodes',SimpleNamespace())
    config=dict(seed=42,data=dict(patch_size=2,time_steps=9,time_interval_hours=6,min_temp=0.,max_temp=35.,satellite='himawari',max_cloud_fraction=.5),training=dict(batch_size=1,num_workers=0,device='cpu'))
    cfg=tmp_path/'config.yaml';cfg.write_text(yaml.safe_dump(config))
    args=SimpleNamespace(config=cfg,himawari_input=raw,gpv_input=gp,output=tmp_path/'out',bounds=[20,26,120,126],min_quality=4,max_frames=None,max_archives=None,patches_per_frame=8)
    manifest=run(args)
    return args,config,manifest


def test_pipeline_and_both_training_stages(prepared):
    args,config,manifest=prepared
    assert len(manifest['samples'])==8
    assert manifest['gpv_pools']['daily'] and manifest['gpv_pools']['weekly']
    assert any(y%2 or x%2 for y,x in (r['origin'] for r in manifest['samples']))
    for stage,key in [('average','assimilation_weekly'),('anomaly','assimilation_sst')]:
        ds=UnifiedSSTDataset(args.output,config,stage=stage)
        sample=ds[0]
        assert sample['sst_volume'].shape==(1,9,2,2)
        assert sample[key].shape==(1,2,2)
        assert sample['monthly_average'].shape==(1,1,2,2)
        assert sample['weekly_mask'].sum()==4
        # Target Apr 30; same-hour 7 days are Apr 24..30 => mean 27 C.
        torch.testing.assert_close(sample['weekly_average'],torch.full((1,2,2),2*27/35-1),atol=1e-5,rtol=0)
        torch.testing.assert_close(sample['sst_volume'][:,-1],sample['target_sst'])
        assert all(torch.isfinite(v).all() for v in sample.values())
        loader=create_unified_loader(args.output,config,stage=stage,shuffle=False)
        assert next(iter(loader))[key].shape==(1,1,2,2)
    with pytest.raises(ValueError,match='empty output'):run(args)


def test_extension_hook_and_config_guard(prepared):
    args,config,_=prepared
    class AddFeature:
        def __call__(self,sample,context):
            assert len(context['lat'])==2
            return dict(sample,color_structure=torch.zeros(3,2,2))
    ds=UnifiedSSTDataset(args.output,config,stage='average',transforms=[AddFeature()])
    assert ds[0]['color_structure'].shape==(3,2,2)
    config['data']['min_temp']=-2
    with pytest.raises(ValueError,match='min_temp'):
        UnifiedSSTDataset(args.output,config,stage='average')


def test_gpv_gap_prevents_weekly_and_records_missing(prepared):
    args,config,_=prepared
    (args.gpv_input/'Z__C_RJTD_20250404000000_OCN_GPV_Rjp_Gll2km_grib2.tar.gz').unlink()
    args.output=args.output.parent/'gap_out'
    manifest=run(args)
    assert manifest['gpv']['weekly']==[]
    assert manifest['gpv']['missing_days']==['2025-04-04']
    assert not json.loads((args.output/'run.json').read_text())['training_ready']
    with pytest.raises(ValueError,match='No eligible'):
        UnifiedSSTDataset(args.output,config,stage='average')


def test_failed_pipeline_is_not_loadable(prepared,monkeypatch):
    args,config,_=prepared
    args.output=args.output.parent/'failed_out'
    def fail(*a,**kw):raise ValueError('Broken GRIB')
    monkeypatch.setattr(gpv,'extract',fail)
    with pytest.raises(ValueError,match='Broken GRIB'):run(args)
    assert json.loads((args.output/'run.json').read_text())['status']=='failed'
    assert not (args.output/'dataset.json').exists()
    with pytest.raises(ValueError,match='not completed'):
        UnifiedSSTDataset(args.output,config,stage='average')


def test_grib_depth_search_and_field_release(tmp_path):
    from datetime import timezone
    meta=dict(discipline=10,parameterCategory=4,parameterNumber=15,
              scaledValueOfFirstFixedSurface=10,scaleFactorOfFirstFixedSurface=1,
              typeOfFirstFixedSurface=160,dataDate=20250401,dataTime=0,stepType='avg',
              startStep=0,endStep=24,validityDate=20250402,validityTime=0,
              gridType='regular_ll',Nj=2,Ni=2,missingValue=9999,bitmapPresent=1)
    fields=iter([dict(meta,scaledValueOfFirstFixedSurface=100),meta,
                 dict(meta,scaledValueOfFirstFixedSurface=200),None])
    released=[]
    ec=SimpleNamespace(codes_grib_multi_support_on=lambda:None,
        codes_grib_new_from_file=lambda f:next(fields),codes_get=lambda g,k:g[k],
        codes_set=lambda *a:None,codes_release=lambda g:released.append(g),
        codes_get_values=lambda g:np.array([273.15,274.15,9999,275.15]),
        codes_get_array=lambda g,k:dict(latitudes=[2,2,1,1],longitudes=[10,11,10,11],bitmap=[1,1,0,1])[k])
    path=tmp_path/'field.bin';path.touch()
    values,_,_,metadata=gpv.read_field(path,ec,datetime(2025,4,1,tzinfo=timezone.utc))
    np.testing.assert_allclose(values,[[0,1],[np.nan,2]],equal_nan=True,atol=1e-5)
    assert metadata['depth_m']==1 and len(released)==3
