from pathlib import Path
import numpy as np
import pytest
from netCDF4 import Dataset
from src.preprocessing.JCOPE_preprocessing import align


def fixture_files(root):
    raw=root/'raw';raw.mkdir();grid=root/'grid';grid.mkdir()
    np.save(grid/'lat.npy',[2.,1.5,1.,0.])
    np.save(grid/'lon.npy',[10.,11.,12.])
    with Dataset(raw/'TT_20240801.nc','w') as d:
        for name,size in [('time',2),('depth',1),('lat',2),('lon',2)]:d.createDimension(name,size)
        for name,values in [('lat',[1.,2.]),('lon',[10.,12.]),('depth',[0.])]:
            v=d.createVariable(name,'f4',(name,));v[:]=values
            if name=='depth':v.units='meters'
        t=d.createVariable('time','f4',('time',));t.units='hours since 2024-08-01';t[:]=[0,1]
        v=d.createVariable('TT','f4',('time','depth','lat','lon'),fill_value=99999.)
        v.units='degC';v[:]=np.array([[[[10,12],[20,22]]],[[[11,13],[21,99999]]]])
    return raw,grid


def test_alignment_mask_orientation_and_manifest(tmp_path):
    raw,grid=fixture_files(tmp_path);out=tmp_path/'out'
    result=align(raw,grid,out,'2024-08-01','2024-08-01')
    assert len(result['frames'])==2 and not result['gaps']
    a=np.load(out/result['frames'][0]['path']/'sst.npy')
    np.testing.assert_allclose(a[:3],[[20,21,22],[15,16,17],[10,11,12]])
    assert np.isnan(a[3]).all()
    b=np.load(out/result['frames'][1]['path']/'sst.npy')
    assert np.isnan(b[1,1]) and b[0,0]==21 and b[2,2]==13
    np.testing.assert_array_equal(np.load(out/'lat.npy'),np.load(grid/'lat.npy'))
    with pytest.raises(ValueError,match='空'):align(raw,grid,out)


def test_wrong_depth_rejected(tmp_path):
    raw,grid=fixture_files(tmp_path)
    with Dataset(raw/'TT_20240801.nc','a') as d:d.variables['depth'][:]=1
    with pytest.raises(ValueError,match='深度0m'):align(raw,grid,tmp_path/'out')


def test_duplicate_time_rejected(tmp_path):
    raw,grid=fixture_files(tmp_path)
    with Dataset(raw/'TT_20240801.nc','a') as d:d.variables['time'][:]=[0,0]
    with pytest.raises(ValueError,match='重複'):align(raw,grid,tmp_path/'out')
