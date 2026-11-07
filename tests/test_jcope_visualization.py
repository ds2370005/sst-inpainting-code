import json
import numpy as np
import pytest
from visualize.visualize_jcope_alignment import visualize,edges


def test_edges_preserve_descending_coordinates():
    np.testing.assert_allclose(edges([2.,1.,0.]),[2.5,1.5,.5,-.5])
    with pytest.raises(ValueError):edges([1.,1.])


def test_images_and_masked_statistics(tmp_path):
    np.save(tmp_path/'lat.npy',[2.,1.]);np.save(tmp_path/'lon.npy',[10.,11.])
    (tmp_path/'run.json').write_text(json.dumps({'status':'complete'}))
    frame=tmp_path/'frames'/'one';frame.mkdir(parents=True)
    np.save(frame/'sst.npy',[[10.,20.],[30.,np.nan]])
    (tmp_path/'simulation.json').write_text(json.dumps(dict(units='degC',frames=[dict(time='2024-08-01T00:00:00Z',path='frames/one')])))
    report=visualize(tmp_path,dpi=50)
    row=report['frames'][0]
    assert row['valid_pixels']==3 and row['missing_pixels']==1
    assert row['mean_c']==20 and row['valid_fraction']==.75
    assert (tmp_path/'visualizations'/row['image']).stat().st_size>0
    assert (tmp_path/'visualizations'/'statistics.json').exists()
