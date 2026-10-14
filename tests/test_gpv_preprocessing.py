"""Offline checks: no server data or ecCodes installation required."""
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "gpv_preprocess", Path(__file__).resolve().parents[1] / "data/preprocess_gpv.py")
gpv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpv)


def test_regrid_descending_latitude_and_outside():
    sst = np.array([[1, np.nan], [3, 4]], dtype=np.float32)
    result = gpv.regrid(sst, np.array([2, 1]), np.array([10, 11]),
                        np.array([2.1, 2, 1]), np.array([10, 11]))
    np.testing.assert_allclose(result, [[np.nan, np.nan], [1, np.nan], [3, 4]], equal_nan=True)


def test_strict_weekly_mask():
    arrays = [np.full((2, 2), i, dtype=np.float32) for i in range(7)]
    arrays[0][0, 0] = np.nan
    mean, count = gpv.weekly_mean(arrays)
    assert np.isnan(mean[0, 0]) and count[0, 0] == 6
    assert mean[1, 1] == 3 and count[1, 1] == 7
    with pytest.raises(ValueError):
        gpv.weekly_mean(arrays[:6])


def test_pipeline_gaps_and_output(tmp_path, monkeypatch):
    source = tmp_path / "input"
    source.mkdir()
    for day in [1, 2, 3, 4, 5, 6, 7, 9]:
        (source / f"Z__C_RJTD_202505{day:02d}000000_OCN_GPV_Rjp_Gll2km_grib2.tar.gz").touch()
    def fake_extract(archive, scratch, ec, stamp):
        return np.full((2, 2), stamp.day, dtype=np.float32), np.array([2., 1.]), np.array([10., 11.]), {
            "source_archive": str(archive), "period_start": stamp.isoformat(),
            "period_end": (stamp + gpv.timedelta(days=1)).isoformat()}
    monkeypatch.setattr(gpv, "extract", fake_extract)
    monkeypatch.setitem(sys.modules, "eccodes", SimpleNamespace())
    output = tmp_path / "out"
    assert gpv.main(["--input", str(source), "--output", str(output),
                     "--bounds", "1", "2", "10", "11", "--patch-size", "2"]) == 0
    assert [p.name for p in (output / "weekly").glob("*.npz")] == ["20250507.npz"]
    with np.load(output / "weekly/20250507.npz") as f:
        np.testing.assert_allclose(f["sst"], 4)
    with np.load(output / "patches/daily/20250501_r00_c00.npz") as f:
        assert f["assimilation_sst"].shape == (1, 2, 2)
        np.testing.assert_allclose(f["assimilation_sst"], 2/35-1)
    report = json.loads((output / "report.json").read_text())
    assert report["missing_archive_days"] == ["2025-05-08"]
    assert "20250509" in report["weekly_unavailable"]
    assert all(p["adversarial_eligible"] for p in report["patches"])


def test_depth_selection_bitmap_and_kelvin(tmp_path):
    metadata = dict(discipline=10, parameterCategory=4, parameterNumber=15,
                    scaledValueOfFirstFixedSurface=10, scaleFactorOfFirstFixedSurface=1,
                    typeOfFirstFixedSurface=160, dataDate=20250501, dataTime=0,
                    stepType="avg", startStep=0, endStep=24, validityDate=20250502,
                    validityTime=0, gridType="regular_ll", Nj=2, Ni=2,
                    missingValue=9999, bitmapPresent=1)
    wrong_depth = dict(metadata, scaledValueOfFirstFixedSurface=100)
    fields = iter([wrong_depth, metadata, None])
    released = []
    ec = SimpleNamespace(
        codes_grib_multi_support_on=lambda: None,
        codes_grib_new_from_file=lambda f: next(fields),
        codes_get=lambda gid, key: gid[key],
        codes_set=lambda *args: None,
        codes_release=lambda gid: released.append(gid),
        codes_get_values=lambda gid: np.array([273.15, 274.15, 9999, 275.15]),
        codes_get_array=lambda gid, key: {
            "latitudes": [2, 2, 1, 1], "longitudes": [10, 11, 10, 11],
            "bitmap": [1, 1, 0, 1]}[key],
    )
    source = tmp_path / "field.bin"
    source.touch()
    sst, lat, lon, meta = gpv.read_field(source, ec, datetime(2025, 5, 1, tzinfo=timezone.utc))
    np.testing.assert_allclose(sst, [[0, 1], [np.nan, 2]], atol=1e-5, equal_nan=True)
    assert len(released) == 2
    assert meta["depth_m"] == 1
