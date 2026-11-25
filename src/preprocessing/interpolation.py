# 役割: 緯度経度格子間で水温をバイリニア補間する共通関数を提供する。
"""緯度経度格子の共通バイリニア補間。"""
# ── 使用するライブラリ・共通処理 ──────────────────
import numpy as np

# ── 処理に使用する関数・クラス ──────────────────
def bilinear(sst, lat, lon, target_lat, target_lon):
    """元格子の周囲4画素を使って、対象座標の水温を求める。

    Args:
        sst: 元格子の水温配列[緯度, 経度]。欠損はNaN。
        lat, lon: 元格子の1次元緯度・経度配列。
        target_lat, target_lon: 揃えたい格子の1次元緯度・経度配列。
    Returns:
        np.ndarray: 対象格子のfloat32水温。範囲外や補間不能な画素はNaN。
    """
    # 緯度・経度が有限値で単調に並んでいるかを確認する。
    for axis in (lat, lon, target_lat, target_lon):
        delta = np.diff(axis)
        if len(axis) < 2 or not np.isfinite(axis).all() or not ((delta>0).all() or (delta<0).all()):
            raise ValueError('Expected finite monotonic 1D coordinates')
    # 元格子を昇順に揃え、水温配列も同じ順序に並べ替える。
    yi,xi = np.argsort(lat),np.argsort(lon)
    lat,lon,sst = lat[yi],lon[xi],sst[np.ix_(yi,xi)]
    # 対象座標を囲む元格子の上下・左右の位置を求める。
    y = np.clip(np.searchsorted(lat,target_lat),1,len(lat)-1)
    x = np.clip(np.searchsorted(lon,target_lon),1,len(lon)-1)
    # 座標間の距離比から、周囲4画素の重みを計算する。
    wy = (target_lat-lat[y-1])/(lat[y]-lat[y-1])
    wx = (target_lon-lon[x-1])/(lon[x]-lon[x-1])
    # 元データの範囲外は外挿せず、欠損として残す。
    valid = ((target_lat>=lat[0]) & (target_lat<=lat[-1]))[:,None] & ((target_lon>=lon[0]) & (target_lon<=lon[-1]))[None,:]
    result = np.zeros((len(target_lat),len(target_lon)),np.float64)
    # 重みがある参照画素に欠損があれば、その補間結果も欠損とする。
    for ry,cy in ((y-1,1-wy),(y,wy)):
        for rx,cx in ((x-1,1-wx),(x,wx)):
            weights = cy[:,None]*cx[None,:]
            field = sst[np.ix_(ry,rx)]
            needed = weights > 1e-12
            valid &= ~needed | np.isfinite(field)
            result += np.where(np.isfinite(field),field,0)*weights
    return np.where(valid,result,np.nan).astype(np.float32)


