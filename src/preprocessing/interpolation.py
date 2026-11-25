"""緯度経度格子の共通バイリニア補間。"""
import numpy as np

def bilinear(sst, lat, lon, target_lat, target_lon):
    """Coordinate-based bilinear interpolation; no extrapolation or coast filling."""
    for axis in (lat, lon, target_lat, target_lon):
        delta = np.diff(axis)
        if len(axis) < 2 or not np.isfinite(axis).all() or not ((delta>0).all() or (delta<0).all()):
            raise ValueError('Expected finite monotonic 1D coordinates')
    yi,xi = np.argsort(lat),np.argsort(lon)
    lat,lon,sst = lat[yi],lon[xi],sst[np.ix_(yi,xi)]
    y = np.clip(np.searchsorted(lat,target_lat),1,len(lat)-1)
    x = np.clip(np.searchsorted(lon,target_lon),1,len(lon)-1)
    wy = (target_lat-lat[y-1])/(lat[y]-lat[y-1])
    wx = (target_lon-lon[x-1])/(lon[x]-lon[x-1])
    valid = ((target_lat>=lat[0]) & (target_lat<=lat[-1]))[:,None] & ((target_lon>=lon[0]) & (target_lon<=lon[-1]))[None,:]
    result = np.zeros((len(target_lat),len(target_lon)),np.float64)
    for ry,cy in ((y-1,1-wy),(y,wy)):
        for rx,cx in ((x-1,1-wx),(x,wx)):
            weights = cy[:,None]*cx[None,:]
            field = sst[np.ix_(ry,rx)]
            needed = weights > 1e-12
            valid &= ~needed | np.isfinite(field)
            result += np.where(np.isfinite(field),field,0)*weights
    return np.where(valid,result,np.nan).astype(np.float32)


