"""JCOPE-Tの毎時表層水温をひまわり座標へバイリニア補間する。"""
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import re
import numpy as np
from netCDF4 import Dataset, num2date
from .common import save_frame, write_json
from .GPV_preprocessing import bilinear


def align(input_root, satellite_root, output, start=None, end=None, max_frames=None):
    """日付の両端を含む。原本は変更せず、水温は1時間分ずつ読み込む。"""
    input_root, satellite_root, output = map(Path, (input_root, satellite_root, output))
    start = datetime.fromisoformat(start) if start else None
    end = datetime.fromisoformat(end) + timedelta(days=1) if end else None
    if start and end and start >= end:
        raise ValueError('開始日と終了日の順序が不正です')
    if max_frames is not None and max_frames < 1:
        raise ValueError('max_framesは正の整数にしてください')
    if output.exists() and any(output.iterdir()):
        raise ValueError('出力先には空のディレクトリを指定してください')
    target = [np.load(satellite_root/f'{key}.npy', allow_pickle=False) for key in ('lat','lon')]
    for axis in target:
        if axis.ndim != 1 or len(axis)<2 or not np.isfinite(axis).all() or not (
                (np.diff(axis)>0).all() or (np.diff(axis)<0).all()):
            raise ValueError('ひまわり座標は有限・単調な1次元配列が必要です')
    files=sorted(input_root.rglob('TT_*.nc'))
    if not files: raise ValueError('TT_*.ncが見つかりません')
    # 日別ファイルの名前で候補を絞り、選択したファイルの実時刻も検証する。
    selected=[]
    seen=set()
    for path in files:
        match=re.fullmatch(r'TT_(\d{8})\.nc', path.name)
        if match is None:
            raise ValueError(f'未対応のファイル名（TT_YYYYMMDD.ncが必要）: {path}')
        file_day=datetime.strptime(match[1], '%Y%m%d')
        if (start and file_day+timedelta(days=1)<=start) or (end and file_day>=end):
            continue
        try:
            dataset=Dataset(path)
        except OSError as exc:
            raise OSError(f'対象期間のNetCDFを開けません: {path}。ファイルの完全性・アクセス環境を確認してください。元のエラー: {exc}') from exc
        with dataset as ds:
            time=ds.variables['time']
            if getattr(time,'calendar','standard') not in ('standard','gregorian','proleptic_gregorian'):
                raise ValueError(f'未対応のカレンダー: {path}')
            raw=time[:]
            if np.ma.getmaskarray(raw).any(): raise ValueError(f'時間座標の欠損: {path}')
            dates=num2date(raw, time.units, calendar=getattr(time,'calendar','standard'))
            for index,d in enumerate(dates):
                stamp=datetime(d.year,d.month,d.day,d.hour,d.minute,d.second,d.microsecond)
                if stamp.date() != file_day.date():
                    raise ValueError(f'ファイル名の日付と内部時刻が不一致: {path}: {stamp}')
                if stamp.minute or stamp.second or stamp.microsecond:
                    raise ValueError(f'正時ではない時刻: {path}: {stamp}')
                if (start and stamp<start) or (end and stamp>=end): continue
                if stamp in seen: raise ValueError(f'時刻の重複: {stamp}')
                seen.add(stamp); selected.append((stamp,path,index))
    selected.sort(key=lambda row: row[0])
    if max_frames: selected=selected[:max_frames]
    if not selected: raise ValueError('指定期間のフレームがありません')
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'run.json',dict(status='running'))
    for key,axis in zip(('lat','lon'),target): np.save(output/f'{key}.npy',axis)
    frames=[]; gaps=[]; current=None; ds=None
    try:
        for n,(stamp,path,index) in enumerate(selected,1):
            if path != current:
                if ds is not None: ds.close()
                ds=Dataset(path);current=path
                var=ds.variables['TT']
                if var.dimensions != ('time','depth','lat','lon'):
                    raise ValueError(f'TTの次元順序が異なります: {path}')
                if str(getattr(var,'units','')).lower() not in ('degc','degree_celsius','degrees_celsius'):
                    raise ValueError(f'TTの単位が℃ではありません: {path}')
                if 'time:' in str(getattr(var,'cell_methods','')).lower():
                    raise ValueError('時間集計済みデータのcell_methodsを確認してください')
                depth=ds.variables['depth']
                if getattr(depth,'units','').lower() not in ('m','meter','meters','metre','metres'):
                    raise ValueError('深度の単位がmではありません')
                depths=np.ma.filled(depth[:],np.nan)
                surface=np.flatnonzero(np.isclose(depths,0,atol=1e-6,rtol=0))
                if len(surface)!=1: raise ValueError('深度0mが一意に見つかりません')
                z=int(surface[0])
                lat=np.ma.filled(ds.variables['lat'][:],np.nan)
                lon=np.ma.filled(ds.variables['lon'][:],np.nan)
            # netCDF4が_FillValue・missing_valueをマスクする。
            field=np.ma.filled(var[index,z,:,:],np.nan).astype(np.float32)
            result=bilinear(field,lat,lon,*target)
            if not np.isfinite(result).any(): raise ValueError(f'補間後の有効画素がありません: {stamp}')
            relative=f'frames/{stamp:%Y%m%dT%H%M%S}'
            save_frame(output,relative,{'sst':result},dict(
                time=stamp.isoformat()+'Z',units='degC',depth_m=0,
                source_file=str(path.resolve()),source_time_index=index,
                product=getattr(ds,'title',''),version=getattr(ds,'references',''),
                interpolation='coordinate_bilinear_no_extrapolation',
                valid_pixels=int(np.isfinite(result).sum())))
            if frames:
                previous=datetime.fromisoformat(frames[-1]['time'].replace('Z',''))
                if stamp-previous != timedelta(hours=1):
                    gaps.append(dict(after=frames[-1]['time'],before=stamp.isoformat()+'Z',
                                     missing_hours=int((stamp-previous).total_seconds()/3600)-1))
            frames.append(dict(time=stamp.isoformat()+'Z',path=relative))
            print(f'[{n}/{len(selected)}] {stamp.isoformat()} valid={np.isfinite(result).sum()}/{result.size}',flush=True)
    finally:
        if ds is not None: ds.close()
    manifest=dict(units='degC',cadence_hours=1,time_basis='UTC',frames=frames,
                  gaps=gaps,interpolation='bilinear',source=str(input_root.resolve()),
                  target_grid=str(satellite_root.resolve()),limited_run=max_frames is not None)
    write_json(output/'simulation.json',manifest)
    write_json(output/'run.json',dict(status='complete',frames=len(frames),gaps=len(gaps)))
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--satellite-root',type=Path,required=True,help='lat.npy・lon.npyのあるひまわり前処理出力')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',help='開始日 YYYY-MM-DD（含む）')
    p.add_argument('--end',help='終了日 YYYY-MM-DD（含む）')
    p.add_argument('--max-frames',type=int,help='動作確認用の出力時刻数上限')
    a=p.parse_args()
    align(a.input,a.satellite_root,a.output,a.start,a.end,a.max_frames)

if __name__=='__main__': main()
