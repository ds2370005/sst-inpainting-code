#!/usr/bin/env python3
"""JCOPEの日別ファイル・時間欠落・読み出しを検査する。陸域欠損は破損扱いしない。"""
import argparse
from datetime import datetime,timedelta
from pathlib import Path
import json
import numpy as np
from netCDF4 import Dataset,num2date


def check(root,start,end,output,deep=False):
    root,output=Path(root),Path(output)
    start=datetime.fromisoformat(start);end=datetime.fromisoformat(end)
    if start>end:raise ValueError('期間が逆です')
    issues=[];records=[];day=start
    while day<=end:
        path=root/f'TT_{day:%Y%m%d}.nc'
        record=dict(file=path.name,status='ok')
        try:
            if not path.is_file():raise FileNotFoundError(str(path))
            record['bytes']=path.stat().st_size
            with Dataset(path) as ds:
                var=ds.variables['TT'];t=ds.variables['time']
                if var.dimensions!=('time','depth','lat','lon') or var.shape[0]!=24:
                    raise ValueError(f'想定外の次元/時刻数: {var.dimensions}, {var.shape}')
                depth=np.ma.filled(ds.variables['depth'][:],np.nan)
                zero=np.flatnonzero(np.isclose(depth,0,rtol=0,atol=1e-6))
                if len(zero)!=1:raise ValueError('深度0mが一意ではありません')
                raw=t[:]
                if np.ma.getmaskarray(raw).any():raise ValueError('時間座標が欠損')
                dates=num2date(raw,t.units,calendar=getattr(t,'calendar','standard'))
                stamps=[datetime(d.year,d.month,d.day,d.hour,d.minute,d.second,d.microsecond) for d in dates]
                if stamps!=[day+timedelta(hours=h) for h in range(24)]:
                    raise ValueError('日付・時刻の欠落/重複/順序不一致')
                if deep:
                    counts=[]
                    for h in range(24):
                        field=np.ma.filled(var[h,int(zero[0]),:,:],np.nan)
                        counts.append(int(np.isfinite(field).sum()))
                        del field
                    record['valid_pixels_per_hour']=counts
                    if min(counts)==0:raise ValueError('全画素欠損の時刻があります')
        except Exception as exc:
            record.update(status='error',error=f'{type(exc).__name__}: {exc}');issues.append(record.copy())
        records.append(record)
        print(f'[{len(records)}] {path.name}: {record["status"]}',flush=True)
        day+=timedelta(days=1)
    output.mkdir(parents=True,exist_ok=True)
    report=dict(root=str(root),start=start.isoformat(),end=end.isoformat(),deep=deep,
                expected_files=len(records),passed=sum(r['status']=='ok' for r in records),issues=issues,files=records)
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'確認数={len(records)} 問題数={len(issues)} 保存先={output/"report.json"}')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--start',required=True)
    p.add_argument('--end',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deep',action='store_true',help='全時刻の全表層画素を順に読む（時間がかかる）')
    a=p.parse_args();r=check(a.root,a.start,a.end,a.output,a.deep)
    if r['issues']:raise SystemExit(1)
if __name__=='__main__':main()
