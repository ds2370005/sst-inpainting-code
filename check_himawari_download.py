#!/usr/bin/env python3
"""Read-only local hourly SST audit; no FTP access or data modification."""
import argparse
import calendar
import csv
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from download_himawari_sst import FILE_PATTERN


def audit(root, start, end, output, deep=False):
    reader = None
    if deep:
        from netCDF4 import Dataset
        reader = Dataset
    expected = set()
    current = datetime.strptime(start, '%Y%m')
    end_month = datetime.strptime(end, '%Y%m')
    if current > end_month:
        raise ValueError('start must be <= end')
    stop = (end_month.replace(day=28) + timedelta(days=4)).replace(day=1)
    while current < stop:
        expected.add(current.strftime('%Y%m%d%H%M%S'))
        current += timedelta(hours=1)
    found = defaultdict(list)
    valid = set()
    issues = []
    def issue(kind, stamp, path, detail=''):
        issues.append((kind, stamp, str(path), detail))
    scanned = 0
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if path.name.endswith('.tmp'):
            issue('temporary', '', rel)
            continue
        if path.suffix != '.nc':
            continue
        match = FILE_PATTERN.fullmatch(path.name)
        if not match:
            issue('unexpected_filename', '', rel)
            continue
        stamp = match[1]
        if stamp not in expected:
            issue('outside_expected_hours', stamp, rel)
            continue
        found[stamp].append(str(rel))
        okay = True
        if rel.parent != Path(stamp[:6]) / stamp[6:8]:
            issue('wrong_directory', stamp, rel)
            okay = False
        try:
            if path.stat().st_size == 0:
                raise ValueError('Empty file')
            with path.open('rb') as f:
                if f.read(8) != b'\x89HDF\r\n\x1a\n':
                    raise ValueError('Expected NetCDF-4/HDF5 header')
            if reader:
                with reader(path) as ds:
                    required = ('sea_surface_temperature', 'quality_level', 'l2p_flags', 'lat', 'lon')
                    for name in required:
                        if name not in ds.variables:
                            raise ValueError(f'Missing variable: {name}')
                        variable = ds.variables[name]
                        # Read each field in strips to limit RAM; force decompression.
                        if variable.ndim < 2:
                            variable[:]
                        else:
                            for row in range(0, variable.shape[-2], 128):
                                selection = (slice(None),) * (variable.ndim-2) + (slice(row,row+128),slice(None))
                                variable[selection]
        except Exception as exc:
            issue('unreadable_or_invalid', stamp, rel, f'{type(exc).__name__}: {exc}')
            okay = False
        if okay:
            valid.add(stamp)
        scanned += 1
        if scanned % 100 == 0:
            print(f'Checked {scanned} files', flush=True)
    for stamp in sorted(expected - found.keys()):
        issue('missing_hour', stamp, '')
    for stamp, paths in found.items():
        if len(paths) > 1:
            issue('duplicate_hour', stamp, ' | '.join(paths))
    output.mkdir(parents=True, exist_ok=True)
    with (output/'issues.csv').open('w', newline='') as f:
        writer=csv.writer(f);writer.writerow(('issue','timestamp_utc','path','detail'));writer.writerows(issues)
    counts = Counter(s[:6] for s in expected)
    present = Counter(s[:6] for s in found)
    passed = Counter(s[:6] for s in valid)
    with (output/'monthly_summary.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(('month','expected_hours','present_hours','passed_hours','missing_hours'))
        for month,n in sorted(counts.items()):
            row=(month,n,present[month],passed[month],n-present[month]);writer.writerow(row)
            print(f'{month}: expected={n}, present={present[month]}, passed={passed[month]}, missing={n-present[month]}')
    print(f'Mode: {"deep variable read" if deep else "inventory + header only"}')
    print(f'Expected hours: {len(expected)}; issues: {len(issues)}; categories: {dict(Counter(i[0] for i in issues))}')
    print(f'Reports: {output.resolve()}')
    print('Missing hours do not prove download failure: source availability is not checked.')
    return 1 if issues else 0


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--start',default='202501')
    parser.add_argument('--end',default='202609')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--deep',action='store_true',help='Read SST, quality, flags and coordinates fully; requires netCDF4')
    args=parser.parse_args()
    if not args.root.is_dir():parser.error('root directory does not exist')
    if args.output.exists():parser.error('Use a new report directory')
    try:
        code=audit(args.root,args.start,args.end,args.output,args.deep)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, str(exc)+'\n')
    raise SystemExit(code)
