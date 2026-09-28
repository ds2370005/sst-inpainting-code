"""千葉県の海洋シミュレーションNetCDFを緯度・経度上に可視化する。

リポジトリ直下からの実行例（pythonは必要なライブラリがある環境を使用）:
    python test/visualize_simulation.py --all-times
    python test/visualize_simulation.py /path/to/Metro3_hs-Std_A20250101.nc4
    python test/visualize_simulation.py /path/to/data.nc4 --all-times
    python test/visualize_simulation.py /path/to/data.nc4 --time-index 12
    python test/visualize_simulation.py /path/to/data.nc4 --variable so --all-times
    python test/visualize_simulation.py /path/to/data.nc4 --vmin 10 --vmax 25

必要ライブラリ: numpy, netCDF4, matplotlib（requirements.txtに記載済み）
入力を省略した場合は、コード内のDEFAULT_INPUTで指定したファイルを読み込む。
標準出力先: outputs/simulation/。GUIのないサーバーでもPNGを保存できる。
水温の標準色範囲は0〜25℃。範囲外は端の色で表示し、元データは変更しない。
塩分の標準色範囲は選択データの最小値〜最大値。

to: sea_water_potential_temperature（ポテンシャル水温、degreeC）
so: sea_water_practical_salinity（実用塩分、無次元）
Z=0 mなら表層の値。衛星が観測するskin SSTとは測定・定義が異なる。
_FillValue等はnetCDF4のマスクを利用し、0は勝手に欠損扱いにしない。
時刻はファイルのunits/calendarから復号し、JST変換や時刻の丸めは行わない。
添付サンプルのtimeはfloat32で、毎正時から約1分ずれた値を含む。
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from netCDF4 import Dataset, num2date
import numpy as np


# サーバー上の入力ファイル。別の日付を標準にする場合はここを変更する。
# コマンドラインでファイルを指定した場合は、その指定が優先される。
DEFAULT_INPUT = Path('/datasets/metro3/Metro3_hs-Std_A20250101.nc4')


VARIABLES = {
    'to': ('Potential temperature', 'deg C', 'turbo'),
    'so': ('Practical salinity', 'dimensionless', 'viridis'),
}


def read_fields(path: Path, variable: str, time_index: int, depth_index: int, all_times: bool):
    """必要な時刻・深さだけを読み、欠損マスクを保持して返す。"""
    with Dataset(path) as dataset:
        for key in ('time', 'Z', 'Latt', 'Lngt', variable):
            if key not in dataset.variables:
                raise ValueError(f'必要な変数がありません: {key}')
        field = dataset[variable]
        if field.dimensions != ('time', 'Z', 'Latt', 'Lngt'):
            raise ValueError(f'未対応の次元順です: {field.dimensions}')
        if not 0 <= depth_index < len(dataset['Z']):
            raise ValueError(f'--depth-indexは0〜{len(dataset["Z"])-1}で指定してください')
        if not all_times and not 0 <= time_index < len(dataset['time']):
            raise ValueError(f'--time-indexは0〜{len(dataset["time"])-1}で指定してください')
        indices = list(range(len(dataset['time']))) if all_times else [time_index]
        values = np.ma.stack([np.ma.masked_invalid(field[i, depth_index, :, :]) for i in indices])
        lat = np.asarray(dataset['Latt'][:], dtype=float)
        lon = np.asarray(dataset['Lngt'][:], dtype=float)
        for key, coordinates in [('Latt', lat), ('Lngt', lon)]:
            differences = np.diff(coordinates)
            if (not np.isfinite(coordinates).all() or len(coordinates) < 2
                    or not (np.all(differences > 0) or np.all(differences < 0))):
                raise ValueError(f'{key}は有限で単調な1次元座標が必要です')
        time = dataset['time']
        timestamps = num2date(time[indices], units=time.units, calendar=getattr(time, 'calendar', 'standard'))
        depth = float(dataset['Z'][depth_index])
        units = getattr(field, 'units', '')
    if values.count() == 0:
        raise ValueError('選択したデータに有効な値がありません')
    return values, lat, lon, timestamps, depth, units


def visualize(path: Path, *, variable='to', time_index=0, depth_index=0,
              all_times=False, output=None, vmin=None, vmax=None, dpi=150):
    values, lat, lon, timestamps, depth, units = read_fields(
        path, variable, time_index, depth_index, all_times,
    )
    title, label_unit, cmap_name = VARIABLES[variable]
    print(f'入力: {path.resolve()}')
    print(f'変数: {variable} ({title}), 単位: {units}, 深さ: {depth:g} m')
    print(f'形状 [時刻, 緯度, 経度]: {values.shape}')
    print(f'緯度: {lat.min():.5f}〜{lat.max():.5f}, 経度: {lon.min():.5f}〜{lon.max():.5f}')
    print(f'時刻（ファイル値）: {timestamps[0]} 〜 {timestamps[-1]}')
    print(f'有効値率: {values.count()/values.size:.2%}, 値域: {values.min():.4f}〜{values.max():.4f}')
    default_lower = 0.0 if variable == 'to' else float(values.min())
    default_upper = 25.0 if variable == 'to' else float(values.max())
    lower = default_lower if vmin is None else vmin
    upper = default_upper if vmax is None else vmax
    if lower == upper and vmin is None and vmax is None:
        lower, upper = lower-.5, upper+.5
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError('色の範囲は有限値でvmin < vmaxにしてください')
    print(f'色の範囲: {lower:g}〜{upper:g} ({units})')
    if dpi <= 0:
        raise ValueError('dpiは正の値で指定してください')
    count = len(timestamps)
    columns = min(4, count)
    rows = math.ceil(count/columns)
    figsize = (9, 8) if count == 1 else (4*columns+1, 3.7*rows+1)
    fig, axes = plt.subplots(rows, columns, figsize=figsize, squeeze=False, constrained_layout=True)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad('lightgray')
    try:
        active_axes = []
        for i, ax in enumerate(axes.flat):
            if i >= count:
                ax.set_visible(False)
                continue
            active_axes.append(ax)
            im = ax.pcolormesh(lon, lat, values[i], shading='nearest', cmap=cmap, vmin=lower, vmax=upper, rasterized=True)
            # 緯度経度の縦横比を中央緯度で近似補正（地図投影ではない）。
            ax.set_aspect(1 / np.cos(np.deg2rad(lat.mean())))
            ax.set_title(timestamps[i].strftime('%Y-%m-%d %H:%M:%S'), fontsize=11)
            ax.set_xlabel('Longitude (deg E)')
            ax.set_ylabel('Latitude (deg N)')
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.ticklabel_format(useOffset=False, style='plain')
            ax.grid(alpha=.2, linewidth=.5)
        below, above = values.min() < lower, values.max() > upper
        extend = 'both' if below and above else 'min' if below else 'max' if above else 'neither'
        fig.colorbar(im, ax=active_axes, label=f'{title} ({label_unit})', shrink=.85, extend=extend)
        fig.suptitle(f'{path.name}\n{title} | depth = {depth:g} m\nGray = missing / masked; time as stored in file', fontsize=13)
        if output is None:
            suffix = 'all_times' if all_times else f't{time_index:02d}'
            output = Path('outputs/simulation') / f'{path.stem}_{variable}_z{depth_index:02d}_{suffix}.png'
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi)
    finally:
        plt.close(fig)
    print(f'保存先: {output.resolve()}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', type=Path, nargs='?', default=DEFAULT_INPUT,
                        help=f'海洋シミュレーションの.nc4ファイル（省略時: {DEFAULT_INPUT}）')
    parser.add_argument('--variable', choices=VARIABLES, default='to', help='to: 水温（既定）, so: 塩分')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--time-index', type=int, default=0, help='時刻の添字（0始まり、既定0）')
    group.add_argument('--all-times', action='store_true', help='全時刻を共通の色範囲で一覧表示')
    parser.add_argument('--depth-index', type=int, default=0, help='深さの添字（既定0）')
    parser.add_argument('--output', type=Path, help='保存先（省略時outputs/simulation/）')
    parser.add_argument('--vmin', type=float, help='色範囲の下限（既定: 水温0℃、塩分は最小値）')
    parser.add_argument('--vmax', type=float, help='色範囲の上限（既定: 水温25℃、塩分は最大値）')
    parser.add_argument('--dpi', type=int, default=150)
    args = parser.parse_args()
    try:
        visualize(args.input, variable=args.variable, time_index=args.time_index,
                  depth_index=args.depth_index, all_times=args.all_times, output=args.output,
                  vmin=args.vmin, vmax=args.vmax, dpi=args.dpi)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'エラー: {exc}\n')


if __name__ == '__main__':
    main()
