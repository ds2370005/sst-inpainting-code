"""1時刻1NPZから同一区画の雲マスクを時系列表示。

python test/visualize_cloud_mask.py --data-root /path/to/frames256
python test/visualize_cloud_mask.py --data-root /path/to/frames256 --target 20250403000000_r05_c05.npz

SST可視化と同じ--data-root/--target/--configで同じ観測列を表示する。
黒=1（雲・欠損海域）、白=0（晴天海域または陸地）。平均NPZは不要。
--time-steps 1で単一観測、--outputで保存先を指定可能。
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from test.frame_visualization import main

DEFAULT_DATA_ROOT = ROOT/'data/frames256'
DEFAULT_CONFIG = ROOT/'config.yaml'

if __name__ == '__main__':
    main('mask',DEFAULT_DATA_ROOT,DEFAULT_CONFIG)
