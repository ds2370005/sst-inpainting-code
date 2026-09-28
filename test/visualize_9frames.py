"""1時刻1NPZから同一区画のSSTを時系列表示（標準: 6時間間隔で9枚）。

python test/visualize_9frames.py --data-root /path/to/frames256
python test/visualize_9frames.py --data-root /path/to/frames256 --target 20250403000000_r05_c05.npz

--targetは最後の時刻。省略時は必要な履歴が揃う最初の対象を選ぶ。
平均NPZは不要。単一観測は--time-steps 1。出力はoutputs/。
色範囲は有効SSTの2〜98パーセンタイル（--vmin/--vmaxで変更可能）。
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# 同じフォルダの共通処理を優先し、標準ライブラリ等のtestとの衝突を避ける。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from frame_visualization import main

# サーバー上の前処理済みNPZの親フォルダに変更可能。CLI指定を優先する。
DEFAULT_DATA_ROOT = ROOT/'data/frames256'
DEFAULT_CONFIG = ROOT/'config.yaml'

if __name__ == '__main__':
    main('sst',DEFAULT_DATA_ROOT,DEFAULT_CONFIG)
