# 旧実装・旧テスト: old_network/old_src/datasetsをPythonパッケージとして扱うための初期化ファイル。
"""SSTアノマリー補完用のデータローダーと合成データの補助処理。"""

# ── 使用するライブラリ・共通処理 ──────────────────
from old_network.old_src.datasets.anomaly_dataset import AnomalyDataset, SyntheticAnomalyDataset
from old_network.old_src.datasets.average_dataset import AverageDataset, SyntheticAverageDataset
from old_network.old_src.datasets.loaders import ArrayFieldLoader, load_index_records

__all__ = [
    "AnomalyDataset",
    "ArrayFieldLoader",
    "AverageDataset",
    "SyntheticAnomalyDataset",
    "SyntheticAverageDataset",
    "load_index_records",
]
