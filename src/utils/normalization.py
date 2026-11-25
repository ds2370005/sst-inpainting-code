# 役割: 偏差の正規化値をSSTの正規化尺度へ変換する倍率を計算する。
"""学習と復元で共通使用する単位変換。"""
from __future__ import annotations

# ── 使用するライブラリ・共通処理 ──────────────────
import math
from collections.abc import Mapping
from typing import Any


# ── 処理に使用する関数・クラス ──────────────────
def anomaly_to_sst_scale(data: Mapping[str, Any]) -> float:
    """単位偏差出力を、正規化SSTの差分尺度へ変換する。

    偏差の±1は±anomaly_range℃に対応する。一方、SSTの正規化では
    max_temp-min_temp℃の幅を2に対応させる。温度差にはオフセットを加えない。"""
    minimum = float(data["min_temp"])
    maximum = float(data["max_temp"])
    anomaly_range = float(data["anomaly_range"])
    if not all(math.isfinite(x) for x in (minimum, maximum, anomaly_range)):
        raise ValueError("Temperature scales must be finite")
    if maximum <= minimum or anomaly_range <= 0:
        raise ValueError("Require max_temp > min_temp and anomaly_range > 0")
    return 2.0 * anomaly_range / (maximum - minimum)
