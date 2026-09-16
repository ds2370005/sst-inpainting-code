"""Shared unit conversions for training and reconstruction."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def anomaly_to_sst_scale(data: Mapping[str, Any]) -> float:
    """Convert a unit anomaly output to the normalized SST difference.

    Anomaly +/-1 means +/-anomaly_range degrees, whereas normalized SST
    spans two units across max_temp-min_temp degrees. No offset is applied
    to temperature differences.
    """
    minimum = float(data["min_temp"])
    maximum = float(data["max_temp"])
    anomaly_range = float(data["anomaly_range"])
    if not all(math.isfinite(x) for x in (minimum, maximum, anomaly_range)):
        raise ValueError("Temperature scales must be finite")
    if maximum <= minimum or anomaly_range <= 0:
        raise ValueError("Require max_temp > min_temp and anomaly_range > 0")
    return 2.0 * anomaly_range / (maximum - minimum)
