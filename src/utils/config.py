# 役割: YAMLの実行設定を読み込み、必須項目を確認する。
"""Configuration loading utilities."""

from __future__ import annotations

# ── 使用するライブラリ・共通処理 ──────────────────
from pathlib import Path
from typing import Any

import yaml


# ── 処理に使用する関数・クラス ──────────────────
def load_config(path: str | Path) -> dict[str, Any]:  # 共通: config.yamlの中身を辞書として読む
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert isinstance(config, dict), f"Expected a YAML mapping in {path}"
    return config

