"""日時関連のユーティリティ。"""

from __future__ import annotations

from datetime import datetime


def now_ts_str(dt: datetime | None = None) -> str:
    """YYYYMMDD_hhmmss 形式のタイムスタンプ文字列を返す。

    Args:
        dt: 基準となる日時。未指定時は現在時刻。

    Returns:
        str: `20251202_153045` のような文字列。
    """

    target = dt or datetime.now()
    return target.strftime("%Y%m%d_%H%M%S")

