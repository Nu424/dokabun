"""Cell value helpers shared across runner modules."""

from __future__ import annotations

from typing import Any

import pandas as pd


def is_empty_value(value: Any) -> bool:
    """セルの値が「空」と見なせるかを判定する。"""

    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return bool(pd.isna(value))
