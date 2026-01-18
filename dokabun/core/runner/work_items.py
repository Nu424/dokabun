"""Work item collection helpers."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from dokabun.core.runner.cells import is_empty_value
from dokabun.core.runner.models import RowWorkItem


def collect_work_items(
    df: pd.DataFrame,
    *,
    structured_columns: Sequence[str],
    ns_columns: Sequence[str],
    embedding_columns: Sequence[str],
    start_row_index: int,
    max_rows: int | None,
) -> list[RowWorkItem]:
    """未入力セルが残る行の一覧を組み立てる。

    Args:
        df: スプレッドシート全体の DataFrame。
        structured_columns: 構造化出力列名のシーケンス。
        ns_columns: 非構造化出力列名のシーケンス。
        embedding_columns: 埋め込み列名のシーケンス。
        start_row_index: 探索を開始する行インデックス（再開用）。
        max_rows: この実行で処理する最大行数。制限なしは ``None``。

    Returns:
        list[RowWorkItem]: まだ処理が必要な行と未入力列のペア。
    """

    work_items: list[RowWorkItem] = []
    for row_index in range(start_row_index, len(df)):
        row = df.iloc[row_index]
        pending_structured = get_pending_columns(row, structured_columns)
        pending_ns = get_pending_columns(row, ns_columns)
        pending_embedding = get_pending_columns(row, embedding_columns)
        if pending_structured or pending_ns or pending_embedding:
            work_items.append(
                RowWorkItem(
                    row_index=row_index,
                    pending_structured_columns=pending_structured,
                    pending_ns_columns=pending_ns,
                    pending_embedding_columns=pending_embedding,
                )
            )
            if max_rows is not None and len(work_items) >= max_rows:
                break
    return work_items


def get_pending_columns(row: pd.Series, output_columns: Sequence[str]) -> list[str]:
    """行の中で空欄になっている出力列のみを抽出する。"""

    pending: list[str] = []
    for column in output_columns:
        value = row.get(column)
        if is_empty_value(value):
            pending.append(column)
    return pending
