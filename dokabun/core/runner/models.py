"""Data models for runner orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RowResult:
    """1 行分の処理結果を保持するデータクラス。

    Attributes:
        row_index: DataFrame 上の行インデックス。
        updates: シートへ書き戻す列名と値のマッピング。
        usage: LLM 応答から得た usage 情報（トークン数やコストなど）。
        error: エラー発生時のメッセージ。成功時は ``None``。
        error_type: エラー種別（例外名など）。成功時は ``None``。
        embedding_vectors: 後段処理が必要な埋め込みベクトル。
    """

    row_index: int
    updates: dict[str, Any]
    usage: dict[str, Any] | None = None
    error: str | None = None
    error_type: str | None = None
    generation_ids: list[str] = field(default_factory=list)
    embedding_vectors: dict[str, list[float]] = field(default_factory=dict)


@dataclass(slots=True)
class RowWorkItem:
    """処理待ち行に関するメタデータ。

    Attributes:
        row_index: DataFrame 上の行インデックス。
        pending_structured_columns: まだ値が入っていない構造化出力列。
        pending_ns_columns: まだ値が入っていない非構造化出力列（nso_/nsof_）。
        pending_embedding_columns: まだ値が入っていない埋め込み列（eo*）。
    """

    row_index: int
    pending_structured_columns: list[str]
    pending_ns_columns: list[str]
    pending_embedding_columns: list[str]


@dataclass(slots=True)
class ColumnClassification:
    """列プレフィックス分類結果を保持するデータクラス。

    Attributes:
        input_columns: `i_` で始まる入力列。
        structured_columns: `so_` で始まる構造化出力列。
        nonstructured_columns: `nso_` で始まる非構造化出力列。
        nsof_index_map: `nsof_` 列に対する 1 始まりのインデックス。
        label_columns: `l_` で始まるラベル列。
        embedding_columns: `eo` で始まる埋め込み列。
        embedding_spec_map: 埋め込み列の仕様。
        e.g.
            {
                "eo_1536": {
                    "pre_method": "n",
                    "pre_dim": 1536,
                    "post_method": "n",
                    "post_dim": 1536,
                    "file_output": True,
                }
            }
        embedding_index_map: 埋め込み列に対する 1 始まりのインデックス。
    """

    input_columns: list[str]
    structured_columns: list[str]
    nonstructured_columns: list[str]
    nsof_index_map: dict[str, int]
    label_columns: list[str]
    embedding_columns: list[str]
    embedding_spec_map: dict[str, dict[str, Any]]
    embedding_index_map: dict[str, int]


@dataclass(slots=True)
class RunState:
    """再開カーソルと完了済み行を管理する。"""

    resume_cursor: int
    completed_rows: set[int] = field(default_factory=set)

    def mark_completed(self, row_index: int) -> None:
        """行の完了を記録し、連続完了カーソルを進める。"""

        self.completed_rows.add(row_index)
        while self.resume_cursor + 1 in self.completed_rows:
            self.resume_cursor += 1
            self.completed_rows.remove(self.resume_cursor)
