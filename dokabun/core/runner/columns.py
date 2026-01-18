"""Column classification and header parsing helpers."""

from __future__ import annotations

import re
from typing import Any, Iterable

import pandas as pd

from dokabun.core.runner.models import ColumnClassification


def classify_columns(df: pd.DataFrame) -> ColumnClassification:
    """DataFrame の列をプレフィックス体系で分類し、妥当性を検証する。

    Returns:
        ColumnClassification: 列分類の結果。
    """

    input_columns: list[str] = []
    structured_columns: list[str] = []
    ns_columns: list[str] = []
    nsof_index_map: dict[str, int] = {}
    nsof_counter = 0
    label_columns: list[str] = []
    embedding_columns: list[str] = []
    embedding_spec_map: dict[str, dict[str, Any]] = {}
    embedding_index_map: dict[str, int] = {}
    embedding_counter = 0
    errors: list[str] = []

    for col in df.columns:
        if not isinstance(col, str):
            errors.append("列名が文字列ではありません。")
            continue

        lower = col.lower()

        if lower.startswith("i_"):
            input_columns.append(col)
            continue

        if lower.startswith("l_"):
            label_columns.append(col)
            continue

        if lower.startswith("so_"):
            structured_columns.append(col)
            continue

        if lower.startswith("nsof_"):
            ns_columns.append(col)
            nsof_counter += 1
            nsof_index_map[col] = nsof_counter
            continue

        if lower.startswith("nso_"):
            ns_columns.append(col)
            continue

        if lower.startswith("eo"):
            try:
                embedding_spec_map[col] = parse_embedding_column(col)
                embedding_columns.append(col)
                embedding_counter += 1
                embedding_index_map[col] = embedding_counter
            except ValueError as exc:  # noqa: BLE001
                errors.append(str(exc))
            continue

        if lower.startswith("c_"):
            errors.append(f"config 列はサポートしていません: {col}")
            continue

        errors.append(
            f"未知の列プレフィックスです: {col}（ラベル列にする場合は l_ を付与してください）"
        )

    if errors:
        raise ValueError("; ".join(errors))

    if not input_columns:
        raise ValueError(
            "入力列 i_ が1つも存在しません。少なくとも1列の i_ を用意してください。"
        )

    return ColumnClassification(
        input_columns=input_columns,
        structured_columns=structured_columns,
        nonstructured_columns=ns_columns,
        nsof_index_map=nsof_index_map,
        label_columns=label_columns,
        embedding_columns=embedding_columns,
        embedding_spec_map=embedding_spec_map,
        embedding_index_map=embedding_index_map,
    )


def strip_prefix_case_insensitive(text: str, prefix: str) -> str:
    """大小文字を無視して prefix を取り除く。"""

    if text.lower().startswith(prefix.lower()):
        return text[len(prefix) :]
    return text


def structured_schema_header(column: str) -> str:
    """so_ を除去し、説明を保持したまま Schema 用ヘッダに変換する。"""

    if "|" in column:
        name, desc = column.split("|", 1)
        stripped_name = strip_prefix_case_insensitive(name, "so_")
        return f"{stripped_name}|{desc}"
    return strip_prefix_case_insensitive(column, "so_")


def validate_structured_property_names(columns: Iterable[str]) -> None:
    """構造化列のプロパティ名衝突を検知する。"""

    seen: set[str] = set()
    for col in columns:
        prop = column_to_property_name(col).lower()
        if prop in seen:
            raise ValueError(f"構造化出力のプロパティ名が重複しています: {prop}")
        seen.add(prop)


def parse_embedding_column(column: str) -> dict[str, Any]:
    """eo* 列名をパースし、前段/後段の次元指定とファイル出力を返す。"""

    pattern = re.compile(
        r"^eo"
        r"(?:(?P<pre_method>n)(?P<pre_dim>\d+)?)?"
        r"(?:(?P<post_method>[ptu])(?P<post_dim>\d+))?"
        r"(?P<fileflag>f)?$",
        re.IGNORECASE,
    )
    match = pattern.match(column)
    if not match:
        raise ValueError(
            f"埋め込み列の形式が不正です: {column} "
            "(例: eo / eof / eon1536 / eop128 / eon1536p128f)"
        )

    pre_method = match.group("pre_method")
    pre_dim = match.group("pre_dim")
    post_method = match.group("post_method")
    post_dim = match.group("post_dim")
    file_output = match.group("fileflag") is not None

    if pre_dim is not None and int(pre_dim) <= 0:
        raise ValueError(f"埋め込み列の次元数が不正です: {column}")
    if post_dim is not None and int(post_dim) <= 0:
        raise ValueError(f"埋め込み列の次元数が不正です: {column}")
    if post_method and not post_dim:
        raise ValueError(f"埋め込み列の後段次元指定が不正です: {column}")

    return {
        "pre_method": pre_method.lower() if pre_method else None,
        "pre_dim": int(pre_dim) if pre_dim else None,
        "post_method": post_method.lower() if post_method else None,
        "post_dim": int(post_dim) if post_dim else None,
        "file_output": file_output,
    }


def column_to_property_name(column: str) -> str:
    """列ヘッダから JSON プロパティ名を抽出する。

    Args:
        column: ``列名`` または ``列名|説明`` 形式のヘッダ文字列（so_ プレフィックスを含む）。

    Returns:
        str: JSON Schema で利用するプロパティ名。
    """

    if "|" in column:
        name, _ = column.split("|", 1)
    else:
        name = column
    name = strip_prefix_case_insensitive(name, "so_")
    return name.strip()
