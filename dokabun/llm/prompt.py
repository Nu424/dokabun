"""LLM へ渡すプロンプトを構築する。"""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from dokabun.target import Target

SYSTEM_MESSAGE = (
    "あなたは入力データから情報を抽出し、指定された JSON Schema に完全準拠した "
    "JSON を返すアシスタントです。必ず schema の required を満たし、"
    "追加のプロパティは出力しないでください。"
)
NONSTRUCTURED_SYSTEM_MESSAGE = "指示に従い、与えられた入力データを処理してください。"


def build_prompt(
    row_index: int,
    row: pd.Series,
    targets: Sequence[Target],
    schema: dict[str, Any],
    instructions: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """1 行分の入力から LLM 用メッセージと response_format を生成する。

    Args:
        row_index: DataFrame 上での行インデックス。
        row: 対象行の Series。ターゲット列・出力列の両方を含む。
        targets: 前処理済みの分析対象 (`TextTarget` / `ImageTarget` など) の並び。
        schema: `llm.schema.build_schema_from_headers` で生成した JSON Schema。
        instructions: 追加でモデルに伝えたい指示文。未指定時は既定文を利用。

    Returns:
        tuple[list[dict[str, Any]], dict[str, Any]]:
            OpenAI SDK に渡す `messages` と `response_format` のペア。

    Raises:
        ValueError: `targets` が空の場合。
    """

    if not targets:
        raise ValueError("targets が空です。少なくとも 1 つの分析対象が必要です。")

    summary = _build_row_summary(row_index, row)
    instruction_text = instructions or "空欄の列をすべて埋める JSON を生成してください。"
    user_text = (
        "以下のスプレッドシート行に基づいて情報を抽出し、"
        "指示された JSON Schema に従って結果を返してください。\n"
        f"{instruction_text}\n\n"
        f"{summary}\n\n"
        "この後に解析対象のテキストや画像コンテンツが続きます。"
    )

    user_content = [{"type": "text", "text": user_text}]
    user_content.extend(target.to_llm_content() for target in targets)

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]

    response_format = {"type": "json_schema", "json_schema": schema}
    return messages, response_format


def _build_row_summary(row_index: int, row: pd.Series) -> str:
    """行の内容を人間が読みやすいテキストにまとめる。

    Args:
        row_index: DataFrame 上での行インデックス。
        row: 行の値を保持する Series。

    Returns:
        str: 行インデックスと列ごとの値を列挙した文字列。
    """

    lines = [f"行インデックス: {row_index}"]
    for column, value in row.items():
        if isinstance(column, str):
            lower = column.lower()
            if lower.startswith(("nso_", "nsof_", "c_", "eo")):
                continue
        if pd.isna(value):
            value_str = ""
        else:
            value_str = str(value)
        lines.append(f"- {column}: {value_str}")
    return "\n".join(lines)


def build_nonstructured_prompt(
    prompt_text: str,
    targets: Sequence[Target],
    instructions: str | None = None,
) -> list[dict[str, Any]]:
    """非構造化出力向けのメッセージを生成する。"""

    if not targets:
        raise ValueError("targets が空です。少なくとも 1 つの分析対象が必要です。")

    user_instruction = instructions or prompt_text
    user_text = (
        f"{user_instruction}\n\n"
        "この後に処理対象のテキストや画像コンテンツが続きます。"
    )
    user_content = [{"type": "text", "text": user_text}]
    user_content.extend(target.to_llm_content() for target in targets)

    messages = [
        {"role": "system", "content": NONSTRUCTURED_SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]
    return messages
