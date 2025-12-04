"""出力列ヘッダから JSON Schema を生成するユーティリティ。"""

from __future__ import annotations

from typing import Any, Iterable


def build_schema_from_headers(headers: Iterable[str], name: str = "dokabun_row") -> dict[str, Any]:
    """スプレッドシートの出力列ヘッダから JSON Schema を構築する。

    Args:
        headers: `{列名}` または `{列名}|{説明}` 形式の文字列群。
        name: `response_format.json_schema.name` に使用する名前。

    Returns:
        dict[str, Any]: OpenAI SDK の `response_format.json_schema` にそのまま渡せる辞書。
    """

    properties: dict[str, dict[str, str]] = {}
    for raw_header in headers:
        if not raw_header:
            continue
        column, description = _split_header(raw_header)
        properties[column] = {"type": "string"}
        if description:
            properties[column]["description"] = description

    if not properties:
        raise ValueError("出力列が指定されていません。")

    schema = {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }

    return {
        "name": name,
        "strict": True,
        "schema": schema,
    }


def _split_header(header: str) -> tuple[str, str]:
    """ヘッダ文字列を列名と説明に分割する。"""

    if "|" not in header:
        return header.strip(), ""

    column, description = header.split("|", 1)
    return column.strip(), description.strip()

