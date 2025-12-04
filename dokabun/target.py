"""LLM に渡す分析対象の内部表現。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Target(Protocol):
    """LLM へ渡す content を生成できるターゲットのインターフェース。"""

    def to_llm_content(self) -> dict[str, Any]:
        """OpenAI SDK に渡せる content 辞書を返す。"""


@dataclass(slots=True)
class TextTarget:
    """テキストを対象とするターゲット。"""

    text: str

    def to_llm_content(self) -> dict[str, Any]:
        """LLM へ渡す content 形式に変換する。"""

        return {"type": "text", "text": self.text}


@dataclass(slots=True)
class ImageTarget:
    """画像を対象とするターゲット。"""

    base64_data: str
    mime_type: str = "image/png"

    def to_llm_content(self) -> dict[str, Any]:
        """LLM へ渡す content 形式に変換する。"""

        url = f"data:{self.mime_type};base64,{self.base64_data}"
        return {"type": "image_url", "image_url": {"url": url}}

