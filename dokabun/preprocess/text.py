"""テキスト向けの前処理。"""

from __future__ import annotations

import re
from pathlib import Path

from dokabun.preprocess.base import Preprocess
from dokabun.target import TextTarget

_CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]")


class PlainTextPreprocess(Preprocess):
    """プレーンテキストの前処理を担当する。"""

    def is_eligible(self, target_text: str) -> bool:
        """常に True を返し、フォールバックとして機能する。"""

        return True

    def preprocess(self, target_text: str, base_dir: Path) -> TextTarget:
        """不要な制御文字・余分な空白を整形して返す。"""

        _ = base_dir  # TextTarget では base_dir を利用しない想定
        normalized = target_text.replace("\r\n", "\n").replace("\r", "\n").strip()
        normalized = _CONTROL_CHARS_PATTERN.sub("", normalized)
        return TextTarget(text=normalized)

