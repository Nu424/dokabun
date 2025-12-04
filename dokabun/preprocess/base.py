"""分析対象を LLM へ渡す形式に整える前処理の抽象クラス。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from dokabun.target import Target


class Preprocess(ABC):
    """各種分析対象の前処理を行う抽象クラス。"""

    @abstractmethod
    def is_eligible(self, target_text: str) -> bool:
        """対象の文字列を担当できるかを判定する。"""

    @abstractmethod
    def preprocess(self, target_text: str, base_dir: Path) -> Target:
        """前処理を実行し、Target を返す。"""

