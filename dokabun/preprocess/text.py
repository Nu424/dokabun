"""テキスト向けの前処理。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from dokabun.logging_utils import get_logger
from dokabun.preprocess.base import Preprocess
from dokabun.target import TextTarget

logger = get_logger(__name__)
_CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]")


class TextFilePreprocess(Preprocess):
    """テキストファイルを読み込んで TextTarget にする前処理。"""

    def __init__(self, extensions: Iterable[str] | None = None, max_bytes: int = 262_144) -> None:
        """インスタンスを初期化する。

        Args:
            extensions: 対象とする拡張子のリスト。デフォルトは主要なテキスト形式。
            max_bytes: 読み込むファイルの最大サイズ（バイト）。デフォルトは 256KiB。
        """
        self.extensions = tuple(
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in (
                extensions
                or (".txt", ".md", ".markdown", ".log", ".csv", ".json", ".yml", ".yaml")
            )
        )
        self.max_bytes = max_bytes

    def is_eligible(self, target_text: str) -> bool:
        """拡張子に基づいて担当可否を判定する。"""

        lower = target_text.strip().lower()
        return any(lower.endswith(ext) for ext in self.extensions)

    def preprocess(self, target_text: str, base_dir: Path) -> TextTarget:
        """テキストファイルを読み込み、内容を TextTarget として返す。

        Args:
            target_text: ファイルパス文字列。
            base_dir: 相対パスを解決する基準ディレクトリ。

        Returns:
            TextTarget: ファイル内容を整形したテキストターゲット。

        Raises:
            FileNotFoundError: ファイルが存在しない場合。
            ValueError: ファイルサイズが上限を超えている、または文字コードが判別できない場合。
            OSError: ファイル読み込みに失敗した場合。
        """
        path = Path(target_text)
        if not path.is_absolute():
            path = (base_dir / path).resolve()

        if not path.exists():
            logger.error("テキストファイルが見つかりません: %s", path)
            raise FileNotFoundError(f"テキストファイルが見つかりません: {path}")

        file_size = path.stat().st_size
        if file_size > self.max_bytes:
            logger.error(
                "テキストファイルが大きすぎます: %s (%d bytes > %d bytes)",
                path,
                file_size,
                self.max_bytes,
            )
            raise ValueError(
                f"テキストファイルが大きすぎます: {path} ({file_size} bytes > {self.max_bytes} bytes)"
            )

        # 文字コードを順に試す
        encodings = ["utf-8-sig", "utf-8", "cp932"]
        content_bytes = path.read_bytes()
        text_content: str | None = None
        last_error: Exception | None = None

        for encoding in encodings:
            try:
                text_content = content_bytes.decode(encoding)
                logger.debug("テキストファイルを読み込みました: %s (encoding: %s)", path, encoding)
                break
            except UnicodeDecodeError as exc:
                last_error = exc
                continue

        if text_content is None:
            logger.error("テキストファイルの文字コードが判別できません: %s", path)
            raise ValueError(
                f"テキストファイルの文字コードが判別できません: {path} (最後のエラー: {last_error})"
            )

        # PlainTextPreprocess と同様の整形を行う
        normalized = text_content.replace("\r\n", "\n").replace("\r", "\n").strip()
        normalized = _CONTROL_CHARS_PATTERN.sub("", normalized)
        return TextTarget(text=normalized)


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

