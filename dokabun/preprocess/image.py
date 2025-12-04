"""画像ファイル向けの前処理。"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Iterable

from dokabun.logging_utils import get_logger
from dokabun.preprocess.base import Preprocess
from dokabun.target import ImageTarget

logger = get_logger(__name__)


class ImagePreprocess(Preprocess):
    """画像ファイルを Base64 へ変換して ImageTarget にする。"""

    def __init__(self, extensions: Iterable[str] | None = None) -> None:
        self.extensions = tuple(
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in (extensions or (".png", ".jpg", ".jpeg", ".webp"))
        ) # [".png", ".jpg", ".jpeg", ".webp"]のようなタプルを作成

    def is_eligible(self, target_text: str) -> bool:
        """拡張子に基づいて担当可否を判定する。"""

        lower = target_text.strip().lower()
        return any(lower.endswith(ext) for ext in self.extensions)

    def preprocess(self, target_text: str, base_dir: Path) -> ImageTarget:
        """画像ファイルを読み込み、Base64 へ変換する。"""

        path = Path(target_text)
        if not path.is_absolute():
            path = (base_dir / path).resolve()

        if not path.exists():
            logger.error("画像ファイルが見つかりません: %s", path)
            raise FileNotFoundError(path)

        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        logger.debug("画像ファイルを読み込みました: %s", path)
        return ImageTarget(base64_data=encoded, mime_type=mime_type)

