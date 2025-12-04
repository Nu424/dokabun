"""前処理パイプラインのヘルパー。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from dokabun.preprocess.base import Preprocess
from dokabun.preprocess.image import ImagePreprocess
from dokabun.preprocess.text import PlainTextPreprocess
from dokabun.target import Target


def build_default_preprocessors() -> list[Preprocess]:
    """既定の前処理リストを生成する。"""

    return [ImagePreprocess(), PlainTextPreprocess()]


def run_preprocess_pipeline(
    target_text: str,
    base_dir: Path,
    preprocessors: Sequence[Preprocess] | None = None,
) -> Target:
    """前処理パイプラインを実行し、最初に適合した結果を返す。"""

    pipeline: Iterable[Preprocess] = preprocessors or build_default_preprocessors()
    for preprocessor in pipeline:
        if preprocessor.is_eligible(target_text):
            return preprocessor.preprocess(target_text, base_dir)

    raise ValueError(f"前処理できない入力です: {target_text[:32]}...")

