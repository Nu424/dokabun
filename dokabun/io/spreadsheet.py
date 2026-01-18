"""スプレッドシートの読み書きと一時ファイル管理。"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from dokabun.config import AppConfig
from dokabun.logging_utils import get_logger
from dokabun.utils.datetime import now_ts_str

logger = get_logger(__name__)


@dataclass(slots=True)
class SpreadsheetPaths:
    """スプレッドシート処理で利用する重要なパス群。"""

    copy_path: Path
    output_path: Path
    meta_path: Path


class SpreadsheetReaderWriter:
    """スプレッドシートファイルの読み書きと一時保存を担当する。"""

    def __init__(self, config: AppConfig) -> None:
        """インスタンスを初期化する。

        Args:
            config: 実行設定。入出力パスやタイムスタンプ設定を含む。
        """

        timestamp = config.timestamp or now_ts_str()
        self.config = config if config.timestamp else config.with_timestamp(timestamp)
        self.timestamp = self.config.timestamp or timestamp

        self.input_path = self.config.input_path
        self.input_dir = self.input_path.parent
        self.output_dir = self.config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.base_stem = self.input_path.stem
        self.source_ext = self.input_path.suffix.lower()
        self.paths = self._build_paths()
        self._df: Optional[pd.DataFrame] = None

    def _build_paths(self) -> SpreadsheetPaths:
        """必要なパスを生成する。

        Returns:
            SpreadsheetPaths: コピー・出力・メタデータの各パス。
        """

        copy_path = self.output_dir / f"{self.base_stem}_{self.timestamp}{self.source_ext}"
        output_path = self.output_dir / f"{self.base_stem}_{self.timestamp}.out.xlsx"
        meta_path = self.output_dir / f"{self.base_stem}_{self.timestamp}.meta.json"
        return SpreadsheetPaths(copy_path=copy_path, output_path=output_path, meta_path=meta_path)

    @property
    def target_base_dir(self) -> Path:
        """i_xxx 列の相対パス解決に利用するディレクトリを返す。"""

        return self.input_dir

    def load(self) -> pd.DataFrame:
        """入力ファイルを読み込み DataFrame を返す。

        Returns:
            pandas.DataFrame: 入力シート内容のコピー。

        Raises:
            FileNotFoundError: 入力ファイルが存在しない場合。
            ValueError: 未対応の拡張子が指定された場合。
        """

        partial_path = self._find_latest_partial_path()
        if partial_path:
            logger.info("既存の一時ファイルを読み込みます: %s", partial_path)
            df = pd.read_excel(partial_path)
        else:
            if not self.input_path.exists():
                raise FileNotFoundError(f"入力ファイルが見つかりません: {self.input_path}")

            shutil.copy2(self.input_path, self.paths.copy_path)
            logger.info("入力ファイルをコピーしました: %s", self.paths.copy_path)

            if self.source_ext == ".xlsx":
                df = pd.read_excel(self.paths.copy_path)
            elif self.source_ext == ".csv":
                df = pd.read_csv(self.paths.copy_path)
            else:
                raise ValueError(f"未対応のファイル形式です: {self.source_ext}")

        self._df = df
        return df.copy()

    def save_output(self, df: pd.DataFrame) -> Path:
        """最終的な結果を Excel 形式で保存する。

        Args:
            df: 保存対象の DataFrame。

        Returns:
            Path: 出力先ファイルパス。
        """

        df.to_excel(self.paths.output_path, index=False)
        logger.info("最終結果を保存しました: %s", self.paths.output_path)
        return self.paths.output_path

    def save_partial(
        self,
        df: pd.DataFrame,
        start_row: int,
        end_row: int,
        *,
        meta: dict[str, Any] | None = None,
    ) -> Path:
        """途中経過を一時ファイルとして保存する。

        Args:
            df: 保存対象の DataFrame（最新状態）。
            start_row: 一時保存対象の開始行インデックス。
            end_row: 一時保存対象の終了行インデックス。

        Returns:
            Path: 一時ファイルのパス。
        """

        partial_name = f"{self.base_stem}_{self.timestamp}.partial.{start_row}-{end_row}.xlsx"
        partial_path = self.output_dir / partial_name
        df.to_excel(partial_path, index=False)
        logger.info("一時ファイルを保存しました: %s", partial_path)
        self.save_meta(meta or {"last_completed_row": end_row})
        return partial_path

    def save_meta(self, meta: dict[str, Any]) -> None:
        """再開用メタデータを保存する。

        Args:
            meta: 保存したいメタ情報（処理済み行など）。
        """

        payload = {"timestamp": self.timestamp, **meta}
        self.paths.meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug("メタデータを保存しました: %s", self.paths.meta_path)

    def load_meta_if_exists(self) -> Optional[dict[str, Any]]:
        """メタデータが存在すれば読み込む。

        Returns:
            dict[str, Any] | None: 読み込んだメタデータ。存在しない・破損している場合は None。
        """

        if not self.paths.meta_path.exists():
            return None
        try:
            data = json.loads(self.paths.meta_path.read_text(encoding="utf-8"))
            return data
        except json.JSONDecodeError as exc:
            logger.warning("メタデータの読み込みに失敗しました (%s): %s", exc, self.paths.meta_path)
            return None

    def _find_latest_partial_path(self) -> Path | None:
        """最新の一時ファイルのパスを返す。存在しなければ None。"""

        pattern = f"{self.base_stem}_{self.timestamp}.partial.*.xlsx"
        candidates: list[tuple[int, Path]] = []
        for path in self.output_dir.glob(pattern):
            range_info = self._parse_partial_range(path.name)
            if range_info is None:
                continue
            _, end_row = range_info
            candidates.append((end_row, path))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        return candidates[-1][1]

    @staticmethod
    def _parse_partial_range(filename: str) -> tuple[int, int] | None:
        """partial ファイル名から (start, end) を取り出す。"""

        try:
            marker = ".partial."
            start = filename.index(marker) + len(marker)
            end = filename.rindex(".xlsx")
            range_part = filename[start:end]
            start_str, end_str = range_part.split("-", 1)
            return int(start_str), int(end_str)
        except (ValueError, IndexError):
            return None

