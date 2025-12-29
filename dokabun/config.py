"""どかぶん全体で利用する設定クラス。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(slots=True)
class AppConfig:
    """アプリケーション設定を保持するデータクラス。

    Attributes:
        input_path: 処理対象となるスプレッドシートファイルのパス。
        output_dir: 結果ファイルを保存するディレクトリ。未指定時は入力ファイルと同じ。
        timestamp: 実行ごとに採番されるタイムスタンプ文字列。フェーズ2で設定。
        partial_interval: 何行処理ごとに一時ファイルを保存するか。
        model: OpenRouter 経由で利用するモデル名。
        temperature: LLM 呼び出し時の温度パラメータ。
        max_tokens: LLM 応答の最大トークン数。
        max_concurrency: 同時並列実行数（後続フェーズで利用）。
        max_rows: 実行あたりの最大処理行数。None なら制限なし。
        log_level: ルートロガーに適用するログレベル文字列。
        max_text_file_bytes: テキストファイル読み込み時の最大サイズ（バイト）。デフォルトは 256KiB。
    """

    input_path: Path
    output_dir: Optional[Path] = None
    timestamp: Optional[str] = None
    partial_interval: int = 100
    model: str = "openai/gpt-4.1-mini"
    temperature: float | None = None
    max_tokens: Optional[int] = None
    max_concurrency: int = 5
    max_rows: Optional[int] = None
    log_level: str = "INFO"
    max_text_file_bytes: int = 262_144
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """パスや数値の正規化を行う。"""

        self.input_path = Path(self.input_path).expanduser().resolve()
        if self.output_dir is None:
            self.output_dir = self.input_path.parent
        else:
            self.output_dir = Path(self.output_dir).expanduser().resolve()

        if self.partial_interval <= 0:
            raise ValueError("partial_interval は 1 以上である必要があります。")

        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency は 1 以上である必要があります。")

        if self.max_rows is not None and self.max_rows <= 0:
            raise ValueError("max_rows は None または 1 以上である必要があります。")

        if self.max_text_file_bytes <= 0:
            raise ValueError("max_text_file_bytes は 1 以上である必要があります。")

    @property
    def log_file(self) -> Path:
        """ログファイルの保存先を返す。"""

        return self.output_dir / "dokabun.log"

    def with_timestamp(self, timestamp: str) -> "AppConfig":
        """タイムスタンプを付与した新しい AppConfig を返す。"""

        return AppConfig(
            input_path=self.input_path,
            output_dir=self.output_dir,
            timestamp=timestamp,
            partial_interval=self.partial_interval,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_concurrency=self.max_concurrency,
            max_rows=self.max_rows,
            log_level=self.log_level,
            max_text_file_bytes=self.max_text_file_bytes,
            extra=self.extra.copy(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        """辞書から AppConfig インスタンスを生成する。

        Args:
            data: CLI や設定ファイルから取得した値。

        Returns:
            AppConfig: 生成された設定オブジェクト。
        """

        known_fields = {
            "input_path",
            "output_dir",
            "timestamp",
            "partial_interval",
            "model",
            "temperature",
            "max_tokens",
            "max_concurrency",
            "max_rows",
            "log_level",
            "max_text_file_bytes",
        }
        filtered = {key: value for key, value in data.items() if key in known_fields}
        extra = {key: value for key, value in data.items() if key not in known_fields}
        return cls(**filtered, extra=extra)
