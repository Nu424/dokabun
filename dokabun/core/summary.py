"""実行結果を集計するサマリークラス。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass(slots=True)
class ExecutionSummary:
    """ランナー全体の実行状況を集計する。

    Attributes:
        started_at: 実行開始時刻。
        finished_at: 実行終了時刻。
        total_rows: 対象となる行数。
        processed_rows: 実際に処理を試みた行数。
        success_rows: 正常に完了した行数。
        failed_rows: エラーとなった行数。
        prompt_tokens: プロンプト側トークン数。
        completion_tokens: 応答側トークン数。
        total_tokens: 合計トークン数。
        total_cost_usd: 推定コスト (USD)。
        error_counts: エラータイプごとの件数。
    """

    started_at: datetime | None = None
    finished_at: datetime | None = None
    total_rows: int = 0
    processed_rows: int = 0
    success_rows: int = 0
    failed_rows: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    error_counts: Dict[str, int] = field(default_factory=dict)

    def start(self, total_rows: int) -> None:
        """集計を開始する。

        Args:
            total_rows: 今回処理対象となる行数。
        """

        self.started_at = datetime.now()
        self.total_rows = total_rows

    def finish(self) -> None:
        """終了時刻を記録し、集計を終了する。"""

        self.finished_at = datetime.now()

    def record_success(self, row_index: int, usage: dict[str, Any] | None) -> None:
        """成功した行の統計情報を反映する。

        Args:
            row_index: 成功した行のインデックス。
            usage: LLM クライアントが返した usage 情報。
        """

        _ = row_index
        self.processed_rows += 1
        self.success_rows += 1
        if usage:
            self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            self.total_tokens += int(
                usage.get("total_tokens", usage.get("prompt_tokens", 0)) or 0
            )
            self.total_cost_usd += float(
                usage.get("total_cost_usd", usage.get("total_cost", 0.0)) or 0.0
            )

    def record_failure(self, row_index: int, error_type: str, message: str) -> None:
        """失敗した行の情報を集計に反映する。

        Args:
            row_index: 失敗した行のインデックス。
            error_type: エラー種別（例外名など）。
            message: エラーメッセージ。
        """

        _ = (row_index, message)
        self.processed_rows += 1
        self.failed_rows += 1
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        """集計結果をシリアライズしやすい辞書に変換する。

        Returns:
            dict[str, Any]: 各種メトリクスを含む辞書。
        """

        return {
            "started_at": self._format_dt(self.started_at),
            "finished_at": self._format_dt(self.finished_at),
            "total_rows": self.total_rows,
            "processed_rows": self.processed_rows,
            "success_rows": self.success_rows,
            "failed_rows": self.failed_rows,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "error_counts": self.error_counts.copy(),
            "duration_seconds": self._calc_duration(),
        }

    def format_text(self) -> str:
        """人間が読みやすいテキスト形式のサマリーを生成する。

        Returns:
            str: 標準出力やログに適した複数行文字列。
        """

        data = self.to_dict()
        lines = [
            "=== Execution Summary ===",
            f"開始時刻:   {data['started_at']}",
            f"終了時刻:   {data['finished_at']}",
            f"処理時間:   {data['duration_seconds']:.1f}s",
            f"対象行数:   {self.total_rows}",
            f"処理済み:   {self.processed_rows} (成功 {self.success_rows} / 失敗 {self.failed_rows})",
            (
                "トークン数: prompt={prompt_tokens} / completion={completion_tokens} / total={total_tokens}"
            ).format(**data),
            f"推定コスト: ${data['total_cost_usd']:.6f}",
        ]

        if self.error_counts:
            lines.append("エラー内訳:")
            for error_type, count in sorted(self.error_counts.items()):
                lines.append(f"  - {error_type}: {count}")

        return "\n".join(lines)

    def _calc_duration(self) -> float:
        """処理に要した時間（秒）を計算する。

        Returns:
            float: 実行時間（終了前の場合は現在時刻まで）。
        """

        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        if self.started_at:
            return (datetime.now() - self.started_at).total_seconds()
        return 0.0

    @staticmethod
    def _format_dt(dt: datetime | None) -> str:
        """日時を表示用に整形する。

        Args:
            dt: ``datetime`` オブジェクトまたは ``None``。

        Returns:
            str: 整形済み文字列。``None`` の場合は ``-``。
        """

        return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "-"
