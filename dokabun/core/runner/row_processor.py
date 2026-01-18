"""Per-row processing logic for the runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from dokabun.config import AppConfig
from dokabun.core.runner.cells import is_empty_value
from dokabun.core.runner.columns import (
    column_to_property_name,
    structured_schema_header,
    validate_structured_property_names,
)
from dokabun.core.runner.embedding import (
    build_embedding_text,
    extract_embedding_vector,
    prepare_embedding_output,
)
from dokabun.core.runner.models import RowResult, RowWorkItem
from dokabun.llm.openrouter_client import AsyncOpenRouterClient
from dokabun.llm.prompt import build_nonstructured_prompt, build_prompt
from dokabun.llm.schema import build_schema_from_headers
from dokabun.logging_utils import get_logger
from dokabun.preprocess import build_default_preprocessors, run_preprocess_pipeline
from dokabun.preprocess.base import Preprocess
from dokabun.target import Target

logger = get_logger(__name__)


class RowProcessor:
    """1 行の処理を担当するクラス。"""

    def __init__(
        self,
        *,
        client: AsyncOpenRouterClient,
        config: AppConfig,
        target_columns: Sequence[str],
        base_dir: Path,
        nsof_index_map: dict[str, int],
        embedding_spec_map: dict[str, dict[str, Any]],
        embedding_index_map: dict[str, int],
        preprocessors: Sequence[Preprocess] | None = None,
    ) -> None:
        """1 行分の処理を担当するクラスを初期化する。

        Args:
            client: OpenRouter へ問い合わせる非同期クライアント。
            config: 温度や最大トークン数などを含むアプリ設定。
            target_columns: `i_` で始まるターゲット列の順序付きリスト。
            base_dir: 画像パスなどを解決する基準ディレクトリ。
            nsof_index_map: nsof 列に対する 1 始まりのインデックス。
            embedding_spec_map: 埋め込み列の仕様。
            embedding_index_map: 埋め込み列に対する 1 始まりのインデックス。
            preprocessors: 前処理パイプライン。
        """
        self.client = client
        self.config = config
        self.target_columns = list(target_columns)
        self.base_dir = base_dir
        self.nsof_index_map = dict(nsof_index_map)
        self.embedding_spec_map = dict(embedding_spec_map)
        self.embedding_index_map = dict(embedding_index_map)
        self.preprocessors = (
            list(preprocessors)
            if preprocessors is not None
            else build_default_preprocessors(config.max_text_file_bytes)
        )

    async def process(self, work_item: RowWorkItem, df: pd.DataFrame) -> RowResult:
        """1 行分のスプレッドシートを非同期で処理する。

        Args:
            work_item: 行インデックスと未入力列の情報。
            df: スプレッドシート全体の DataFrame。

        Returns:
            RowResult: 更新内容・usage・エラー情報を含む結果。
        """

        row = df.iloc[work_item.row_index]
        try:
            # ---ターゲット列を前処理する
            targets = self._build_targets(row)
            if not targets:
                raise ValueError("有効なターゲット列が存在しません。")

            updates: dict[str, Any] = {}
            usage_total: dict[str, Any] | None = None
            errors: list[str] = []
            generation_ids: list[str] = []
            embedding_vectors: dict[str, list[float]] = {}
            row_no_1based = work_item.row_index + 1

            # ---構造化出力（JSON Schema）が必要な場合のみ実行する
            if work_item.pending_structured_columns:
                validate_structured_property_names(work_item.pending_structured_columns)
                structured_headers = [
                    structured_schema_header(col)
                    for col in work_item.pending_structured_columns
                ]
                # ---出力形式を、ヘッダーから生成する
                schema_payload = build_schema_from_headers(
                    structured_headers,
                    name=f"dokabun_row_{work_item.row_index}",
                )
                # ---プロンプトを生成する
                messages, response_format = build_prompt(
                    work_item.row_index,
                    row,
                    targets,
                    schema_payload,
                )
                # ---LLMに問い合わせる
                response = await self.client.create_completion(
                    messages=messages,
                    json_schema=response_format["json_schema"],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                # ---レスポンスをパースする
                parsed = _extract_parsed_json(response)
                usage = _strip_cost_keys(
                    _coerce_usage_dict(getattr(response, "usage", None))
                )
                generation_id = getattr(response, "id", None)
                if generation_id:
                    generation_ids.append(str(generation_id))
                # ---更新内容をビルドする
                updates.update(
                    _build_updates_from_parsed(
                        parsed, work_item.pending_structured_columns
                    )
                )
                usage_total = _merge_usage(usage_total, usage)

            # ---非構造化出力（nso_/nsof_）が必要な場合のみ実行する
            if work_item.pending_ns_columns:
                # ---ファイル名を推測する
                target_file_stem = _guess_target_file_stem(
                    row, self.target_columns, self.base_dir
                )
                for ns_column in work_item.pending_ns_columns:
                    try:
                        # ---非構造化出力用のプロンプトを用意し、LLMに問い合わせる
                        prompt_text = _parse_ns_prompt(ns_column)
                        ns_messages = build_nonstructured_prompt(prompt_text, targets)
                        ns_response = await self.client.create_completion_text(
                            messages=ns_messages,
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_tokens,
                        )
                        # ---使用量・コストを取得する
                        ns_usage = _strip_cost_keys(
                            _coerce_usage_dict(getattr(ns_response, "usage", None))
                        )
                        ns_generation_id = getattr(ns_response, "id", None)
                        if ns_generation_id:
                            generation_ids.append(str(ns_generation_id))
                        # ---出力結果を取得する
                        content_text = _extract_text_content(ns_response)

                        if ns_column.lower().startswith("nsof_"):
                            # ---nsof_列の場合、出力結果をファイルに保存する
                            nsf_index = self.nsof_index_map.get(ns_column, 1)
                            filename = _build_nsof_filename(
                                nsf_index=nsf_index,
                                row_no=row_no_1based,
                                ext=self.config.nsof_ext,
                                target_file_stem=target_file_stem,
                                use_filetarget_template=target_file_stem is not None,
                                name_template=self.config.nsof_name_template,
                                name_template_filetarget=self.config.nsof_name_template_filetarget,
                            )
                            # ---ファイルに保存する
                            output_path = (self.config.output_dir / filename).resolve()
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            output_path.write_text(content_text, encoding="utf-8")
                            updates[ns_column] = filename
                        else:
                            updates[ns_column] = content_text

                        usage_total = _merge_usage(usage_total, ns_usage)
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"{ns_column}: {exc}")
            # ---埋め込み（eo*）を列ごとに実行する
            if work_item.pending_embedding_columns:
                try:
                    # ---埋め込み用のプロンプトを生成する
                    embedding_text = build_embedding_text(targets)
                except Exception as exc:
                    for column in work_item.pending_embedding_columns:
                        errors.append(f"{column}: {exc}")
                else:
                    for column in work_item.pending_embedding_columns:
                        spec = self.embedding_spec_map.get(column, {})
                        try:
                            # ---前段次元数を取得する
                            pre_dim = (
                                spec.get("pre_dim")
                                if spec.get("pre_method") == "n"
                                else None
                            )
                            response = await self.client.create_embedding(
                                input_text=embedding_text,
                                dimensions=pre_dim,
                            )
                            vector = extract_embedding_vector(response)
                            usage = _coerce_usage_dict(getattr(response, "usage", None))
                            usage = _normalize_embedding_usage(usage)
                            usage_total = _merge_usage(usage_total, usage)

                            # ---後処理をする場合、得たベクトルを保存する
                            post_method = spec.get("post_method")
                            post_dim = spec.get("post_dim")
                            if post_method and post_dim:
                                embedding_vectors[column] = vector
                                continue

                            # ---後処理をしない場合、セルまたはファイルに出力する
                            output_value = prepare_embedding_output(
                                vector,
                                output_dir=self.config.output_dir,
                                row_no=row_no_1based,
                                embedding_index=self.embedding_index_map.get(column, 1),
                                force_file=bool(spec.get("file_output")),
                            )
                            updates[column] = output_value
                        except Exception as exc:
                            errors.append(f"{column}: {exc}")

            error_msg = "; ".join(errors) if errors else None
            error_type = "PartialError" if errors else None
            return RowResult(
                row_index=work_item.row_index,
                updates=updates,
                usage=usage_total,
                error=error_msg,
                error_type=error_type,
                generation_ids=generation_ids,
                embedding_vectors=embedding_vectors,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("行 %s の処理に失敗しました: %s", work_item.row_index, exc)
            return RowResult(
                row_index=work_item.row_index,
                updates={},
                usage=None,
                error=str(exc),
                error_type=exc.__class__.__name__,
                generation_ids=[],
                embedding_vectors={},
            )

    def _build_targets(self, row: pd.Series) -> list[Target]:
        """ターゲット列を前処理し、Targetオブジェクトのリストを返す。"""

        targets: list[Target] = []
        for column in self.target_columns:
            value = row.get(column)
            if is_empty_value(value):
                continue
            try:
                target = run_preprocess_pipeline(
                    str(value), self.base_dir, preprocessors=self.preprocessors
                )
                targets.append(target)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ターゲット列 %s の前処理に失敗しました: %s", column, exc
                )
                raise
        return targets


def _build_updates_from_parsed(
    parsed: dict[str, Any],
    pending_columns: Iterable[str],
) -> dict[str, Any]:
    """LLM から返った JSON を DataFrame の列へマッピングする。"""

    updates: dict[str, Any] = {}
    for column in pending_columns:
        property_name = column_to_property_name(column)
        value = parsed.get(property_name, "")
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        elif value is None:
            value = ""
        updates[column] = value
    return updates


def _extract_parsed_json(response: object) -> dict[str, Any]:
    """OpenAI SDK のレスポンスから JSON を取り出す。"""

    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("LLM 応答に choices が含まれていません。")

    message = getattr(choices[0], "message", None)
    if message is None:
        raise ValueError("LLM 応答に message が含まれていません。")

    parsed = getattr(message, "parsed", None)
    if parsed is not None:
        return dict(parsed)

    content = getattr(message, "content", None)
    if not content:
        raise ValueError("LLM 応答から JSON を取得できませんでした。")

    if isinstance(content, str):
        return json.loads(content)

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        if not text_parts:
            raise ValueError("LLM 応答の content から JSON を抽出できませんでした。")
        return json.loads("".join(text_parts))

    raise ValueError("未対応の LLM 応答形式です。")


def _coerce_usage_dict(usage: Any) -> dict[str, Any] | None:
    """SDK 固有の usage オブジェクトを辞書に変換する。"""

    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "__dict__"):
        return usage.__dict__
    return None


def _strip_cost_keys(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    """usage 辞書からコスト系キーを除外する。"""

    if usage is None:
        return None
    cleaned = dict(usage)
    cleaned.pop("total_cost_usd", None)
    cleaned.pop("total_cost", None)
    return cleaned


def _normalize_embedding_usage(
    usage: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """埋め込みレスポンスの usage を集計しやすい形に整形する。"""

    if usage is None:
        return None
    normalized = dict(usage)
    if "total_cost" not in normalized and "cost" in normalized:
        normalized["total_cost"] = normalized.get("cost")
    return normalized


def _merge_usage(
    base: dict[str, Any] | None, add: dict[str, Any] | None
) -> dict[str, Any] | None:
    """usage を合算する。"""

    if add is None:
        return base
    if base is None:
        return dict(add)

    merged = dict(base)
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        merged[key] = int(merged.get(key, 0) or 0) + int(add.get(key, 0) or 0)
    for cost_key in ("total_cost_usd", "total_cost"):
        merged[cost_key] = float(merged.get(cost_key, 0.0) or 0.0) + float(
            add.get(cost_key, 0.0) or 0.0
        )
    return merged


def _parse_ns_prompt(column: str) -> str:
    """nso_/nsof_ 列ヘッダーからプロンプト文を抽出する。"""

    stripped = column
    lower = stripped.lower()
    if lower.startswith("nsof_"):
        stripped = stripped[len("nsof_") :]
    elif lower.startswith("nso_"):
        stripped = stripped[len("nso_") :]
    if "|" in stripped:
        stripped, _ = stripped.split("|", 1)
    prompt = stripped.strip()
    return prompt or "与えられた入力を処理してください。"


def _extract_text_content(response: object) -> str:
    """OpenAI SDK のレスポンスからテキストを取り出す。"""

    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("LLM 応答に choices が含まれていません。")

    message = getattr(choices[0], "message", None)
    if message is None:
        raise ValueError("LLM 応答に message が含まれていません。")

    content = getattr(message, "content", None)
    if content is None:
        raise ValueError("LLM 応答から content を取得できませんでした。")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        if text_parts:
            return "".join(text_parts)

    raise ValueError("未対応の LLM 応答形式です。")


def _guess_target_file_stem(
    row: pd.Series, target_columns: Sequence[str], base_dir: Path
) -> str | None:
    """i_ 列が 1 つだけで、ファイルパスなら stem を返す。"""

    if len(target_columns) != 1:
        return None
    raw = row.get(target_columns[0])
    if is_empty_value(raw):
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if path.exists():
        return path.stem
    return None


def _build_nsof_filename(
    *,
    nsf_index: int,
    row_no: int,
    ext: str,
    target_file_stem: str | None,
    use_filetarget_template: bool,
    name_template: str,
    name_template_filetarget: str,
) -> str:
    """nsof 用のファイル名を生成する。"""

    clean_ext = ext.lstrip(".")
    template = name_template_filetarget if use_filetarget_template else name_template
    filename = template.format(
        nsf_index=nsf_index,
        row_no=row_no,
        ext=clean_ext,
        target_file_stem=target_file_stem or "",
    )
    filename = _sanitize_filename(filename)
    if not filename.lower().endswith(f".{clean_ext}"):
        filename = f"{filename}.{clean_ext}"
    return filename


def _sanitize_filename(name: str) -> str:
    """Windows でも扱えるように簡易サニタイズする。"""

    import re

    sanitized = re.sub(r'[<>:"/\\|?*\r\n]', "_", name).strip()
    return sanitized or "output"


