"""スプレッドシートを処理して LLM で未入力セルを埋めるコアランナー。"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from dokabun.config import AppConfig
from dokabun.core.summary import ExecutionSummary
from dokabun.io.spreadsheet import SpreadsheetReaderWriter
from dokabun.llm.openrouter_client import AsyncOpenRouterClient
from dokabun.llm.prompt import build_nonstructured_prompt, build_prompt
from dokabun.llm.schema import build_schema_from_headers
from dokabun.logging_utils import get_logger
from dokabun.preprocess import build_default_preprocessors, run_preprocess_pipeline
from dokabun.preprocess.base import Preprocess
from dokabun.target import Target, TextTarget

logger = get_logger(__name__)

EMBEDDING_CELL_CHAR_LIMIT = 32767 # 埋め込みベクトルをセルに出力する場合の文字数制限
EMBEDDING_FILE_EXT = "npy" # 埋め込みベクトルをファイルに出力する場合のファイル拡張子


@dataclass(slots=True)
class RowResult:
    """1 行分の処理結果を保持するデータクラス。

    Attributes:
        row_index: DataFrame 上の行インデックス。
        updates: シートへ書き戻す列名と値のマッピング。
        usage: LLM 応答から得た usage 情報（トークン数やコストなど）。
        error: エラー発生時のメッセージ。成功時は ``None``。
        error_type: エラー種別（例外名など）。成功時は ``None``。
        embedding_vectors: 後段処理が必要な埋め込みベクトル。
    """

    row_index: int
    updates: dict[str, Any]
    usage: dict[str, Any] | None = None
    error: str | None = None
    error_type: str | None = None
    generation_ids: list[str] = field(default_factory=list)
    embedding_vectors: dict[str, list[float]] = field(default_factory=dict)


@dataclass(slots=True)
class RowWorkItem:
    """処理待ち行に関するメタデータ。

    Attributes:
        row_index: DataFrame 上の行インデックス。
        pending_structured_columns: まだ値が入っていない構造化出力列。
        pending_ns_columns: まだ値が入っていない非構造化出力列（nso_/nsof_）。
        pending_embedding_columns: まだ値が入っていない埋め込み列（eo*）。
    """

    row_index: int
    pending_structured_columns: list[str]
    pending_ns_columns: list[str]
    pending_embedding_columns: list[str]


@dataclass(slots=True)
class ColumnClassification:
    """列プレフィックス分類結果を保持するデータクラス。
    
    Attributes:
        input_columns: `i_` で始まる入力列。
        structured_columns: `so_` で始まる構造化出力列。
        nonstructured_columns: `nso_` で始まる非構造化出力列。
        nsof_index_map: `nsof_` 列に対する 1 始まりのインデックス。
        label_columns: `l_` で始まるラベル列。
        embedding_columns: `eo` で始まる埋め込み列。
        embedding_spec_map: 埋め込み列の仕様。
        e.g.
            {
                "eo_1536": {
                    "pre_method": "n",
                    "pre_dim": 1536,
                    "post_method": "n",
                    "post_dim": 1536,
                    "file_output": True,
                }
            }
        embedding_index_map: 埋め込み列に対する 1 始まりのインデックス。
    """

    input_columns: list[str]
    structured_columns: list[str]
    nonstructured_columns: list[str]
    nsof_index_map: dict[str, int]
    label_columns: list[str]
    embedding_columns: list[str]
    embedding_spec_map: dict[str, dict[str, Any]]
    embedding_index_map: dict[str, int]


async def process_row_async(
    work_item: RowWorkItem,
    df: pd.DataFrame,
    *,
    target_columns: Sequence[str],
    base_dir: Path,
    client: AsyncOpenRouterClient,
    config: AppConfig,
    nsof_index_map: dict[str, int],
    embedding_spec_map: dict[str, dict[str, Any]],
    embedding_index_map: dict[str, int],
    preprocessors: Sequence[Preprocess] | None = None,
) -> RowResult:
    """1 行分のスプレッドシートを非同期で処理する。

    Args:
        work_item: 行インデックスと未入力列の情報。
        df: スプレッドシート全体の DataFrame。
        target_columns: `i_` で始まるターゲット列の順序付きリスト。
        base_dir: 画像パスなどを解決する基準ディレクトリ。
        client: OpenRouter へ問い合わせる非同期クライアント。
        config: 温度や最大トークン数などを含むアプリ設定。
        nsof_index_map: nsof 列に対する 1 始まりのインデックス。
        embedding_spec_map: 埋め込み列の仕様。
        embedding_index_map: 埋め込み列に対する 1 始まりのインデックス。

    Returns:
        RowResult: 更新内容・usage・エラー情報を含む結果。
    """

    row = df.iloc[work_item.row_index]
    try:
        # ---ターゲット列を前処理する
        targets = _build_targets(row, target_columns, base_dir, preprocessors)
        if not targets:
            raise ValueError("有効なターゲット列が存在しません。")

        updates: dict[str, Any] = {}
        usage_total: dict[str, Any] | None = None
        errors: list[str] = []
        generation_ids: list[str] = []
        embedding_vectors: dict[str, list[float]] = {} # 後処理用の埋め込みベクトル
        row_no_1based = work_item.row_index + 1

        # ---構造化出力（JSON Schema）が必要な場合のみ実行する
        if work_item.pending_structured_columns:
            _validate_structured_property_names(work_item.pending_structured_columns)
            structured_headers = [
                _structured_schema_header(col)
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
            response = await client.create_completion(
                messages=messages,
                json_schema=response_format["json_schema"],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
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
                _build_updates_from_parsed(parsed, work_item.pending_structured_columns)
            )
            usage_total = _merge_usage(usage_total, usage)

        # ---非構造化出力（nso_/nsof_）を列ごとに実行する
        if work_item.pending_ns_columns:
            target_file_stem = _guess_target_file_stem(row, target_columns, base_dir)
            for ns_column in work_item.pending_ns_columns:
                try:
                    # ---非構造化出力用のプロンプトを用意し、LLMに問い合わせる
                    prompt_text = _parse_ns_prompt(ns_column)
                    ns_messages = build_nonstructured_prompt(prompt_text, targets)
                    ns_response = await client.create_completion_text(
                        messages=ns_messages,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
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

                    # ---nsof_列の場合、出力結果をファイルに保存する
                    if ns_column.lower().startswith("nsof_"):
                        # ---ファイル名を準備する
                        nsf_index = nsof_index_map.get(ns_column, 1)
                        filename = _build_nsof_filename(
                            nsf_index=nsf_index,
                            row_no=row_no_1based,
                            ext=config.nsof_ext,
                            target_file_stem=target_file_stem,
                            use_filetarget_template=target_file_stem is not None,
                            name_template=config.nsof_name_template,
                            name_template_filetarget=config.nsof_name_template_filetarget,
                        )
                        # ---ファイルに保存する
                        output_path = (config.output_dir / filename).resolve()
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
                embedding_text = _build_embedding_text(targets)
            except Exception as exc:
                for column in work_item.pending_embedding_columns:
                    errors.append(f"{column}: {exc}")
            else:
                for column in work_item.pending_embedding_columns:
                    spec = embedding_spec_map.get(column, {})
                    try:
                        pre_dim = (
                            spec.get("pre_dim")
                            if spec.get("pre_method") == "n"
                            else None
                        )
                        response = await client.create_embedding(
                            input_text=embedding_text,
                            dimensions=pre_dim,
                        )
                        vector = _extract_embedding_vector(response)
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
                        output_value = _prepare_embedding_output(
                            vector,
                            output_dir=config.output_dir,
                            row_no=row_no_1based,
                            embedding_index=embedding_index_map.get(column, 1),
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


async def run_async(config: AppConfig, api_key: str) -> None:
    """スプレッドシート全体を非同期で処理する。

    Args:
        config: 実行に関するアプリ設定。
        api_key: OpenRouter API キー。
    """

    logger.info(
        "処理を開始します: input=%s model=%s base_url=%s concurrency=%s",
        config.input_path,
        config.model,
        config.base_url,
        config.max_concurrency,
    )

    # ---スプレッドシートを読み込む(中断再開付き)
    reader = SpreadsheetReaderWriter(config)
    meta = reader.load_meta_if_exists() or {}
    last_completed_row = meta.get("last_completed_row")
    start_row_index = (
        int(last_completed_row) + 1 if last_completed_row is not None else 0
    )

    # ---generation_id 永続化用のパスを用意する
    generation_log_path = (
        reader.output_dir / f"{reader.base_stem}_{reader.timestamp}.generations.jsonl"
    )
    cost_cache_path = (
        reader.output_dir
        / f"{reader.base_stem}_{reader.timestamp}.generation_costs.jsonl"
    )
    generation_log_path.parent.mkdir(parents=True, exist_ok=True)
    generation_log_path.touch(exist_ok=True)
    cost_cache_path.parent.mkdir(parents=True, exist_ok=True)

    # ---処理対象・出力列をわけ、処理が必要な部分を抽出する
    df = reader.load()
    classification = _classify_columns(df)
    target_columns = classification.input_columns
    structured_columns = classification.structured_columns
    ns_columns = classification.nonstructured_columns
    nsof_index_map = classification.nsof_index_map
    embedding_columns = classification.embedding_columns
    embedding_spec_map = classification.embedding_spec_map
    embedding_index_map = classification.embedding_index_map
    work_items = _collect_work_items(
        df,
        structured_columns=structured_columns,
        ns_columns=ns_columns,
        embedding_columns=embedding_columns,
        start_row_index=start_row_index,
        max_rows=config.max_rows,
    )

    # ---LLM処理(並列)を準備する
    client = AsyncOpenRouterClient(
        api_key=api_key,
        model=config.model,
        base_url=config.base_url,
    )
    semaphore = asyncio.Semaphore(config.max_concurrency)
    summary = ExecutionSummary()
    summary.start(total_rows=len(work_items))
    preprocessors = build_default_preprocessors(
        max_text_file_bytes=config.max_text_file_bytes
    )

    # 処理対象が無い場合でも generation のコストだけ再計算する
    if not work_items:
        logger.info("処理対象の行が存在しません。コスト再取得のみ実行します。")
        run_cost_usd, total_cost_usd, pending_count = await _reconcile_generation_costs(
            client=client,
            generation_log_path=generation_log_path,
            cost_cache_path=cost_cache_path,
            run_generation_ids=[],
            max_concurrency=min(config.max_concurrency * 2, 20),
        )
        if total_cost_usd or pending_count:
            logger.info(
                "コスト再取得: total=$%.6f pending=%s", total_cost_usd, pending_count
            )
        return

    async def _bounded_process(item: RowWorkItem) -> RowResult:
        async with semaphore:
            return await process_row_async(
                item,
                df,
                target_columns=target_columns,
                base_dir=reader.target_base_dir,
                client=client,
                config=config,
                nsof_index_map=nsof_index_map,
                embedding_spec_map=embedding_spec_map,
                embedding_index_map=embedding_index_map,
                preprocessors=preprocessors,
            )

    # ---処理を並列実行する
    tasks = [asyncio.create_task(_bounded_process(item)) for item in work_items]
    processed = 0
    chunk_first_row: int | None = None
    chunk_last_row: int | None = None
    new_generation_ids_run: list[str] = []
    embedding_reduction_inputs: dict[str, dict[int, list[float]]] = {}

    # ---処理結果を取得する
    for future in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Processing rows",
        unit="row",
    ):
        result = await future
        processed += 1

        # ---成功分の更新はエラーがあっても反映する
        for column, value in result.updates.items():
            df.loc[result.row_index, column] = value

        if result.error:
            logger.error(
                "行 %s の処理に失敗しました: %s", result.row_index, result.error
            )
            summary.record_failure(
                result.row_index,
                result.error_type or "RowError",
                result.error,
            )
        else:
            summary.record_success(result.row_index, result.usage)

        if result.generation_ids:
            _append_jsonl(
                generation_log_path, [{"id": gid} for gid in result.generation_ids]
            )
            new_generation_ids_run.extend(result.generation_ids)

        # ---後処理対象の埋め込みベクトルを、embedding_reduction_inputs に追加する
        if result.embedding_vectors:
            for column, vector in result.embedding_vectors.items():
                embedding_reduction_inputs.setdefault(column, {})[
                    result.row_index
                ] = vector

        # ---部分的な保存のため、処理範囲を更新する
        chunk_first_row = (
            result.row_index
            if chunk_first_row is None
            else min(chunk_first_row, result.row_index)
        )
        chunk_last_row = (
            result.row_index
            if chunk_last_row is None
            else max(chunk_last_row, result.row_index)
        )

        # ---部分的に保存する
        if processed % config.partial_interval == 0:
            reader.save_partial(df, chunk_first_row, chunk_last_row)
            chunk_first_row = None
            chunk_last_row = None

    # ---後処理対象の埋め込みベクトルを、実際に後処理し、DFを更新する
    if embedding_reduction_inputs:
        _apply_embedding_reductions(
            df=df,
            embedding_vectors=embedding_reduction_inputs,
            embedding_spec_map=embedding_spec_map,
            embedding_index_map=embedding_index_map,
            output_dir=reader.output_dir,
        )

    if chunk_first_row is not None and chunk_last_row is not None:
        reader.save_partial(df, chunk_first_row, chunk_last_row)

    # ---最終版を保存する
    reader.save_output(df)

    # ---コストを末尾でまとめて再取得する
    run_cost_usd, total_cost_usd, pending_count = await _reconcile_generation_costs(
        client=client,
        generation_log_path=generation_log_path,
        cost_cache_path=cost_cache_path,
        run_generation_ids=new_generation_ids_run,
        max_concurrency=min(config.max_concurrency * 2, 20),
    )
    summary.total_cost_usd += run_cost_usd
    summary.finish()
    summary_text = summary.format_text()
    logger.info("処理が完了しました。対象行: %s", len(work_items))
    logger.info("\n%s", summary_text)
    logger.info(
        "コスト再取得: run=$%.6f total=$%.6f pending=%s",
        run_cost_usd,
        total_cost_usd,
        pending_count,
    )
    print(summary_text)


def run(config: AppConfig, api_key: str) -> None:
    """`asyncio.run` を用いてランナーを同期的に起動する。

    Args:
        config: 実行に関するアプリ設定。
        api_key: OpenRouter API キー。
    """

    try:
        asyncio.run(run_async(config, api_key))
    except KeyboardInterrupt:
        logger.warning("ユーザーにより処理が中断されました。")
    except Exception:
        logger.exception("処理中に致命的なエラーが発生しました。")
        raise


def _classify_columns(df: pd.DataFrame) -> ColumnClassification:
    """DataFrame の列を新プレフィックス体系で分類し、妥当性を検証する。

    Returns:
        ColumnClassification: 列分類の結果。
    """

    input_columns: list[str] = []
    structured_columns: list[str] = []
    ns_columns: list[str] = []
    nsof_index_map: dict[str, int] = {}
    nsof_counter = 0
    label_columns: list[str] = []
    embedding_columns: list[str] = []
    embedding_spec_map: dict[str, dict[str, Any]] = {}
    embedding_index_map: dict[str, int] = {}
    embedding_counter = 0
    errors: list[str] = []

    for col in df.columns:
        if not isinstance(col, str):
            errors.append("列名が文字列ではありません。")
            continue

        lower = col.lower()

        if lower.startswith("i_"):
            input_columns.append(col)
            continue

        if lower.startswith("l_"):
            label_columns.append(col)
            continue

        if lower.startswith("so_"):
            structured_columns.append(col)
            continue

        if lower.startswith("nsof_"):
            ns_columns.append(col)
            nsof_counter += 1
            nsof_index_map[col] = nsof_counter
            continue

        if lower.startswith("nso_"):
            ns_columns.append(col)
            continue

        if lower.startswith("eo"):
            try:
                embedding_spec_map[col] = _parse_embedding_column(col)
                embedding_columns.append(col)
                embedding_counter += 1
                embedding_index_map[col] = embedding_counter
            except ValueError as exc:  # noqa: BLE001
                errors.append(str(exc))
            continue

        errors.append(
            f"未知の列プレフィックスです: {col}（ラベル列にする場合は l_ を付与してください）"
        )

    if errors:
        raise ValueError("; ".join(errors))

    if not input_columns:
        raise ValueError(
            "入力列 i_ が1つも存在しません。少なくとも1列の i_ を用意してください。"
        )

    return ColumnClassification(
        input_columns=input_columns,
        structured_columns=structured_columns,
        nonstructured_columns=ns_columns,
        nsof_index_map=nsof_index_map,
        label_columns=label_columns,
        embedding_columns=embedding_columns,
        embedding_spec_map=embedding_spec_map,
        embedding_index_map=embedding_index_map,
    )


def _strip_prefix_case_insensitive(text: str, prefix: str) -> str:
    """大小文字を無視して prefix を取り除く。"""

    if text.lower().startswith(prefix.lower()):
        return text[len(prefix) :]
    return text


def _structured_schema_header(column: str) -> str:
    """so_ を除去し、説明を保持したまま Schema 用ヘッダに変換する。"""

    if "|" in column:
        name, desc = column.split("|", 1)
        stripped_name = _strip_prefix_case_insensitive(name, "so_")
        return f"{stripped_name}|{desc}"
    return _strip_prefix_case_insensitive(column, "so_")


def _validate_structured_property_names(columns: Iterable[str]) -> None:
    """構造化列のプロパティ名衝突を検知する。"""

    seen: set[str] = set()
    for col in columns:
        prop = _column_to_property_name(col).lower()
        if prop in seen:
            raise ValueError(f"構造化出力のプロパティ名が重複しています: {prop}")
        seen.add(prop)


def _parse_embedding_column(column: str) -> dict[str, Any]:
    """eo* 列名をパースし、前段/後段の次元指定とファイル出力を返す。"""

    pattern = re.compile(
        r"^eo"
        r"(?:(?P<pre_method>n)(?P<pre_dim>\d+)?)?"
        r"(?:(?P<post_method>[ptu])(?P<post_dim>\d+))?"
        r"(?P<fileflag>f)?$",
        re.IGNORECASE,
    )
    match = pattern.match(column)
    if not match:
        raise ValueError(
            f"埋め込み列の形式が不正です: {column} "
            "(例: eo / eof / eon1536 / eop128 / eon1536p128f)"
        )

    pre_method = match.group("pre_method")
    pre_dim = match.group("pre_dim")
    post_method = match.group("post_method")
    post_dim = match.group("post_dim")
    file_output = match.group("fileflag") is not None

    if pre_dim is not None and int(pre_dim) <= 0:
        raise ValueError(f"埋め込み列の次元数が不正です: {column}")
    if post_dim is not None and int(post_dim) <= 0:
        raise ValueError(f"埋め込み列の次元数が不正です: {column}")
    if post_method and not post_dim:
        raise ValueError(f"埋め込み列の後段次元指定が不正です: {column}")

    return {
        "pre_method": pre_method.lower() if pre_method else None,
        "pre_dim": int(pre_dim) if pre_dim else None,
        "post_method": post_method.lower() if post_method else None,
        "post_dim": int(post_dim) if post_dim else None,
        "file_output": file_output,
    }


def _collect_work_items(
    df: pd.DataFrame,
    *,
    structured_columns: Sequence[str],
    ns_columns: Sequence[str],
    embedding_columns: Sequence[str],
    start_row_index: int,
    max_rows: int | None,
) -> list[RowWorkItem]:
    """未入力セルが残る行の一覧を組み立てる。

    Args:
        df: スプレッドシート全体の DataFrame。
        structured_columns: 構造化出力列名のシーケンス。
        ns_columns: 非構造化出力列名のシーケンス。
        embedding_columns: 埋め込み列名のシーケンス。
        start_row_index: 探索を開始する行インデックス（再開用）。
        max_rows: この実行で処理する最大行数。制限なしは ``None``。

    Returns:
        list[RowWorkItem]: まだ処理が必要な行と未入力列のペア。
    """

    work_items: list[RowWorkItem] = []
    for row_index in range(start_row_index, len(df)):
        row = df.iloc[row_index]
        pending_structured = _get_pending_columns(row, structured_columns)
        pending_ns = _get_pending_columns(row, ns_columns)
        pending_embedding = _get_pending_columns(row, embedding_columns)
        if pending_structured or pending_ns or pending_embedding:
            work_items.append(
                RowWorkItem(
                    row_index=row_index,
                    pending_structured_columns=pending_structured,
                    pending_ns_columns=pending_ns,
                    pending_embedding_columns=pending_embedding,
                )
            )
            if max_rows is not None and len(work_items) >= max_rows:
                break
    return work_items


def _get_pending_columns(row: pd.Series, output_columns: Sequence[str]) -> list[str]:
    """行の中で空欄になっている出力列のみを抽出する。

    Args:
        row: 対象行の Pandas Series。
        output_columns: 出力列名のシーケンス。

    Returns:
        list[str]: まだ値が入っていない列名のリスト。
    """

    pending: list[str] = []
    for column in output_columns:
        value = row.get(column)
        if _is_empty_value(value):
            pending.append(column)
    return pending


def _is_empty_value(value: Any) -> bool:
    """セルの値が「空」と見なせるかを判定する。

    Args:
        value: 判定対象の値。

    Returns:
        bool: 空・NaN・空文字列なら ``True``、それ以外は ``False``。
    """

    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return bool(pd.isna(value))


def _build_targets(
    row: pd.Series,
    target_columns: Sequence[str],
    base_dir: Path,
    preprocessors: Sequence[Preprocess] | None = None,
) -> list[Target]:
    """ターゲット列を前処理して Target オブジェクトを作成する。

    Args:
        row: 対象行の Pandas Series。
        target_columns: `t_` で始まるターゲット列名のシーケンス。
        base_dir: 相対パスを解決するための基準ディレクトリ。
        preprocessors: 使用する前処理クラスのリスト。None の場合はデフォルトを使用。

    Returns:
        list[Target]: 前処理済みターゲットのリスト。

    Raises:
        Exception: 前処理が失敗した場合、その例外をそのまま送出。
    """

    targets: list[Target] = []
    for column in target_columns:
        value = row.get(column)
        if _is_empty_value(value):
            continue
        try:
            target = run_preprocess_pipeline(
                str(value), base_dir, preprocessors=preprocessors
            )
            targets.append(target)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ターゲット列 %s の前処理に失敗しました: %s", column, exc)
            raise
    return targets


def _build_updates_from_parsed(
    parsed: dict[str, Any],
    pending_columns: Iterable[str],
) -> dict[str, Any]:
    """LLM から返った JSON を DataFrame の列へマッピングする。

    Args:
        parsed: LLM 応答の JSON オブジェクト。
        pending_columns: 今回書き込むべき出力列。

    Returns:
        dict[str, Any]: DataFrame に代入可能な列名と値の辞書。
    """

    updates: dict[str, Any] = {}
    for column in pending_columns:
        property_name = _column_to_property_name(column)
        value = parsed.get(property_name, "")
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        elif value is None:
            value = ""
        updates[column] = value
    return updates


def _column_to_property_name(column: str) -> str:
    """列ヘッダから JSON プロパティ名を抽出する。

    Args:
        column: ``列名`` または ``列名|説明`` 形式のヘッダ文字列（so_ プレフィックスを含む）。

    Returns:
        str: JSON Schema で利用するプロパティ名。
    """

    if "|" in column:
        name, _ = column.split("|", 1)
    else:
        name = column
    name = _strip_prefix_case_insensitive(name, "so_")
    return name.strip()


def _extract_parsed_json(response: Any) -> dict[str, Any]:
    """OpenAI SDK のレスポンスから JSON を取り出す。

    Args:
        response: ``create_completion`` が返すレスポンスオブジェクト。

    Returns:
        dict[str, Any]: パース済みの JSON オブジェクト。

    Raises:
        ValueError: JSON を取得できない場合。
    """

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
    """SDK 固有の usage オブジェクトを辞書に変換する。

    Args:
        usage: OpenAI/OpenRouter が返した usage 情報。

    Returns:
        dict[str, Any] | None: 変換に成功した場合は辞書、失敗時は ``None``。
    """

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
    """usage 辞書からコスト系キーを除外する。

    OpenRouter の /generation で後から確定させるため、LLM 応答直後の
    total_cost* はここで捨てる。
    """

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


def _build_embedding_text(targets: Sequence[Target]) -> str:
    """埋め込み用の入力テキストを構築する。"""

    text_parts = [target.text for target in targets if isinstance(target, TextTarget)]
    if not text_parts:
        raise ValueError("埋め込み用のテキストが存在しません。")
    return "\n\n".join(text_parts)


def _extract_embedding_vector(response: Any) -> list[float]:
    """埋め込み API のレスポンスからベクトルを取り出す。"""

    data = getattr(response, "data", None)
    if not data:
        raise ValueError("埋め込み応答に data が含まれていません。")
    first = data[0]
    embedding = getattr(first, "embedding", None)
    if embedding is None and isinstance(first, dict):
        embedding = first.get("embedding")
    if embedding is None:
        raise ValueError("埋め込み応答に embedding が含まれていません。")
    return list(embedding)


def _serialize_embedding_vector(vector: Sequence[float]) -> str:
    """埋め込みベクトルをセル書き込み用にシリアライズする。"""
    # テキストを小さくするため、カンマとコロンのみを使用(不要なスペースを削除)
    return json.dumps(list(vector), ensure_ascii=False, separators=(",", ":"))


def _write_embedding_file(path: Path, vector: Sequence[float]) -> None:
    """埋め込みベクトルをファイルへ保存する。"""

    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(vector, dtype="float32")
    np.save(path, array)


def _build_eof_filename(*, embedding_index: int, row_no: int) -> str:
    """埋め込みベクトル出力ファイルの名前を生成する。"""

    return f"eof{embedding_index}_{row_no}.{EMBEDDING_FILE_EXT}"


def _prepare_embedding_output(
    vector: Sequence[float],
    *,
    output_dir: Path,
    row_no: int,
    embedding_index: int,
    force_file: bool,
) -> str:
    """埋め込みベクトルをセルまたはファイルへ出力する。"""

    serialized = _serialize_embedding_vector(vector)
    if not force_file and len(serialized) <= EMBEDDING_CELL_CHAR_LIMIT:
        return serialized

    filename = _build_eof_filename(embedding_index=embedding_index, row_no=row_no)
    output_path = (output_dir / filename).resolve()
    _write_embedding_file(output_path, vector)
    if not force_file:
        logger.info(
            "埋め込み出力が長すぎるためファイル出力にフォールバックしました: %s",
            filename,
        )
    return filename


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


def _extract_text_content(response: Any) -> str:
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


def _apply_embedding_reductions(
    *,
    df: pd.DataFrame,
    embedding_vectors: dict[str, dict[int, list[float]]],
    embedding_spec_map: dict[str, dict[str, Any]],
    embedding_index_map: dict[str, int],
    output_dir: Path,
) -> None:
    """後段の次元削減を実行して DataFrame を更新する。
    
    Args:
        df: DataFrame
        embedding_vectors: 後処理用の埋め込みベクトル ( {列名: {行インデックス: 埋め込みベクトル}} )
        embedding_spec_map: 埋め込み列の仕様 ( {列名: {post_method: 後処理方法, post_dim: 後処理次元, file_output: ファイル出力フラグ}} )
        embedding_index_map: 埋め込み列に対する 1 始まりのインデックス
        output_dir: 出力ディレクトリ
    """

    for column, row_vectors in embedding_vectors.items():
        spec = embedding_spec_map.get(column, {})
        post_method = spec.get("post_method")
        post_dim = spec.get("post_dim")
        if not post_method or not post_dim:
            continue

        row_indices = sorted(row_vectors.keys())
        vectors = [row_vectors[row_index] for row_index in row_indices] # forでその列の埋め込みベクトルをすべて取得
        try:
            # ---後処理(次元削減)を実行する
            reduced_vectors = _reduce_embeddings(
                vectors, method=str(post_method), dim=int(post_dim)
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "埋め込みの後段次元削減に失敗しました: %s (%s)", column, exc
            )
            continue

        # ---後処理後の埋め込みベクトルを、セルまたはファイルに出力する
        for offset, row_index in enumerate(row_indices):
            output_value = _prepare_embedding_output(
                reduced_vectors[offset], # row_indexは行番号なだけ。順番にはなっていない可能性があるので、offsetを使う
                output_dir=output_dir,
                row_no=row_index + 1,
                embedding_index=embedding_index_map.get(column, 1),
                force_file=bool(spec.get("file_output")),
            )
            df.loc[row_index, column] = output_value # 後処理後の埋め込みベクトルを、DataFrameに書き込む


def _reduce_embeddings(
    vectors: Sequence[Sequence[float]],
    *,
    method: str,
    dim: int,
) -> list[list[float]]:
    """埋め込みベクトルを指定の手法で次元削減する。"""

    import numpy as np

    # ---前確認
    if dim <= 0:
        raise ValueError("後段次元数が不正です。")
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("埋め込みベクトルが空です。")
    n_samples, n_features = matrix.shape
    if dim > n_features:
        raise ValueError("後段次元数が元の次元数を超えています。")

    method = method.lower()
    if method == "p": # PCA
        if n_samples < 2:
            logger.warning("PCA を実行できないため先頭切り出しで代替します。")
            return _truncate_embeddings(matrix, dim)
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=dim, random_state=0)
        reduced = reducer.fit_transform(matrix)
        return reduced.astype("float32").tolist()
    if method == "t": # t-SNE
        if n_samples < 2:
            logger.warning("t-SNE を実行できないため先頭切り出しで代替します。")
            return _truncate_embeddings(matrix, dim)
        from sklearn.manifold import TSNE

        perplexity = min(30, max(1, n_samples - 1))
        reducer = TSNE(
            n_components=dim,
            perplexity=perplexity,
            random_state=0,
            init="pca",
        )
        reduced = reducer.fit_transform(matrix)
        return reduced.astype("float32").tolist()
    if method == "u": # UMAP
        if n_samples < 3:
            logger.warning("UMAP を実行できないため先頭切り出しで代替します。")
            return _truncate_embeddings(matrix, dim)
        import umap

        n_neighbors = min(15, n_samples - 1)
        reducer = umap.UMAP(
            n_components=dim,
            n_neighbors=n_neighbors,
            random_state=0,
        )
        reduced = reducer.fit_transform(matrix)
        return reduced.astype("float32").tolist()

    raise ValueError(f"未対応の後段次元削減方式です: {method}")


def _truncate_embeddings(matrix: np.ndarray, dim: int) -> list[list[float]]:
    """埋め込みを先頭から切り出して指定次元に合わせる。"""

    if dim > matrix.shape[1]:
        raise ValueError("後段次元数が元の次元数を超えています。")
    return matrix[:, :dim].astype("float32").tolist()


def _guess_target_file_stem(
    row: pd.Series, target_columns: Sequence[str], base_dir: Path
) -> str | None:
    """i_ 列が 1 つだけで、ファイルパスなら stem を返す。nsof の出力ファイル名の判定に用いる"""

    if len(target_columns) != 1:
        return None
    raw = row.get(target_columns[0])
    if _is_empty_value(raw):
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
    """nsof 用のファイル名を生成する。

    Args:
        nsf_index: nsf_index
        row_no: 行番号
        ext: 拡張子
        target_file_stem: i_列のファイル名のstem
        use_filetarget_template: filenameに、i_列のファイル名のstemを使用するかどうか({ファイル名}_nsf{nsf_index}.{ext}にするか？)
        name_template: ファイル名のテンプレート
        name_template_filetarget: i_列のファイル名のstemを使用する場合のファイル名のテンプレート
    """

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

    sanitized = re.sub(r'[<>:"/\\|?*\r\n]', "_", name).strip()
    return sanitized or "output"


# ---generation / cost 取得の補助関数群


def _append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """JSONL 形式でレコードを追記する。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONL ファイルを読み込む。"""

    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _unique_preserve(seq: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


async def _reconcile_generation_costs(
    *,
    client: AsyncOpenRouterClient,
    generation_log_path: Path,
    cost_cache_path: Path,
    run_generation_ids: Sequence[str],
    max_concurrency: int = 10,
    max_rounds: int = 4,
    initial_delay: float = 0.5,
) -> tuple[float, float, int]:
    """generation_id に対するコストをまとめて再取得する。

    Returns:
        (run_cost_usd, total_cost_usd, pending_count)
    """

    # ---生成のログjsonlを読み込み、generation_idを取得する
    all_generation_records = _read_jsonl(generation_log_path)
    generation_ids_all = _unique_preserve(
        [rec.get("id", "") for rec in all_generation_records if rec.get("id")]
    )
    if not generation_ids_all:
        return 0.0, 0.0, 0

    # ---cost_cache_pathのjsonlを読み込み、キャッシュからgeneration_idとcostを取得する
    cost_records = _read_jsonl(cost_cache_path)
    cache: dict[str, dict[str, Any]] = {
        rec["id"]: rec
        for rec in cost_records
        if isinstance(rec, dict) and rec.get("id")
    }

    # ---キャッシュにないgeneration_idをpending_idsに追加する
    pending_ids: list[str] = []
    for gen_id in generation_ids_all:
        rec = cache.get(gen_id)
        if rec is None or rec.get("status") != "ok":
            pending_ids.append(gen_id)

    # ---すべてコスト取得済みの場合、最終的なコストを計算・返却する
    if not pending_ids:
        total_cost = sum(
            float(cache[gid].get("total_cost", 0.0) or 0.0) for gid in cache
        )
        run_cost = sum(
            float(cache.get(gid, {}).get("total_cost", 0.0) or 0.0)
            for gid in _unique_preserve(run_generation_ids)
            if cache.get(gid, {}).get("status") == "ok"
        )
        pending_count = len(
            [
                gid
                for gid in generation_ids_all
                if cache.get(gid, {}).get("status") != "ok"
            ]
        )
        return run_cost, total_cost, pending_count

    # ---コスト取得を並列実行する
    semaphore = asyncio.Semaphore(max_concurrency)  # セマフォで並列数を制限する

    async def _fetch(gen_id: str) -> tuple[str, float | None]:
        async with semaphore:
            cost = await client.fetch_generation_cost(gen_id)
            return gen_id, cost

    current_pending = pending_ids
    for round_index in range(max_rounds):
        results = await asyncio.gather(*[_fetch(gen_id) for gen_id in current_pending])
        next_pending: list[str] = []
        for gen_id, cost in results:
            if cost is None or (cost == 0 and round_index < max_rounds - 1):
                cache[gen_id] = {"id": gen_id, "total_cost": cost, "status": "pending"}
                next_pending.append(gen_id)
                continue
            cache[gen_id] = {"id": gen_id, "total_cost": cost or 0.0, "status": "ok"}
        if not next_pending:
            break
        current_pending = next_pending
        await asyncio.sleep(initial_delay * (2**round_index))

    # 書き戻す（generation_log の順序を優先）
    ordered_records: list[dict[str, Any]] = []
    for gen_id in generation_ids_all:
        rec = cache.get(gen_id, {"id": gen_id, "status": "pending", "total_cost": None})
        ordered_records.append(rec)
    cost_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cost_cache_path.open("w", encoding="utf-8") as f:  # キャッシュを更新する
        for rec in ordered_records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")

    # ---最終的なコストを計算・返却する
    total_cost = sum(
        float(rec.get("total_cost", 0.0) or 0.0)
        for rec in ordered_records
        if rec.get("status") == "ok"
    )
    run_cost = sum(
        float(cache.get(gid, {}).get("total_cost", 0.0) or 0.0)
        for gid in _unique_preserve(run_generation_ids)
        if cache.get(gid, {}).get("status") == "ok"
    )
    pending_count = len([rec for rec in ordered_records if rec.get("status") != "ok"])
    return run_cost, total_cost, pending_count
