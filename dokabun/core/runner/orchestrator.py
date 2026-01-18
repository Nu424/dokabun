"""Runner orchestration and concurrency control."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from dokabun.config import AppConfig
from dokabun.core.runner.columns import classify_columns
from dokabun.core.runner.embedding import apply_embedding_reductions
from dokabun.core.runner.generation_costs import (
    append_jsonl,
    reconcile_generation_costs,
)
from dokabun.core.runner.models import RowResult, RowWorkItem, RunState
from dokabun.core.runner.row_processor import RowProcessor
from dokabun.core.runner.work_items import collect_work_items
from dokabun.core.summary import ExecutionSummary
from dokabun.io.spreadsheet import SpreadsheetReaderWriter
from dokabun.llm.openrouter_client import AsyncOpenRouterClient
from dokabun.logging_utils import get_logger

logger = get_logger(__name__)


async def run_async(config: AppConfig, api_key: str) -> None:
    """スプレッドシート全体を非同期で処理する。"""

    logger.info(
        "処理を開始します: input=%s model=%s base_url=%s concurrency=%s",
        config.input_path,
        config.model,
        config.base_url,
        config.max_concurrency,
    )

    # ---スプレッドシートを読み込む
    reader = SpreadsheetReaderWriter(config)
    meta = reader.load_meta_if_exists() or {}
    resume_cursor = _load_resume_cursor(meta)
    start_row_index = max(resume_cursor + 1, 0)

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
    classification = classify_columns(df)
    work_items = collect_work_items(
        df,
        structured_columns=classification.structured_columns,
        ns_columns=classification.nonstructured_columns,
        embedding_columns=classification.embedding_columns,
        start_row_index=start_row_index,
        max_rows=config.max_rows,
    )

    # ---LLM処理(並列)を準備する
    client = AsyncOpenRouterClient(
        api_key=api_key,
        model=config.model,
        base_url=config.base_url,
    )
    summary = ExecutionSummary()
    summary.start(total_rows=len(work_items))
    processor = RowProcessor(
        client=client,
        config=config,
        target_columns=classification.input_columns,
        base_dir=reader.target_base_dir,
        nsof_index_map=classification.nsof_index_map,
        embedding_spec_map=classification.embedding_spec_map,
        embedding_index_map=classification.embedding_index_map,
    )

    # 処理対象が無い場合でも generation のコストだけ再計算する
    if not work_items:
        logger.info("処理対象の行が存在しません。コスト再取得のみ実行します。")
        run_cost_usd, total_cost_usd, pending_count = await reconcile_generation_costs(
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

    # ---処理対象を並列処理するためのキューを用意する
    work_queue: asyncio.Queue[object] = asyncio.Queue()
    result_queue: asyncio.Queue[RowResult] = asyncio.Queue()
    total_items = len(work_items)
    for item in work_items:
        work_queue.put_nowait(item)
    for _ in range(config.max_concurrency):
        work_queue.put_nowait(None)

    # ---処理対象を並列処理するためのワーカーを用意する
    async def _worker() -> None:
        while True:
            item = await work_queue.get()
            try:
                if item is None:
                    return
                assert isinstance(item, RowWorkItem)
                try:
                    result = await processor.process(item, df)
                except Exception as exc:  # noqa: BLE001
                    result = RowResult(
                        row_index=item.row_index,
                        updates={},
                        usage=None,
                        error=str(exc),
                        error_type=exc.__class__.__name__,
                        generation_ids=[],
                        embedding_vectors={},
                    )
                await result_queue.put(result)
            finally:
                work_queue.task_done()

    # ---処理を並列実行する
    workers = [asyncio.create_task(_worker()) for _ in range(config.max_concurrency)]

    processed = 0
    chunk_first_row: int | None = None
    chunk_last_row: int | None = None
    new_generation_ids_run: list[str] = []
    embedding_reduction_inputs: dict[str, dict[int, list[float]]] = {}
    run_state = RunState(resume_cursor=resume_cursor)

    # ---処理結果を取得する
    for _ in tqdm(
        range(total_items),
        total=total_items,
        desc="Processing rows",
        unit="row",
    ):
        result = await result_queue.get()
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
            append_jsonl(
                generation_log_path, [{"id": gid} for gid in result.generation_ids]
            )
            new_generation_ids_run.extend(result.generation_ids)

        # ---後処理対象の埋め込みベクトルを、embedding_reduction_inputs に追加する
        if result.embedding_vectors:
            for column, vector in result.embedding_vectors.items():
                embedding_reduction_inputs.setdefault(column, {})[result.row_index] = (
                    vector
                )

        run_state.mark_completed(result.row_index)

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
            _save_partial_with_resume(
                reader,
                df,
                chunk_first_row,
                chunk_last_row,
                run_state,
            )
            chunk_first_row = None
            chunk_last_row = None

    await work_queue.join()
    await asyncio.gather(*workers, return_exceptions=True)

    # ---後処理対象の埋め込みベクトルを、実際に後処理し、DFを更新する
    if embedding_reduction_inputs:
        apply_embedding_reductions(
            df=df,
            embedding_vectors=embedding_reduction_inputs,
            embedding_spec_map=classification.embedding_spec_map,
            embedding_index_map=classification.embedding_index_map,
            output_dir=reader.output_dir,
        )

    if chunk_first_row is not None and chunk_last_row is not None:
        _save_partial_with_resume(
            reader,
            df,
            chunk_first_row,
            chunk_last_row,
            run_state,
        )

    # ---最終版を保存する
    reader.save_output(df)
    reader.save_meta(_build_resume_meta(run_state))

    # ---コストを末尾でまとめて再取得する
    run_cost_usd, total_cost_usd, pending_count = await reconcile_generation_costs(
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
    """`asyncio.run` を用いてランナーを同期的に起動する。"""

    try:
        asyncio.run(run_async(config, api_key))
    except KeyboardInterrupt:
        logger.warning("ユーザーにより処理が中断されました。")
    except Exception:
        logger.exception("処理中に致命的なエラーが発生しました。")
        raise


def _load_resume_cursor(meta: dict[str, Any] | None) -> int:
    if not meta:
        return -1
    for key in ("resume_cursor", "last_completed_row"):
        value = meta.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            break
    return -1


def _build_resume_meta(run_state: RunState) -> dict[str, int]:
    return {
        "resume_cursor": run_state.resume_cursor,
        "last_completed_row": run_state.resume_cursor,
    }


def _save_partial_with_resume(
    reader: SpreadsheetReaderWriter,
    df: pd.DataFrame,
    start_row: int | None,
    end_row: int | None,
    run_state: RunState,
) -> Path | None:
    if start_row is None or end_row is None:
        return None
    return reader.save_partial(
        df,
        start_row,
        end_row,
        meta=_build_resume_meta(run_state),
    )
