"""スプレッドシートを処理して LLM で未入力セルを埋めるコアランナー。"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
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
from dokabun.target import Target

logger = get_logger(__name__)


@dataclass(slots=True)
class RowResult:
    """1 行分の処理結果を保持するデータクラス。

    Attributes:
        row_index: DataFrame 上の行インデックス。
        updates: シートへ書き戻す列名と値のマッピング。
        usage: LLM 応答から得た usage 情報（トークン数やコストなど）。
        error: エラー発生時のメッセージ。成功時は ``None``。
        error_type: エラー種別（例外名など）。成功時は ``None``。
    """

    row_index: int
    updates: dict[str, Any]
    usage: dict[str, Any] | None = None
    error: str | None = None
    error_type: str | None = None
    generation_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RowWorkItem:
    """処理待ち行に関するメタデータ。

    Attributes:
        row_index: DataFrame 上の行インデックス。
        pending_structured_columns: まだ値が入っていない構造化出力列。
        pending_ns_columns: まだ値が入っていない非構造化出力列。
    """

    row_index: int
    pending_structured_columns: list[str]
    pending_ns_columns: list[str]


async def process_row_async(
    work_item: RowWorkItem,
    df: pd.DataFrame,
    *,
    target_columns: Sequence[str],
    base_dir: Path,
    client: AsyncOpenRouterClient,
    config: AppConfig,
    nsf_index_map: dict[str, int],
    preprocessors: Sequence[Preprocess] | None = None,
) -> RowResult:
    """1 行分のスプレッドシートを非同期で処理する。

    Args:
        work_item: 行インデックスと未入力列の情報。
        df: スプレッドシート全体の DataFrame。
        target_columns: `t_` で始まるターゲット列の順序付きリスト。
        base_dir: 画像パスなどを解決する基準ディレクトリ。
        client: OpenRouter へ問い合わせる非同期クライアント。
        config: 温度や最大トークン数などを含むアプリ設定。
        nsf_index_map: nsf 列に対する 1 始まりのインデックス。

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

        # ---構造化出力（JSON Schema）が必要な場合のみ実行する
        if work_item.pending_structured_columns:
            # ---出力形式を、ヘッダーから生成する
            schema_payload = build_schema_from_headers(
                work_item.pending_structured_columns,
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
            usage = _strip_cost_keys(_coerce_usage_dict(getattr(response, "usage", None)))
            generation_id = getattr(response, "id", None)
            if generation_id:
                generation_ids.append(str(generation_id))
            # ---更新内容をビルドする
            updates.update(_build_updates_from_parsed(parsed, work_item.pending_structured_columns))
            usage_total = _merge_usage(usage_total, usage)

        # ---非構造化出力（ns_/nsf_）を列ごとに実行する
        if work_item.pending_ns_columns:
            row_no_1based = work_item.row_index + 1
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

                    # ---nsf_列の場合、出力結果をファイルに保存する
                    if ns_column.startswith("nsf_"):
                        # ---ファイル名を準備する
                        nsf_index = nsf_index_map.get(ns_column, 1)
                        filename = _build_nsf_filename(
                            nsf_index=nsf_index,
                            row_no=row_no_1based,
                            ext=config.nsf_ext,
                            target_file_stem=target_file_stem,
                            use_filetarget_template=target_file_stem is not None,
                            name_template=config.nsf_name_template,
                            name_template_filetarget=config.nsf_name_template_filetarget,
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

        error_msg = "; ".join(errors) if errors else None
        error_type = "PartialError" if errors else None
        return RowResult(
            row_index=work_item.row_index,
            updates=updates,
            usage=usage_total,
            error=error_msg,
            error_type=error_type,
            generation_ids=generation_ids,
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
    start_row_index = int(last_completed_row) + 1 if last_completed_row is not None else 0

    # ---generation_id 永続化用のパスを用意する
    generation_log_path = (
        reader.output_dir / f"{reader.base_stem}_{reader.timestamp}.generations.jsonl"
    )
    cost_cache_path = (
        reader.output_dir / f"{reader.base_stem}_{reader.timestamp}.generation_costs.jsonl"
    )
    generation_log_path.parent.mkdir(parents=True, exist_ok=True)
    generation_log_path.touch(exist_ok=True)
    cost_cache_path.parent.mkdir(parents=True, exist_ok=True)

    # ---処理対象・出力列をわけ、処理が必要な部分を抽出する
    df = reader.load()
    (
        target_columns,
        structured_columns,
        ns_columns,
        nsf_index_map,
    ) = _classify_columns(df)
    work_items = _collect_work_items(
        df,
        structured_columns=structured_columns,
        ns_columns=ns_columns,
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
    preprocessors = build_default_preprocessors(max_text_file_bytes=config.max_text_file_bytes)

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
                nsf_index_map=nsf_index_map,
                preprocessors=preprocessors,
            )

    # ---処理を並列実行する
    tasks = [asyncio.create_task(_bounded_process(item)) for item in work_items]
    processed = 0
    chunk_first_row: int | None = None
    chunk_last_row: int | None = None
    new_generation_ids_run: list[str] = []

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
            logger.error("行 %s の処理に失敗しました: %s", result.row_index, result.error)
            summary.record_failure(
                result.row_index,
                result.error_type or "RowError",
                result.error,
            )
        else:
            summary.record_success(result.row_index, result.usage)

        if result.generation_ids:
            _append_jsonl(generation_log_path, [{"id": gid} for gid in result.generation_ids])
            new_generation_ids_run.extend(result.generation_ids)

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


def _classify_columns(
    df: pd.DataFrame,
) -> tuple[list[str], list[str], list[str], dict[str, int]]:
    """DataFrame の列をターゲット列・構造化出力列・非構造化列に分類する。

    Args:
        df: スプレッドシートから読み込んだ DataFrame。

    Returns:
        tuple[list[str], list[str], list[str], dict[str, int]]:
            (ターゲット列, 構造化出力列, 非構造化列, nsf列のインデックスマップ)
    Notes:
        nsf列のインデックスマップは、以下のような形式である
        ```
        {
            "{nsfな列名}": {その列のインデックス(1始まり)},
            "nsf_2": 2,
            "nsf_3": 3,
            ...
        }
        ```
    """

    target_columns: list[str] = []
    structured_columns: list[str] = []
    ns_columns: list[str] = []
    nsf_index_map: dict[str, int] = {}
    nsf_counter = 0

    for col in df.columns:
        if not isinstance(col, str):
            continue
        if col.startswith("t_"):
            target_columns.append(col)
            continue
        if col.startswith("nsf_"):
            ns_columns.append(col)
            nsf_counter += 1
            nsf_index_map[col] = nsf_counter
            continue
        if col.startswith("ns_"):
            ns_columns.append(col)
            continue
        structured_columns.append(col)

    return target_columns, structured_columns, ns_columns, nsf_index_map


def _collect_work_items(
    df: pd.DataFrame,
    *,
    structured_columns: Sequence[str],
    ns_columns: Sequence[str],
    start_row_index: int,
    max_rows: int | None,
) -> list[RowWorkItem]:
    """未入力セルが残る行の一覧を組み立てる。

    Args:
        df: スプレッドシート全体の DataFrame。
        structured_columns: 構造化出力列名のシーケンス。
        ns_columns: 非構造化出力列名のシーケンス。
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
        if pending_structured or pending_ns:
            work_items.append(
                RowWorkItem(
                    row_index=row_index,
                    pending_structured_columns=pending_structured,
                    pending_ns_columns=pending_ns,
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
            target = run_preprocess_pipeline(str(value), base_dir, preprocessors=preprocessors)
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
        column: ``列名`` または ``列名|説明`` 形式のヘッダ文字列。

    Returns:
        str: JSON Schema で利用するプロパティ名。
    """

    if "|" not in column:
        return column.strip()
    name, _ = column.split("|", 1)
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


def _parse_ns_prompt(column: str) -> str:
    """ns_/nsf_ 列ヘッダーからプロンプト文を抽出する。"""

    stripped = column
    if stripped.startswith("nsf_"):
        stripped = stripped[len("nsf_") :]
    elif stripped.startswith("ns_"):
        stripped = stripped[len("ns_") :]
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


def _guess_target_file_stem(
    row: pd.Series, target_columns: Sequence[str], base_dir: Path
) -> str | None:
    """t_ 列が 1 つだけで、ファイルパスなら stem を返す。nsfの出力ファイル名の判定に用いる"""

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


def _build_nsf_filename(
    *,
    nsf_index: int,
    row_no: int,
    ext: str,
    target_file_stem: str | None,
    use_filetarget_template: bool,
    name_template: str,
    name_template_filetarget: str,
) -> str:
    """nsf 用のファイル名を生成する。
    
    Args:
        nsf_index: nsf_index
        row_no: 行番号
        ext: 拡張子
        target_file_stem: t_列のファイル名のstem
        use_filetarget_template: filenameに、t_列のファイル名のstemを使用するかどうか({ファイル名}_nsf{nsf_index}.{ext}にするか？)
        name_template: ファイル名のテンプレート
        name_template_filetarget: t_列のファイル名のstemを使用する場合のファイル名のテンプレート
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
        rec["id"]: rec for rec in cost_records if isinstance(rec, dict) and rec.get("id")
    }

    # ---キャッシュにないgeneration_idをpending_idsに追加する
    pending_ids: list[str] = []
    for gen_id in generation_ids_all:
        rec = cache.get(gen_id)
        if rec is None or rec.get("status") != "ok":
            pending_ids.append(gen_id)

    # ---すべてコスト取得済みの場合、最終的なコストを計算・返却する
    if not pending_ids:
        total_cost = sum(float(cache[gid].get("total_cost", 0.0) or 0.0) for gid in cache)
        run_cost = sum(
            float(cache.get(gid, {}).get("total_cost", 0.0) or 0.0)
            for gid in _unique_preserve(run_generation_ids)
            if cache.get(gid, {}).get("status") == "ok"
        )
        pending_count = len(
            [gid for gid in generation_ids_all if cache.get(gid, {}).get("status") != "ok"]
        )
        return run_cost, total_cost, pending_count

    # ---コスト取得を並列実行する
    semaphore = asyncio.Semaphore(max_concurrency) # セマフォで並列数を制限する

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
    with cost_cache_path.open("w", encoding="utf-8") as f: # キャッシュを更新する
        for rec in ordered_records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")

    # ---最終的なコストを計算・返却する
    total_cost = sum(
        float(rec.get("total_cost", 0.0) or 0.0) for rec in ordered_records if rec.get("status") == "ok"
    )
    run_cost = sum(
        float(cache.get(gid, {}).get("total_cost", 0.0) or 0.0)
        for gid in _unique_preserve(run_generation_ids)
        if cache.get(gid, {}).get("status") == "ok"
    )
    pending_count = len([rec for rec in ordered_records if rec.get("status") != "ok"])
    return run_cost, total_cost, pending_count
