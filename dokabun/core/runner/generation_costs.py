"""Generation cost reconciliation helpers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from dokabun.llm.openrouter_client import AsyncOpenRouterClient


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """JSONL 形式でレコードを追記する。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def unique_preserve(seq: Iterable[str]) -> list[str]:
    """順序を維持したまま重複を取り除く。"""

    seen: set[str] = set()
    ordered: list[str] = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


async def reconcile_generation_costs(
    *,
    client: AsyncOpenRouterClient,
    generation_log_path: Path,
    cost_cache_path: Path,
    run_generation_ids: Sequence[str],
    max_concurrency: int = 10,
    max_rounds: int = 4,
    initial_delay: float = 0.5,
) -> tuple[float, float, int]:
    """generation_id に対するコストをまとめて再取得する。"""

    # ---生成のログjsonlを読み込み、generation_idを取得する
    all_generation_records = read_jsonl(generation_log_path)
    generation_ids_all = unique_preserve(
        [rec.get("id", "") for rec in all_generation_records if rec.get("id")]
    )
    if not generation_ids_all:
        return 0.0, 0.0, 0

    # ---cost_cache_pathのjsonlを読み込み、キャッシュからgeneration_idとcostを取得する
    cost_records = read_jsonl(cost_cache_path)
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
            for gid in unique_preserve(run_generation_ids)
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
    semaphore = asyncio.Semaphore(max_concurrency)

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
    with cost_cache_path.open("w", encoding="utf-8") as f:
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
        for gid in unique_preserve(run_generation_ids)
        if cache.get(gid, {}).get("status") == "ok"
    )
    pending_count = len([rec for rec in ordered_records if rec.get("status") != "ok"])
    return run_cost, total_cost, pending_count
