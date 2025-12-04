import asyncio
from dataclasses import dataclass

import pandas as pd
import pytest

from dokabun.config import AppConfig
from dokabun.core.runner import RowWorkItem, process_row_async


@dataclass
class _FakeMessage:
    parsed: dict


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]
    usage: dict
    id: str


class _FakeClient:
    def __init__(self, parsed: dict, total_cost: float | None = 0.001):
        self._parsed = parsed
        self._total_cost = total_cost

    async def create_completion(self, **_: object) -> _FakeResponse:
        return _FakeResponse(
            choices=[_FakeChoice(_FakeMessage(parsed=self._parsed))],
            usage={"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
            id="gen-test-1",
        )

    async def fetch_generation_cost(self, generation_id: str) -> float | None:
        assert generation_id == "gen-test-1"
        return self._total_cost


@pytest.mark.asyncio
async def test_process_row_async_returns_updates(tmp_path):
    df = pd.DataFrame({"t_content": ["今日は晴れです"], "summary|本文の要約": [None]})
    work_item = RowWorkItem(row_index=0, pending_columns=["summary|本文の要約"])
    config = AppConfig.from_dict({"input_path": tmp_path / "dummy.xlsx", "output_dir": tmp_path})
    client = _FakeClient({"summary": "短い要約"})

    result = await process_row_async(
        work_item,
        df,
        target_columns=["t_content"],
        base_dir=tmp_path,
        client=client,
        config=config,
    )

    assert result.error is None
    assert result.updates["summary|本文の要約"] == "短い要約"
    assert result.usage["total_tokens"] == 8
    assert result.usage["total_cost_usd"] == pytest.approx(0.001)

