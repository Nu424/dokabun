from dataclasses import dataclass

import pandas as pd
import pytest

from dokabun.config import AppConfig
from dokabun.core.runner import RowWorkItem, _classify_columns, process_row_async


@dataclass
class _FakeMessage:
    parsed: dict
    content: str | None = None


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

    async def create_completion_text(self, **_: object) -> _FakeResponse:
        return _FakeResponse(
            choices=[_FakeChoice(_FakeMessage(parsed={}, content="non structured response"))],
            usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            id="gen-test-2",
        )

    async def fetch_generation_cost(self, generation_id: str) -> float | None:
        assert generation_id in {"gen-test-1", "gen-test-2"}
        return self._total_cost


@pytest.mark.asyncio
async def test_process_row_async_returns_updates(tmp_path):
    df = pd.DataFrame({"i_content": ["今日は晴れです"], "so_summary|本文の要約": [None]})
    work_item = RowWorkItem(
        row_index=0,
        pending_structured_columns=["so_summary|本文の要約"],
        pending_ns_columns=[],
    )
    config = AppConfig.from_dict({"input_path": tmp_path / "dummy.xlsx", "output_dir": tmp_path})
    client = _FakeClient({"summary": "短い要約"})

    result = await process_row_async(
        work_item,
        df,
        target_columns=["i_content"],
        base_dir=tmp_path,
        client=client,
        config=config,
        nsof_index_map={},
    )

    assert result.error is None
    assert result.updates["so_summary|本文の要約"] == "短い要約"
    assert result.usage["total_tokens"] == 8
    assert result.generation_ids == ["gen-test-1"]


@pytest.mark.asyncio
async def test_process_row_async_handles_ns_and_nsf(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    df = pd.DataFrame(
        {
            "i_content": ["hello"],
            "nso_summary": [None],
            "nsof_detail": [None],
        }
    )
    work_item = RowWorkItem(
        row_index=0,
        pending_structured_columns=[],
        pending_ns_columns=["nso_summary", "nsof_detail"],
    )
    config = AppConfig.from_dict({"input_path": tmp_path / "dummy.xlsx", "output_dir": output_dir})
    client = _FakeClient({"summary": "短い要約"})

    result = await process_row_async(
        work_item,
        df,
        target_columns=["i_content"],
        base_dir=tmp_path,
        client=client,
        config=config,
        nsof_index_map={"nsof_detail": 1},
    )

    assert "nso_summary" in result.updates
    assert result.updates["nso_summary"] == "non structured response"
    saved_path = output_dir / result.updates["nsof_detail"]
    assert saved_path.exists()
    assert saved_path.read_text(encoding="utf-8") == "non structured response"
    assert result.generation_ids == ["gen-test-1", "gen-test-2"]


def test_classify_columns_accepts_new_prefix_and_case():
    df = pd.DataFrame(
        {
            "I_content": ["x"],
            "SO_summary": [None],
            "nso_title": [None],
            "NSOF_detail": [None],
            "l_id": [123],
            "EON1536F": [None],
        }
    )

    classification = _classify_columns(df)

    assert classification.input_columns == ["I_content"]
    assert classification.structured_columns == ["SO_summary"]
    assert classification.nonstructured_columns == ["nso_title", "NSOF_detail"]
    assert classification.nsof_index_map == {"NSOF_detail": 1}
    assert classification.label_columns == ["l_id"]
    assert classification.embedding_columns == ["EON1536F"]
    assert classification.embedding_spec_map["EON1536F"]["dim"] == 1536
    assert classification.embedding_spec_map["EON1536F"]["file_output"] is True


def test_classify_columns_rejects_old_prefix():
    df = pd.DataFrame({"t_content": ["x"], "so_summary": [None]})
    with pytest.raises(ValueError) as excinfo:
        _classify_columns(df)
    assert "t_" in str(excinfo.value)


def test_classify_columns_rejects_config_prefix():
    df = pd.DataFrame({"i_content": ["x"], "c_temperature": [0.2], "so_summary": [None]})
    with pytest.raises(ValueError) as excinfo:
        _classify_columns(df)
    assert "config列" in str(excinfo.value)
