from dataclasses import dataclass

import pandas as pd
import pytest

from dokabun.config import AppConfig
from dokabun.core.runner.columns import classify_columns
from dokabun.core.runner.models import RowWorkItem, RunState
from dokabun.core.runner.row_processor import RowProcessor


@dataclass
class _FakeMessage:
    parsed: dict | None = None
    content: str | None = None


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]
    usage: dict
    id: str


@dataclass
class _FakeEmbeddingData:
    embedding: list[float]


@dataclass
class _FakeEmbeddingResponse:
    data: list[_FakeEmbeddingData]
    usage: dict


class _FakeClient:
    def __init__(
        self,
        parsed: dict,
        *,
        text_content: str = "non structured response",
        embedding_vector: list[float] | None = None,
    ):
        self._parsed = parsed
        self._text_content = text_content
        self._embedding_vector = embedding_vector or [0.1, 0.2]
        self._text_calls = 0

    async def create_completion(self, **_: object) -> _FakeResponse:
        return _FakeResponse(
            choices=[_FakeChoice(_FakeMessage(parsed=self._parsed))],
            usage={"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
            id="gen-structured",
        )

    async def create_completion_text(self, **_: object) -> _FakeResponse:
        self._text_calls += 1
        return _FakeResponse(
            choices=[_FakeChoice(_FakeMessage(content=self._text_content))],
            usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            id=f"gen-ns-{self._text_calls}",
        )

    async def create_embedding(self, **_: object) -> _FakeEmbeddingResponse:
        return _FakeEmbeddingResponse(
            data=[_FakeEmbeddingData(self._embedding_vector)],
            usage={"total_tokens": 0, "cost": 0.001},
        )


@pytest.mark.asyncio
async def test_row_processor_structured_updates(tmp_path):
    df = pd.DataFrame({"i_content": ["今日は晴れです"], "so_summary|本文の要約": [None]})
    work_item = RowWorkItem(
        row_index=0,
        pending_structured_columns=["so_summary|本文の要約"],
        pending_ns_columns=[],
        pending_embedding_columns=[],
    )
    config = AppConfig.from_dict(
        {"input_path": tmp_path / "dummy.xlsx", "output_dir": tmp_path}
    )
    client = _FakeClient({"summary": "短い要約"})
    processor = RowProcessor(
        client=client,
        config=config,
        target_columns=["i_content"],
        base_dir=tmp_path,
        nsof_index_map={},
        embedding_spec_map={},
        embedding_index_map={},
    )

    result = await processor.process(work_item, df)

    assert result.error is None
    assert result.updates["so_summary|本文の要約"] == "短い要約"
    assert result.usage["total_tokens"] == 8
    assert result.generation_ids == ["gen-structured"]


@pytest.mark.asyncio
async def test_row_processor_handles_ns_and_nsof(tmp_path):
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
        pending_embedding_columns=[],
    )
    config = AppConfig.from_dict(
        {"input_path": tmp_path / "dummy.xlsx", "output_dir": output_dir}
    )
    client = _FakeClient({"summary": "短い要約"})
    processor = RowProcessor(
        client=client,
        config=config,
        target_columns=["i_content"],
        base_dir=tmp_path,
        nsof_index_map={"nsof_detail": 1},
        embedding_spec_map={},
        embedding_index_map={},
    )

    result = await processor.process(work_item, df)

    assert result.updates["nso_summary"] == "non structured response"
    saved_path = output_dir / result.updates["nsof_detail"]
    assert saved_path.exists()
    assert saved_path.read_text(encoding="utf-8") == "non structured response"
    assert result.generation_ids == ["gen-ns-1", "gen-ns-2"]


@pytest.mark.asyncio
async def test_row_processor_handles_embedding(tmp_path):
    df = pd.DataFrame({"i_content": ["hello"], "eo": [None]})
    work_item = RowWorkItem(
        row_index=0,
        pending_structured_columns=[],
        pending_ns_columns=[],
        pending_embedding_columns=["eo"],
    )
    config = AppConfig.from_dict(
        {"input_path": tmp_path / "dummy.xlsx", "output_dir": tmp_path}
    )
    client = _FakeClient({"summary": "短い要約"}, embedding_vector=[0.1, 0.2])
    processor = RowProcessor(
        client=client,
        config=config,
        target_columns=["i_content"],
        base_dir=tmp_path,
        nsof_index_map={},
        embedding_spec_map={"eo": {"file_output": False}},
        embedding_index_map={"eo": 1},
    )

    result = await processor.process(work_item, df)

    assert result.error is None
    assert result.updates["eo"] == "[0.1,0.2]"


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

    classification = classify_columns(df)

    assert classification.input_columns == ["I_content"]
    assert classification.structured_columns == ["SO_summary"]
    assert classification.nonstructured_columns == ["nso_title", "NSOF_detail"]
    assert classification.nsof_index_map == {"NSOF_detail": 1}
    assert classification.label_columns == ["l_id"]
    assert classification.embedding_columns == ["EON1536F"]
    assert classification.embedding_spec_map["EON1536F"]["pre_dim"] == 1536
    assert classification.embedding_spec_map["EON1536F"]["file_output"] is True


def test_classify_columns_rejects_config_prefix():
    df = pd.DataFrame({"i_content": ["x"], "c_temperature": [0.2], "so_summary": [None]})
    with pytest.raises(ValueError) as excinfo:
        classify_columns(df)
    assert "config" in str(excinfo.value)


def test_run_state_advances_only_contiguous():
    state = RunState(resume_cursor=-1)
    state.mark_completed(2)
    assert state.resume_cursor == -1
    state.mark_completed(0)
    assert state.resume_cursor == 0
    state.mark_completed(1)
    assert state.resume_cursor == 2
