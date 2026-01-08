from types import SimpleNamespace

import pytest

from dokabun.llm.openrouter_client import AsyncOpenRouterClient


class _FakeCompletions:
    def __init__(self, recorder: dict) -> None:
        self.recorder = recorder

    async def create(self, **payload):  # type: ignore[override]
        self.recorder["payload"] = payload
        message = SimpleNamespace(parsed={}, content=None)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice], usage={}, id="gen-non-or")


class _FakeChat:
    def __init__(self, recorder: dict) -> None:
        self.completions = _FakeCompletions(recorder)


class _FakeClient:
    def __init__(self, recorder: dict) -> None:
        self.chat = _FakeChat(recorder)


@pytest.mark.asyncio
async def test_create_completion_skips_extra_body_on_non_openrouter():
    recorder: dict = {}
    client = AsyncOpenRouterClient(
        api_key="test-key",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
    )
    client.client = _FakeClient(recorder)

    await client.create_completion(
        messages=[{"role": "user", "content": "hi"}],
        json_schema={
            "name": "dummy",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    )

    assert "extra_body" not in recorder["payload"]


@pytest.mark.asyncio
async def test_fetch_generation_cost_returns_none_for_non_openrouter(monkeypatch):
    client = AsyncOpenRouterClient(
        api_key="test-key",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
    )

    called = {"value": False}

    class _RaiseClient:
        async def __aenter__(self):
            called["value"] = True
            raise AssertionError("should not be called")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        "dokabun.llm.openrouter_client.httpx.AsyncClient", _RaiseClient
    )

    result = await client.fetch_generation_cost("gen-test")

    assert result is None
    assert called["value"] is False
