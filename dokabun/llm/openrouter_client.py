"""OpenRouter へのチャット補完呼び出しラッパー。"""

from __future__ import annotations

import asyncio
from typing import Any, Mapping, Sequence

import httpx
from openai import AsyncOpenAI

from dokabun.logging_utils import get_logger

logger = get_logger(__name__)


class AsyncOpenRouterClient:
    """OpenRouter API を介して LLM を呼び出す非同期クライアント。"""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float | None = 60.0,
        max_retries: int = 3,
    ) -> None:
        """クライアントを初期化する。

        Args:
            api_key: OpenRouter で発行された API キー。
            model: 呼び出しに利用するモデル名。
            base_url: OpenRouter API のベース URL。
            timeout: リクエストタイムアウト（秒）。None なら SDK 既定値。
            max_retries: 一時的なエラー発生時の最大リトライ回数。
        """

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.model = model
        self.max_retries = max_retries
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    async def create_completion(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        json_schema: Mapping[str, Any],
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> Any:
        """構造化出力を伴うチャット補完を実行する。

        Args:
            messages: OpenAI SDK 互換のメッセージ配列。
            json_schema: `response_format.json_schema` に渡す辞書。
            temperature: 応答多様性を制御する温度。None ならモデル既定値。
            max_tokens: 応答の最大トークン数。None ならモデル既定値。
            extra_headers: OpenRouter に追加で渡すヘッダ。

        Returns:
            Any: OpenAI SDK が返すレスポンスオブジェクト。

        Raises:
            Exception: 連続リトライ後も失敗した場合に例外をそのまま送出。
        """

        payload = {
            "model": self.model,
            "messages": list(messages),
            "response_format": {"type": "json_schema", "json_schema": json_schema},
            "extra_headers": extra_headers or {},
            "extra_body": {"usage": {"include": True}},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        attempt = 0
        delay = 1.0
        while True:
            try:
                return await self.client.chat.completions.create(**payload)  # type: ignore[arg-type]
            except Exception as exc:  # noqa: BLE001
                attempt += 1
                if attempt > self.max_retries:
                    logger.error("OpenRouter 呼び出しが失敗しました: %s", exc)
                    raise
                logger.warning("OpenRouter 呼び出しに失敗しました。再試行します (%s/%s)", attempt, self.max_retries)
                await asyncio.sleep(delay)
                delay *= 2

    async def create_completion_text(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> Any:
        """プレーンテキスト応答のチャット補完を実行する。

        Args:
            messages: OpenAI SDK 互換のメッセージ配列。
            temperature: 応答多様性を制御する温度。None ならモデル既定値。
            max_tokens: 応答の最大トークン数。None ならモデル既定値。
            extra_headers: OpenRouter に追加で渡すヘッダ。

        Returns:
            Any: OpenAI SDK が返すレスポンスオブジェクト。
        """

        payload = {
            "model": self.model,
            "messages": list(messages),
            "extra_headers": extra_headers or {},
            "extra_body": {"usage": {"include": True}},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        attempt = 0
        delay = 1.0
        while True:
            try:
                return await self.client.chat.completions.create(**payload)  # type: ignore[arg-type]
            except Exception as exc:  # noqa: BLE001
                attempt += 1
                if attempt > self.max_retries:
                    logger.error("OpenRouter 呼び出しが失敗しました: %s", exc)
                    raise
                logger.warning("OpenRouter 呼び出しに失敗しました。再試行します (%s/%s)", attempt, self.max_retries)
                await asyncio.sleep(delay)
                delay *= 2

    async def fetch_generation_cost(self, generation_id: str) -> float | None:
        """GET /generation で合計コスト (USD) を取得する。

        OpenRouter API reference: https://openrouter.ai/docs/api/api-reference/generations/get-generation

        Args:
            generation_id: `chat.completions.create` が返したリクエスト ID。

        Returns:
            float | None: 取得できた場合は USD 建ての合計コスト。失敗・未提供時は None。
        """

        url = f"{self.base_url}/generation"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {"id": generation_id}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
        except httpx.HTTPError as exc:  # noqa: PERF203
            logger.warning("生成コストの取得に失敗しました: %s", exc)
            return None

        payload = response.json()
        total_cost = payload.get("data", {}).get("total_cost")
        if isinstance(total_cost, (int, float)):
            return float(total_cost)
        return None
