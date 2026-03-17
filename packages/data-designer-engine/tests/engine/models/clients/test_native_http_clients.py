# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data_designer.engine.models.clients.adapters.anthropic import AnthropicClient
from data_designer.engine.models.clients.adapters.openai_compatible import OpenAICompatibleClient
from data_designer.engine.models.clients.types import ChatCompletionRequest

_OPENAI_PROVIDER = "test-provider"
_OPENAI_MODEL = "gpt-test"
_OPENAI_ENDPOINT = "https://api.example.com/v1"
_ANTHROPIC_PROVIDER = "anthropic-prod"
_ANTHROPIC_MODEL = "claude-test"
_ANTHROPIC_ENDPOINT = "https://api.anthropic.com"
_SYNC_CLIENT_PATCH = "data_designer.engine.models.clients.adapters.http_model_client.lazy.httpx.Client"
_ASYNC_CLIENT_PATCH = "data_designer.engine.models.clients.adapters.http_model_client.lazy.httpx.AsyncClient"


def _make_openai_client(
    *,
    sync_client: MagicMock | None = None,
    async_client: MagicMock | None = None,
) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        provider_name=_OPENAI_PROVIDER,
        model_id=_OPENAI_MODEL,
        endpoint=_OPENAI_ENDPOINT,
        api_key="sk-test-key",
        sync_client=sync_client,
        async_client=async_client,
    )


def _make_anthropic_client(
    *,
    sync_client: MagicMock | None = None,
    async_client: MagicMock | None = None,
) -> AnthropicClient:
    return AnthropicClient(
        provider_name=_ANTHROPIC_PROVIDER,
        model_id=_ANTHROPIC_MODEL,
        endpoint=_ANTHROPIC_ENDPOINT,
        api_key="sk-ant-test",
        sync_client=sync_client,
        async_client=async_client,
    )


def _mock_httpx_response(json_data: dict[str, Any], status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_data
    response.text = ""
    response.headers = {}
    return response


def _make_openai_chat_response(text: str = "lazy result") -> dict[str, Any]:
    return {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _make_anthropic_chat_response(text: str = "lazy result") -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _make_chat_request(model_name: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(model=model_name, messages=[{"role": "user", "content": "Hi"}])


_CLIENT_CASES = [
    pytest.param(_make_openai_client, _OPENAI_MODEL, id="openai"),
    pytest.param(_make_anthropic_client, _ANTHROPIC_MODEL, id="anthropic"),
]

_LAZY_INIT_CASES = [
    pytest.param(_make_openai_client, _OPENAI_MODEL, _make_openai_chat_response(), id="openai"),
    pytest.param(_make_anthropic_client, _ANTHROPIC_MODEL, _make_anthropic_chat_response(), id="anthropic"),
]


@pytest.mark.parametrize(("client_factory", "model_name"), _CLIENT_CASES)
def test_close_delegates_to_httpx_client(client_factory: Callable[..., Any], model_name: str) -> None:
    sync_mock = MagicMock()
    client = client_factory(sync_client=sync_mock)

    client.close()

    sync_mock.close.assert_called_once()


@pytest.mark.parametrize(("client_factory", "model_name"), _CLIENT_CASES)
@pytest.mark.asyncio
async def test_aclose_closes_both_clients(client_factory: Callable[..., Any], model_name: str) -> None:
    sync_mock = MagicMock()
    async_mock = MagicMock()
    async_mock.aclose = AsyncMock()
    client = client_factory(sync_client=sync_mock, async_client=async_mock)

    await client.aclose()

    async_mock.aclose.assert_awaited_once()
    sync_mock.close.assert_called_once()


@pytest.mark.parametrize(("client_factory", "model_name"), _CLIENT_CASES)
def test_close_is_idempotent(client_factory: Callable[..., Any], model_name: str) -> None:
    sync_mock = MagicMock()
    client = client_factory(sync_client=sync_mock)

    client.close()
    client.close()

    sync_mock.close.assert_called_once()


@pytest.mark.parametrize(("client_factory", "model_name"), _CLIENT_CASES)
@pytest.mark.asyncio
async def test_aclose_is_idempotent(client_factory: Callable[..., Any], model_name: str) -> None:
    sync_mock = MagicMock()
    async_mock = MagicMock()
    async_mock.aclose = AsyncMock()
    client = client_factory(sync_client=sync_mock, async_client=async_mock)

    await client.aclose()
    await client.aclose()

    async_mock.aclose.assert_awaited_once()
    sync_mock.close.assert_called_once()


@pytest.mark.parametrize(("client_factory", "model_name"), _CLIENT_CASES)
def test_close_noop_when_no_client_created(client_factory: Callable[..., Any], model_name: str) -> None:
    client = client_factory()

    client.close()


@pytest.mark.parametrize(("client_factory", "model_name"), _CLIENT_CASES)
def test_completion_raises_after_close(client_factory: Callable[..., Any], model_name: str) -> None:
    client = client_factory()
    client.close()

    with pytest.raises(RuntimeError, match="Model client is closed\\."):
        client.completion(_make_chat_request(model_name))


@pytest.mark.parametrize(("client_factory", "model_name"), _CLIENT_CASES)
@pytest.mark.asyncio
async def test_aclose_noop_when_no_client_created(client_factory: Callable[..., Any], model_name: str) -> None:
    client = client_factory()

    await client.aclose()


@pytest.mark.parametrize(("client_factory", "model_name"), _CLIENT_CASES)
@pytest.mark.asyncio
async def test_acompletion_raises_after_aclose(client_factory: Callable[..., Any], model_name: str) -> None:
    client = client_factory()
    await client.aclose()

    with pytest.raises(RuntimeError, match="Model client is closed\\."):
        await client.acompletion(_make_chat_request(model_name))


@pytest.mark.parametrize(("client_factory", "model_name", "response_json"), _LAZY_INIT_CASES)
def test_completion_lazy_initializes_sync_client(
    client_factory: Callable[..., Any],
    model_name: str,
    response_json: dict[str, Any],
) -> None:
    sync_mock = MagicMock()
    sync_mock.post = MagicMock(return_value=_mock_httpx_response(response_json))

    with patch(_SYNC_CLIENT_PATCH, return_value=sync_mock) as mock_ctor:
        client = client_factory()
        result = client.completion(_make_chat_request(model_name))

    mock_ctor.assert_called_once()
    assert result.message.content == "lazy result"


@pytest.mark.parametrize(("client_factory", "model_name", "response_json"), _LAZY_INIT_CASES)
@pytest.mark.asyncio
async def test_acompletion_lazy_initializes_async_client(
    client_factory: Callable[..., Any],
    model_name: str,
    response_json: dict[str, Any],
) -> None:
    async_mock = MagicMock()
    async_mock.post = AsyncMock(return_value=_mock_httpx_response(response_json))

    with patch(_ASYNC_CLIENT_PATCH, return_value=async_mock) as mock_ctor:
        client = client_factory()
        result = await client.acompletion(_make_chat_request(model_name))

    mock_ctor.assert_called_once()
    assert result.message.content == "lazy result"
