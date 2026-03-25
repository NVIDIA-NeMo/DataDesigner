# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data_designer.engine.models.clients.adapters.anthropic import AnthropicClient
from data_designer.engine.models.clients.adapters.http_model_client import ClientConcurrencyMode
from data_designer.engine.models.clients.adapters.openai_compatible import OpenAICompatibleClient
from data_designer.engine.models.clients.types import ChatCompletionRequest
from tests.engine.models.clients.conftest import mock_httpx_response

_OPENAI_PROVIDER = "test-provider"
_OPENAI_MODEL = "gpt-test"
_OPENAI_ENDPOINT = "https://api.example.com/v1"
_ANTHROPIC_PROVIDER = "anthropic-prod"
_ANTHROPIC_MODEL = "claude-test"
_ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1"
_SYNC_CLIENT_PATCH = "data_designer.engine.models.clients.adapters.http_model_client.lazy.httpx.Client"
_ASYNC_CLIENT_PATCH = "data_designer.engine.models.clients.adapters.http_model_client.lazy.httpx.AsyncClient"


def _make_openai_client(
    *,
    concurrency_mode: ClientConcurrencyMode = ClientConcurrencyMode.SYNC,
    sync_client: MagicMock | None = None,
    async_client: MagicMock | None = None,
) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        provider_name=_OPENAI_PROVIDER,
        endpoint=_OPENAI_ENDPOINT,
        api_key="sk-test-key",
        concurrency_mode=concurrency_mode,
        sync_client=sync_client,
        async_client=async_client,
    )


def _make_anthropic_client(
    *,
    concurrency_mode: ClientConcurrencyMode = ClientConcurrencyMode.SYNC,
    sync_client: MagicMock | None = None,
    async_client: MagicMock | None = None,
) -> AnthropicClient:
    return AnthropicClient(
        provider_name=_ANTHROPIC_PROVIDER,
        endpoint=_ANTHROPIC_ENDPOINT,
        api_key="sk-ant-test",
        concurrency_mode=concurrency_mode,
        sync_client=sync_client,
        async_client=async_client,
    )


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


_CLIENT_FACTORY_CASES = [
    pytest.param(_make_openai_client, _OPENAI_MODEL, id="openai"),
    pytest.param(_make_anthropic_client, _ANTHROPIC_MODEL, id="anthropic"),
]

_SYNC_LAZY_INIT_CASES = [
    pytest.param(_make_openai_client, _OPENAI_MODEL, _make_openai_chat_response(), id="openai"),
    pytest.param(_make_anthropic_client, _ANTHROPIC_MODEL, _make_anthropic_chat_response(), id="anthropic"),
]

_ASYNC_LAZY_INIT_CASES = [
    pytest.param(_make_openai_client, _OPENAI_MODEL, _make_openai_chat_response(), id="openai"),
    pytest.param(_make_anthropic_client, _ANTHROPIC_MODEL, _make_anthropic_chat_response(), id="anthropic"),
]


# ---------------------------------------------------------------------------
# Sync-mode lifecycle tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("client_factory", "model_name"), _CLIENT_FACTORY_CASES)
def test_sync_close_delegates_to_httpx_client(client_factory: Callable[..., Any], model_name: str) -> None:
    sync_mock = MagicMock()
    client = client_factory(concurrency_mode=ClientConcurrencyMode.SYNC, sync_client=sync_mock)

    client.close()

    sync_mock.close.assert_called_once()


@pytest.mark.parametrize(("client_factory", "model_name"), _CLIENT_FACTORY_CASES)
def test_sync_close_is_idempotent(client_factory: Callable[..., Any], model_name: str) -> None:
    """Second close() should be a no-op — the underlying httpx client is only closed once."""
    sync_mock = MagicMock()
    client = client_factory(concurrency_mode=ClientConcurrencyMode.SYNC, sync_client=sync_mock)

    client.close()
    client.close()

    sync_mock.close.assert_called_once()


@pytest.mark.parametrize(("client_factory", "model_name"), _CLIENT_FACTORY_CASES)
def test_sync_close_noop_when_no_client_created(client_factory: Callable[..., Any], model_name: str) -> None:
    client = client_factory(concurrency_mode=ClientConcurrencyMode.SYNC)
    client.close()


@pytest.mark.parametrize(("client_factory", "model_name"), _CLIENT_FACTORY_CASES)
def test_sync_completion_raises_after_close(client_factory: Callable[..., Any], model_name: str) -> None:
    client = client_factory(concurrency_mode=ClientConcurrencyMode.SYNC)
    client.close()

    with pytest.raises(RuntimeError, match="Model client is closed\\."):
        client.completion(_make_chat_request(model_name))


@pytest.mark.parametrize(("client_factory", "model_name"), _CLIENT_FACTORY_CASES)
@pytest.mark.asyncio
async def test_aclose_is_noop_on_sync_mode_client(client_factory: Callable[..., Any], model_name: str) -> None:
    sync_mock = MagicMock()
    client = client_factory(concurrency_mode=ClientConcurrencyMode.SYNC, sync_client=sync_mock)

    await client.aclose()

    sync_mock.close.assert_not_called()


# ---------------------------------------------------------------------------
# Async-mode lifecycle tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("client_factory", "model_name"), _CLIENT_FACTORY_CASES)
@pytest.mark.asyncio
async def test_async_aclose_delegates_to_httpx_async_client(
    client_factory: Callable[..., Any], model_name: str
) -> None:
    async_mock = MagicMock()
    async_mock.aclose = AsyncMock()
    client = client_factory(concurrency_mode=ClientConcurrencyMode.ASYNC, async_client=async_mock)

    await client.aclose()

    async_mock.aclose.assert_awaited_once()


@pytest.mark.parametrize(("client_factory", "model_name"), _CLIENT_FACTORY_CASES)
@pytest.mark.asyncio
async def test_async_aclose_is_idempotent(client_factory: Callable[..., Any], model_name: str) -> None:
    """Second aclose() should be a no-op — the underlying httpx async client is only closed once."""
    async_mock = MagicMock()
    async_mock.aclose = AsyncMock()
    client = client_factory(concurrency_mode=ClientConcurrencyMode.ASYNC, async_client=async_mock)

    await client.aclose()
    await client.aclose()

    async_mock.aclose.assert_awaited_once()


@pytest.mark.parametrize(("client_factory", "model_name"), _CLIENT_FACTORY_CASES)
@pytest.mark.asyncio
async def test_async_aclose_noop_when_no_client_created(client_factory: Callable[..., Any], model_name: str) -> None:
    client = client_factory(concurrency_mode=ClientConcurrencyMode.ASYNC)
    await client.aclose()


@pytest.mark.parametrize(("client_factory", "model_name"), _CLIENT_FACTORY_CASES)
@pytest.mark.asyncio
async def test_async_acompletion_raises_after_aclose(client_factory: Callable[..., Any], model_name: str) -> None:
    client = client_factory(concurrency_mode=ClientConcurrencyMode.ASYNC)
    await client.aclose()

    with pytest.raises(RuntimeError, match="Model client is closed\\."):
        await client.acompletion(_make_chat_request(model_name))


@pytest.mark.parametrize(("client_factory", "model_name"), _CLIENT_FACTORY_CASES)
def test_close_is_noop_on_async_mode_client(client_factory: Callable[..., Any], model_name: str) -> None:
    async_mock = MagicMock()
    client = client_factory(concurrency_mode=ClientConcurrencyMode.ASYNC, async_client=async_mock)

    client.close()

    async_mock.aclose.assert_not_called()


# ---------------------------------------------------------------------------
# Mode enforcement tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("client_factory", "model_name"), _CLIENT_FACTORY_CASES)
def test_sync_mode_blocks_async_methods(client_factory: Callable[..., Any], model_name: str) -> None:
    client = client_factory(concurrency_mode=ClientConcurrencyMode.SYNC)

    with pytest.raises(RuntimeError, match="Async methods are not available"):
        client._get_async_client()


@pytest.mark.parametrize(("client_factory", "model_name"), _CLIENT_FACTORY_CASES)
def test_async_mode_blocks_sync_methods(client_factory: Callable[..., Any], model_name: str) -> None:
    client = client_factory(concurrency_mode=ClientConcurrencyMode.ASYNC)

    with pytest.raises(RuntimeError, match="Sync methods are not available"):
        client._get_sync_client()


@pytest.mark.parametrize(("client_factory", "model_name"), _CLIENT_FACTORY_CASES)
def test_mode_property_reflects_constructor_arg(client_factory: Callable[..., Any], model_name: str) -> None:
    sync_client = client_factory(concurrency_mode=ClientConcurrencyMode.SYNC)
    async_client = client_factory(concurrency_mode=ClientConcurrencyMode.ASYNC)

    assert sync_client.concurrency_mode == ClientConcurrencyMode.SYNC
    assert async_client.concurrency_mode == ClientConcurrencyMode.ASYNC


# ---------------------------------------------------------------------------
# Constructor validation tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("client_factory", "model_name"), _CLIENT_FACTORY_CASES)
def test_sync_mode_rejects_async_client_injection(client_factory: Callable[..., Any], model_name: str) -> None:
    with pytest.raises(ValueError, match="async_client must not be provided"):
        client_factory(concurrency_mode=ClientConcurrencyMode.SYNC, async_client=MagicMock())


@pytest.mark.parametrize(("client_factory", "model_name"), _CLIENT_FACTORY_CASES)
def test_async_mode_rejects_sync_client_injection(client_factory: Callable[..., Any], model_name: str) -> None:
    with pytest.raises(ValueError, match="sync_client must not be provided"):
        client_factory(concurrency_mode=ClientConcurrencyMode.ASYNC, sync_client=MagicMock())


# ---------------------------------------------------------------------------
# Lazy initialization tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("client_factory", "model_name", "response_json"), _SYNC_LAZY_INIT_CASES)
def test_completion_lazy_initializes_sync_client(
    client_factory: Callable[..., Any],
    model_name: str,
    response_json: dict[str, Any],
) -> None:
    sync_mock = MagicMock()
    sync_mock.post = MagicMock(return_value=mock_httpx_response(response_json))

    with patch(_SYNC_CLIENT_PATCH, return_value=sync_mock) as mock_ctor:
        client = client_factory(concurrency_mode=ClientConcurrencyMode.SYNC)
        result = client.completion(_make_chat_request(model_name))

    mock_ctor.assert_called_once()
    assert result.message.content == "lazy result"


@pytest.mark.parametrize(("client_factory", "model_name", "response_json"), _ASYNC_LAZY_INIT_CASES)
@pytest.mark.asyncio
async def test_acompletion_lazy_initializes_async_client(
    client_factory: Callable[..., Any],
    model_name: str,
    response_json: dict[str, Any],
) -> None:
    async_mock = MagicMock()
    async_mock.post = AsyncMock(return_value=mock_httpx_response(response_json))

    with patch(_ASYNC_CLIENT_PATCH, return_value=async_mock) as mock_ctor:
        client = client_factory(concurrency_mode=ClientConcurrencyMode.ASYNC)
        result = await client.acompletion(_make_chat_request(model_name))

    mock_ctor.assert_called_once()
    assert result.message.content == "lazy result"


# ---------------------------------------------------------------------------
# Connection pool size regression tests (issue #459)
# ---------------------------------------------------------------------------


def test_sync_client_pool_size_respects_max_parallel_requests() -> None:
    """Connection pool max_connections must be 2*max_parallel_requests, not the httpx default of 100."""
    client = OpenAICompatibleClient(
        provider_name=_OPENAI_PROVIDER,
        endpoint=_OPENAI_ENDPOINT,
        api_key="sk-test",
        max_parallel_requests=300,
        concurrency_mode=ClientConcurrencyMode.SYNC,
    )
    with patch(_SYNC_CLIENT_PATCH):
        client._get_sync_client()

    # Attribute chain explained:
    #   RetryTransport._sync_transport  → the httpx.HTTPTransport we injected
    #   HTTPTransport._pool             → the underlying httpcore.ConnectionPool
    #   ConnectionPool._max_connections → the hard cap configured via Limits
    # pool_max = max(32, 2 * 300) = 600
    assert client._transport._sync_transport._pool._max_connections == 600


@pytest.mark.asyncio
async def test_async_client_pool_size_respects_max_parallel_requests() -> None:
    """Async connection pool max_connections must be 2*max_parallel_requests, not the httpx default of 100."""
    client = OpenAICompatibleClient(
        provider_name=_OPENAI_PROVIDER,
        endpoint=_OPENAI_ENDPOINT,
        api_key="sk-test",
        max_parallel_requests=300,
        concurrency_mode=ClientConcurrencyMode.ASYNC,
    )
    with patch(_ASYNC_CLIENT_PATCH):
        client._get_async_client()

    # Attribute chain explained:
    #   RetryTransport._async_transport → the httpx.AsyncHTTPTransport we injected
    #   AsyncHTTPTransport._pool        → the underlying httpcore.AsyncConnectionPool
    #   AsyncConnectionPool._max_connections → the hard cap configured via Limits
    # pool_max = max(32, 2 * 300) = 600
    assert client._transport._async_transport._pool._max_connections == 600
