# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from data_designer.engine.models.clients.adapters.openai_compatible import OpenAICompatibleClient
from data_designer.engine.models.clients.errors import ProviderError, ProviderErrorKind
from data_designer.engine.models.clients.types import (
    ChatCompletionRequest,
    EmbeddingRequest,
    ImageGenerationRequest,
)

PROVIDER = "test-provider"
MODEL = "gpt-test"
ENDPOINT = "https://api.example.com/v1"


def _mock_httpx_response(json_data: dict[str, Any], status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = json.dumps(json_data)
    resp.headers = {}
    return resp


def _make_sync_client(response_json: dict[str, Any], status_code: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.post = MagicMock(return_value=_mock_httpx_response(response_json, status_code))
    return mock


def _make_async_client(response_json: dict[str, Any], status_code: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.post = AsyncMock(return_value=_mock_httpx_response(response_json, status_code))
    return mock


def _make_client(
    *,
    sync_client: MagicMock | None = None,
    async_client: MagicMock | None = None,
    api_key: str | None = "sk-test-key",
) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        provider_name=PROVIDER,
        model_id=MODEL,
        endpoint=ENDPOINT,
        api_key=api_key,
        sync_client=sync_client,
        async_client=async_client,
    )


# --- Response helpers ---


def _chat_response(
    content: str = "Hello!",
    reasoning: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning"] = reasoning
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _embedding_response() -> dict[str, Any]:
    return {
        "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }


def _image_response() -> dict[str, Any]:
    return {"data": [{"b64_json": "aW1hZ2VkYXRh"}]}


# --- Chat completion ---


def test_completion_maps_canonical_fields() -> None:
    response_json = _chat_response(content="Hello!", reasoning="step-by-step")
    client = _make_client(sync_client=_make_sync_client(response_json))

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    result = client.completion(request)

    assert result.message.content == "Hello!"
    assert result.message.reasoning_content == "step-by-step"
    assert result.usage is not None
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5


def test_completion_with_tool_calls() -> None:
    tool_calls = [{"id": "tc1", "type": "function", "function": {"name": "search", "arguments": '{"q": "x"}'}}]
    client = _make_client(sync_client=_make_sync_client(_chat_response(tool_calls=tool_calls)))

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Search"}])
    result = client.completion(request)

    assert len(result.message.tool_calls) == 1
    assert result.message.tool_calls[0].name == "search"
    assert result.message.tool_calls[0].arguments_json == '{"q": "x"}'


def test_completion_posts_to_chat_completions_route() -> None:
    sync_mock = _make_sync_client(_chat_response())
    client = _make_client(sync_client=sync_mock)

    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": "Hi"}],
        temperature=0.7,
        extra_body={"seed": 42},
        extra_headers={"X-Trace": "1"},
    )
    client.completion(request)

    call_args = sync_mock.post.call_args
    assert "/chat/completions" in call_args.args[0]
    payload = call_args.kwargs["json"]
    assert payload["model"] == MODEL
    assert payload["temperature"] == 0.7
    assert payload["seed"] == 42
    assert call_args.kwargs["headers"]["X-Trace"] == "1"


@pytest.mark.asyncio
async def test_acompletion_maps_canonical_fields() -> None:
    client = _make_client(async_client=_make_async_client(_chat_response(content="async result")))

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    result = await client.acompletion(request)

    assert result.message.content == "async result"


# --- Embeddings ---


def test_embeddings_maps_vectors_and_usage() -> None:
    client = _make_client(sync_client=_make_sync_client(_embedding_response()))

    request = EmbeddingRequest(model=MODEL, inputs=["hello world"])
    result = client.embeddings(request)

    assert result.vectors == [[0.1, 0.2, 0.3]]
    assert result.usage is not None
    assert result.usage.input_tokens == 5


def test_embeddings_posts_to_embeddings_route() -> None:
    sync_mock = _make_sync_client(_embedding_response())
    client = _make_client(sync_client=sync_mock)

    request = EmbeddingRequest(model=MODEL, inputs=["hello"])
    client.embeddings(request)

    call_url = sync_mock.post.call_args.args[0]
    assert "/embeddings" in call_url


@pytest.mark.asyncio
async def test_aembeddings_maps_vectors() -> None:
    client = _make_client(async_client=_make_async_client(_embedding_response()))

    request = EmbeddingRequest(model=MODEL, inputs=["hello"])
    result = await client.aembeddings(request)

    assert len(result.vectors) == 1


# --- Image generation ---


def test_generate_image_diffusion_route() -> None:
    sync_mock = _make_sync_client(_image_response())
    client = _make_client(sync_client=sync_mock)

    request = ImageGenerationRequest(model=MODEL, prompt="a sunset")
    result = client.generate_image(request)

    assert len(result.images) == 1
    assert result.images[0].b64_data == "aW1hZ2VkYXRh"
    call_url = sync_mock.post.call_args.args[0]
    assert "/images/generations" in call_url


def test_generate_image_chat_route_when_messages_present() -> None:
    chat_img_response = {
        "choices": [{"message": {"content": None, "images": [{"b64_json": "Y2hhdGltZw=="}]}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
    }
    sync_mock = _make_sync_client(chat_img_response)
    client = _make_client(sync_client=sync_mock)

    request = ImageGenerationRequest(
        model=MODEL,
        prompt="a sunset",
        messages=[{"role": "user", "content": "draw a sunset"}],
    )
    result = client.generate_image(request)

    assert len(result.images) == 1
    call_url = sync_mock.post.call_args.args[0]
    assert "/chat/completions" in call_url


@pytest.mark.asyncio
async def test_agenerate_image_maps_images() -> None:
    client = _make_client(async_client=_make_async_client(_image_response()))

    request = ImageGenerationRequest(model=MODEL, prompt="a cat")
    result = await client.agenerate_image(request)

    assert len(result.images) == 1


# --- Auth headers ---


def test_auth_header_present_when_api_key_set() -> None:
    client = _make_client()
    headers = client._build_headers({})
    assert headers["Authorization"] == "Bearer sk-test-key"
    assert headers["Content-Type"] == "application/json"


def test_no_auth_header_when_api_key_none() -> None:
    client = _make_client(api_key=None)
    headers = client._build_headers({})
    assert "Authorization" not in headers


def test_extra_headers_merged_into_auth_headers() -> None:
    client = _make_client()
    headers = client._build_headers({"X-Custom": "val"})
    assert headers["X-Custom"] == "val"
    assert headers["Authorization"] == "Bearer sk-test-key"


# --- Error mapping ---


@pytest.mark.parametrize(
    "status_code,expected_kind",
    [
        (429, ProviderErrorKind.RATE_LIMIT),
        (401, ProviderErrorKind.AUTHENTICATION),
        (403, ProviderErrorKind.PERMISSION_DENIED),
        (404, ProviderErrorKind.NOT_FOUND),
        (500, ProviderErrorKind.INTERNAL_SERVER),
    ],
    ids=["rate-limit", "auth", "permission", "not-found", "server-error"],
)
def test_http_error_maps_to_provider_error(
    status_code: int,
    expected_kind: ProviderErrorKind,
) -> None:
    client = _make_client(sync_client=_make_sync_client({"error": {"message": "fail"}}, status_code=status_code))

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    with pytest.raises(ProviderError) as exc_info:
        client.completion(request)

    assert exc_info.value.kind == expected_kind


def test_transport_timeout_raises_timeout_error() -> None:
    sync_mock = MagicMock()
    sync_mock.post = MagicMock(side_effect=TimeoutError("timed out"))
    client = _make_client(sync_client=sync_mock)

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    with pytest.raises(ProviderError) as exc_info:
        client.completion(request)

    assert exc_info.value.kind == ProviderErrorKind.TIMEOUT


def test_transport_connection_error_raises_connection_error() -> None:
    sync_mock = MagicMock()
    sync_mock.post = MagicMock(side_effect=ConnectionError("refused"))
    client = _make_client(sync_client=sync_mock)

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    with pytest.raises(ProviderError) as exc_info:
        client.completion(request)

    assert exc_info.value.kind == ProviderErrorKind.API_CONNECTION


# --- Lifecycle ---


def test_close_delegates_to_httpx_client() -> None:
    sync_mock = MagicMock()
    client = _make_client(sync_client=sync_mock)
    client.close()
    sync_mock.close.assert_called_once()


@pytest.mark.asyncio
async def test_aclose_delegates_to_httpx_async_client() -> None:
    async_mock = MagicMock()
    async_mock.aclose = AsyncMock()
    client = _make_client(async_client=async_mock)
    await client.aclose()
    async_mock.aclose.assert_awaited_once()


def test_close_noop_when_no_client_created() -> None:
    client = _make_client()
    client.close()  # should not raise


@pytest.mark.asyncio
async def test_aclose_noop_when_no_client_created() -> None:
    client = _make_client()
    await client.aclose()  # should not raise


# --- Capabilities ---


@pytest.mark.parametrize(
    "method",
    ["supports_chat_completion", "supports_embeddings", "supports_image_generation"],
)
def test_capability_checks_return_true(method: str) -> None:
    client = _make_client()
    assert getattr(client, method)() is True
