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


@pytest.fixture
def client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        provider_name=PROVIDER,
        model_id=MODEL,
        endpoint=ENDPOINT,
        api_key="sk-test-key",
    )


@pytest.fixture
def client_no_key() -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        provider_name=PROVIDER,
        model_id=MODEL,
        endpoint=ENDPOINT,
        api_key=None,
    )


# --- Chat completion ---


def test_completion_maps_canonical_fields(client: OpenAICompatibleClient) -> None:
    response_json = _chat_response(content="Hello!", reasoning="step-by-step")
    _stub_sync_post(client, response_json)

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    result = client.completion(request)

    assert result.message.content == "Hello!"
    assert result.message.reasoning_content == "step-by-step"
    assert result.usage is not None
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5


def test_completion_with_tool_calls(client: OpenAICompatibleClient) -> None:
    tool_calls = [{"id": "tc1", "type": "function", "function": {"name": "search", "arguments": '{"q": "x"}'}}]
    _stub_sync_post(client, _chat_response(tool_calls=tool_calls))

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Search"}])
    result = client.completion(request)

    assert len(result.message.tool_calls) == 1
    assert result.message.tool_calls[0].name == "search"
    assert result.message.tool_calls[0].arguments_json == '{"q": "x"}'


def test_completion_posts_to_chat_completions_route(client: OpenAICompatibleClient) -> None:
    _stub_sync_post(client, _chat_response())

    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": "Hi"}],
        temperature=0.7,
        extra_body={"seed": 42},
        extra_headers={"X-Trace": "1"},
    )
    client.completion(request)

    call_args = client._client.post.call_args
    assert "/chat/completions" in call_args.args[0]
    payload = call_args.kwargs["json"]
    assert payload["model"] == MODEL
    assert payload["temperature"] == 0.7
    assert payload["seed"] == 42
    assert call_args.kwargs["headers"]["X-Trace"] == "1"


@pytest.mark.asyncio
async def test_acompletion_maps_canonical_fields(client: OpenAICompatibleClient) -> None:
    _stub_async_post(client, _chat_response(content="async result"))

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    result = await client.acompletion(request)

    assert result.message.content == "async result"


# --- Embeddings ---


def test_embeddings_maps_vectors_and_usage(client: OpenAICompatibleClient) -> None:
    _stub_sync_post(client, _embedding_response())

    request = EmbeddingRequest(model=MODEL, inputs=["hello world"])
    result = client.embeddings(request)

    assert result.vectors == [[0.1, 0.2, 0.3]]
    assert result.usage is not None
    assert result.usage.input_tokens == 5


def test_embeddings_posts_to_embeddings_route(client: OpenAICompatibleClient) -> None:
    _stub_sync_post(client, _embedding_response())

    request = EmbeddingRequest(model=MODEL, inputs=["hello"])
    client.embeddings(request)

    call_url = client._client.post.call_args.args[0]
    assert "/embeddings" in call_url


@pytest.mark.asyncio
async def test_aembeddings_maps_vectors(client: OpenAICompatibleClient) -> None:
    _stub_async_post(client, _embedding_response())

    request = EmbeddingRequest(model=MODEL, inputs=["hello"])
    result = await client.aembeddings(request)

    assert len(result.vectors) == 1


# --- Image generation ---


def test_generate_image_diffusion_route(client: OpenAICompatibleClient) -> None:
    _stub_sync_post(client, _image_response())

    request = ImageGenerationRequest(model=MODEL, prompt="a sunset")
    result = client.generate_image(request)

    assert len(result.images) == 1
    assert result.images[0].b64_data == "aW1hZ2VkYXRh"
    call_url = client._client.post.call_args.args[0]
    assert "/images/generations" in call_url


def test_generate_image_chat_route_when_messages_present(client: OpenAICompatibleClient) -> None:
    chat_img_response = {
        "choices": [{"message": {"content": None, "images": [{"b64_json": "Y2hhdGltZw=="}]}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
    }
    _stub_sync_post(client, chat_img_response)

    request = ImageGenerationRequest(
        model=MODEL,
        prompt="a sunset",
        messages=[{"role": "user", "content": "draw a sunset"}],
    )
    result = client.generate_image(request)

    assert len(result.images) == 1
    call_url = client._client.post.call_args.args[0]
    assert "/chat/completions" in call_url


@pytest.mark.asyncio
async def test_agenerate_image_maps_images(client: OpenAICompatibleClient) -> None:
    _stub_async_post(client, _image_response())

    request = ImageGenerationRequest(model=MODEL, prompt="a cat")
    result = await client.agenerate_image(request)

    assert len(result.images) == 1


# --- Auth headers ---


def test_auth_header_present_when_api_key_set(client: OpenAICompatibleClient) -> None:
    headers = client._build_headers({})
    assert headers["Authorization"] == "Bearer sk-test-key"
    assert headers["Content-Type"] == "application/json"


def test_no_auth_header_when_api_key_none(client_no_key: OpenAICompatibleClient) -> None:
    headers = client_no_key._build_headers({})
    assert "Authorization" not in headers


def test_extra_headers_merged_into_auth_headers(client: OpenAICompatibleClient) -> None:
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
    client: OpenAICompatibleClient,
    status_code: int,
    expected_kind: ProviderErrorKind,
) -> None:
    _stub_sync_post(client, {"error": {"message": "fail"}}, status_code=status_code)

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    with pytest.raises(ProviderError) as exc_info:
        client.completion(request)

    assert exc_info.value.kind == expected_kind


def test_transport_timeout_raises_timeout_error(client: OpenAICompatibleClient) -> None:
    client._client.post = MagicMock(side_effect=TimeoutError("timed out"))

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    with pytest.raises(ProviderError) as exc_info:
        client.completion(request)

    assert exc_info.value.kind == ProviderErrorKind.TIMEOUT


def test_transport_connection_error_raises_connection_error(client: OpenAICompatibleClient) -> None:
    client._client.post = MagicMock(side_effect=ConnectionError("refused"))

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    with pytest.raises(ProviderError) as exc_info:
        client.completion(request)

    assert exc_info.value.kind == ProviderErrorKind.API_CONNECTION


# --- Lifecycle ---


def test_close_delegates_to_httpx_client(client: OpenAICompatibleClient) -> None:
    client._client = MagicMock()
    client.close()
    client._client.close.assert_called_once()


@pytest.mark.asyncio
async def test_aclose_delegates_to_httpx_async_client(client: OpenAICompatibleClient) -> None:
    client._aclient = MagicMock()
    client._aclient.aclose = AsyncMock()
    await client.aclose()
    client._aclient.aclose.assert_awaited_once()


# --- Capabilities ---


@pytest.mark.parametrize(
    "method",
    ["supports_chat_completion", "supports_embeddings", "supports_image_generation"],
)
def test_capability_checks_return_true(client: OpenAICompatibleClient, method: str) -> None:
    assert getattr(client, method)() is True


# --- Helpers ---


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


def _mock_httpx_response(json_data: dict[str, Any], status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = json.dumps(json_data)
    resp.headers = {}
    return resp


def _stub_sync_post(
    client: OpenAICompatibleClient,
    response_json: dict[str, Any],
    status_code: int = 200,
) -> None:
    client._client.post = MagicMock(return_value=_mock_httpx_response(response_json, status_code))


def _stub_async_post(
    client: OpenAICompatibleClient,
    response_json: dict[str, Any],
    status_code: int = 200,
) -> None:
    client._aclient.post = AsyncMock(return_value=_mock_httpx_response(response_json, status_code))
