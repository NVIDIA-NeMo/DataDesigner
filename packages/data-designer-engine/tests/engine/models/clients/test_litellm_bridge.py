# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from data_designer.engine.models.clients.adapters.litellm_bridge import LiteLLMBridgeClient
from data_designer.engine.models.clients.errors import ProviderError, ProviderErrorKind
from data_designer.engine.models.clients.types import (
    ChatCompletionRequest,
    EmbeddingRequest,
    ImageGenerationRequest,
)


def test_completion_maps_canonical_fields_from_litellm_response(
    mock_router: MagicMock,
    bridge_client: LiteLLMBridgeClient,
) -> None:
    response = _build_chat_response(
        content="final answer",
        reasoning_content="reasoning trace",
        tool_calls=[{"id": "call-1", "function": {"name": "lookup", "arguments": '{"query":"foo"}'}}],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=13, total_tokens=24),
    )
    mock_router.completion.return_value = response

    request = ChatCompletionRequest(
        model="stub-model",
        messages=[{"role": "user", "content": "hello"}],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        temperature=0.2,
        top_p=0.8,
        max_tokens=256,
        extra_body={"foo": "bar"},
        extra_headers={"x-trace": "1"},
        metadata={"trace_id": "abc"},
    )
    result = bridge_client.completion(request)

    assert result.message.content == "final answer"
    assert result.message.reasoning_content == "reasoning trace"
    assert len(result.message.tool_calls) == 1
    assert result.message.tool_calls[0].id == "call-1"
    assert result.message.tool_calls[0].name == "lookup"
    assert result.message.tool_calls[0].arguments_json == '{"query":"foo"}'
    assert result.usage is not None
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 13
    assert result.usage.total_tokens == 24
    assert result.raw is response

    mock_router.completion.assert_called_once_with(
        model="stub-model",
        messages=[{"role": "user", "content": "hello"}],
        extra_headers={"x-trace": "1"},
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        temperature=0.2,
        top_p=0.8,
        max_tokens=256,
        metadata={"trace_id": "abc"},
        foo="bar",
    )


@pytest.mark.asyncio
async def test_acompletion_maps_canonical_fields_from_litellm_response(
    mock_router: MagicMock,
    bridge_client: LiteLLMBridgeClient,
) -> None:
    response = _build_chat_response(content="async result", reasoning_content=None, tool_calls=[], usage=None)
    mock_router.acompletion = AsyncMock(return_value=response)

    request = ChatCompletionRequest(model="stub-model", messages=[{"role": "user", "content": "hello"}])
    result = await bridge_client.acompletion(request)

    assert result.message.content == "async result"
    assert result.usage is None
    mock_router.acompletion.assert_awaited_once_with(
        model="stub-model",
        messages=[{"role": "user", "content": "hello"}],
        extra_headers=None,
    )


def test_embeddings_maps_vectors_and_usage(
    mock_router: MagicMock,
    bridge_client: LiteLLMBridgeClient,
) -> None:
    response = SimpleNamespace(
        data=[{"embedding": [1, 2]}, SimpleNamespace(embedding=[3.5, 4.5])],
        usage=SimpleNamespace(prompt_tokens=4, total_tokens=4),
    )
    mock_router.embedding.return_value = response

    request = EmbeddingRequest(model="stub-model", inputs=["a", "b"], dimensions=32, encoding_format="float")
    result = bridge_client.embeddings(request)

    assert result.vectors == [[1.0, 2.0], [3.5, 4.5]]
    assert result.usage is not None
    assert result.usage.input_tokens == 4
    assert result.usage.output_tokens is None
    assert result.raw is response
    mock_router.embedding.assert_called_once_with(
        model="stub-model",
        input=["a", "b"],
        extra_headers=None,
        encoding_format="float",
        dimensions=32,
    )


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "generate image"}],
        [],
    ],
    ids=["with-content", "empty-list"],
)
def test_generate_image_uses_chat_completion_path_when_messages_provided(
    mock_router: MagicMock,
    bridge_client: LiteLLMBridgeClient,
    messages: list[dict[str, Any]],
) -> None:
    response = _build_chat_response(
        content=None,
        reasoning_content=None,
        tool_calls=None,
        images=[{"image_url": {"url": "data:image/png;base64,aGVsbG8="}}],
        usage=None,
    )
    mock_router.completion.return_value = response

    request = ImageGenerationRequest(
        model="stub-model",
        prompt="unused because messages are supplied",
        messages=messages,
        n=1,
    )
    result = bridge_client.generate_image(request)

    assert len(result.images) == 1
    assert result.images[0].b64_data == "aGVsbG8="
    mock_router.completion.assert_called_once_with(
        model="stub-model",
        messages=messages,
        extra_headers=None,
        n=1,
    )
    mock_router.image_generation.assert_not_called()


def test_generate_image_uses_diffusion_path_without_messages(
    mock_router: MagicMock,
    bridge_client: LiteLLMBridgeClient,
) -> None:
    response = SimpleNamespace(
        data=[
            SimpleNamespace(b64_json="Zmlyc3Q="),
            {"url": "data:image/jpeg;base64,c2Vjb25k"},
        ],
        usage=SimpleNamespace(input_tokens=9, output_tokens=12),
    )
    mock_router.image_generation.return_value = response

    request = ImageGenerationRequest(model="stub-model", prompt="make an image", n=2)
    result = bridge_client.generate_image(request)

    assert [image.b64_data for image in result.images] == ["Zmlyc3Q=", "c2Vjb25k"]
    assert [image.mime_type for image in result.images] == [None, "image/jpeg"]
    assert result.usage is not None
    assert result.usage.input_tokens == 9
    assert result.usage.output_tokens == 12
    assert result.usage.total_tokens == 21
    assert result.usage.generated_images == 2
    mock_router.image_generation.assert_called_once_with(
        prompt="make an image", model="stub-model", extra_headers=None, n=2
    )


@pytest.mark.asyncio
async def test_aembeddings_maps_vectors_and_usage(
    mock_router: MagicMock,
    bridge_client: LiteLLMBridgeClient,
) -> None:
    response = SimpleNamespace(
        data=[{"embedding": [0.1, 0.2]}, SimpleNamespace(embedding=[0.3, 0.4])],
        usage=SimpleNamespace(prompt_tokens=5, total_tokens=5),
    )
    mock_router.aembedding = AsyncMock(return_value=response)

    request = EmbeddingRequest(model="stub-model", inputs=["x", "y"])
    result = await bridge_client.aembeddings(request)

    assert result.vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert result.usage is not None
    assert result.usage.input_tokens == 5
    assert result.raw is response
    mock_router.aembedding.assert_awaited_once_with(model="stub-model", input=["x", "y"], extra_headers=None)


def test_completion_coerces_list_content_blocks_to_string(
    mock_router: MagicMock,
    bridge_client: LiteLLMBridgeClient,
) -> None:
    response = _build_chat_response(
        content=[{"type": "text", "text": "first"}, {"type": "text", "text": "second"}],
        reasoning_content=None,
        tool_calls=[],
        usage=None,
    )
    mock_router.completion.return_value = response

    request = ChatCompletionRequest(model="stub-model", messages=[{"role": "user", "content": "hello"}])
    result = bridge_client.completion(request)

    assert result.message.content == "first\nsecond"


def test_close_and_aclose_are_callable(bridge_client: LiteLLMBridgeClient) -> None:
    bridge_client.close()


@pytest.mark.asyncio
async def test_aclose_is_callable(bridge_client: LiteLLMBridgeClient) -> None:
    await bridge_client.aclose()


@pytest.mark.asyncio
async def test_agenerate_image_uses_diffusion_path_without_messages(
    mock_router: MagicMock,
    bridge_client: LiteLLMBridgeClient,
) -> None:
    response = SimpleNamespace(
        data=[SimpleNamespace(b64_json="YXN5bmM=")],
        usage=SimpleNamespace(input_tokens=3, output_tokens=7),
    )
    mock_router.aimage_generation = AsyncMock(return_value=response)

    request = ImageGenerationRequest(model="stub-model", prompt="async image", n=1)
    result = await bridge_client.agenerate_image(request)

    assert len(result.images) == 1
    assert result.images[0].b64_data == "YXN5bmM="
    assert result.usage is not None
    assert result.usage.generated_images == 1
    mock_router.aimage_generation.assert_awaited_once_with(
        prompt="async image", model="stub-model", extra_headers=None, n=1
    )


def test_completion_with_empty_choices_returns_empty_message(
    mock_router: MagicMock,
    bridge_client: LiteLLMBridgeClient,
) -> None:
    response = SimpleNamespace(choices=[], usage=None)
    mock_router.completion.return_value = response

    request = ChatCompletionRequest(model="stub-model", messages=[{"role": "user", "content": "hello"}])
    result = bridge_client.completion(request)

    assert result.message.content is None
    assert result.message.tool_calls == []
    assert result.message.images == []


def test_completion_with_tool_call_dict_arguments(
    mock_router: MagicMock,
    bridge_client: LiteLLMBridgeClient,
) -> None:
    response = _build_chat_response(
        content=None,
        reasoning_content=None,
        tool_calls=[{"id": "call-2", "function": {"name": "search", "arguments": {"q": "test"}}}],
        usage=None,
    )
    mock_router.completion.return_value = response

    request = ChatCompletionRequest(model="stub-model", messages=[{"role": "user", "content": "hello"}])
    result = bridge_client.completion(request)

    assert len(result.message.tool_calls) == 1
    assert result.message.tool_calls[0].arguments_json == '{"q": "test"}'


# --- Exception wrapping tests ---


def test_completion_wraps_router_exception_with_status_code(
    mock_router: MagicMock,
    bridge_client: LiteLLMBridgeClient,
) -> None:
    exc = Exception("Rate limit exceeded")
    exc.status_code = 429  # type: ignore[attr-defined]
    mock_router.completion.side_effect = exc

    request = ChatCompletionRequest(model="stub-model", messages=[{"role": "user", "content": "hello"}])
    with pytest.raises(ProviderError) as exc_info:
        bridge_client.completion(request)

    assert exc_info.value.kind == ProviderErrorKind.RATE_LIMIT
    assert exc_info.value.status_code == 429
    assert exc_info.value.provider_name == "stub-provider"
    assert exc_info.value.__cause__ is exc


def test_completion_wraps_generic_router_exception(
    mock_router: MagicMock,
    bridge_client: LiteLLMBridgeClient,
) -> None:
    mock_router.completion.side_effect = RuntimeError("something broke")

    request = ChatCompletionRequest(model="stub-model", messages=[{"role": "user", "content": "hello"}])
    with pytest.raises(ProviderError) as exc_info:
        bridge_client.completion(request)

    assert exc_info.value.kind == ProviderErrorKind.API_ERROR
    assert "something broke" in exc_info.value.message
    assert exc_info.value.status_code is None


def test_completion_passes_through_provider_error(
    mock_router: MagicMock,
    bridge_client: LiteLLMBridgeClient,
) -> None:
    original = ProviderError(kind=ProviderErrorKind.AUTHENTICATION, message="bad key")
    mock_router.completion.side_effect = original

    request = ChatCompletionRequest(model="stub-model", messages=[{"role": "user", "content": "hello"}])
    with pytest.raises(ProviderError) as exc_info:
        bridge_client.completion(request)

    assert exc_info.value is original


@pytest.mark.asyncio
async def test_acompletion_wraps_router_exception(
    mock_router: MagicMock,
    bridge_client: LiteLLMBridgeClient,
) -> None:
    mock_router.acompletion = AsyncMock(side_effect=ConnectionError("connection refused"))

    request = ChatCompletionRequest(model="stub-model", messages=[{"role": "user", "content": "hello"}])
    with pytest.raises(ProviderError) as exc_info:
        await bridge_client.acompletion(request)

    assert exc_info.value.kind == ProviderErrorKind.API_CONNECTION


def test_embeddings_wraps_router_exception(
    mock_router: MagicMock,
    bridge_client: LiteLLMBridgeClient,
) -> None:
    exc = Exception("server error")
    exc.status_code = 500  # type: ignore[attr-defined]
    mock_router.embedding.side_effect = exc

    request = EmbeddingRequest(model="stub-model", inputs=["a"])
    with pytest.raises(ProviderError) as exc_info:
        bridge_client.embeddings(request)

    assert exc_info.value.kind == ProviderErrorKind.INTERNAL_SERVER


def test_generate_image_wraps_router_exception(
    mock_router: MagicMock,
    bridge_client: LiteLLMBridgeClient,
) -> None:
    mock_router.image_generation.side_effect = TimeoutError("timed out")

    request = ImageGenerationRequest(model="stub-model", prompt="make an image")
    with pytest.raises(ProviderError) as exc_info:
        bridge_client.generate_image(request)

    assert exc_info.value.kind == ProviderErrorKind.TIMEOUT


@pytest.mark.asyncio
async def test_agenerate_image_wraps_router_exception(
    mock_router: MagicMock,
    bridge_client: LiteLLMBridgeClient,
) -> None:
    mock_router.aimage_generation = AsyncMock(side_effect=RuntimeError("boom"))

    request = ImageGenerationRequest(model="stub-model", prompt="async image")
    with pytest.raises(ProviderError) as exc_info:
        await bridge_client.agenerate_image(request)

    assert exc_info.value.kind == ProviderErrorKind.API_ERROR


@pytest.mark.asyncio
async def test_aembeddings_wraps_router_exception(
    mock_router: MagicMock,
    bridge_client: LiteLLMBridgeClient,
) -> None:
    mock_router.aembedding = AsyncMock(side_effect=RuntimeError("network error"))

    request = EmbeddingRequest(model="stub-model", inputs=["a"])
    with pytest.raises(ProviderError) as exc_info:
        await bridge_client.aembeddings(request)

    assert exc_info.value.kind == ProviderErrorKind.API_ERROR


# --- Helpers ---


def _build_chat_response(
    *,
    content: Any,
    reasoning_content: str | None,
    tool_calls: list[dict[str, Any]] | None,
    usage: Any,
    images: list[dict[str, Any]] | None = None,
) -> Any:
    message = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
        images=images,
    )
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=usage)
