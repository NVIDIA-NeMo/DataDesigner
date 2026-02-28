# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from data_designer.engine.models.clients.adapters.litellm_bridge import LiteLLMBridgeClient
from data_designer.engine.models.clients.types import (
    ChatCompletionRequest,
    EmbeddingRequest,
    ImageGenerationRequest,
)


def test_completion_maps_canonical_fields_from_litellm_response() -> None:
    response = _build_chat_response(
        content="final answer",
        reasoning_content="reasoning trace",
        tool_calls=[{"id": "call-1", "function": {"name": "lookup", "arguments": '{"query":"foo"}'}}],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=13, total_tokens=24),
    )
    router = MagicMock()
    router.completion.return_value = response
    client = LiteLLMBridgeClient(provider_name="stub-provider", router=router)

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
    result = client.completion(request)

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

    router.completion.assert_called_once_with(
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


@pytest.mark.asyncio
async def test_acompletion_maps_canonical_fields_from_litellm_response() -> None:
    response = _build_chat_response(content="async result", reasoning_content=None, tool_calls=[], usage=None)
    router = MagicMock()
    router.acompletion = AsyncMock(return_value=response)
    client = LiteLLMBridgeClient(provider_name="stub-provider", router=router)

    request = ChatCompletionRequest(model="stub-model", messages=[{"role": "user", "content": "hello"}])
    result = await client.acompletion(request)

    assert result.message.content == "async result"
    assert result.usage is None
    router.acompletion.assert_awaited_once_with(
        model="stub-model",
        messages=[{"role": "user", "content": "hello"}],
    )


def test_embeddings_maps_vectors_and_usage() -> None:
    response = SimpleNamespace(
        data=[{"embedding": [1, 2]}, SimpleNamespace(embedding=[3.5, 4.5])],
        usage=SimpleNamespace(prompt_tokens=4, total_tokens=4),
    )
    router = MagicMock()
    router.embedding.return_value = response
    client = LiteLLMBridgeClient(provider_name="stub-provider", router=router)

    request = EmbeddingRequest(model="stub-model", inputs=["a", "b"], dimensions=32, encoding_format="float")
    result = client.embeddings(request)

    assert result.vectors == [[1.0, 2.0], [3.5, 4.5]]
    assert result.usage is not None
    assert result.usage.input_tokens == 4
    assert result.usage.output_tokens is None
    assert result.raw is response
    router.embedding.assert_called_once_with(
        model="stub-model",
        input=["a", "b"],
        encoding_format="float",
        dimensions=32,
    )


def test_generate_image_uses_chat_completion_path_when_messages_are_present() -> None:
    response = _build_chat_response(
        content=None,
        reasoning_content=None,
        tool_calls=None,
        images=[{"image_url": {"url": "data:image/png;base64,aGVsbG8="}}],
        usage=None,
    )
    router = MagicMock()
    router.completion.return_value = response
    client = LiteLLMBridgeClient(provider_name="stub-provider", router=router)

    request = ImageGenerationRequest(
        model="stub-model",
        prompt="unused because messages are supplied",
        messages=[{"role": "user", "content": "generate image"}],
        n=1,
    )
    result = client.generate_image(request)

    assert len(result.images) == 1
    assert result.images[0].b64_data == "aGVsbG8="
    assert result.images[0].mime_type == "image/png"
    assert result.usage is not None
    assert result.usage.generated_images == 1
    router.completion.assert_called_once_with(
        model="stub-model",
        messages=[{"role": "user", "content": "generate image"}],
        n=1,
    )
    router.image_generation.assert_not_called()


def test_generate_image_uses_chat_completion_path_when_messages_is_empty_list() -> None:
    response = _build_chat_response(
        content=None,
        reasoning_content=None,
        tool_calls=None,
        images=[{"image_url": {"url": "data:image/png;base64,aGVsbG8="}}],
        usage=None,
    )
    router = MagicMock()
    router.completion.return_value = response
    client = LiteLLMBridgeClient(provider_name="stub-provider", router=router)

    request = ImageGenerationRequest(
        model="stub-model",
        prompt="unused because messages are supplied",
        messages=[],
        n=1,
    )
    result = client.generate_image(request)

    assert len(result.images) == 1
    router.completion.assert_called_once_with(
        model="stub-model",
        messages=[],
        n=1,
    )
    router.image_generation.assert_not_called()


def test_generate_image_uses_diffusion_path_without_messages() -> None:
    response = SimpleNamespace(
        data=[
            SimpleNamespace(b64_json="Zmlyc3Q="),
            {"url": "data:image/jpeg;base64,c2Vjb25k"},
        ],
        usage=SimpleNamespace(input_tokens=9, output_tokens=12),
    )
    router = MagicMock()
    router.image_generation.return_value = response
    client = LiteLLMBridgeClient(provider_name="stub-provider", router=router)

    request = ImageGenerationRequest(model="stub-model", prompt="make an image", n=2)
    result = client.generate_image(request)

    assert [image.b64_data for image in result.images] == ["Zmlyc3Q=", "c2Vjb25k"]
    assert [image.mime_type for image in result.images] == [None, "image/jpeg"]
    assert result.usage is not None
    assert result.usage.input_tokens == 9
    assert result.usage.output_tokens == 12
    assert result.usage.total_tokens == 21
    assert result.usage.generated_images == 2
    router.image_generation.assert_called_once_with(prompt="make an image", model="stub-model", n=2)


@pytest.mark.asyncio
async def test_aembeddings_maps_vectors_and_usage() -> None:
    response = SimpleNamespace(
        data=[{"embedding": [0.1, 0.2]}, SimpleNamespace(embedding=[0.3, 0.4])],
        usage=SimpleNamespace(prompt_tokens=5, total_tokens=5),
    )
    router = MagicMock()
    router.aembedding = AsyncMock(return_value=response)
    client = LiteLLMBridgeClient(provider_name="stub-provider", router=router)

    request = EmbeddingRequest(model="stub-model", inputs=["x", "y"])
    result = await client.aembeddings(request)

    assert result.vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert result.usage is not None
    assert result.usage.input_tokens == 5
    assert result.raw is response
    router.aembedding.assert_awaited_once_with(model="stub-model", input=["x", "y"])


def test_completion_coerces_list_content_blocks_to_string() -> None:
    response = _build_chat_response(
        content=[{"type": "text", "text": "first"}, {"type": "text", "text": "second"}],
        reasoning_content=None,
        tool_calls=[],
        usage=None,
    )
    router = MagicMock()
    router.completion.return_value = response
    client = LiteLLMBridgeClient(provider_name="stub-provider", router=router)

    request = ChatCompletionRequest(model="stub-model", messages=[{"role": "user", "content": "hello"}])
    result = client.completion(request)

    assert result.message.content == "first\nsecond"


def test_close_and_aclose_are_callable() -> None:
    router = MagicMock()
    client = LiteLLMBridgeClient(provider_name="stub-provider", router=router)
    client.close()


@pytest.mark.asyncio
async def test_aclose_is_callable() -> None:
    router = MagicMock()
    client = LiteLLMBridgeClient(provider_name="stub-provider", router=router)
    await client.aclose()


@pytest.mark.asyncio
async def test_agenerate_image_uses_diffusion_path_without_messages() -> None:
    response = SimpleNamespace(
        data=[SimpleNamespace(b64_json="YXN5bmM=")],
        usage=SimpleNamespace(input_tokens=3, output_tokens=7),
    )
    router = MagicMock()
    router.aimage_generation = AsyncMock(return_value=response)
    client = LiteLLMBridgeClient(provider_name="stub-provider", router=router)

    request = ImageGenerationRequest(model="stub-model", prompt="async image", n=1)
    result = await client.agenerate_image(request)

    assert len(result.images) == 1
    assert result.images[0].b64_data == "YXN5bmM="
    assert result.usage is not None
    assert result.usage.generated_images == 1
    router.aimage_generation.assert_awaited_once_with(prompt="async image", model="stub-model", n=1)


def test_completion_with_empty_choices_returns_empty_message() -> None:
    response = SimpleNamespace(choices=[], usage=None)
    router = MagicMock()
    router.completion.return_value = response
    client = LiteLLMBridgeClient(provider_name="stub-provider", router=router)

    request = ChatCompletionRequest(model="stub-model", messages=[{"role": "user", "content": "hello"}])
    result = client.completion(request)

    assert result.message.content is None
    assert result.message.tool_calls == []
    assert result.message.images == []


def test_completion_with_tool_call_dict_arguments() -> None:
    response = _build_chat_response(
        content=None,
        reasoning_content=None,
        tool_calls=[{"id": "call-2", "function": {"name": "search", "arguments": {"q": "test"}}}],
        usage=None,
    )
    router = MagicMock()
    router.completion.return_value = response
    client = LiteLLMBridgeClient(provider_name="stub-provider", router=router)

    request = ChatCompletionRequest(model="stub-model", messages=[{"role": "user", "content": "hello"}])
    result = client.completion(request)

    assert len(result.message.tool_calls) == 1
    assert result.message.tool_calls[0].arguments_json == '{"q": "test"}'


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
