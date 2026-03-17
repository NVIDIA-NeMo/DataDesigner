# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from data_designer.engine.mcp.registry import MCPToolDefinition
from data_designer.engine.models.clients.adapters.anthropic import AnthropicClient
from data_designer.engine.models.clients.errors import ProviderError, ProviderErrorKind
from data_designer.engine.models.clients.types import (
    ChatCompletionRequest,
    EmbeddingRequest,
    ImageGenerationRequest,
)
from data_designer.engine.models.utils import ChatMessage

PROVIDER = "anthropic-prod"
MODEL = "claude-test"
ENDPOINT = "https://api.anthropic.com"


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
    api_key: str | None = "sk-ant-test",
) -> AnthropicClient:
    return AnthropicClient(
        provider_name=PROVIDER,
        model_id=MODEL,
        endpoint=ENDPOINT,
        api_key=api_key,
        sync_client=sync_client,
        async_client=async_client,
    )


# --- Response helpers ---


def _text_response(text: str = "Hello!") -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "stop_reason": "end_turn",
    }


def _tool_use_response() -> dict[str, Any]:
    return {
        "content": [
            {"type": "text", "text": "Let me search for that."},
            {
                "type": "tool_use",
                "id": "toolu_01",
                "name": "search",
                "input": {"query": "weather"},
            },
        ],
        "usage": {"input_tokens": 15, "output_tokens": 20},
        "stop_reason": "tool_use",
    }


def _thinking_response() -> dict[str, Any]:
    return {
        "content": [
            {"type": "thinking", "thinking": "Let me reason step by step."},
            {"type": "text", "text": "The answer is 42."},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 15},
        "stop_reason": "end_turn",
    }


# --- Chat completion ---


def test_completion_maps_text_content() -> None:
    client = _make_client(sync_client=_make_sync_client(_text_response(text="Hello from Claude!")))

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    result = client.completion(request)

    assert result.message.content == "Hello from Claude!"
    assert result.usage is not None
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5


def test_completion_maps_tool_use_blocks() -> None:
    client = _make_client(sync_client=_make_sync_client(_tool_use_response()))

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Weather?"}])
    result = client.completion(request)

    assert result.message.content == "Let me search for that."
    assert len(result.message.tool_calls) == 1
    assert result.message.tool_calls[0].id == "toolu_01"
    assert result.message.tool_calls[0].name == "search"
    assert json.loads(result.message.tool_calls[0].arguments_json) == {"query": "weather"}


def test_completion_maps_thinking_blocks() -> None:
    client = _make_client(sync_client=_make_sync_client(_thinking_response()))

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "What is 6*7?"}])
    result = client.completion(request)

    assert result.message.content == "The answer is 42."
    assert result.message.reasoning_content == "Let me reason step by step."


def test_completion_extracts_system_to_top_level() -> None:
    sync_mock = _make_sync_client(_text_response())
    client = _make_client(sync_client=sync_mock)

    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ],
    )
    client.completion(request)

    payload = sync_mock.post.call_args.kwargs["json"]
    assert payload["system"] == "You are helpful."
    assert all(msg["role"] != "system" for msg in payload["messages"])
    assert len(payload["messages"]) == 1


def test_completion_concatenates_multiple_system_messages() -> None:
    sync_mock = _make_sync_client(_text_response())
    client = _make_client(sync_client=sync_mock)

    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Be concise."},
            {"role": "system", "content": "Answer in English."},
            {"role": "user", "content": "Hi"},
        ],
    )
    client.completion(request)

    payload = sync_mock.post.call_args.kwargs["json"]
    assert payload["system"] == "Be concise.\n\nAnswer in English."


def test_completion_extracts_system_from_chat_message_blocks() -> None:
    sync_mock = _make_sync_client(_text_response())
    client = _make_client(sync_client=sync_mock)

    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            ChatMessage.as_system("You are helpful.").to_dict(),
            ChatMessage.as_user("Hi").to_dict(),
        ],
    )
    client.completion(request)

    payload = sync_mock.post.call_args.kwargs["json"]
    assert payload["system"] == "You are helpful."
    assert payload["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]


def test_completion_posts_to_messages_route() -> None:
    sync_mock = _make_sync_client(_text_response())
    client = _make_client(sync_client=sync_mock)

    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": "Hi"}],
        temperature=0.7,
    )
    client.completion(request)

    call_url = sync_mock.post.call_args.args[0]
    assert "/v1/messages" in call_url
    payload = sync_mock.post.call_args.kwargs["json"]
    assert payload["model"] == MODEL
    assert payload["temperature"] == 0.7


def test_completion_defaults_max_tokens() -> None:
    sync_mock = _make_sync_client(_text_response())
    client = _make_client(sync_client=sync_mock)

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    client.completion(request)

    payload = sync_mock.post.call_args.kwargs["json"]
    assert payload["max_tokens"] == 4096


def test_completion_forwards_explicit_max_tokens() -> None:
    sync_mock = _make_sync_client(_text_response())
    client = _make_client(sync_client=sync_mock)

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}], max_tokens=1024)
    client.completion(request)

    payload = sync_mock.post.call_args.kwargs["json"]
    assert payload["max_tokens"] == 1024


def test_completion_maps_stop_to_stop_sequences() -> None:
    sync_mock = _make_sync_client(_text_response())
    client = _make_client(sync_client=sync_mock)

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}], stop=["END", "STOP"])
    client.completion(request)

    payload = sync_mock.post.call_args.kwargs["json"]
    assert payload["stop_sequences"] == ["END", "STOP"]
    assert "stop" not in payload


def test_completion_maps_stop_string_to_list() -> None:
    sync_mock = _make_sync_client(_text_response())
    client = _make_client(sync_client=sync_mock)

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}], stop="END")
    client.completion(request)

    payload = sync_mock.post.call_args.kwargs["json"]
    assert payload["stop_sequences"] == ["END"]


def test_completion_translates_openai_tool_schema_to_anthropic() -> None:
    sync_mock = _make_sync_client(_text_response())
    client = _make_client(sync_client=sync_mock)

    tools = [
        MCPToolDefinition(
            name="search",
            description="Search the knowledge base.",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        ).to_openai_tool_schema()
    ]
    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}], tools=tools)
    client.completion(request)

    payload = sync_mock.post.call_args.kwargs["json"]
    assert payload["tools"] == [
        {
            "name": "search",
            "description": "Search the knowledge base.",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
        }
    ]


def test_completion_translates_tool_turns_from_chat_messages() -> None:
    sync_mock = _make_sync_client(_text_response())
    client = _make_client(sync_client=sync_mock)

    assistant_message = ChatMessage.as_assistant(
        content="Let me check.",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"query": "weather"}'},
            }
        ],
    )
    tool_message = ChatMessage.as_tool(content="Sunny and 72F", tool_call_id="call_1")
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            ChatMessage.as_user("What's the weather?").to_dict(),
            assistant_message.to_dict(),
            tool_message.to_dict(),
        ],
        tools=[
            MCPToolDefinition(
                name="search",
                description="Search the knowledge base.",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            ).to_openai_tool_schema()
        ],
    )
    client.completion(request)

    payload = sync_mock.post.call_args.kwargs["json"]
    assert payload["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "What's the weather?"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "call_1", "name": "search", "input": {"query": "weather"}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "Sunny and 72F"}],
        },
    ]


def test_completion_merges_parallel_tool_results_into_single_user_message() -> None:
    sync_mock = _make_sync_client(_text_response())
    client = _make_client(sync_client=sync_mock)

    assistant_message = ChatMessage.as_assistant(
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup_city", "arguments": '{"city": "Paris"}'},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "lookup_weather", "arguments": '{"city": "Paris"}'},
            },
        ]
    )
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            ChatMessage.as_user("Plan my day.").to_dict(),
            assistant_message.to_dict(),
            ChatMessage.as_tool(content="City found", tool_call_id="call_1").to_dict(),
            ChatMessage.as_tool(content="Sunny", tool_call_id="call_2").to_dict(),
        ],
    )
    client.completion(request)

    payload = sync_mock.post.call_args.kwargs["json"]
    assert payload["messages"][-1] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "call_1", "content": "City found"},
            {"type": "tool_result", "tool_use_id": "call_2", "content": "Sunny"},
        ],
    }


def test_completion_excludes_openai_specific_params() -> None:
    sync_mock = _make_sync_client(_text_response())
    client = _make_client(sync_client=sync_mock)

    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": "Hi"}],
        response_format={"type": "json_object"},
        frequency_penalty=0.5,
        presence_penalty=0.5,
        seed=42,
    )
    client.completion(request)

    payload = sync_mock.post.call_args.kwargs["json"]
    for field in ("response_format", "frequency_penalty", "presence_penalty", "seed"):
        assert field not in payload, f"{field!r} should be excluded from Anthropic payload"


def test_completion_empty_content_returns_none() -> None:
    response = {"content": [], "usage": {"input_tokens": 5, "output_tokens": 0}}
    client = _make_client(sync_client=_make_sync_client(response))

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    result = client.completion(request)

    assert result.message.content is None
    assert result.message.reasoning_content is None
    assert result.message.tool_calls == []


@pytest.mark.asyncio
async def test_acompletion_maps_text_content() -> None:
    client = _make_client(async_client=_make_async_client(_text_response(text="async result")))

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    result = await client.acompletion(request)

    assert result.message.content == "async result"


# --- Image content block translation ---


def test_completion_translates_data_uri_image_blocks() -> None:
    sync_mock = _make_sync_client(_text_response())
    client = _make_client(sync_client=sync_mock)

    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBOR...", "format": "png"},
                    },
                    {"type": "text", "text": "What is this?"},
                ],
            },
        ],
    )
    client.completion(request)

    payload = sync_mock.post.call_args.kwargs["json"]
    content = payload["messages"][0]["content"]
    assert content[0] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "iVBOR..."},
    }
    assert content[1] == {"type": "text", "text": "What is this?"}


def test_completion_translates_url_string_image_blocks() -> None:
    sync_mock = _make_sync_client(_text_response())
    client = _make_client(sync_client=sync_mock)

    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": "https://example.com/cat.png"},
                    {"type": "text", "text": "Describe this."},
                ],
            },
        ],
    )
    client.completion(request)

    payload = sync_mock.post.call_args.kwargs["json"]
    content = payload["messages"][0]["content"]
    assert content[0] == {
        "type": "image",
        "source": {"type": "url", "url": "https://example.com/cat.png"},
    }


def test_completion_translates_data_uri_string_image_blocks() -> None:
    sync_mock = _make_sync_client(_text_response())
    client = _make_client(sync_client=sync_mock)

    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": "data:image/jpeg;base64,/9j/4AAQ..."},
                    {"type": "text", "text": "What is this?"},
                ],
            },
        ],
    )
    client.completion(request)

    payload = sync_mock.post.call_args.kwargs["json"]
    content = payload["messages"][0]["content"]
    assert content[0] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": "/9j/4AAQ..."},
    }


def test_completion_preserves_non_image_content_blocks() -> None:
    sync_mock = _make_sync_client(_text_response())
    client = _make_client(sync_client=sync_mock)

    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "custom_block", "data": "something"},
                ],
            },
        ],
    )
    client.completion(request)

    payload = sync_mock.post.call_args.kwargs["json"]
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Hello"}
    assert content[1] == {"type": "custom_block", "data": "something"}


def test_completion_passes_string_content_unchanged() -> None:
    sync_mock = _make_sync_client(_text_response())
    client = _make_client(sync_client=sync_mock)

    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": "plain text"}],
    )
    client.completion(request)

    payload = sync_mock.post.call_args.kwargs["json"]
    assert payload["messages"][0]["content"] == "plain text"


# --- Auth headers ---


def test_auth_headers_use_x_api_key() -> None:
    sync_mock = _make_sync_client(_text_response())
    client = _make_client(sync_client=sync_mock)

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    client.completion(request)

    headers = sync_mock.post.call_args.kwargs["headers"]
    assert headers["x-api-key"] == "sk-ant-test"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["Content-Type"] == "application/json"
    assert "Authorization" not in headers


def test_no_api_key_header_when_key_is_none() -> None:
    sync_mock = _make_sync_client(_text_response())
    client = _make_client(sync_client=sync_mock, api_key=None)

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    client.completion(request)

    headers = sync_mock.post.call_args.kwargs["headers"]
    assert "x-api-key" not in headers
    assert headers["anthropic-version"] == "2023-06-01"


def test_extra_headers_merged() -> None:
    sync_mock = _make_sync_client(_text_response())
    client = _make_client(sync_client=sync_mock)

    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": "Hi"}],
        extra_headers={"X-Custom": "val"},
    )
    client.completion(request)

    headers = sync_mock.post.call_args.kwargs["headers"]
    assert headers["X-Custom"] == "val"
    assert headers["x-api-key"] == "sk-ant-test"


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
def test_http_error_maps_to_provider_error(status_code: int, expected_kind: ProviderErrorKind) -> None:
    client = _make_client(
        sync_client=_make_sync_client({"error": {"type": "error", "message": "fail"}}, status_code=status_code)
    )

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    with pytest.raises(ProviderError) as exc_info:
        client.completion(request)

    assert exc_info.value.kind == expected_kind


def test_transport_timeout_raises_provider_error() -> None:
    sync_mock = MagicMock()
    sync_mock.post = MagicMock(side_effect=TimeoutError("timed out"))
    client = _make_client(sync_client=sync_mock)

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    with pytest.raises(ProviderError) as exc_info:
        client.completion(request)

    assert exc_info.value.kind == ProviderErrorKind.TIMEOUT


# --- Unsupported capabilities ---


def test_embeddings_raises_unsupported() -> None:
    client = _make_client()
    with pytest.raises(ProviderError) as exc_info:
        client.embeddings(EmbeddingRequest(model=MODEL, inputs=["hello"]))

    assert exc_info.value.kind == ProviderErrorKind.UNSUPPORTED_CAPABILITY
    assert "embeddings" in exc_info.value.message


def test_generate_image_raises_unsupported() -> None:
    client = _make_client()
    with pytest.raises(ProviderError) as exc_info:
        client.generate_image(ImageGenerationRequest(model=MODEL, prompt="a cat"))

    assert exc_info.value.kind == ProviderErrorKind.UNSUPPORTED_CAPABILITY
    assert "image-generation" in exc_info.value.message


@pytest.mark.asyncio
async def test_aembeddings_raises_unsupported() -> None:
    client = _make_client()
    with pytest.raises(ProviderError) as exc_info:
        await client.aembeddings(EmbeddingRequest(model=MODEL, inputs=["hello"]))

    assert exc_info.value.kind == ProviderErrorKind.UNSUPPORTED_CAPABILITY


@pytest.mark.asyncio
async def test_agenerate_image_raises_unsupported() -> None:
    client = _make_client()
    with pytest.raises(ProviderError) as exc_info:
        await client.agenerate_image(ImageGenerationRequest(model=MODEL, prompt="a cat"))

    assert exc_info.value.kind == ProviderErrorKind.UNSUPPORTED_CAPABILITY


# --- Lifecycle ---


def test_close_delegates_to_httpx_client() -> None:
    sync_mock = MagicMock()
    client = _make_client(sync_client=sync_mock)
    client.close()
    sync_mock.close.assert_called_once()


@pytest.mark.asyncio
async def test_aclose_closes_both_clients() -> None:
    sync_mock = MagicMock()
    async_mock = MagicMock()
    async_mock.aclose = AsyncMock()
    client = _make_client(sync_client=sync_mock, async_client=async_mock)

    await client.aclose()

    async_mock.aclose.assert_awaited_once()
    sync_mock.close.assert_called_once()


def test_close_noop_when_no_client_created() -> None:
    client = _make_client()
    client.close()


def test_completion_raises_after_close_without_recreating_client() -> None:
    client = _make_client()
    client.close()

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    with pytest.raises(RuntimeError, match="Model client is closed\\."):
        client.completion(request)

    assert client._client is None


@pytest.mark.asyncio
async def test_aclose_noop_when_no_client_created() -> None:
    client = _make_client()
    await client.aclose()


@pytest.mark.asyncio
async def test_acompletion_raises_after_aclose_without_recreating_client() -> None:
    client = _make_client()
    await client.aclose()

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    with pytest.raises(RuntimeError, match="Model client is closed\\."):
        await client.acompletion(request)

    assert client._aclient is None


# --- Capabilities ---


@pytest.mark.parametrize(
    "method,expected",
    [
        ("supports_chat_completion", True),
        ("supports_embeddings", False),
        ("supports_image_generation", False),
    ],
)
def test_capability_checks(method: str, expected: bool) -> None:
    client = _make_client()
    assert getattr(client, method)() is expected
