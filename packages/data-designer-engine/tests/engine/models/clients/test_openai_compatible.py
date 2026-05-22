# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from data_designer.config.models import Modality
from data_designer.config.utils.media_helpers import get_media_base64_context, get_media_url_context
from data_designer.engine.models.clients.adapters.http_model_client import ClientConcurrencyMode
from data_designer.engine.models.clients.adapters.openai_compatible import OpenAICompatibleClient
from data_designer.engine.models.clients.errors import ProviderError, ProviderErrorKind
from data_designer.engine.models.clients.types import (
    ChatCompletionRequest,
    EmbeddingRequest,
    ImageGenerationRequest,
)
from data_designer.engine.models.usage import TokenCountSource
from tests.engine.models.clients.conftest import make_mock_async_client, make_mock_sync_client

PROVIDER = "test-provider"
MODEL = "gpt-test"
ENDPOINT = "https://api.example.com/v1"


def _make_client(
    *,
    sync_client: MagicMock | None = None,
    async_client: MagicMock | None = None,
    api_key: str | None = "sk-test-key",
) -> OpenAICompatibleClient:
    concurrency_mode = ClientConcurrencyMode.ASYNC if async_client is not None else ClientConcurrencyMode.SYNC
    return OpenAICompatibleClient(
        provider_name=PROVIDER,
        endpoint=ENDPOINT,
        api_key=api_key,
        concurrency_mode=concurrency_mode,
        sync_client=sync_client,
        async_client=async_client,
    )


# --- Response helpers ---


def _chat_response(
    content: str = "Hello!",
    reasoning: str | None = None,
    reasoning_tokens: int | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning"] = reasoning
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens} if reasoning_tokens is not None else {},
            "total_tokens": 15,
        },
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
    response_json = _chat_response(content="Hello!", reasoning="step-by-step", reasoning_tokens=3)
    client = _make_client(sync_client=make_mock_sync_client(response_json))

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    result = client.completion(request)

    assert result.message.content == "Hello!"
    assert result.message.reasoning_content == "step-by-step"
    assert result.usage is not None
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5
    assert result.usage.reasoning_tokens == 3
    assert result.usage.reasoning_token_count_source == TokenCountSource.PROVIDER


def test_completion_with_tool_calls() -> None:
    tool_calls = [{"id": "tc1", "type": "function", "function": {"name": "search", "arguments": '{"q": "x"}'}}]
    client = _make_client(sync_client=make_mock_sync_client(_chat_response(tool_calls=tool_calls)))

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Search"}])
    result = client.completion(request)

    assert len(result.message.tool_calls) == 1
    assert result.message.tool_calls[0].name == "search"
    assert result.message.tool_calls[0].arguments_json == '{"q": "x"}'


def test_completion_posts_to_chat_completions_route() -> None:
    sync_mock = make_mock_sync_client(_chat_response())
    client = _make_client(sync_client=sync_mock)

    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": "Hi"}],
        n=4,
        temperature=0.7,
        extra_body={"seed": 42},
        extra_headers={"X-Trace": "1"},
    )
    client.completion(request)

    call_args = sync_mock.post.call_args
    assert "/chat/completions" in call_args.args[0]
    payload = call_args.kwargs["json"]
    assert payload["model"] == MODEL
    assert payload["n"] == 4
    assert payload["temperature"] == 0.7
    assert payload["seed"] == 42
    assert "timeout" not in payload
    assert call_args.kwargs["headers"]["X-Trace"] == "1"


def test_timeout_excluded_from_body_and_used_as_http_timeout() -> None:
    sync_mock = make_mock_sync_client(_chat_response())
    client = _make_client(sync_client=sync_mock)

    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": "Hi"}],
        timeout=120.0,
    )
    client.completion(request)

    call_args = sync_mock.post.call_args
    payload = call_args.kwargs["json"]
    assert "timeout" not in payload
    http_timeout = call_args.kwargs["timeout"]
    assert http_timeout.connect == 120.0
    assert http_timeout.read == 120.0


def test_default_timeout_used_when_request_timeout_is_none() -> None:
    sync_mock = make_mock_sync_client(_chat_response())
    client = OpenAICompatibleClient(
        provider_name=PROVIDER,
        endpoint=ENDPOINT,
        timeout_s=45.0,
        sync_client=sync_mock,
    )

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    client.completion(request)

    call_args = sync_mock.post.call_args
    http_timeout = call_args.kwargs["timeout"]
    assert http_timeout.connect == 45.0


@pytest.mark.asyncio
async def test_acompletion_maps_canonical_fields() -> None:
    client = _make_client(async_client=make_mock_async_client(_chat_response(content="async result")))

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    result = await client.acompletion(request)

    assert result.message.content == "async result"


# --- Embeddings ---


def test_embeddings_maps_vectors_and_usage() -> None:
    client = _make_client(sync_client=make_mock_sync_client(_embedding_response()))

    request = EmbeddingRequest(model=MODEL, inputs=["hello world"])
    result = client.embeddings(request)

    assert result.vectors == [[0.1, 0.2, 0.3]]
    assert result.usage is not None
    assert result.usage.input_tokens == 5


def test_embeddings_posts_to_embeddings_route() -> None:
    sync_mock = make_mock_sync_client(_embedding_response())
    client = _make_client(sync_client=sync_mock)

    request = EmbeddingRequest(model=MODEL, inputs=["hello"])
    client.embeddings(request)

    call_url = sync_mock.post.call_args.args[0]
    assert "/embeddings" in call_url


@pytest.mark.asyncio
async def test_aembeddings_maps_vectors() -> None:
    client = _make_client(async_client=make_mock_async_client(_embedding_response()))

    request = EmbeddingRequest(model=MODEL, inputs=["hello"])
    result = await client.aembeddings(request)

    assert len(result.vectors) == 1


# --- Image generation ---


def test_generate_image_diffusion_route() -> None:
    sync_mock = make_mock_sync_client(_image_response())
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
    sync_mock = make_mock_sync_client(chat_img_response)
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
    client = _make_client(async_client=make_mock_async_client(_image_response()))

    request = ImageGenerationRequest(model=MODEL, prompt="a cat")
    result = await client.agenerate_image(request)

    assert len(result.images) == 1


# --- Image URL passthrough ---


def test_completion_forwards_image_url_dict_unchanged() -> None:
    """OpenAI expects image_url blocks as {"url": ...} dicts — verify they pass through as-is."""
    sync_mock = make_mock_sync_client(_chat_response())
    client = _make_client(sync_client=sync_mock)

    image_block = {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}
    text_block = {"type": "text", "text": "Describe this."}
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": [image_block, text_block]}],
    )
    client.completion(request)

    payload = sync_mock.post.call_args.kwargs["json"]
    content = payload["messages"][0]["content"]
    assert content == [image_block, text_block]


def test_completion_forwards_base64_image_url_dict_unchanged() -> None:
    """Base64 data-URI image blocks should also pass through unmodified."""
    sync_mock = make_mock_sync_client(_chat_response())
    client = _make_client(sync_client=sync_mock)

    image_block = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}}
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": [image_block, {"type": "text", "text": "What is this?"}]}],
    )
    client.completion(request)

    payload = sync_mock.post.call_args.kwargs["json"]
    content = payload["messages"][0]["content"]
    assert content[0] == image_block


def test_completion_forwards_multimodal_tool_result_content_unchanged() -> None:
    """OpenAI-compatible VLM backends receive canonical multimodal tool content."""
    sync_mock = make_mock_sync_client(_chat_response())
    client = _make_client(sync_client=sync_mock)

    content = [
        {"type": "text", "text": "Rendered page:"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}},
    ]
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "tool", "tool_call_id": "call-1", "content": content}],
    )
    client.completion(request)

    payload = sync_mock.post.call_args.kwargs["json"]
    assert payload["messages"][0]["content"] == content


def test_completion_translates_canonical_image_blocks() -> None:
    sync_mock = make_mock_sync_client(_chat_response())
    client = _make_client(sync_client=sync_mock)

    image_block = get_media_base64_context(Modality.IMAGE.value, "image/png", "iVBOR...")
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": [image_block, {"type": "text", "text": "What is this?"}]}],
    )
    client.completion(request)

    payload = sync_mock.post.call_args.kwargs["json"]
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}}


def test_completion_translates_base64_audio_blocks() -> None:
    sync_mock = make_mock_sync_client(_chat_response())
    client = _make_client(sync_client=sync_mock)

    audio_block = get_media_base64_context(Modality.AUDIO.value, "audio/mpeg", "abc123")
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": [audio_block, {"type": "text", "text": "Transcribe this."}]}],
    )
    client.completion(request)

    payload = sync_mock.post.call_args.kwargs["json"]
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "input_audio", "input_audio": {"data": "abc123", "format": "mp3"}}


def test_completion_rejects_unsupported_canonical_audio_media_type() -> None:
    sync_mock = make_mock_sync_client(_chat_response())
    client = _make_client(sync_client=sync_mock)

    audio_block = get_media_base64_context(Modality.AUDIO.value, "audio/flac", "abc123")
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": [audio_block, {"type": "text", "text": "Transcribe this."}]}],
    )

    with pytest.raises(ProviderError) as exc_info:
        client.completion(request)

    assert exc_info.value.kind == ProviderErrorKind.BAD_REQUEST
    assert exc_info.value.provider_name == PROVIDER
    assert exc_info.value.model_name == MODEL
    assert "audio/flac" in exc_info.value.message
    sync_mock.post.assert_not_called()


def test_completion_translates_audio_url_blocks() -> None:
    sync_mock = make_mock_sync_client(_chat_response())
    client = _make_client(sync_client=sync_mock)

    audio_block = get_media_url_context(Modality.AUDIO.value, "https://example.com/download?id=123")
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": [audio_block, {"type": "text", "text": "Transcribe this."}]}],
    )
    client.completion(request)

    payload = sync_mock.post.call_args.kwargs["json"]
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "audio_url", "audio_url": {"url": "https://example.com/download?id=123"}}


def test_completion_translates_local_audio_path_blocks() -> None:
    sync_mock = make_mock_sync_client(_chat_response())
    client = _make_client(sync_client=sync_mock)

    audio_block = get_media_url_context(Modality.AUDIO.value, "recordings/speech.wav")
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": [audio_block, {"type": "text", "text": "Transcribe this."}]}],
    )
    client.completion(request)

    payload = sync_mock.post.call_args.kwargs["json"]
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "audio_url", "audio_url": {"url": "recordings/speech.wav"}}


def test_completion_translates_video_blocks() -> None:
    sync_mock = make_mock_sync_client(_chat_response())
    client = _make_client(sync_client=sync_mock)

    video_block = get_media_url_context(Modality.VIDEO.value, "https://example.com/download?id=123")
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": [video_block, {"type": "text", "text": "Describe this."}]}],
    )
    client.completion(request)

    payload = sync_mock.post.call_args.kwargs["json"]
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "video_url", "video_url": {"url": "https://example.com/download?id=123"}}


def test_completion_translates_local_video_path_blocks() -> None:
    sync_mock = make_mock_sync_client(_chat_response())
    client = _make_client(sync_client=sync_mock)

    video_block = get_media_url_context(Modality.VIDEO.value, "clips/screen_recording.mp4")
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": [video_block, {"type": "text", "text": "Describe this."}]}],
    )
    client.completion(request)

    payload = sync_mock.post.call_args.kwargs["json"]
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "video_url", "video_url": {"url": "clips/screen_recording.mp4"}}


def test_completion_translates_base64_video_blocks() -> None:
    sync_mock = make_mock_sync_client(_chat_response())
    client = _make_client(sync_client=sync_mock)

    video_block = get_media_base64_context(Modality.VIDEO.value, "video/mp4", "abc123")
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": [video_block, {"type": "text", "text": "Describe this."}]}],
    )
    client.completion(request)

    payload = sync_mock.post.call_args.kwargs["json"]
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,abc123"}}


# --- Auth headers ---


def test_auth_header_present_when_api_key_set() -> None:
    sync_mock = make_mock_sync_client(_chat_response())
    client = _make_client(sync_client=sync_mock)

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    client.completion(request)

    headers = sync_mock.post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer sk-test-key"
    assert headers["Content-Type"] == "application/json"


def test_no_auth_header_when_api_key_none() -> None:
    sync_mock = make_mock_sync_client(_chat_response())
    client = _make_client(sync_client=sync_mock, api_key=None)

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    client.completion(request)

    headers = sync_mock.post.call_args.kwargs["headers"]
    assert "Authorization" not in headers


def test_extra_headers_merged_into_auth_headers() -> None:
    sync_mock = make_mock_sync_client(_chat_response())
    client = _make_client(sync_client=sync_mock)

    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": "Hi"}],
        extra_headers={"X-Custom": "val"},
    )
    client.completion(request)

    headers = sync_mock.post.call_args.kwargs["headers"]
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
    client = _make_client(sync_client=make_mock_sync_client({"error": {"message": "fail"}}, status_code=status_code))

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


def test_non_json_response_raises_provider_error() -> None:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.side_effect = ValueError("not json")
    sync_mock = MagicMock()
    sync_mock.post = MagicMock(return_value=resp)
    client = _make_client(sync_client=sync_mock)

    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "Hi"}])
    with pytest.raises(ProviderError) as exc_info:
        client.completion(request)

    assert exc_info.value.kind == ProviderErrorKind.API_ERROR
    assert "non-JSON" in exc_info.value.message


# --- Capabilities ---


@pytest.mark.parametrize(
    "method",
    ["supports_chat_completion", "supports_embeddings", "supports_image_generation"],
)
def test_capability_checks_return_true(method: str) -> None:
    client = _make_client()
    assert getattr(client, method)() is True
