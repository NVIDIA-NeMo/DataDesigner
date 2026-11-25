# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, NoReturn

import httpx
import pytest
import requests

from data_designer.config.models import ImageInferenceParams, ModelConfig, ModelProvider
from data_designer.engine.model_provider import ModelProviderRegistry
from data_designer.engine.models.clients.adapters.http_model_client import ClientConcurrencyMode
from data_designer.engine.models.clients.adapters.openai_compatible import OpenAICompatibleClient
from data_designer.engine.models.clients.parsing import (
    aparse_chat_completion_response,
    aparse_image_payload,
    parse_chat_completion_response,
    parse_image_payload,
)
from data_designer.engine.models.clients.types import ImageGenerationRequest
from data_designer.engine.models.errors import ImageGenerationError
from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.models.usage_events import TokenUsageEvent, subscribe_token_usage
from tests.engine.models.clients.conftest import make_mock_async_client, make_mock_sync_client

INLINE_IMAGE = "aW1hZ2VkYXRh" * 10


@pytest.fixture
def media_downloads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    attempts: list[str] = []

    def reject_download(*args: Any, **kwargs: Any) -> NoReturn:
        attempts.append("download")
        raise AssertionError("Response parsing must not download image URLs")

    monkeypatch.setattr(requests, "get", reject_download)
    monkeypatch.setattr(httpx.Client, "get", reject_download)
    monkeypatch.setattr(httpx.AsyncClient, "get", reject_download)
    return attempts


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/private.json",
        "http://169.254.169.254/latest/meta-data/",
        "https://images.example.com/image.png?signature=synthetic",
    ],
)
@pytest.mark.parametrize("shape", ["string", "url", "image_url", "nested_image_url", "object"])
async def test_image_url_payloads_are_not_downloaded(
    is_async: bool, url: str, shape: str, media_downloads: list[str]
) -> None:
    raw = {
        "string": url,
        "url": {"url": url},
        "image_url": {"image_url": url},
        "nested_image_url": {"image_url": {"url": url}},
        "object": SimpleNamespace(url=url),
    }[shape]

    result = await aparse_image_payload(raw) if is_async else parse_image_payload(raw)

    assert result is None
    assert media_downloads == []


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "raw",
    [
        INLINE_IMAGE,
        {"b64_json": INLINE_IMAGE},
        SimpleNamespace(b64_json=INLINE_IMAGE),
        f"data:image/png;base64,{INLINE_IMAGE}",
        {"image_url": {"url": f"data:image/png;base64,{INLINE_IMAGE}"}},
        {"b64_json": INLINE_IMAGE, "url": "http://127.0.0.1/private.json"},
    ],
)
async def test_inline_image_payloads_remain_supported(is_async: bool, raw: Any, media_downloads: list[str]) -> None:
    result = await aparse_image_payload(raw) if is_async else parse_image_payload(raw)

    assert result is not None
    assert result.b64_data == INLINE_IMAGE
    assert media_downloads == []


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("has_image_field", [False, True], ids=["text-only", "image-field"])
async def test_chat_url_content_is_preserved_without_downloads(
    is_async: bool, has_image_field: bool, media_downloads: list[str]
) -> None:
    url = "http://127.0.0.1:8080/private.json"
    message: dict[str, Any] = {"role": "assistant", "content": url}
    if has_image_field:
        message["images"] = [{"image_url": {"url": url}}]
    response = {"choices": [{"message": message}]}

    result = await aparse_chat_completion_response(response) if is_async else parse_chat_completion_response(response)

    assert result.message.content == url
    assert result.message.images == []
    assert media_downloads == []


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("is_chat_route", [False, True], ids=["images-api", "chat-api"])
@pytest.mark.parametrize("include_inline", [False, True], ids=["url-only", "mixed-inline-and-url"])
async def test_image_generation_responses_use_only_inline_images(
    is_async: bool, is_chat_route: bool, include_inline: bool, media_downloads: list[str]
) -> None:
    raw_images = [{"url": "http://127.0.0.1:8080/private.json"}]
    if include_inline:
        raw_images.append({"b64_json": INLINE_IMAGE})
    response = (
        {"choices": [{"message": {"content": None, "images": raw_images}}]} if is_chat_route else {"data": raw_images}
    )
    client = OpenAICompatibleClient(
        provider_name="synthetic",
        endpoint="https://provider.example.com/v1",
        concurrency_mode=ClientConcurrencyMode.ASYNC if is_async else ClientConcurrencyMode.SYNC,
        async_client=make_mock_async_client(response) if is_async else None,
        sync_client=make_mock_sync_client(response) if not is_async else None,
    )
    request = ImageGenerationRequest(
        model="synthetic-image-model",
        prompt="Generate an image",
        messages=[{"role": "user", "content": "Generate an image"}] if is_chat_route else None,
    )

    result = await client.agenerate_image(request) if is_async else client.generate_image(request)

    assert [image.b64_data for image in result.images] == ([INLINE_IMAGE] if include_inline else [])
    assert media_downloads == []


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("model", "extra_body", "route"),
    [
        ("gpt-image-1.5", {"size": "1024x1024", "quality": "high"}, "/images/generations"),
        ("dall-e-3", {"response_format": "b64_json"}, "/images/generations"),
        ("black-forest-labs/flux.2-pro", {"height": 512, "width": 512}, "/chat/completions"),
    ],
)
async def test_documented_image_models_request_inline_output_on_their_supported_route(
    is_async: bool, model: str, extra_body: dict[str, str | int], route: str, media_downloads: list[str]
) -> None:
    raw_images = [{"b64_json": INLINE_IMAGE}]
    response = (
        {"data": raw_images} if route == "/images/generations" else {"choices": [{"message": {"images": raw_images}}]}
    )
    http_client = make_mock_async_client(response) if is_async else make_mock_sync_client(response)
    provider = ModelProvider(name="example", endpoint="https://provider.example.com/v1")
    client = OpenAICompatibleClient(
        provider_name=provider.name,
        endpoint=provider.endpoint,
        concurrency_mode=ClientConcurrencyMode.ASYNC if is_async else ClientConcurrencyMode.SYNC,
        async_client=http_client if is_async else None,
        sync_client=http_client if not is_async else None,
    )
    facade = ModelFacade(
        model_config=ModelConfig(
            alias="image",
            model=model,
            provider=provider.name,
            inference_parameters=ImageInferenceParams(extra_body=extra_body),
        ),
        model_provider_registry=ModelProviderRegistry(providers=[provider]),
        client=client,
    )

    images = (
        await facade.agenerate_image(prompt="Draw a circle")
        if is_async
        else facade.generate_image(prompt="Draw a circle")
    )

    assert images == [INLINE_IMAGE]
    assert http_client.post.call_args.args[0] == provider.endpoint + route
    payload = http_client.post.call_args.kwargs["json"]
    assert payload["model"] == model
    for key, value in extra_body.items():
        assert payload[key] == value
    if "response_format" not in extra_body:
        assert "response_format" not in payload
    assert media_downloads == []


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("is_chat_route", [False, True], ids=["images-api", "chat-api"])
@pytest.mark.parametrize("skip_usage_tracking", [False, True], ids=["track-usage", "skip-usage"])
async def test_url_only_image_failure_preserves_provider_token_usage(
    is_async: bool, is_chat_route: bool, skip_usage_tracking: bool, media_downloads: list[str]
) -> None:
    raw_images = [{"url": "http://127.0.0.1:8080/private.json"}]
    response = {"choices": [{"message": {"images": raw_images}}]} if is_chat_route else {"data": raw_images}
    response["usage"] = {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46}
    provider = ModelProvider(name="example", endpoint="https://provider.example.com/v1")
    client = OpenAICompatibleClient(
        provider_name=provider.name,
        endpoint=provider.endpoint,
        concurrency_mode=ClientConcurrencyMode.ASYNC if is_async else ClientConcurrencyMode.SYNC,
        async_client=make_mock_async_client(response) if is_async else None,
        sync_client=make_mock_sync_client(response) if not is_async else None,
    )
    facade = ModelFacade(
        model_config=ModelConfig(
            alias="image",
            model="black-forest-labs/flux.2-pro" if is_chat_route else "gpt-image-1.5",
            provider=provider.name,
            inference_parameters=ImageInferenceParams(),
        ),
        model_provider_registry=ModelProviderRegistry(providers=[provider]),
        client=client,
    )
    events: list[TokenUsageEvent] = []
    unsubscribe = subscribe_token_usage(events.append)
    try:
        with pytest.raises(ImageGenerationError) as exc_info:
            if is_async:
                await facade.agenerate_image(prompt="Draw a circle", skip_usage_tracking=skip_usage_tracking)
            else:
                facade.generate_image(prompt="Draw a circle", skip_usage_tracking=skip_usage_tracking)
    finally:
        unsubscribe()

    message = str(exc_info.value)
    assert 'ImageInferenceParams(extra_body={"response_format": "b64_json"})' in message
    assert "Images API" in message
    assert "OpenRouter" in message
    assert "object" in message
    assert facade.usage_stats.request_usage.failed_requests == (0 if skip_usage_tracking else 1)
    assert facade.usage_stats.request_usage.successful_requests == 0
    assert facade.usage_stats.image_usage.total_images == 0
    assert facade.usage_stats.token_usage.input_tokens == (0 if skip_usage_tracking else 12)
    assert facade.usage_stats.token_usage.output_tokens == (0 if skip_usage_tracking else 34)
    assert len(events) == (0 if skip_usage_tracking else 1)
    if events:
        assert events[0].input_tokens == 12
        assert events[0].output_tokens == 34
    assert media_downloads == []
