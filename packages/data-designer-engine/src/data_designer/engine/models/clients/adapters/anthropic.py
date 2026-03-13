# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import re
import threading
from typing import TYPE_CHECKING, Any

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.models.clients.base import ModelClient
from data_designer.engine.models.clients.errors import (
    ProviderError,
    ProviderErrorKind,
    infer_error_kind_from_exception,
    map_http_error_to_provider_error,
)
from data_designer.engine.models.clients.parsing import extract_usage
from data_designer.engine.models.clients.retry import RetryConfig, create_retry_transport
from data_designer.engine.models.clients.types import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ToolCall,
    TransportKwargs,
    Usage,
)

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 4096
_ANTHROPIC_VERSION = "2023-06-01"

# Fields handled explicitly and excluded from TransportKwargs forwarding.
_ANTHROPIC_EXCLUDE = frozenset({"stop", "max_tokens"})


class AnthropicClient(ModelClient):
    """Native HTTP adapter for the Anthropic Messages API.

    Uses ``httpx`` with ``httpx_retries.RetryTransport`` for resilient HTTP
    calls.  Concurrency / throttle policy is an orchestration concern and
    is not managed here — see ``ThrottleManager`` and ``AsyncTaskScheduler``.
    """

    _ROUTE_MESSAGES = "/v1/messages"

    def __init__(
        self,
        *,
        provider_name: str,
        model_id: str,
        endpoint: str,
        api_key: str | None = None,
        retry_config: RetryConfig | None = None,
        max_parallel_requests: int = 32,
        timeout_s: float = 60.0,
        sync_client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.provider_name = provider_name
        self._model_id = model_id
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._retry_config = retry_config

        pool_max = max(32, 2 * max_parallel_requests)
        pool_keepalive = max(16, max_parallel_requests)
        self._limits = lazy.httpx.Limits(
            max_connections=pool_max,
            max_keepalive_connections=pool_keepalive,
        )
        self._transport = create_retry_transport(self._retry_config)
        self._client: httpx.Client | None = sync_client
        self._aclient: httpx.AsyncClient | None = async_client
        self._init_lock = threading.Lock()

    def _get_sync_client(self) -> httpx.Client:
        if self._client is None:
            with self._init_lock:
                if self._client is None:
                    self._client = lazy.httpx.Client(
                        transport=self._transport,
                        limits=self._limits,
                        timeout=lazy.httpx.Timeout(self._timeout_s),
                    )
        return self._client

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._aclient is None:
            with self._init_lock:
                if self._aclient is None:
                    self._aclient = lazy.httpx.AsyncClient(
                        transport=self._transport,
                        limits=self._limits,
                        timeout=lazy.httpx.Timeout(self._timeout_s),
                    )
        return self._aclient

    # -------------------------------------------------------------------
    # Capability checks
    # -------------------------------------------------------------------

    def supports_chat_completion(self) -> bool:
        return True

    def supports_embeddings(self) -> bool:
        return False

    def supports_image_generation(self) -> bool:
        return False

    # -------------------------------------------------------------------
    # Chat completion
    # -------------------------------------------------------------------

    def completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        payload = _build_anthropic_payload(request)
        transport = TransportKwargs.from_request(request, exclude=_ANTHROPIC_EXCLUDE)
        payload.update(transport.body)
        response_json = self._post_sync(
            self._ROUTE_MESSAGES, payload, transport.headers, request.model, transport.timeout
        )
        return _parse_anthropic_response(response_json)

    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        payload = _build_anthropic_payload(request)
        transport = TransportKwargs.from_request(request, exclude=_ANTHROPIC_EXCLUDE)
        payload.update(transport.body)
        response_json = await self._apost(
            self._ROUTE_MESSAGES, payload, transport.headers, request.model, transport.timeout
        )
        return _parse_anthropic_response(response_json)

    # -------------------------------------------------------------------
    # Unsupported capabilities
    # -------------------------------------------------------------------

    def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise ProviderError.unsupported_capability(provider_name=self.provider_name, operation="embeddings")

    async def aembeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise ProviderError.unsupported_capability(provider_name=self.provider_name, operation="embeddings")

    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        raise ProviderError.unsupported_capability(provider_name=self.provider_name, operation="image-generation")

    async def agenerate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        raise ProviderError.unsupported_capability(provider_name=self.provider_name, operation="image-generation")

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        if self._aclient is not None:
            await self._aclient.aclose()
            self._aclient = None
        if self._client is not None:
            self._client.close()
            self._client = None

    # -------------------------------------------------------------------
    # HTTP helpers
    # -------------------------------------------------------------------

    def _build_headers(self, extra_headers: dict[str, str]) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "anthropic-version": _ANTHROPIC_VERSION,
        }
        if self._api_key:
            headers["x-api-key"] = self._api_key
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _resolve_timeout(self, per_request: float | None) -> httpx.Timeout:
        return lazy.httpx.Timeout(per_request if per_request is not None else self._timeout_s)

    def _post_sync(
        self,
        route: str,
        payload: dict[str, Any],
        extra_headers: dict[str, str],
        model_name: str,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        url = f"{self._endpoint}{route}"
        headers = self._build_headers(extra_headers)
        try:
            response = self._get_sync_client().post(
                url, json=payload, headers=headers, timeout=self._resolve_timeout(timeout)
            )
        except Exception as exc:
            raise _wrap_transport_error(exc, self.provider_name, model_name) from exc
        if response.status_code >= 400:
            raise map_http_error_to_provider_error(
                response=response, provider_name=self.provider_name, model_name=model_name
            )
        return _parse_json_body(response, self.provider_name, model_name)

    async def _apost(
        self,
        route: str,
        payload: dict[str, Any],
        extra_headers: dict[str, str],
        model_name: str,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        url = f"{self._endpoint}{route}"
        headers = self._build_headers(extra_headers)
        try:
            response = await self._get_async_client().post(
                url, json=payload, headers=headers, timeout=self._resolve_timeout(timeout)
            )
        except Exception as exc:
            raise _wrap_transport_error(exc, self.provider_name, model_name) from exc
        if response.status_code >= 400:
            raise map_http_error_to_provider_error(
                response=response, provider_name=self.provider_name, model_name=model_name
            )
        return _parse_json_body(response, self.provider_name, model_name)


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------


def _build_anthropic_payload(request: ChatCompletionRequest) -> dict[str, Any]:
    """Convert a canonical ``ChatCompletionRequest`` into an Anthropic Messages API payload.

    Extracts system messages to the top-level ``system`` parameter, translates
    OpenAI-format ``image_url`` content blocks to Anthropic ``image`` blocks,
    and maps ``stop`` to ``stop_sequences`` per Anthropic's API contract.
    """
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    for msg in request.messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                system_parts.append(content)
        else:
            messages.append(_translate_message_content(msg))

    payload: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "max_tokens": request.max_tokens if request.max_tokens is not None else _DEFAULT_MAX_TOKENS,
    }

    if system_parts:
        payload["system"] = "\n\n".join(system_parts)

    if request.tools:
        payload["tools"] = request.tools

    if request.stop is not None:
        if isinstance(request.stop, str):
            payload["stop_sequences"] = [request.stop]
        else:
            payload["stop_sequences"] = list(request.stop)

    return payload


def _translate_message_content(msg: dict[str, Any]) -> dict[str, Any]:
    """Translate OpenAI-format content blocks to Anthropic format.

    Rewrites ``image_url`` blocks (emitted by ``MultiModalContext.get_contexts()``)
    to Anthropic's ``image`` block with a nested ``source`` object.  Non-list
    content and non-image blocks pass through unchanged.
    """
    content = msg.get("content")
    if not isinstance(content, list):
        return msg

    translated: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "image_url":
            anthropic_block = _translate_image_url_block(block)
            if anthropic_block is not None:
                translated.append(anthropic_block)
                continue
        translated.append(block)

    return {**msg, "content": translated}


_DATA_URI_RE = re.compile(r"^data:(?P<media_type>[^;]+);base64,(?P<data>.+)$", re.DOTALL)


def _translate_image_url_block(block: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a single OpenAI ``image_url`` block to an Anthropic ``image`` block.

    Handles the three shapes emitted by ``MultiModalContext.get_contexts()``:
    1. Data URI dict: ``{"url": "data:image/png;base64,...", "format": "png"}``
    2. Plain URL string in ``image_url``
    3. Data URI string directly in ``image_url``
    """
    image_url = block.get("image_url")
    if image_url is None:
        return None

    if isinstance(image_url, dict):
        url = image_url.get("url", "")
    else:
        url = str(image_url)

    match = _DATA_URI_RE.match(url)
    if match:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": match.group("media_type"),
                "data": match.group("data"),
            },
        }

    return {
        "type": "image",
        "source": {"type": "url", "url": url},
    }


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_anthropic_response(response_json: dict[str, Any]) -> ChatCompletionResponse:
    """Convert an Anthropic Messages API response into canonical ``ChatCompletionResponse``."""
    content_blocks = response_json.get("content") or []

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in content_blocks:
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text", "")
            if text:
                text_parts.append(text)
        elif block_type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments_json=json.dumps(block.get("input", {})),
                )
            )
        elif block_type == "thinking":
            thinking = block.get("thinking", "")
            if thinking:
                thinking_parts.append(thinking)

    message = AssistantMessage(
        content="\n".join(text_parts) if text_parts else None,
        reasoning_content="\n".join(thinking_parts) if thinking_parts else None,
        tool_calls=tool_calls,
    )

    raw_usage = response_json.get("usage")
    usage: Usage | None = None
    if raw_usage:
        usage = extract_usage(raw_usage)

    return ChatCompletionResponse(message=message, usage=usage, raw=response_json)


# ---------------------------------------------------------------------------
# HTTP helpers (module-level)
# ---------------------------------------------------------------------------


def _parse_json_body(response: httpx.Response, provider_name: str, model_name: str) -> dict[str, Any]:
    """Parse JSON from a successful HTTP response, wrapping decode errors as ``ProviderError``."""
    try:
        return response.json()
    except Exception as exc:
        raise ProviderError(
            kind=ProviderErrorKind.API_ERROR,
            message=f"Provider {provider_name!r} returned a non-JSON response (status {response.status_code}).",
            status_code=response.status_code,
            provider_name=provider_name,
            model_name=model_name,
            cause=exc,
        ) from exc


def _wrap_transport_error(exc: Exception, provider_name: str, model_name: str) -> ProviderError:
    """Convert httpx transport exceptions into canonical ``ProviderError``."""
    return ProviderError(
        kind=infer_error_kind_from_exception(exc),
        message=str(exc) or f"Transport error from provider {provider_name!r}",
        provider_name=provider_name,
        model_name=model_name,
        cause=exc,
    )
