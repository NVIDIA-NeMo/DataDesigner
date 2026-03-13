# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
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
from data_designer.engine.models.clients.parsing import (
    aextract_images_from_chat_response,
    aextract_images_from_image_response,
    aparse_chat_completion_response,
    extract_embedding_vector,
    extract_images_from_chat_response,
    extract_images_from_image_response,
    extract_usage,
    parse_chat_completion_response,
)
from data_designer.engine.models.clients.retry import RetryConfig, create_retry_transport
from data_designer.engine.models.clients.types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    TransportKwargs,
)

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


class OpenAICompatibleClient(ModelClient):
    """Native HTTP adapter for OpenAI-compatible provider APIs.

    Uses ``httpx`` with ``httpx_retries.RetryTransport`` for resilient HTTP
    calls.  Concurrency / throttle policy is an orchestration concern and
    is not managed here — see ``ThrottleManager`` and ``AsyncTaskScheduler``.
    """

    _ROUTE_CHAT = "/chat/completions"
    _ROUTE_EMBEDDING = "/embeddings"
    _ROUTE_IMAGE = "/images/generations"
    _IMAGE_EXCLUDE = frozenset({"messages", "prompt"})

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

        # 2x headroom for burst traffic across domains; floor of 32/16 for low-concurrency configs.
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
    # Capability checks — adapter-level (see ModelClient docstring)
    # -------------------------------------------------------------------

    def supports_chat_completion(self) -> bool:
        return True

    def supports_embeddings(self) -> bool:
        return True

    def supports_image_generation(self) -> bool:
        return True

    # -------------------------------------------------------------------
    # Chat completion
    # -------------------------------------------------------------------

    def completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        transport = TransportKwargs.from_request(request)
        payload = {"model": request.model, "messages": request.messages, **transport.body}
        response_json = self._post_sync(self._ROUTE_CHAT, payload, transport.headers, request.model, transport.timeout)
        return parse_chat_completion_response(response_json)

    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        transport = TransportKwargs.from_request(request)
        payload = {"model": request.model, "messages": request.messages, **transport.body}
        response_json = await self._apost(
            self._ROUTE_CHAT, payload, transport.headers, request.model, transport.timeout
        )
        return await aparse_chat_completion_response(response_json)

    # -------------------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------------------

    def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        transport = TransportKwargs.from_request(request)
        payload = {"model": request.model, "input": request.inputs, **transport.body}
        response_json = self._post_sync(
            self._ROUTE_EMBEDDING, payload, transport.headers, request.model, transport.timeout
        )
        return _parse_embedding_json(response_json)

    async def aembeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        transport = TransportKwargs.from_request(request)
        payload = {"model": request.model, "input": request.inputs, **transport.body}
        response_json = await self._apost(
            self._ROUTE_EMBEDDING, payload, transport.headers, request.model, transport.timeout
        )
        return _parse_embedding_json(response_json)

    # -------------------------------------------------------------------
    # Image generation
    # -------------------------------------------------------------------

    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        transport = TransportKwargs.from_request(request, exclude=self._IMAGE_EXCLUDE)
        if request.messages is not None:
            route = self._ROUTE_CHAT
            payload = {"model": request.model, "messages": request.messages, **transport.body}
        else:
            route = self._ROUTE_IMAGE
            payload = {"model": request.model, "prompt": request.prompt, **transport.body}
        response_json = self._post_sync(route, payload, transport.headers, request.model, transport.timeout)
        return _parse_image_json(response_json, is_chat_route=request.messages is not None)

    async def agenerate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        transport = TransportKwargs.from_request(request, exclude=self._IMAGE_EXCLUDE)
        if request.messages is not None:
            route = self._ROUTE_CHAT
            payload = {"model": request.model, "messages": request.messages, **transport.body}
        else:
            route = self._ROUTE_IMAGE
            payload = {"model": request.model, "prompt": request.prompt, **transport.body}
        response_json = await self._apost(route, payload, transport.headers, request.model, transport.timeout)
        return await _aparse_image_json(response_json, is_chat_route=request.messages is not None)

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
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
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
# Response parsing helpers
# ---------------------------------------------------------------------------


def _parse_embedding_json(response_json: dict[str, Any]) -> EmbeddingResponse:
    data = response_json.get("data") or []
    vectors = [extract_embedding_vector(item) for item in data]
    usage = extract_usage(response_json.get("usage"))
    return EmbeddingResponse(vectors=vectors, usage=usage, raw=response_json)


def _parse_image_json(response_json: dict[str, Any], *, is_chat_route: bool) -> ImageGenerationResponse:
    if is_chat_route:
        images = extract_images_from_chat_response(response_json)
    else:
        images = extract_images_from_image_response(response_json)
    usage = extract_usage(response_json.get("usage"), generated_images=len(images))
    return ImageGenerationResponse(images=images, usage=usage, raw=response_json)


async def _aparse_image_json(response_json: dict[str, Any], *, is_chat_route: bool) -> ImageGenerationResponse:
    if is_chat_route:
        images = await aextract_images_from_chat_response(response_json)
    else:
        images = await aextract_images_from_image_response(response_json)
    usage = extract_usage(response_json.get("usage"), generated_images=len(images))
    return ImageGenerationResponse(images=images, usage=usage, raw=response_json)


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
