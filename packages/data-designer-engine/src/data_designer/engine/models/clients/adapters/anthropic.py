# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.models.clients.adapters.anthropic_translation import (
    build_anthropic_payload,
    parse_anthropic_response,
)
from data_designer.engine.models.clients.adapters.http_helpers import (
    parse_json_body,
    resolve_timeout,
    wrap_transport_error,
)
from data_designer.engine.models.clients.base import ModelClient
from data_designer.engine.models.clients.errors import (
    ProviderError,
    ProviderErrorKind,
    map_http_error_to_provider_error,
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


class AnthropicClient(ModelClient):
    """Native HTTP adapter for the Anthropic Messages API.

    Uses ``httpx`` with ``httpx_retries.RetryTransport`` for resilient HTTP
    calls.  Concurrency / throttle policy is an orchestration concern and
    is not managed here — see ``ThrottleManager`` and ``AsyncTaskScheduler``.
    """

    _ROUTE_MESSAGES = "/v1/messages"
    _ANTHROPIC_VERSION = "2023-06-01"
    # Fields handled explicitly and excluded from TransportKwargs forwarding.
    _TRANSPORT_EXCLUDE = frozenset(
        {
            "stop",
            "max_tokens",
            "tools",
            "response_format",
            "frequency_penalty",
            "presence_penalty",
            "seed",
        }
    )

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
        payload = self._build_payload_or_raise(request)
        transport = TransportKwargs.from_request(request, exclude=self._TRANSPORT_EXCLUDE)
        payload.update(transport.body)
        response_json = self._post_sync(
            self._ROUTE_MESSAGES, payload, transport.headers, request.model, transport.timeout
        )
        return parse_anthropic_response(response_json)

    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        payload = self._build_payload_or_raise(request)
        transport = TransportKwargs.from_request(request, exclude=self._TRANSPORT_EXCLUDE)
        payload.update(transport.body)
        response_json = await self._apost(
            self._ROUTE_MESSAGES, payload, transport.headers, request.model, transport.timeout
        )
        return parse_anthropic_response(response_json)

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
        with self._init_lock:
            client = self._client
            self._client = None
        if client is not None:
            client.close()

    async def aclose(self) -> None:
        with self._init_lock:
            async_client = self._aclient
            sync_client = self._client
            self._aclient = None
            self._client = None
        if async_client is not None:
            await async_client.aclose()
        if sync_client is not None:
            sync_client.close()

    # -------------------------------------------------------------------
    # HTTP helpers
    # -------------------------------------------------------------------

    def _build_payload_or_raise(self, request: ChatCompletionRequest) -> dict[str, Any]:
        try:
            return build_anthropic_payload(request)
        except ValueError as exc:
            raise ProviderError(
                kind=ProviderErrorKind.BAD_REQUEST,
                message=str(exc),
                provider_name=self.provider_name,
                model_name=request.model,
                cause=exc,
            ) from exc

    def _build_headers(self, extra_headers: dict[str, str]) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "anthropic-version": self._ANTHROPIC_VERSION,
        }
        if self._api_key:
            headers["x-api-key"] = self._api_key
        if extra_headers:
            headers.update(extra_headers)
        return headers

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
                url,
                json=payload,
                headers=headers,
                timeout=resolve_timeout(self._timeout_s, timeout),
            )
        except Exception as exc:
            raise wrap_transport_error(exc, self.provider_name, model_name) from exc
        if response.status_code >= 400:
            raise map_http_error_to_provider_error(
                response=response, provider_name=self.provider_name, model_name=model_name
            )
        return parse_json_body(response, self.provider_name, model_name)

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
                url,
                json=payload,
                headers=headers,
                timeout=resolve_timeout(self._timeout_s, timeout),
            )
        except Exception as exc:
            raise wrap_transport_error(exc, self.provider_name, model_name) from exc
        if response.status_code >= 400:
            raise map_http_error_to_provider_error(
                response=response, provider_name=self.provider_name, model_name=model_name
            )
        return parse_json_body(response, self.provider_name, model_name)
