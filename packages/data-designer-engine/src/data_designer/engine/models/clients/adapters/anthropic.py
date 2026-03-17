# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from data_designer.engine.models.clients.adapters.anthropic_translation import (
    build_anthropic_payload,
    parse_anthropic_response,
)
from data_designer.engine.models.clients.adapters.http_model_client import (
    HttpModelClient,
)
from data_designer.engine.models.clients.errors import (
    ProviderError,
    ProviderErrorKind,
)
from data_designer.engine.models.clients.types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    TransportKwargs,
)


class AnthropicClient(HttpModelClient):
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
