# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.engine.models.clients.adapters.anthropic_translation import (
    UnsupportedAnthropicMediaBlockError,
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
from data_designer.engine.models.clients.streaming import AnthropicMessageStream
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
    calls. Concurrency and request-admission policy are orchestration concerns
    and are not managed here.
    """

    _ROUTE_MESSAGES = "/messages"
    _API_VERSION_PATH = "/v1"
    _ANTHROPIC_VERSION = "2023-06-01"
    # Fields handled explicitly and excluded from TransportKwargs forwarding.
    _TRANSPORT_EXCLUDE = frozenset(
        {
            "stop",
            "max_tokens",
            "tools",
            "n",
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
        """Send request and return a complete Messages response, aggregating SSE if enabled.

        Args:
            request: Canonical chat parameters including messages and optional streaming.

        Raises:
            ProviderError: If translation, HTTP transport, or stream assembly fails.
        """
        transport, stream = self._prepare_completion(request)
        response_json = self._post_sync(
            self._get_messages_route(),
            transport.body,
            transport.headers,
            request.model,
            transport.timeout,
            stream=stream,
        )
        return parse_anthropic_response(response_json)

    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Asynchronously send request and return an assembled Messages response.

        Args:
            request: Canonical chat parameters including messages and optional streaming.

        Raises:
            ProviderError: If translation, HTTP transport, or stream assembly fails.
            asyncio.CancelledError: If cancelled; the stream connection is released.
        """
        transport, stream = self._prepare_completion(request)
        response_json = await self._apost(
            self._get_messages_route(),
            transport.body,
            transport.headers,
            request.model,
            transport.timeout,
            stream=stream,
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

    def _prepare_completion(
        self, request: ChatCompletionRequest
    ) -> tuple[TransportKwargs, AnthropicMessageStream | None]:
        """Translate request into Messages HTTP arguments and an optional fresh SSE accumulator.

        Args:
            request: Canonical chat parameters; extra_body overrides translated fields.

        Returns:
            Complete transport arguments and a stream when the final body enables it.

        Raises:
            ProviderError: If messages or media cannot be represented by the Messages API.
        """
        try:
            payload = build_anthropic_payload(request)
        except UnsupportedAnthropicMediaBlockError as exc:
            raise ProviderError.unsupported_capability(
                provider_name=self.provider_name,
                model_name=request.model,
                operation=f"{exc.modality}-context",
                message=(
                    f"Provider {self.provider_name!r} does not support {exc.modality} context "
                    f"for model {request.model!r}."
                ),
            ) from exc
        except ValueError as exc:
            raise ProviderError(
                kind=ProviderErrorKind.BAD_REQUEST,
                message=str(exc),
                provider_name=self.provider_name,
                model_name=request.model,
                cause=exc,
            ) from exc

        transport = TransportKwargs.from_request(request, exclude=self._TRANSPORT_EXCLUDE)
        transport.body = {**payload, **transport.body}
        stream = AnthropicMessageStream(self.provider_name, request.model) if transport.body.get("stream") else None
        return transport, stream

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

    def _get_messages_route(self) -> str:
        if self._endpoint.endswith(self._API_VERSION_PATH):
            return self._ROUTE_MESSAGES
        return f"{self._API_VERSION_PATH}{self._ROUTE_MESSAGES}"
