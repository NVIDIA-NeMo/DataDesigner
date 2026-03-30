# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from data_designer.engine.models.clients.adapters.http_model_client import (
    HttpModelClient,
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
from data_designer.engine.models.clients.types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    TransportKwargs,
)


class OpenAICompatibleClient(HttpModelClient):
    """Native HTTP adapter for OpenAI-compatible provider APIs.

    Uses ``httpx`` with ``httpx_retries.RetryTransport`` for resilient HTTP
    calls.  Concurrency / throttle policy is an orchestration concern and
    is not managed here — see ``ThrottleManager`` and ``AsyncTaskScheduler``.
    """

    _ROUTE_CHAT = "/chat/completions"
    _ROUTE_EMBEDDING = "/embeddings"
    _ROUTE_IMAGE = "/images/generations"
    _IMAGE_EXCLUDE = frozenset({"messages", "prompt"})

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

    def _build_headers(self, extra_headers: dict[str, str]) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if extra_headers:
            headers.update(extra_headers)
        return headers


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
