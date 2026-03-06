# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
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
from data_designer.engine.models.clients.throttle import ThrottleDomain, ThrottleManager
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
    calls and a shared ``ThrottleManager`` for adaptive concurrency control.
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
        throttle_manager: ThrottleManager | None = None,
        max_parallel_requests: int = 32,
        timeout_s: float = 60.0,
    ) -> None:
        self.provider_name = provider_name
        self._model_id = model_id
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._throttle = throttle_manager
        self._timeout_s = timeout_s

        # 2x headroom for burst traffic across domains; floor of 32/16 for low-concurrency configs.
        pool_max = max(32, 2 * max_parallel_requests)
        pool_keepalive = max(16, max_parallel_requests)
        limits = lazy.httpx.Limits(
            max_connections=pool_max,
            max_keepalive_connections=pool_keepalive,
        )
        self._client: httpx.Client = lazy.httpx.Client(
            transport=create_retry_transport(retry_config),
            limits=limits,
            timeout=lazy.httpx.Timeout(timeout_s),
        )
        self._aclient: httpx.AsyncClient = lazy.httpx.AsyncClient(
            transport=create_retry_transport(retry_config),
            limits=limits,
            timeout=lazy.httpx.Timeout(timeout_s),
        )

    # -------------------------------------------------------------------
    # Capability checks
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
        response_json = self._throttled_post_sync(
            self._ROUTE_CHAT, payload, transport.headers, request.model, ThrottleDomain.CHAT
        )
        return parse_chat_completion_response(response_json)

    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        transport = TransportKwargs.from_request(request)
        payload = {"model": request.model, "messages": request.messages, **transport.body}
        response_json = await self._athrottled_post(
            self._ROUTE_CHAT, payload, transport.headers, request.model, ThrottleDomain.CHAT
        )
        return await aparse_chat_completion_response(response_json)

    # -------------------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------------------

    def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        transport = TransportKwargs.from_request(request)
        payload = {"model": request.model, "input": request.inputs, **transport.body}
        response_json = self._throttled_post_sync(
            self._ROUTE_EMBEDDING, payload, transport.headers, request.model, ThrottleDomain.EMBEDDING
        )
        return _parse_embedding_json(response_json)

    async def aembeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        transport = TransportKwargs.from_request(request)
        payload = {"model": request.model, "input": request.inputs, **transport.body}
        response_json = await self._athrottled_post(
            self._ROUTE_EMBEDDING, payload, transport.headers, request.model, ThrottleDomain.EMBEDDING
        )
        return _parse_embedding_json(response_json)

    # -------------------------------------------------------------------
    # Image generation
    # -------------------------------------------------------------------

    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        transport = TransportKwargs.from_request(request, exclude=self._IMAGE_EXCLUDE)
        if request.messages is not None:
            route, domain = self._ROUTE_CHAT, ThrottleDomain.CHAT
            payload = {"model": request.model, "messages": request.messages, **transport.body}
        else:
            route, domain = self._ROUTE_IMAGE, ThrottleDomain.IMAGE
            payload = {"model": request.model, "prompt": request.prompt, **transport.body}
        response_json = self._throttled_post_sync(route, payload, transport.headers, request.model, domain)
        return _parse_image_json(response_json, is_chat_route=request.messages is not None)

    async def agenerate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        transport = TransportKwargs.from_request(request, exclude=self._IMAGE_EXCLUDE)
        if request.messages is not None:
            route, domain = self._ROUTE_CHAT, ThrottleDomain.CHAT
            payload = {"model": request.model, "messages": request.messages, **transport.body}
        else:
            route, domain = self._ROUTE_IMAGE, ThrottleDomain.IMAGE
            payload = {"model": request.model, "prompt": request.prompt, **transport.body}
        response_json = await self._athrottled_post(route, payload, transport.headers, request.model, domain)
        return await _aparse_image_json(response_json, is_chat_route=request.messages is not None)

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    async def aclose(self) -> None:
        await self._aclient.aclose()

    # -------------------------------------------------------------------
    # HTTP helpers
    # -------------------------------------------------------------------

    def _throttled_post_sync(
        self,
        route: str,
        payload: dict[str, Any],
        extra_headers: dict[str, str],
        model_name: str,
        domain: ThrottleDomain,
    ) -> dict[str, Any]:
        """POST with throttle acquire/release lifecycle."""
        self._acquire_throttle_sync(domain)
        try:
            result = self._post_sync(route, payload, extra_headers, model_name)
        except ProviderError as exc:
            self._handle_provider_error(exc, domain)
            raise
        except Exception:
            self._release_throttle_failure(domain)
            raise
        self._release_throttle_success(domain)
        return result

    async def _athrottled_post(
        self,
        route: str,
        payload: dict[str, Any],
        extra_headers: dict[str, str],
        model_name: str,
        domain: ThrottleDomain,
    ) -> dict[str, Any]:
        """Async POST with throttle acquire/release lifecycle."""
        await self._acquire_throttle_async(domain)
        try:
            result = await self._apost(route, payload, extra_headers, model_name)
        except ProviderError as exc:
            self._handle_provider_error(exc, domain)
            raise
        except Exception:
            self._release_throttle_failure(domain)
            raise
        self._release_throttle_success(domain)
        return result

    def _build_headers(self, extra_headers: dict[str, str]) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _post_sync(
        self,
        route: str,
        payload: dict[str, Any],
        extra_headers: dict[str, str],
        model_name: str,
    ) -> dict[str, Any]:
        url = f"{self._endpoint}{route}"
        headers = self._build_headers(extra_headers)
        try:
            response = self._client.post(url, json=payload, headers=headers)
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
    ) -> dict[str, Any]:
        url = f"{self._endpoint}{route}"
        headers = self._build_headers(extra_headers)
        try:
            response = await self._aclient.post(url, json=payload, headers=headers)
        except Exception as exc:
            raise _wrap_transport_error(exc, self.provider_name, model_name) from exc
        if response.status_code >= 400:
            raise map_http_error_to_provider_error(
                response=response, provider_name=self.provider_name, model_name=model_name
            )
        return _parse_json_body(response, self.provider_name, model_name)

    # -------------------------------------------------------------------
    # Throttle helpers
    # -------------------------------------------------------------------

    def _acquire_throttle_sync(self, domain: ThrottleDomain) -> None:
        if self._throttle is not None:
            self._throttle.acquire_sync(provider_name=self.provider_name, model_id=self._model_id, domain=domain)

    async def _acquire_throttle_async(self, domain: ThrottleDomain) -> None:
        if self._throttle is not None:
            await self._throttle.acquire_async(provider_name=self.provider_name, model_id=self._model_id, domain=domain)

    def _release_throttle_success(self, domain: ThrottleDomain) -> None:
        if self._throttle is not None:
            self._throttle.release_success(provider_name=self.provider_name, model_id=self._model_id, domain=domain)

    def _release_throttle_failure(self, domain: ThrottleDomain) -> None:
        if self._throttle is not None:
            self._throttle.release_failure(provider_name=self.provider_name, model_id=self._model_id, domain=domain)

    def _handle_provider_error(self, exc: ProviderError, domain: ThrottleDomain) -> None:
        if self._throttle is None:
            return
        if exc.kind == ProviderErrorKind.RATE_LIMIT:
            self._throttle.release_rate_limited(
                provider_name=self.provider_name,
                model_id=self._model_id,
                domain=domain,
                retry_after=exc.retry_after,
            )
        else:
            self._release_throttle_failure(domain)


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
            status_code=getattr(response, "status_code", None),
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
