# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any, Protocol

from data_designer.engine.models.clients.base import ModelClient
from data_designer.engine.models.clients.errors import ProviderError, ProviderErrorKind
from data_designer.engine.models.clients.parsing import (
    aextract_images_from_chat_response,
    aextract_images_from_image_response,
    collect_non_none_optional_fields,
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
)

logger = logging.getLogger(__name__)


class LiteLLMRouter(Protocol):
    """Structural type for the LiteLLM router methods the bridge depends on."""

    def completion(self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any: ...

    async def acompletion(self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any: ...

    def embedding(self, *, model: str, input: list[str], **kwargs: Any) -> Any: ...

    async def aembedding(self, *, model: str, input: list[str], **kwargs: Any) -> Any: ...

    def image_generation(self, *, prompt: str, model: str, **kwargs: Any) -> Any: ...

    async def aimage_generation(self, *, prompt: str, model: str, **kwargs: Any) -> Any: ...


class LiteLLMBridgeClient(ModelClient):
    """Bridge adapter that wraps the existing LiteLLM router behind canonical client types."""

    # "messages" and "prompt" have None defaults but are passed explicitly to choose
    # between the chat-completion and diffusion code paths, so exclude them from the
    # automatic optional-field forwarding.
    _IMAGE_EXCLUDE = frozenset({"messages", "prompt"})

    def __init__(self, *, provider_name: str, router: LiteLLMRouter) -> None:
        self.provider_name = provider_name
        self._router = router

    def supports_chat_completion(self) -> bool:
        return True

    def supports_embeddings(self) -> bool:
        return True

    def supports_image_generation(self) -> bool:
        return True

    def completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            response = self._router.completion(
                model=request.model,
                messages=request.messages,
                **collect_non_none_optional_fields(request),
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise _wrap_router_error(exc, provider_name=self.provider_name) from exc
        return parse_chat_completion_response(response)

    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            response = await self._router.acompletion(
                model=request.model,
                messages=request.messages,
                **collect_non_none_optional_fields(request),
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise _wrap_router_error(exc, provider_name=self.provider_name) from exc
        return parse_chat_completion_response(response)

    def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        try:
            response = self._router.embedding(
                model=request.model,
                input=request.inputs,
                **collect_non_none_optional_fields(request),
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise _wrap_router_error(exc, provider_name=self.provider_name) from exc
        vectors = [extract_embedding_vector(item) for item in getattr(response, "data", [])]
        return EmbeddingResponse(vectors=vectors, usage=extract_usage(getattr(response, "usage", None)), raw=response)

    async def aembeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        try:
            response = await self._router.aembedding(
                model=request.model,
                input=request.inputs,
                **collect_non_none_optional_fields(request),
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise _wrap_router_error(exc, provider_name=self.provider_name) from exc
        vectors = [extract_embedding_vector(item) for item in getattr(response, "data", [])]
        return EmbeddingResponse(vectors=vectors, usage=extract_usage(getattr(response, "usage", None)), raw=response)

    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        image_kwargs = collect_non_none_optional_fields(request, exclude=self._IMAGE_EXCLUDE)
        try:
            if request.messages is not None:
                response = self._router.completion(
                    model=request.model,
                    messages=request.messages,
                    **image_kwargs,
                )
                images = extract_images_from_chat_response(response)
            else:
                response = self._router.image_generation(
                    prompt=request.prompt,
                    model=request.model,
                    **image_kwargs,
                )
                images = extract_images_from_image_response(response)
        except ProviderError:
            raise
        except Exception as exc:
            raise _wrap_router_error(exc, provider_name=self.provider_name) from exc

        usage = extract_usage(getattr(response, "usage", None), generated_images=len(images))
        return ImageGenerationResponse(images=images, usage=usage, raw=response)

    async def agenerate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        image_kwargs = collect_non_none_optional_fields(request, exclude=self._IMAGE_EXCLUDE)
        try:
            if request.messages is not None:
                response = await self._router.acompletion(
                    model=request.model,
                    messages=request.messages,
                    **image_kwargs,
                )
                images = await aextract_images_from_chat_response(response)
            else:
                response = await self._router.aimage_generation(
                    prompt=request.prompt,
                    model=request.model,
                    **image_kwargs,
                )
                images = await aextract_images_from_image_response(response)
        except ProviderError:
            raise
        except Exception as exc:
            raise _wrap_router_error(exc, provider_name=self.provider_name) from exc

        usage = extract_usage(getattr(response, "usage", None), generated_images=len(images))
        return ImageGenerationResponse(images=images, usage=usage, raw=response)

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


def _wrap_router_error(exc: Exception, *, provider_name: str) -> ProviderError:
    """Normalize a raw router/LiteLLM exception into a canonical ProviderError."""
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        from data_designer.engine.models.clients.errors import map_http_status_to_provider_error_kind

        kind = map_http_status_to_provider_error_kind(status_code=status_code, body_text=str(exc))
    else:
        kind = _infer_error_kind(exc)

    return ProviderError(
        kind=kind,
        message=str(exc),
        status_code=status_code if isinstance(status_code, int) else None,
        provider_name=provider_name,
        cause=exc,
    )


def _infer_error_kind(exc: Exception) -> ProviderErrorKind:
    """Infer error kind from exception type name when no status code is available."""
    type_name = type(exc).__name__.lower()
    if "timeout" in type_name:
        return ProviderErrorKind.TIMEOUT
    if "connection" in type_name or "connect" in type_name:
        return ProviderErrorKind.API_CONNECTION
    if "auth" in type_name:
        return ProviderErrorKind.AUTHENTICATION
    if "ratelimit" in type_name:
        return ProviderErrorKind.RATE_LIMIT
    return ProviderErrorKind.API_ERROR
