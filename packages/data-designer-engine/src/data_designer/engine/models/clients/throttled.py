# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

from data_designer.engine.models.clients.base import ModelClient
from data_designer.engine.models.clients.errors import ProviderError, ProviderErrorKind
from data_designer.engine.models.clients.throttle_manager import ThrottleDomain
from data_designer.engine.models.clients.types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from data_designer.engine.models.clients.throttle_manager import ThrottleManager


logger = logging.getLogger(__name__)


class ThrottledModelClient(ModelClient):
    """Wraps a ``ModelClient`` with per-request throttle acquire/release.

    Inherits from ``ModelClient`` (a ``Protocol``) so that static type
    checkers verify conformance and flag missing methods if the protocol
    evolves.

    Every outbound HTTP call acquires a throttle permit from the
    ``ThrottleManager`` and releases it on success, rate-limit, or failure.
    The ``ThrottleDomain`` is determined by the method:

    - ``completion`` / ``acompletion`` -> ``CHAT``
    - ``embeddings`` / ``aembeddings`` -> ``EMBEDDING``
    - ``generate_image`` / ``agenerate_image`` -> ``IMAGE`` when
      ``request.messages is None`` (diffusion), ``CHAT`` when messages are set
    """

    def __init__(
        self,
        inner: ModelClient,
        throttle_manager: ThrottleManager,
        provider_name: str,
        model_id: str,
    ) -> None:
        self._inner = inner
        self._tm = throttle_manager
        self._provider_name = provider_name
        self._model_id = model_id

    # --- ModelClient protocol delegation ---

    @property
    def provider_name(self) -> str:
        return self._inner.provider_name

    def supports_chat_completion(self) -> bool:
        return self._inner.supports_chat_completion()

    def supports_embeddings(self) -> bool:
        return self._inner.supports_embeddings()

    def supports_image_generation(self) -> bool:
        return self._inner.supports_image_generation()

    def close(self) -> None:
        self._inner.close()

    async def aclose(self) -> None:
        await self._inner.aclose()

    # --- Throttled methods ---

    def completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        with self._throttled_sync(ThrottleDomain.CHAT):
            return self._inner.completion(request)

    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        async with self._athrottled(ThrottleDomain.CHAT):
            return await self._inner.acompletion(request)

    def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        with self._throttled_sync(ThrottleDomain.EMBEDDING):
            return self._inner.embeddings(request)

    async def aembeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        async with self._athrottled(ThrottleDomain.EMBEDDING):
            return await self._inner.aembeddings(request)

    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        domain = self._image_domain(request)
        with self._throttled_sync(domain):
            return self._inner.generate_image(request)

    async def agenerate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        domain = self._image_domain(request)
        async with self._athrottled(domain):
            return await self._inner.agenerate_image(request)

    # --- Context managers ---

    @contextlib.contextmanager
    def _throttled_sync(self, domain: ThrottleDomain) -> Iterator[None]:
        try:
            self._tm.acquire_sync(
                provider_name=self._provider_name,
                model_id=self._model_id,
                domain=domain,
            )
        except TimeoutError as exc:
            raise ProviderError(
                kind=ProviderErrorKind.TIMEOUT,
                message=str(exc),
                provider_name=self._provider_name,
                model_name=self._model_id,
            ) from exc
        exc_to_reraise: BaseException | None = None
        try:
            yield
        except ProviderError as exc:
            exc_to_reraise = exc
            try:
                self._release_on_provider_error(domain, exc)
            except Exception:
                logger.exception("ThrottleManager release failed; permit may leak")
        except BaseException as exc:
            exc_to_reraise = exc
            try:
                self._tm.release_failure(
                    provider_name=self._provider_name,
                    model_id=self._model_id,
                    domain=domain,
                )
            except Exception:
                logger.exception("ThrottleManager release failed; permit may leak")
        else:
            try:
                self._tm.release_success(
                    provider_name=self._provider_name,
                    model_id=self._model_id,
                    domain=domain,
                )
            except Exception:
                logger.exception("ThrottleManager release_success failed")
        if exc_to_reraise is not None:
            raise exc_to_reraise

    @contextlib.asynccontextmanager
    async def _athrottled(self, domain: ThrottleDomain) -> AsyncIterator[None]:
        try:
            await self._tm.acquire_async(
                provider_name=self._provider_name,
                model_id=self._model_id,
                domain=domain,
            )
        except TimeoutError as exc:
            raise ProviderError(
                kind=ProviderErrorKind.TIMEOUT,
                message=str(exc),
                provider_name=self._provider_name,
                model_name=self._model_id,
            ) from exc
        exc_to_reraise: BaseException | None = None
        try:
            yield
        except ProviderError as exc:
            exc_to_reraise = exc
            try:
                self._release_on_provider_error(domain, exc)
            except Exception:
                logger.exception("ThrottleManager release failed; permit may leak")
        except BaseException as exc:
            exc_to_reraise = exc
            try:
                self._tm.release_failure(
                    provider_name=self._provider_name,
                    model_id=self._model_id,
                    domain=domain,
                )
            except Exception:
                logger.exception("ThrottleManager release failed; permit may leak")
        else:
            try:
                self._tm.release_success(
                    provider_name=self._provider_name,
                    model_id=self._model_id,
                    domain=domain,
                )
            except Exception:
                logger.exception("ThrottleManager release_success failed")
        if exc_to_reraise is not None:
            raise exc_to_reraise

    # --- Private helpers ---

    def _release_on_provider_error(self, domain: ThrottleDomain, exc: ProviderError) -> None:
        if exc.kind == ProviderErrorKind.RATE_LIMIT:
            self._tm.release_rate_limited(
                provider_name=self._provider_name,
                model_id=self._model_id,
                domain=domain,
                retry_after=exc.retry_after,
            )
        else:
            self._tm.release_failure(
                provider_name=self._provider_name,
                model_id=self._model_id,
                domain=domain,
            )

    @staticmethod
    def _image_domain(request: ImageGenerationRequest) -> ThrottleDomain:
        return ThrottleDomain.CHAT if request.messages is not None else ThrottleDomain.IMAGE
