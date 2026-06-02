# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, TypeVar

from data_designer.engine.models.clients.base import ModelClient
from data_designer.engine.models.clients.errors import ProviderError, ProviderErrorKind
from data_designer.engine.models.clients.retry import RetryConfig
from data_designer.engine.models.clients.types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
)
from data_designer.engine.models.request_admission.controller import (
    RequestAdmissionController,
    RequestAdmissionError,
    RequestAdmissionLease,
)
from data_designer.engine.models.request_admission.outcomes import RequestReleaseOutcome
from data_designer.engine.models.request_admission.resolver import RequestResourceResolver
from data_designer.engine.models.request_admission.resources import (
    RequestAdmissionItem,
    RequestDomain,
    RequestEventContext,
    RequestGroupSpec,
)
from data_designer.engine.observability import (
    RequestAdmissionEvent,
    RequestAdmissionEventSink,
    runtime_correlation_provider,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_T = TypeVar("_T")

logger = logging.getLogger(__name__)


class ModelRequestExecutor(ModelClient):
    """Model-call boundary that acquires/releases request-admission leases."""

    def __init__(
        self,
        inner: ModelClient,
        request_admission: RequestAdmissionController,
        provider_name: str,
        model_id: str,
        event_sink: RequestAdmissionEventSink | None = None,
        resource_resolver: RequestResourceResolver | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        self._inner = inner
        self._request_admission = request_admission
        self._provider_name = provider_name
        self._model_id = model_id
        self._event_sink = event_sink
        self._resource_resolver = resource_resolver or RequestResourceResolver()
        self._retry_config = retry_config or RetryConfig()
        self._event_sequence = 0

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

    def completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return self._execute_sync(RequestDomain.CHAT, lambda: self._inner.completion(request))

    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return await self._execute_async(RequestDomain.CHAT, lambda: self._inner.acompletion(request))

    def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        return self._execute_sync(RequestDomain.EMBEDDING, lambda: self._inner.embeddings(request))

    async def aembeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        return await self._execute_async(RequestDomain.EMBEDDING, lambda: self._inner.aembeddings(request))

    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        return self._execute_sync(self._image_domain(request), lambda: self._inner.generate_image(request))

    async def agenerate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        return await self._execute_async(self._image_domain(request), lambda: self._inner.agenerate_image(request))

    def _execute_sync(self, domain: RequestDomain, call: Callable[[], _T]) -> _T:
        for attempt in range(self._max_attempts()):
            try:
                return self._execute_sync_attempt(domain, call)
            except ProviderError as exc:
                if not self._should_retry(exc, attempt):
                    raise
                self._sleep_before_retry(attempt)
        raise RuntimeError("unreachable request retry state")

    def _execute_sync_attempt(self, domain: RequestDomain, call: Callable[[], _T]) -> _T:
        item = self._item(domain)
        try:
            lease = self._request_admission.acquire_sync(item)
        except RequestAdmissionError as exc:
            raise self._provider_error_from_request_admission(exc) from exc
        try:
            self._emit_model_event("model_request_started", item=item, lease=lease)
            result = call()
        except ProviderError as exc:
            self._release_provider_error(lease, exc)
            self._emit_model_event(
                "model_request_completed", item=item, lease=lease, diagnostics={"outcome": exc.kind.value}
            )
            raise
        except TimeoutError:
            self._request_admission.release(lease, RequestReleaseOutcome(kind="provider_timeout"))
            self._emit_model_event(
                "model_request_completed", item=item, lease=lease, diagnostics={"outcome": "provider_timeout"}
            )
            raise
        except BaseException as exc:
            outcome = "local_cancelled" if isinstance(exc, KeyboardInterrupt) else "unexpected_exception"
            self._request_admission.release(lease, RequestReleaseOutcome(kind=outcome))
            self._emit_model_event("model_request_completed", item=item, lease=lease, diagnostics={"outcome": outcome})
            raise
        else:
            self._request_admission.release(lease, RequestReleaseOutcome(kind="success"))
            self._emit_model_event(
                "model_request_completed", item=item, lease=lease, diagnostics={"outcome": "success"}
            )
            return result

    async def _execute_async(self, domain: RequestDomain, call: Callable[[], Awaitable[_T]]) -> _T:
        for attempt in range(self._max_attempts()):
            try:
                return await self._execute_async_attempt(domain, call)
            except ProviderError as exc:
                if not self._should_retry(exc, attempt):
                    raise
                await self._async_sleep_before_retry(attempt)
        raise RuntimeError("unreachable request retry state")

    async def _execute_async_attempt(self, domain: RequestDomain, call: Callable[[], Awaitable[_T]]) -> _T:
        item = self._item(domain)
        try:
            lease = await self._request_admission.acquire_async(item)
        except RequestAdmissionError as exc:
            raise self._provider_error_from_request_admission(exc) from exc
        except asyncio.CancelledError:
            raise
        try:
            self._emit_model_event("model_request_started", item=item, lease=lease)
            result = await call()
        except asyncio.CancelledError:
            self._request_admission.release(lease, RequestReleaseOutcome(kind="local_cancelled"))
            self._emit_model_event(
                "model_request_completed", item=item, lease=lease, diagnostics={"outcome": "local_cancelled"}
            )
            raise
        except ProviderError as exc:
            self._release_provider_error(lease, exc)
            self._emit_model_event(
                "model_request_completed", item=item, lease=lease, diagnostics={"outcome": exc.kind.value}
            )
            raise
        except TimeoutError:
            self._request_admission.release(lease, RequestReleaseOutcome(kind="provider_timeout"))
            self._emit_model_event(
                "model_request_completed", item=item, lease=lease, diagnostics={"outcome": "provider_timeout"}
            )
            raise
        except BaseException as exc:
            outcome = "local_cancelled" if isinstance(exc, KeyboardInterrupt) else "unexpected_exception"
            self._request_admission.release(lease, RequestReleaseOutcome(kind=outcome))
            self._emit_model_event("model_request_completed", item=item, lease=lease, diagnostics={"outcome": outcome})
            raise
        else:
            self._request_admission.release(lease, RequestReleaseOutcome(kind="success"))
            self._emit_model_event(
                "model_request_completed", item=item, lease=lease, diagnostics={"outcome": "success"}
            )
            return result

    def _max_attempts(self) -> int:
        return max(1, self._retry_config.max_retries + 1)

    def _should_retry(self, exc: ProviderError, attempt: int) -> bool:
        if attempt >= self._max_attempts() - 1:
            return False
        if exc.kind == ProviderErrorKind.REQUEST_ADMISSION_TIMEOUT:
            return False
        if exc.kind == ProviderErrorKind.RATE_LIMIT:
            return False
        if exc.status_code is not None:
            return exc.status_code in self._retry_config.retryable_status_codes
        return exc.kind == ProviderErrorKind.API_CONNECTION

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = self._retry_delay_seconds(attempt)
        if delay > 0.0:
            time.sleep(delay)

    async def _async_sleep_before_retry(self, attempt: int) -> None:
        delay = self._retry_delay_seconds(attempt)
        if delay > 0.0:
            await asyncio.sleep(delay)

    def _retry_delay_seconds(self, attempt: int) -> float:
        if self._retry_config.backoff_factor <= 0.0:
            return 0.0
        delay = self._retry_config.backoff_factor * (2**attempt)
        return min(delay, self._retry_config.max_backoff_wait)

    def _release_provider_error(self, lease: RequestAdmissionLease, exc: ProviderError) -> None:
        if exc.kind == ProviderErrorKind.RATE_LIMIT:
            outcome = RequestReleaseOutcome(kind="rate_limited", retry_after_seconds=exc.retry_after)
        elif exc.kind == ProviderErrorKind.TIMEOUT:
            outcome = RequestReleaseOutcome(kind="provider_timeout")
        else:
            outcome = RequestReleaseOutcome(kind="provider_failure")
        self._request_admission.release(lease, outcome)

    def _provider_error_from_request_admission(self, exc: RequestAdmissionError) -> ProviderError:
        kind = (
            ProviderErrorKind.REQUEST_ADMISSION_TIMEOUT
            if exc.decision.reason == "queue_timeout"
            else ProviderErrorKind.TIMEOUT
        )
        return ProviderError(
            kind=kind,
            message=str(exc),
            provider_name=self._provider_name,
            model_name=self._model_id,
        )

    def _item(self, domain: RequestDomain) -> RequestAdmissionItem:
        resolved = self._resource_resolver.resolve(
            provider_name=self._provider_name,
            model_id=self._model_id,
            domain=domain,
        )
        resource = resolved.resource
        correlation = runtime_correlation_provider.current()
        return RequestAdmissionItem(
            resource=resource,
            group=RequestGroupSpec(key=resource),
            event_context=RequestEventContext(
                captured_correlation=correlation,
                task_execution_id=correlation.task_execution_id if correlation is not None else None,
                request_attempt_id=f"request-{uuid.uuid4().hex}",
            ),
        )

    @staticmethod
    def _image_domain(request: ImageGenerationRequest) -> RequestDomain:
        return RequestDomain.CHAT if request.messages is not None else RequestDomain.IMAGE

    def _emit_model_event(
        self,
        event_kind: str,
        *,
        item: RequestAdmissionItem,
        lease: RequestAdmissionLease,
        diagnostics: dict[str, object] | None = None,
    ) -> None:
        if self._event_sink is None:
            return
        self._event_sequence += 1
        context = item.event_context
        try:
            self._event_sink.emit_request_event(
                RequestAdmissionEvent.capture(
                    event_kind,  # type: ignore[arg-type]
                    sequence=self._event_sequence,
                    correlation=context.captured_correlation
                    if context is not None
                    else runtime_correlation_provider.current(),
                    request_attempt_id=context.request_attempt_id if context is not None else None,
                    request_lease_id=lease.lease_id,
                    request_resource_key=item.resource,
                    request_group_key=item.group.key,
                    pressure_snapshot=self._request_admission.pressure.snapshot(item.resource),
                    diagnostics=diagnostics or {},
                )
            )
        except Exception:
            logger.warning("Model request event sink raised; dropping event.", exc_info=True)
            return
