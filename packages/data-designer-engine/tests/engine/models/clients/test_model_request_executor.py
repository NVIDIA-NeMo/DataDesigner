# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging

import pytest

from data_designer.engine.models.clients.errors import ProviderError, ProviderErrorKind
from data_designer.engine.models.clients.model_request_executor import ModelRequestExecutor
from data_designer.engine.models.clients.retry import RetryConfig
from data_designer.engine.models.clients.types import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImagePayload,
)
from data_designer.engine.models.request_admission.config import RequestAdmissionConfig
from data_designer.engine.models.request_admission.controller import (
    AdaptiveRequestAdmissionController,
    RequestAdmissionDenied,
    RequestAdmissionError,
)
from data_designer.engine.models.request_admission.resources import RequestAdmissionItem, RequestDomain
from data_designer.engine.testing import InMemoryAdmissionEventSink


class _Client:
    provider_name = "nvidia"

    def __init__(self) -> None:
        self.error: Exception | None = None

    def supports_chat_completion(self) -> bool:
        return True

    def supports_embeddings(self) -> bool:
        return True

    def supports_image_generation(self) -> bool:
        return True

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None

    def completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if self.error is not None:
            raise self.error
        return ChatCompletionResponse(AssistantMessage(content="ok"))

    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if self.error is not None:
            raise self.error
        return ChatCompletionResponse(AssistantMessage(content="ok"))

    def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        return EmbeddingResponse(vectors=[[1.0]])

    async def aembeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        return EmbeddingResponse(vectors=[[1.0]])

    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        return ImageGenerationResponse(images=[ImagePayload("abc")])

    async def agenerate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        return ImageGenerationResponse(images=[ImagePayload("abc")])


class _BrokenSink:
    def emit_request_event(self, _event: object) -> None:
        raise RuntimeError("sink boom")


class _GatedAsyncClient(_Client):
    def __init__(self) -> None:
        super().__init__()
        self.chat_started = asyncio.Event()
        self.embedding_started = asyncio.Event()
        self.image_started = asyncio.Event()
        self.release_chat = asyncio.Event()
        self.release_embedding = asyncio.Event()
        self.release_image = asyncio.Event()

    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.chat_started.set()
        await self.release_chat.wait()
        return ChatCompletionResponse(AssistantMessage(content="chat"))

    async def aembeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.embedding_started.set()
        await self.release_embedding.wait()
        return EmbeddingResponse(vectors=[[1.0]])

    async def agenerate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        self.image_started.set()
        await self.release_image.wait()
        return ImageGenerationResponse(images=[ImagePayload("image")])


class _FlakyClient(_Client):
    def __init__(
        self,
        *,
        failures: int,
        kind: ProviderErrorKind = ProviderErrorKind.INTERNAL_SERVER,
        status_code: int | None = 503,
    ) -> None:
        super().__init__()
        self.failures = failures
        self.calls = 0
        self.kind = kind
        self.status_code = status_code

    def _maybe_fail(self) -> None:
        self.calls += 1
        if self.calls <= self.failures:
            raise ProviderError(
                kind=self.kind,
                message="temporarily unavailable",
                status_code=self.status_code,
                provider_name="nvidia",
                model_name="nemotron",
            )

    def completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self._maybe_fail()
        return ChatCompletionResponse(AssistantMessage(content="ok"))

    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self._maybe_fail()
        return ChatCompletionResponse(AssistantMessage(content="ok"))


class _DenyingAdmissionController:
    def __init__(self, reason: str) -> None:
        self.reason = reason
        self.acquire_sync_calls = 0

    def acquire_sync(self, item: RequestAdmissionItem) -> object:
        self.acquire_sync_calls += 1
        raise RequestAdmissionError(RequestAdmissionDenied(item=item, reason=self.reason))


def _executor() -> tuple[ModelRequestExecutor, AdaptiveRequestAdmissionController, _Client]:
    controller = AdaptiveRequestAdmissionController()
    controller.register(provider_name="nvidia", model_id="nemotron", alias="default", max_parallel_requests=1)
    client = _Client()
    return ModelRequestExecutor(client, controller, "nvidia", "nemotron"), controller, client


def test_model_request_executor_releases_successful_request() -> None:
    executor, controller, _client = _executor()

    response = executor.completion(ChatCompletionRequest(model="nemotron", messages=[]))

    assert response.message.content == "ok"
    snapshot = controller.pressure.snapshot(next(iter(controller.pressure.snapshots())))
    assert snapshot is not None
    assert snapshot.active_lease_count == 0
    assert snapshot.last_outcome == "success"


def test_model_request_executor_classifies_rate_limit() -> None:
    executor, controller, client = _executor()
    client.error = ProviderError(
        kind=ProviderErrorKind.RATE_LIMIT,
        message="rate limited",
        provider_name="nvidia",
        model_name="nemotron",
        retry_after=1.0,
    )

    with pytest.raises(ProviderError):
        executor.completion(ChatCompletionRequest(model="nemotron", messages=[]))

    snapshot = controller.pressure.snapshot(next(iter(controller.pressure.snapshots())))
    assert snapshot is not None
    assert snapshot.last_outcome == "rate_limited"
    assert snapshot.cooldown_remaining_seconds > 0


def test_model_request_executor_retries_provider_503_with_fresh_leases() -> None:
    sink = InMemoryAdmissionEventSink()
    controller = AdaptiveRequestAdmissionController(event_sink=sink)
    controller.register(provider_name="nvidia", model_id="nemotron", alias="default", max_parallel_requests=1)
    client = _FlakyClient(failures=1)
    executor = ModelRequestExecutor(
        client,
        controller,
        "nvidia",
        "nemotron",
        event_sink=sink,
        retry_config=RetryConfig(max_retries=1, backoff_factor=0.0),
    )

    response = executor.completion(ChatCompletionRequest(model="nemotron", messages=[]))

    assert response.message.content == "ok"
    assert client.calls == 2
    acquired = [event for event in sink.request_events if event.event_kind == "request_lease_acquired"]
    released = [event for event in sink.request_events if event.event_kind == "request_lease_released"]
    assert len(acquired) == 2
    assert len(released) == 2
    assert {event.request_lease_id for event in acquired} == {event.request_lease_id for event in released}


def test_model_request_executor_does_not_retry_provider_timeout_without_status() -> None:
    controller = AdaptiveRequestAdmissionController()
    controller.register(provider_name="nvidia", model_id="nemotron", alias="default", max_parallel_requests=1)
    client = _FlakyClient(failures=2, kind=ProviderErrorKind.TIMEOUT, status_code=None)
    executor = ModelRequestExecutor(
        client,
        controller,
        "nvidia",
        "nemotron",
        retry_config=RetryConfig(max_retries=2, backoff_factor=0.0),
    )

    with pytest.raises(ProviderError) as exc_info:
        executor.completion(ChatCompletionRequest(model="nemotron", messages=[]))

    assert exc_info.value.kind == ProviderErrorKind.TIMEOUT
    assert client.calls == 1


def test_model_request_executor_classifies_sync_request_admission_queue_timeout() -> None:
    controller = AdaptiveRequestAdmissionController(RequestAdmissionConfig(default_queue_wait_timeout_seconds=0.0))
    controller.register(provider_name="nvidia", model_id="nemotron", alias="default", max_parallel_requests=1)
    executor = ModelRequestExecutor(_Client(), controller, "nvidia", "nemotron")

    with pytest.raises(ProviderError) as exc_info:
        executor.completion(ChatCompletionRequest(model="nemotron", messages=[]))

    assert exc_info.value.kind == ProviderErrorKind.REQUEST_ADMISSION_TIMEOUT


def test_model_request_executor_does_not_retry_non_queue_request_admission_errors() -> None:
    controller = _DenyingAdmissionController("hard_policy_denial")
    executor = ModelRequestExecutor(
        _Client(),
        controller,
        "nvidia",
        "nemotron",
        retry_config=RetryConfig(max_retries=2, backoff_factor=0.0),
    )

    with pytest.raises(ProviderError) as exc_info:
        executor.completion(ChatCompletionRequest(model="nemotron", messages=[]))

    assert exc_info.value.kind == ProviderErrorKind.TIMEOUT
    assert controller.acquire_sync_calls == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_model_request_executor_retries_async_provider_503_with_fresh_leases() -> None:
    sink = InMemoryAdmissionEventSink()
    controller = AdaptiveRequestAdmissionController(event_sink=sink)
    controller.register(provider_name="nvidia", model_id="nemotron", alias="default", max_parallel_requests=1)
    client = _FlakyClient(failures=1)
    executor = ModelRequestExecutor(
        client,
        controller,
        "nvidia",
        "nemotron",
        event_sink=sink,
        retry_config=RetryConfig(max_retries=1, backoff_factor=0.0),
    )

    response = await executor.acompletion(ChatCompletionRequest(model="nemotron", messages=[]))

    assert response.message.content == "ok"
    assert client.calls == 2
    acquired = [event for event in sink.request_events if event.event_kind == "request_lease_acquired"]
    released = [event for event in sink.request_events if event.event_kind == "request_lease_released"]
    assert len(acquired) == 2
    assert len(released) == 2
    assert {event.request_lease_id for event in acquired} == {event.request_lease_id for event in released}


@pytest.mark.asyncio(loop_scope="session")
async def test_model_request_executor_classifies_async_request_admission_queue_timeout() -> None:
    controller = AdaptiveRequestAdmissionController(RequestAdmissionConfig(default_queue_wait_timeout_seconds=0.0))
    controller.register(provider_name="nvidia", model_id="nemotron", alias="default", max_parallel_requests=1)
    executor = ModelRequestExecutor(_Client(), controller, "nvidia", "nemotron")

    with pytest.raises(ProviderError) as exc_info:
        await executor.acompletion(ChatCompletionRequest(model="nemotron", messages=[]))

    assert exc_info.value.kind == ProviderErrorKind.REQUEST_ADMISSION_TIMEOUT


@pytest.mark.asyncio(loop_scope="session")
async def test_model_request_executor_releases_async_cancellation() -> None:
    class _SlowClient(_Client):
        async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
            await asyncio.sleep(30)
            return ChatCompletionResponse(AssistantMessage(content="late"))

    controller = AdaptiveRequestAdmissionController()
    controller.register(provider_name="nvidia", model_id="nemotron", alias="default", max_parallel_requests=1)
    executor = ModelRequestExecutor(_SlowClient(), controller, "nvidia", "nemotron")

    task = asyncio.create_task(executor.acompletion(ChatCompletionRequest(model="nemotron", messages=[])))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    snapshot = controller.pressure.snapshot(next(iter(controller.pressure.snapshots())))
    assert snapshot is not None
    assert snapshot.active_lease_count == 0
    assert snapshot.last_outcome == "local_cancelled"


@pytest.mark.asyncio(loop_scope="session")
async def test_model_request_executor_classifies_async_keyboard_interrupt_as_cancelled() -> None:
    class _InterruptingClient(_Client):
        async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
            raise KeyboardInterrupt

    sink = InMemoryAdmissionEventSink()
    controller = AdaptiveRequestAdmissionController(event_sink=sink)
    controller.register(provider_name="nvidia", model_id="nemotron", alias="default", max_parallel_requests=1)
    executor = ModelRequestExecutor(_InterruptingClient(), controller, "nvidia", "nemotron", event_sink=sink)

    with pytest.raises(KeyboardInterrupt):
        await executor.acompletion(ChatCompletionRequest(model="nemotron", messages=[]))

    snapshot = controller.pressure.snapshot(next(iter(controller.pressure.snapshots())))
    assert snapshot is not None
    assert snapshot.active_lease_count == 0
    assert snapshot.last_outcome == "local_cancelled"
    completed = [event for event in sink.request_events if event.event_kind == "model_request_completed"]
    assert completed[-1].diagnostics["outcome"] == "local_cancelled"


def test_model_request_executor_maps_image_chat_domain() -> None:
    executor, controller, _client = _executor()

    executor.generate_image(ImageGenerationRequest(model="nemotron", prompt="p", messages=[]))

    resources = controller.pressure.snapshots()
    assert any(resource.domain == RequestDomain.CHAT for resource in resources)


@pytest.mark.asyncio(loop_scope="session")
async def test_model_request_executor_shares_provider_model_cap_across_async_domains() -> None:
    controller = AdaptiveRequestAdmissionController()
    controller.register(provider_name="nvidia", model_id="nemotron", alias="default", max_parallel_requests=1)
    client = _GatedAsyncClient()
    executor = ModelRequestExecutor(client, controller, "nvidia", "nemotron")

    chat_task = asyncio.create_task(executor.acompletion(ChatCompletionRequest(model="nemotron", messages=[])))
    await asyncio.wait_for(client.chat_started.wait(), timeout=1.0)
    embedding_task = asyncio.create_task(executor.aembeddings(EmbeddingRequest(model="nemotron", inputs=["x"])))
    image_task = asyncio.create_task(executor.agenerate_image(ImageGenerationRequest(model="nemotron", prompt="image")))
    await _wait_for_request_waiters(controller, expected=2)

    global_snapshot = controller.pressure.global_snapshot("nvidia", "nemotron")
    assert global_snapshot is not None
    assert global_snapshot.aggregate_in_flight == 1
    assert not client.embedding_started.is_set()
    assert not client.image_started.is_set()

    client.release_chat.set()
    await asyncio.wait_for(client.embedding_started.wait(), timeout=1.0)
    assert not client.image_started.is_set()
    assert (await chat_task).message.content == "chat"

    global_snapshot = controller.pressure.global_snapshot("nvidia", "nemotron")
    assert global_snapshot is not None
    assert global_snapshot.aggregate_in_flight == 1
    client.release_embedding.set()
    await asyncio.wait_for(client.image_started.wait(), timeout=1.0)
    assert (await embedding_task).vectors == [[1.0]]

    client.release_image.set()
    assert (await image_task).images[0].b64_data == "image"
    global_snapshot = controller.pressure.global_snapshot("nvidia", "nemotron")
    assert global_snapshot is not None
    assert global_snapshot.aggregate_in_flight == 0


async def _wait_for_request_waiters(controller: AdaptiveRequestAdmissionController, *, expected: int) -> None:
    for _ in range(50):
        waiters = sum(snapshot.waiters for snapshot in controller.pressure.snapshots().values())
        if waiters == expected:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"expected {expected} request waiters")


def test_model_request_executor_emits_attempt_events_with_correlation_fields() -> None:
    sink = InMemoryAdmissionEventSink()
    controller = AdaptiveRequestAdmissionController(event_sink=sink)
    controller.register(provider_name="nvidia", model_id="nemotron", alias="default", max_parallel_requests=1)
    executor = ModelRequestExecutor(_Client(), controller, "nvidia", "nemotron", event_sink=sink)

    executor.completion(ChatCompletionRequest(model="nemotron", messages=[]))

    kinds = [event.event_kind for event in sink.request_events]
    assert "request_wait_started" in kinds
    assert "request_lease_acquired" in kinds
    assert "model_request_started" in kinds
    assert "model_request_completed" in kinds
    assert "request_lease_released" in kinds
    attempts = {event.request_attempt_id for event in sink.request_events if event.request_attempt_id is not None}
    assert len(attempts) == 1
    assert all(event.request_resource_key is not None for event in sink.request_events)
    assert all(event.pressure_snapshot is not None for event in sink.request_events)
    attempt_events = [event for event in sink.request_events if event.request_attempt_id is not None]
    assert attempt_events
    assert all(event.request_group_key is not None for event in attempt_events)
    for event in attempt_events:
        assert isinstance(event.pressure_snapshot, dict)
        assert event.pressure_snapshot["resource"] == event.request_resource_key


def test_model_request_executor_logs_sink_failures(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="data_designer.engine.models.clients.model_request_executor")
    controller = AdaptiveRequestAdmissionController()
    controller.register(provider_name="nvidia", model_id="nemotron", alias="default", max_parallel_requests=1)
    executor = ModelRequestExecutor(_Client(), controller, "nvidia", "nemotron", event_sink=_BrokenSink())

    executor.completion(ChatCompletionRequest(model="nemotron", messages=[]))

    assert "Model request event sink raised; dropping event." in caplog.text
