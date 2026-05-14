# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest

from data_designer.engine.models.clients.errors import ProviderError, ProviderErrorKind
from data_designer.engine.models.clients.model_request_executor import ModelRequestExecutor
from data_designer.engine.models.clients.request_admission import AdaptiveRequestAdmissionController, RequestDomain
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
from data_designer.engine.observability import InMemoryAdmissionEventSink


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


def test_model_request_executor_maps_image_chat_domain() -> None:
    executor, controller, _client = _executor()

    executor.generate_image(ImageGenerationRequest(model="nemotron", prompt="p", messages=[]))

    resources = controller.pressure.snapshots()
    assert any(resource.domain == RequestDomain.CHAT for resource in resources)


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
