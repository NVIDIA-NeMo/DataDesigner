# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data_designer.config.run_config import ThrottleConfig
from data_designer.engine.models.clients.adapters.http_model_client import ClientConcurrencyMode
from data_designer.engine.models.clients.adapters.openai_compatible import OpenAICompatibleClient
from data_designer.engine.models.clients.errors import ProviderError, ProviderErrorKind
from data_designer.engine.models.clients.throttle_manager import DomainThrottleState, ThrottleDomain, ThrottleManager
from data_designer.engine.models.clients.throttled import ThrottledModelClient
from data_designer.engine.models.clients.types import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    Usage,
)
from tests.engine.models.clients.conftest import mock_httpx_response

PROVIDER = "test-provider"
MODEL_ID = "test-model"
ENDPOINT = "https://api.example.com/v1"


@dataclass(frozen=True)
class ColdServerSample:
    elapsed: float
    in_flight: int
    allowed: int
    status_code: int


class ColdServerAsyncHTTPClient:
    def __init__(
        self,
        *,
        max_parallel: int,
        server_ramp_seconds: float,
        service_seconds: float,
        retry_after: float,
    ) -> None:
        self._max_parallel = max_parallel
        self._server_ramp_seconds = server_ramp_seconds
        self._service_seconds = service_seconds
        self._retry_after = retry_after
        self._lock = asyncio.Lock()
        self._started_at = time.monotonic()
        self._in_flight = 0
        self.samples: list[ColdServerSample] = []

    @property
    def rate_limits(self) -> int:
        return sum(1 for sample in self.samples if sample.status_code == 429)

    @property
    def peak_in_flight(self) -> int:
        if not self.samples:
            return 0
        return max(sample.in_flight for sample in self.samples)

    def peak_in_flight_before(self, seconds: float) -> int:
        samples = [sample for sample in self.samples if sample.elapsed <= seconds]
        if not samples:
            return 0
        return max(sample.in_flight for sample in samples)

    async def post(self, *_args: object, **_kwargs: object) -> MagicMock:
        async with self._lock:
            self._in_flight += 1
            elapsed = time.monotonic() - self._started_at
            allowed = self._allowed(elapsed)
            status_code = 200 if self._in_flight <= allowed else 429
            self.samples.append(
                ColdServerSample(
                    elapsed=elapsed,
                    in_flight=self._in_flight,
                    allowed=allowed,
                    status_code=status_code,
                )
            )
        try:
            if status_code == 429:
                response = mock_httpx_response({"error": {"message": "synthetic rate limit"}}, status_code=429)
                response.headers = {"Retry-After": str(self._retry_after)}
                return response
            await asyncio.sleep(self._service_seconds)
            return mock_httpx_response(
                {
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            )
        finally:
            async with self._lock:
                self._in_flight -= 1

    async def aclose(self) -> None:
        return None

    def _allowed(self, elapsed: float) -> int:
        if self._server_ramp_seconds <= 0:
            return self._max_parallel
        fraction = min(1.0, max(0.0, elapsed / self._server_ramp_seconds))
        ramp_slots = math.floor((self._max_parallel - 1) * fraction)
        return max(1, min(self._max_parallel, 1 + ramp_slots))


@pytest.fixture
def throttle_manager() -> ThrottleManager:
    tm = ThrottleManager()
    tm.register(provider_name=PROVIDER, model_id=MODEL_ID, alias="alias", max_parallel_requests=10)
    return tm


@pytest.fixture
def inner_client() -> MagicMock:
    client = MagicMock()
    client.provider_name = PROVIDER
    client.supports_chat_completion.return_value = True
    client.supports_embeddings.return_value = True
    client.supports_image_generation.return_value = True
    client.completion.return_value = ChatCompletionResponse(message=AssistantMessage(content="ok"), usage=Usage())
    client.acompletion = AsyncMock(
        return_value=ChatCompletionResponse(message=AssistantMessage(content="ok"), usage=Usage())
    )
    client.embeddings.return_value = EmbeddingResponse(vectors=[[0.1]], usage=Usage())
    client.aembeddings = AsyncMock(return_value=EmbeddingResponse(vectors=[[0.1]], usage=Usage()))
    client.generate_image.return_value = ImageGenerationResponse(images=[])
    client.agenerate_image = AsyncMock(return_value=ImageGenerationResponse(images=[]))
    client.close.return_value = None
    client.aclose = AsyncMock()
    return client


@pytest.fixture
def throttled_client(inner_client: MagicMock, throttle_manager: ThrottleManager) -> ThrottledModelClient:
    return ThrottledModelClient(
        inner=inner_client,
        throttle_manager=throttle_manager,
        provider_name=PROVIDER,
        model_id=MODEL_ID,
    )


# --- Protocol delegation ---


def test_provider_name_delegates(throttled_client: ThrottledModelClient) -> None:
    assert throttled_client.provider_name == PROVIDER


def test_supports_methods_delegate(throttled_client: ThrottledModelClient) -> None:
    assert throttled_client.supports_chat_completion() is True
    assert throttled_client.supports_embeddings() is True
    assert throttled_client.supports_image_generation() is True


def test_close_delegates(throttled_client: ThrottledModelClient, inner_client: MagicMock) -> None:
    throttled_client.close()
    inner_client.close.assert_called_once()


@pytest.mark.asyncio(loop_scope="session")
async def test_aclose_delegates(throttled_client: ThrottledModelClient, inner_client: MagicMock) -> None:
    await throttled_client.aclose()
    inner_client.aclose.assert_awaited_once()


# --- Sync: acquire/release on success ---


def test_completion_success_releases_success(
    throttled_client: ThrottledModelClient, throttle_manager: ThrottleManager
) -> None:
    request = ChatCompletionRequest(model=MODEL_ID, messages=[])
    result = throttled_client.completion(request)
    assert result.message.content == "ok"
    state = throttle_manager.get_domain_state(PROVIDER, MODEL_ID, ThrottleDomain.CHAT)
    assert state is not None
    assert state.in_flight == 0
    assert state.success_streak == 1


def test_embeddings_success_releases_success(
    throttled_client: ThrottledModelClient, throttle_manager: ThrottleManager
) -> None:
    request = EmbeddingRequest(model=MODEL_ID, inputs=["hello"])
    result = throttled_client.embeddings(request)
    assert result.vectors == [[0.1]]
    state = throttle_manager.get_domain_state(PROVIDER, MODEL_ID, ThrottleDomain.EMBEDDING)
    assert state is not None
    assert state.in_flight == 0
    assert state.success_streak == 1


def test_generate_image_diffusion_uses_image_domain(
    throttled_client: ThrottledModelClient, throttle_manager: ThrottleManager
) -> None:
    request = ImageGenerationRequest(model=MODEL_ID, prompt="a cat", messages=None)
    throttled_client.generate_image(request)
    state = throttle_manager.get_domain_state(PROVIDER, MODEL_ID, ThrottleDomain.IMAGE)
    assert state is not None
    assert state.success_streak == 1


def test_generate_image_chat_backed_uses_chat_domain(
    throttled_client: ThrottledModelClient, throttle_manager: ThrottleManager
) -> None:
    request = ImageGenerationRequest(model=MODEL_ID, prompt="a cat", messages=[{"role": "user", "content": "draw"}])
    throttled_client.generate_image(request)
    state = throttle_manager.get_domain_state(PROVIDER, MODEL_ID, ThrottleDomain.CHAT)
    assert state is not None
    assert state.success_streak == 1


# --- Async: acquire/release on success ---


@pytest.mark.asyncio(loop_scope="session")
async def test_acompletion_success_releases_success(
    throttled_client: ThrottledModelClient, throttle_manager: ThrottleManager
) -> None:
    request = ChatCompletionRequest(model=MODEL_ID, messages=[])
    result = await throttled_client.acompletion(request)
    assert result.message.content == "ok"
    state = throttle_manager.get_domain_state(PROVIDER, MODEL_ID, ThrottleDomain.CHAT)
    assert state is not None
    assert state.in_flight == 0
    assert state.success_streak == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_aembeddings_success_releases_success(
    throttled_client: ThrottledModelClient, throttle_manager: ThrottleManager
) -> None:
    request = EmbeddingRequest(model=MODEL_ID, inputs=["hello"])
    result = await throttled_client.aembeddings(request)
    assert result.vectors == [[0.1]]
    state = throttle_manager.get_domain_state(PROVIDER, MODEL_ID, ThrottleDomain.EMBEDDING)
    assert state is not None
    assert state.in_flight == 0
    assert state.success_streak == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_agenerate_image_diffusion_uses_image_domain(
    throttled_client: ThrottledModelClient, throttle_manager: ThrottleManager
) -> None:
    request = ImageGenerationRequest(model=MODEL_ID, prompt="a cat", messages=None)
    await throttled_client.agenerate_image(request)
    state = throttle_manager.get_domain_state(PROVIDER, MODEL_ID, ThrottleDomain.IMAGE)
    assert state is not None
    assert state.success_streak == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_agenerate_image_chat_backed_uses_chat_domain(
    throttled_client: ThrottledModelClient, throttle_manager: ThrottleManager
) -> None:
    request = ImageGenerationRequest(model=MODEL_ID, prompt="a cat", messages=[{"role": "user", "content": "draw"}])
    await throttled_client.agenerate_image(request)
    state = throttle_manager.get_domain_state(PROVIDER, MODEL_ID, ThrottleDomain.CHAT)
    assert state is not None
    assert state.success_streak == 1


# --- Rate-limit error: release_rate_limited with retry_after ---


def test_completion_rate_limit_calls_release_rate_limited(
    throttled_client: ThrottledModelClient, throttle_manager: ThrottleManager
) -> None:
    throttled_client._inner.completion.side_effect = ProviderError(
        kind=ProviderErrorKind.RATE_LIMIT,
        message="429",
        status_code=429,
        retry_after=5.0,
    )
    with pytest.raises(ProviderError, match="429"):
        throttled_client.completion(ChatCompletionRequest(model=MODEL_ID, messages=[]))

    state = throttle_manager.get_domain_state(PROVIDER, MODEL_ID, ThrottleDomain.CHAT)
    assert state is not None
    assert state.in_flight == 0
    assert state.blocked_until > 0


@pytest.mark.asyncio(loop_scope="session")
async def test_acompletion_rate_limit_calls_release_rate_limited(
    throttled_client: ThrottledModelClient, throttle_manager: ThrottleManager
) -> None:
    throttled_client._inner.acompletion = AsyncMock(
        side_effect=ProviderError(
            kind=ProviderErrorKind.RATE_LIMIT,
            message="429",
            status_code=429,
            retry_after=3.0,
        )
    )
    with pytest.raises(ProviderError, match="429"):
        await throttled_client.acompletion(ChatCompletionRequest(model=MODEL_ID, messages=[]))

    state = throttle_manager.get_domain_state(PROVIDER, MODEL_ID, ThrottleDomain.CHAT)
    assert state is not None
    assert state.in_flight == 0
    assert state.blocked_until > 0


# --- Non-rate-limit ProviderError: release_failure ---


def test_completion_non_rate_limit_error_calls_release_failure(
    throttled_client: ThrottledModelClient, throttle_manager: ThrottleManager
) -> None:
    throttled_client._inner.completion.side_effect = ProviderError(
        kind=ProviderErrorKind.INTERNAL_SERVER,
        message="500",
        status_code=500,
    )
    with pytest.raises(ProviderError, match="500"):
        throttled_client.completion(ChatCompletionRequest(model=MODEL_ID, messages=[]))

    state = throttle_manager.get_domain_state(PROVIDER, MODEL_ID, ThrottleDomain.CHAT)
    assert state is not None
    assert state.in_flight == 0
    assert state.success_streak == 0


# --- Non-ProviderError exception: release_failure ---


def test_completion_generic_exception_calls_release_failure(
    throttled_client: ThrottledModelClient, throttle_manager: ThrottleManager
) -> None:
    throttled_client._inner.completion.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        throttled_client.completion(ChatCompletionRequest(model=MODEL_ID, messages=[]))

    state = throttle_manager.get_domain_state(PROVIDER, MODEL_ID, ThrottleDomain.CHAT)
    assert state is not None
    assert state.in_flight == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_acompletion_generic_exception_calls_release_failure(
    throttled_client: ThrottledModelClient, throttle_manager: ThrottleManager
) -> None:
    throttled_client._inner.acompletion = AsyncMock(side_effect=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        await throttled_client.acompletion(ChatCompletionRequest(model=MODEL_ID, messages=[]))

    state = throttle_manager.get_domain_state(PROVIDER, MODEL_ID, ThrottleDomain.CHAT)
    assert state is not None
    assert state.in_flight == 0


# --- Acquire timeout: normalized to ProviderError(kind=TIMEOUT), no release ---


def test_sync_acquire_timeout_normalized_to_provider_error(inner_client: MagicMock) -> None:
    tm = ThrottleManager()
    tm.register(provider_name=PROVIDER, model_id=MODEL_ID, alias="alias", max_parallel_requests=1)
    client = ThrottledModelClient(inner=inner_client, throttle_manager=tm, provider_name=PROVIDER, model_id=MODEL_ID)

    with patch.object(tm, "acquire_sync", side_effect=TimeoutError("timed out")):
        with pytest.raises(ProviderError) as exc_info:
            client.completion(ChatCompletionRequest(model=MODEL_ID, messages=[]))
        assert exc_info.value.kind == ProviderErrorKind.TIMEOUT

    inner_client.completion.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_async_acquire_timeout_normalized_to_provider_error(inner_client: MagicMock) -> None:
    tm = ThrottleManager()
    tm.register(provider_name=PROVIDER, model_id=MODEL_ID, alias="alias", max_parallel_requests=1)
    client = ThrottledModelClient(inner=inner_client, throttle_manager=tm, provider_name=PROVIDER, model_id=MODEL_ID)

    with patch.object(tm, "acquire_async", side_effect=TimeoutError("timed out")):
        with pytest.raises(ProviderError) as exc_info:
            await client.acompletion(ChatCompletionRequest(model=MODEL_ID, messages=[]))
        assert exc_info.value.kind == ProviderErrorKind.TIMEOUT

    inner_client.acompletion.assert_not_awaited()


# --- Cancellation: release_failure on CancelledError ---


@pytest.mark.asyncio(loop_scope="session")
async def test_acompletion_cancelled_releases_permit(throttle_manager: ThrottleManager) -> None:
    """CancelledError during an in-flight async request releases the throttle permit."""
    blocked = asyncio.Event()

    async def slow_acompletion(_request: ChatCompletionRequest) -> ChatCompletionResponse:
        blocked.set()
        await asyncio.sleep(60)
        return ChatCompletionResponse(message=AssistantMessage(content="ok"), usage=Usage())

    inner = MagicMock()
    inner.provider_name = PROVIDER
    inner.acompletion = slow_acompletion

    client = ThrottledModelClient(
        inner=inner, throttle_manager=throttle_manager, provider_name=PROVIDER, model_id=MODEL_ID
    )
    request = ChatCompletionRequest(model=MODEL_ID, messages=[])

    task = asyncio.create_task(client.acompletion(request))
    await blocked.wait()

    state = throttle_manager.get_domain_state(PROVIDER, MODEL_ID, ThrottleDomain.CHAT)
    assert state is not None
    assert state.in_flight == 1

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert state.in_flight == 0


# --- E2E: full AIMD feedback loop ---


async def _complete_with_rate_limit_retries(
    client: ThrottledModelClient,
    request: ChatCompletionRequest,
    *,
    max_attempts: int = 50,
) -> int:
    for attempt in range(1, max_attempts + 1):
        try:
            await client.acompletion(request)
            return attempt
        except ProviderError as exc:
            if exc.kind != ProviderErrorKind.RATE_LIMIT:
                raise
            await asyncio.sleep(exc.retry_after or 0.01)
    raise AssertionError(f"request did not complete after {max_attempts} attempts")


async def _run_cold_server_scenario(
    *,
    throttle_ramp_seconds: float,
    server_ramp_seconds: float,
    max_parallel: int = 4,
    tasks: int = 12,
) -> tuple[ColdServerAsyncHTTPClient, ThrottleManager, ThrottledModelClient]:
    async_http_client = ColdServerAsyncHTTPClient(
        max_parallel=max_parallel,
        server_ramp_seconds=server_ramp_seconds,
        service_seconds=0.04,
        retry_after=0.02,
    )
    inner = OpenAICompatibleClient(
        provider_name=PROVIDER,
        endpoint=ENDPOINT,
        api_key="sk-test-key",
        concurrency_mode=ClientConcurrencyMode.ASYNC,
        async_client=async_http_client,
    )
    throttle_manager = ThrottleManager(
        ThrottleConfig(
            reduce_factor=0.5,
            additive_increase=1,
            success_window=2,
            cooldown_seconds=0.02,
            rampup_seconds=throttle_ramp_seconds,
        )
    )
    throttle_manager.register(
        provider_name=PROVIDER,
        model_id=MODEL_ID,
        alias="cold-server",
        max_parallel_requests=max_parallel,
    )
    client = ThrottledModelClient(
        inner=inner,
        throttle_manager=throttle_manager,
        provider_name=PROVIDER,
        model_id=MODEL_ID,
    )
    requests = [
        ChatCompletionRequest(model=MODEL_ID, messages=[{"role": "user", "content": f"request {i}"}])
        for i in range(tasks)
    ]
    await asyncio.gather(*(_complete_with_rate_limit_retries(client, request) for request in requests))
    return async_http_client, throttle_manager, client


@pytest.mark.asyncio(loop_scope="session")
async def test_startup_ramp_integration_eases_into_cold_server_without_429s() -> None:
    no_ramp_client, _, _ = await _run_cold_server_scenario(
        throttle_ramp_seconds=0.0,
        server_ramp_seconds=0.3,
    )
    ramped_client, throttle_manager, throttled_client = await _run_cold_server_scenario(
        throttle_ramp_seconds=0.3,
        server_ramp_seconds=0.1,
    )

    assert no_ramp_client.rate_limits > 0
    assert no_ramp_client.peak_in_flight > 1
    assert ramped_client.rate_limits == 0
    assert ramped_client.peak_in_flight_before(0.05) <= 1
    assert ramped_client.peak_in_flight > 1

    await asyncio.sleep(0.31)
    await _complete_with_rate_limit_retries(
        throttled_client,
        ChatCompletionRequest(model=MODEL_ID, messages=[{"role": "user", "content": "final ramp probe"}]),
    )
    state = throttle_manager.get_domain_state(PROVIDER, MODEL_ID, ThrottleDomain.CHAT)
    assert state is not None
    assert state.current_limit == 4
    assert state.rampup_active is False


@pytest.mark.asyncio(loop_scope="session")
async def test_startup_ramp_integration_overaggressive_ramp_aborts_to_aimd() -> None:
    cold_client, throttle_manager, _ = await _run_cold_server_scenario(
        throttle_ramp_seconds=0.05,
        server_ramp_seconds=0.5,
    )

    assert cold_client.rate_limits > 0

    state = throttle_manager.get_domain_state(PROVIDER, MODEL_ID, ThrottleDomain.CHAT)
    assert state is not None
    assert state.rampup_active is False


@pytest.mark.asyncio(loop_scope="session")
async def test_aimd_feedback_loop_rate_limit_reduces_then_successes_recover() -> None:
    """Verify the full AIMD cycle: success -> rate-limit halves limit -> successes recover.

    Uses a real ThrottleManager with aggressive tuning (success_window=2,
    additive_increase=1) so the test can drive a full decrease+increase cycle
    with a small number of calls.

    Sequence:
    1. Register model with max_parallel_requests=4.
    2. Make 1 successful async completion -> limit stays 4, streak=1.
    3. Hit a 429 with retry_after=0.01s -> limit halves to 2, cooldown applied.
    4. Wait for cooldown to expire.
    5. Make 2 more successes -> streak reaches success_window=2, limit increases to 3.
    6. Make 2 more successes -> limit increases to 4 (full recovery).
    """
    tm = ThrottleManager(
        ThrottleConfig(
            reduce_factor=0.5,
            additive_increase=1,
            success_window=2,
            cooldown_seconds=0.01,
        )
    )
    tm.register(provider_name=PROVIDER, model_id=MODEL_ID, alias="a", max_parallel_requests=4)

    call_count = 0
    rate_limit_on_call = 2

    async def mock_acompletion(request: ChatCompletionRequest) -> ChatCompletionResponse:
        nonlocal call_count
        call_count += 1
        if call_count == rate_limit_on_call:
            raise ProviderError(
                kind=ProviderErrorKind.RATE_LIMIT,
                message="429 Too Many Requests",
                status_code=429,
                retry_after=0.01,
            )
        return ChatCompletionResponse(message=AssistantMessage(content="ok"), usage=Usage())

    inner = MagicMock()
    inner.provider_name = PROVIDER
    inner.acompletion = mock_acompletion

    client = ThrottledModelClient(inner=inner, throttle_manager=tm, provider_name=PROVIDER, model_id=MODEL_ID)
    request = ChatCompletionRequest(model=MODEL_ID, messages=[])

    def get_state() -> DomainThrottleState:
        s = tm.get_domain_state(PROVIDER, MODEL_ID, ThrottleDomain.CHAT)
        assert s is not None
        return s

    # Step 1: first success
    await client.acompletion(request)
    assert get_state().current_limit == 4
    assert get_state().success_streak == 1

    # Step 2: 429 -> AIMD decrease
    with pytest.raises(ProviderError):
        await client.acompletion(request)
    assert get_state().current_limit == 2
    assert get_state().success_streak == 0
    assert get_state().in_flight == 0

    # Step 3: wait for cooldown
    await asyncio.sleep(0.02)

    # Step 4: two successes -> additive increase (limit 2 -> 3)
    await client.acompletion(request)
    assert get_state().success_streak == 1
    await client.acompletion(request)
    assert get_state().current_limit == 3
    assert get_state().success_streak == 0

    # Step 5: two more successes -> additive increase (limit 3 -> 4, full recovery)
    await client.acompletion(request)
    await client.acompletion(request)
    assert get_state().current_limit == 4
    assert get_state().success_streak == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_requests_bounded_by_throttle_limit() -> None:
    """Concurrent async requests are bounded by the throttle limit.

    Registers a model with max_parallel_requests=2, fires 5 concurrent
    acompletion calls that each sleep briefly, and verifies that the
    ThrottleManager never had more than 2 in-flight at once.
    """
    tm = ThrottleManager()
    tm.register(provider_name=PROVIDER, model_id=MODEL_ID, alias="a", max_parallel_requests=2)

    peak_in_flight = 0
    lock = asyncio.Lock()

    async def mock_acompletion(request: ChatCompletionRequest) -> ChatCompletionResponse:
        nonlocal peak_in_flight
        state = tm.get_domain_state(PROVIDER, MODEL_ID, ThrottleDomain.CHAT)
        if state is not None:
            async with lock:
                peak_in_flight = max(peak_in_flight, state.in_flight)
        await asyncio.sleep(0.02)
        return ChatCompletionResponse(message=AssistantMessage(content="ok"), usage=Usage())

    inner = MagicMock()
    inner.provider_name = PROVIDER
    inner.acompletion = mock_acompletion

    client = ThrottledModelClient(inner=inner, throttle_manager=tm, provider_name=PROVIDER, model_id=MODEL_ID)
    request = ChatCompletionRequest(model=MODEL_ID, messages=[])

    tasks = [asyncio.create_task(client.acompletion(request)) for _ in range(5)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successes = [r for r in results if not isinstance(r, Exception)]
    assert len(successes) == 5
    assert peak_in_flight <= 2

    state = tm.get_domain_state(PROVIDER, MODEL_ID, ThrottleDomain.CHAT)
    assert state is not None
    assert state.in_flight == 0
