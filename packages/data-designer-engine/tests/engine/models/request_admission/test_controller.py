# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import threading
import time

import pytest

from data_designer.engine.models.request_admission.config import RequestAdmissionConfig
from data_designer.engine.models.request_admission.controller import (
    AdaptiveRequestAdmissionController,
    RequestAdmissionDenied,
    RequestAdmissionError,
    RequestAdmissionLease,
)
from data_designer.engine.models.request_admission.outcomes import RequestReleaseOutcome
from data_designer.engine.models.request_admission.resources import (
    RequestAdmissionItem,
    RequestDomain,
    RequestGroupSpec,
    RequestResourceKey,
)


def _item(domain: RequestDomain = RequestDomain.CHAT, timeout: float | None = None) -> RequestAdmissionItem:
    resource = RequestResourceKey("nvidia", "nemotron", domain)
    return RequestAdmissionItem(
        resource=resource,
        group=RequestGroupSpec(resource),
        queue_wait_timeout_seconds=timeout,
    )


def _controller(cap: int = 2, config: RequestAdmissionConfig | None = None) -> AdaptiveRequestAdmissionController:
    controller = AdaptiveRequestAdmissionController(config)
    controller.register(provider_name="nvidia", model_id="nemotron", alias="default", max_parallel_requests=cap)
    return controller


class _BrokenRequestSink:
    def emit_request_event(self, _event: object) -> None:
        raise RuntimeError("sink boom")


def test_request_admission_acquires_and_releases_lease() -> None:
    controller = _controller(cap=1)
    item = _item()

    decision = controller.try_acquire(item)

    assert isinstance(decision, RequestAdmissionLease)
    assert controller.pressure.snapshot(item.resource).in_flight_count == 1  # type: ignore[union-attr]
    result = controller.release(decision, RequestReleaseOutcome(kind="success"))
    assert result.released is True
    assert controller.pressure.snapshot(item.resource).in_flight_count == 0  # type: ignore[union-attr]


def test_request_admission_enforces_provider_model_aggregate_cap() -> None:
    controller = _controller(cap=1)
    chat = _item(RequestDomain.CHAT)
    embedding = _item(RequestDomain.EMBEDDING)
    lease = controller.try_acquire(chat)
    assert isinstance(lease, RequestAdmissionLease)

    denied = controller.try_acquire(embedding)

    assert isinstance(denied, RequestAdmissionDenied)
    assert denied.reason == "no_capacity"


def test_request_admission_duplicate_release_does_not_corrupt_counts() -> None:
    controller = _controller(cap=1)
    item = _item()
    lease = controller.try_acquire(item)
    assert isinstance(lease, RequestAdmissionLease)

    first = controller.release(lease, RequestReleaseOutcome(kind="success"))
    second = controller.release(lease, RequestReleaseOutcome(kind="success"))

    assert first.released is True
    assert second.released is False
    assert second.reason == "duplicate"
    assert controller.pressure.snapshot(item.resource).active_lease_count == 0  # type: ignore[union-attr]


def test_request_admission_rate_limit_decreases_and_sets_cooldown() -> None:
    controller = _controller(
        cap=4,
        config=RequestAdmissionConfig(
            multiplicative_decrease_factor=0.5,
            cooldown_seconds=10,
        ),
    )
    item = _item()
    lease = controller.try_acquire(item)
    assert isinstance(lease, RequestAdmissionLease)

    controller.release(lease, RequestReleaseOutcome(kind="rate_limited", retry_after_seconds=1.0))
    denied = controller.try_acquire(item)
    snapshot = controller.pressure.snapshot(item.resource)

    assert isinstance(denied, RequestAdmissionDenied)
    assert denied.reason == "cooldown"
    assert snapshot is not None
    assert snapshot.current_limit == 2
    assert snapshot.cooldown_remaining_seconds > 0


def test_request_admission_additive_recovery_after_successes() -> None:
    item = _item()
    controller = _controller(
        cap=3,
        config=RequestAdmissionConfig(
            initial_limits={item.resource: 1},
            increase_after_successes=1,
            additive_increase_step=1,
        ),
    )

    lease = controller.try_acquire(item)
    assert isinstance(lease, RequestAdmissionLease)
    controller.release(lease, RequestReleaseOutcome(kind="success"))

    assert controller.pressure.snapshot(item.resource).current_limit == 2  # type: ignore[union-attr]


def test_request_admission_blocking_timeout_raises_typed_error() -> None:
    controller = _controller(cap=1)
    first = _item()
    second = _item(timeout=0.01)
    lease = controller.try_acquire(first)
    assert isinstance(lease, RequestAdmissionLease)

    with pytest.raises(RequestAdmissionError) as exc_info:
        controller.acquire_sync(second)

    assert exc_info.value.decision.reason == "queue_timeout"


def test_request_admission_zero_sync_timeout_is_immediate() -> None:
    controller = _controller(cap=1)
    lease = controller.try_acquire(_item())
    assert isinstance(lease, RequestAdmissionLease)

    with pytest.raises(RequestAdmissionError) as exc_info:
        controller.acquire_sync(_item(RequestDomain.EMBEDDING, timeout=0.0))

    assert exc_info.value.decision.reason == "queue_timeout"
    snapshot = controller.pressure.snapshot(RequestResourceKey("nvidia", "nemotron", RequestDomain.EMBEDDING))
    assert snapshot is not None
    assert snapshot.waiters == 0
    controller.release(lease, RequestReleaseOutcome(kind="success"))


def test_request_admission_logs_sink_failures(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="data_designer.engine.models.request_admission.controller")
    controller = AdaptiveRequestAdmissionController(event_sink=_BrokenRequestSink())

    controller.register(provider_name="nvidia", model_id="nemotron", alias="default", max_parallel_requests=1)

    assert "Request admission event sink raised; dropping event." in caplog.text


@pytest.mark.asyncio(loop_scope="session")
async def test_acquire_sync_rejects_running_event_loop() -> None:
    controller = _controller(cap=1)

    with pytest.raises(RuntimeError, match="would block the running event loop"):
        controller.acquire_sync(_item())


@pytest.mark.asyncio(loop_scope="session")
async def test_try_acquire_does_not_bypass_queued_waiter_for_same_provider_model() -> None:
    controller = _controller(cap=1)
    first = _item(RequestDomain.CHAT)
    queued = _item(RequestDomain.EMBEDDING, timeout=2)
    incoming = _item(RequestDomain.IMAGE)
    lease = controller.try_acquire(first)
    assert isinstance(lease, RequestAdmissionLease)

    queued_task = asyncio.create_task(controller.acquire_async(queued))
    await asyncio.sleep(0)

    denied = controller.try_acquire(incoming)

    assert isinstance(denied, RequestAdmissionDenied)
    assert denied.reason == "no_capacity"
    snapshot = controller.pressure.snapshot(queued.resource)
    assert snapshot is not None
    assert snapshot.waiters == 1
    controller.release(lease, RequestReleaseOutcome(kind="success"))
    queued_lease = await queued_task
    controller.release(queued_lease, RequestReleaseOutcome(kind="success"))


@pytest.mark.asyncio(loop_scope="session")
async def test_request_admission_zero_async_timeout_is_immediate() -> None:
    controller = _controller(cap=1)
    lease = controller.try_acquire(_item())
    assert isinstance(lease, RequestAdmissionLease)

    with pytest.raises(RequestAdmissionError) as exc_info:
        await controller.acquire_async(_item(RequestDomain.EMBEDDING, timeout=0.0))

    assert exc_info.value.decision.reason == "queue_timeout"
    snapshot = controller.pressure.snapshot(RequestResourceKey("nvidia", "nemotron", RequestDomain.EMBEDDING))
    assert snapshot is not None
    assert snapshot.waiters == 0
    controller.release(lease, RequestReleaseOutcome(kind="success"))


@pytest.mark.asyncio(loop_scope="session")
async def test_acquire_async_does_not_assign_expired_waiter_after_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _controller(cap=1)
    monkeypatch.setattr(controller, "_wait_seconds_locked", lambda _item, _now, _deadline: 10.0)
    lease = controller.try_acquire(_item(RequestDomain.CHAT))
    assert isinstance(lease, RequestAdmissionLease)
    queued = _item(RequestDomain.EMBEDDING, timeout=0.01)

    queued_task = asyncio.create_task(controller.acquire_async(queued))
    for _ in range(20):
        snapshot = controller.pressure.snapshot(queued.resource)
        if snapshot is not None and snapshot.waiters == 1:
            break
        await asyncio.sleep(0)
    else:
        raise AssertionError("async waiter did not enqueue")

    def release_after_deadline() -> None:
        time.sleep(0.03)
        controller.release(lease, RequestReleaseOutcome(kind="success"))

    release_thread = threading.Thread(target=release_after_deadline)
    release_thread.start()
    try:
        time.sleep(0.06)
        with pytest.raises(RequestAdmissionError) as exc_info:
            await asyncio.wait_for(queued_task, timeout=0.5)
    finally:
        release_thread.join()

    assert exc_info.value.decision.reason == "queue_timeout"
    snapshot = controller.pressure.snapshot(queued.resource)
    assert snapshot is not None
    assert snapshot.waiters == 0
    assert snapshot.active_lease_count == 0
    assert snapshot.in_flight_count == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_acquire_async_wakes_when_release_assigns_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = _controller(cap=1)
    monkeypatch.setattr(controller, "_wait_seconds_locked", lambda _item, _now, _deadline: 10.0)
    lease = controller.try_acquire(_item(RequestDomain.CHAT))
    assert isinstance(lease, RequestAdmissionLease)
    queued = _item(RequestDomain.EMBEDDING, timeout=30.0)

    queued_task = asyncio.create_task(controller.acquire_async(queued))
    for _ in range(20):
        snapshot = controller.pressure.snapshot(queued.resource)
        if snapshot is not None and snapshot.waiters == 1:
            break
        await asyncio.sleep(0)
    else:
        raise AssertionError("async waiter did not enqueue")

    controller.release(lease, RequestReleaseOutcome(kind="success"))
    queued_lease = await asyncio.wait_for(queued_task, timeout=0.5)

    controller.release(queued_lease, RequestReleaseOutcome(kind="success"))


@pytest.mark.asyncio(loop_scope="session")
async def test_register_wakes_unregistered_async_waiter(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = AdaptiveRequestAdmissionController()
    monkeypatch.setattr(controller, "_wait_seconds_locked", lambda _item, _now, _deadline: 10.0)
    queued = _item(RequestDomain.CHAT, timeout=30.0)

    queued_task = asyncio.create_task(controller.acquire_async(queued))
    for _ in range(20):
        snapshot = controller.pressure.snapshot(queued.resource)
        if snapshot is not None and snapshot.waiters == 1:
            break
        await asyncio.sleep(0)
    else:
        raise AssertionError("async waiter did not enqueue")

    controller.register(provider_name="nvidia", model_id="nemotron", alias="default", max_parallel_requests=1)
    queued_lease = await asyncio.wait_for(queued_task, timeout=0.5)

    controller.release(queued_lease, RequestReleaseOutcome(kind="success"))


@pytest.mark.asyncio(loop_scope="session")
async def test_async_cancellation_after_waiter_assignment_releases_lease() -> None:
    controller = _controller(cap=1)
    first = _item(RequestDomain.CHAT)
    queued = _item(RequestDomain.EMBEDDING, timeout=1.0)
    lease = controller.try_acquire(first)
    assert isinstance(lease, RequestAdmissionLease)

    queued_task = asyncio.create_task(controller.acquire_async(queued))
    await asyncio.sleep(0)
    controller.release(lease, RequestReleaseOutcome(kind="success"))
    queued_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await queued_task

    snapshot = controller.pressure.snapshot(queued.resource)
    assert snapshot is not None
    assert snapshot.active_lease_count == 0
    assert snapshot.in_flight_count == 0
    assert snapshot.waiters == 0
