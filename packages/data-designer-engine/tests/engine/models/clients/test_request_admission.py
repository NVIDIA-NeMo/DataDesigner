# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest

from data_designer.engine.models.clients.request_admission import (
    AdaptiveRequestAdmissionController,
    RequestAdmissionConfig,
    RequestAdmissionDenied,
    RequestAdmissionError,
    RequestAdmissionItem,
    RequestAdmissionLease,
    RequestDomain,
    RequestGroupSpec,
    RequestReleaseOutcome,
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
