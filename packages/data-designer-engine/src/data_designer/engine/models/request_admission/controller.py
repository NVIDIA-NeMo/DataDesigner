# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
import uuid
from collections import Counter, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from data_designer.engine.models.request_admission.config import RequestAdmissionConfig
from data_designer.engine.models.request_admission.limits import AdaptiveRequestLimitState
from data_designer.engine.models.request_admission.outcomes import ReleaseResult, RequestReleaseOutcome
from data_designer.engine.models.request_admission.pressure import (
    ProviderModelPressureSnapshot,
    RequestPressureSnapshot,
    RequestPressureSnapshotProvider,
)
from data_designer.engine.models.request_admission.queue import RequestFairQueue, RequestWaiter
from data_designer.engine.models.request_admission.resources import (
    RequestAdmissionItem,
    RequestDomain,
    RequestResourceKey,
)
from data_designer.engine.models.resources import ProviderModelKey
from data_designer.engine.observability import (
    RequestAdmissionEvent,
    RequestAdmissionEventSink,
    emit_request_admission_event,
    runtime_correlation_provider,
)

logger = logging.getLogger(__name__)

DEFAULT_MIN_LIMIT: int = 1
RequestDenyReason = Literal[
    "no_capacity",
    "cooldown",
    "queue_timeout",
    "queued_waiters_ahead",
    "cancellation",
    "shutdown",
    "hard_policy_denial",
]
RELEASED_LEASE_HISTORY_LIMIT = 8192
_TERMINAL_DENIAL_REASONS: frozenset[RequestDenyReason] = frozenset({"hard_policy_denial", "shutdown"})


@dataclass(frozen=True)
class RequestAdmissionDenied:
    item: RequestAdmissionItem
    reason: RequestDenyReason
    retry_after_seconds: float | None = None
    available_after_monotonic: float | None = None
    snapshot: object | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RequestAdmissionLease:
    lease_id: str
    item: RequestAdmissionItem
    acquired_at: float
    current_adaptive_limit: int
    effective_max: int
    controller_generation: str


RequestAdmissionDecision = RequestAdmissionLease | RequestAdmissionDenied


class RequestAdmissionError(RuntimeError):
    """Raised by blocking acquire paths when no request lease is acquired."""

    def __init__(self, decision: RequestAdmissionDenied) -> None:
        super().__init__(f"Request admission failed: {decision.reason}")
        self.decision = decision


class RequestAdmissionController(Protocol):
    def try_acquire(self, item: RequestAdmissionItem) -> RequestAdmissionDecision: ...

    def acquire_sync(self, item: RequestAdmissionItem) -> RequestAdmissionLease: ...

    async def acquire_async(self, item: RequestAdmissionItem) -> RequestAdmissionLease: ...

    def release(self, lease: RequestAdmissionLease, outcome: RequestReleaseOutcome) -> ReleaseResult: ...

    @property
    def pressure(self) -> RequestPressureSnapshotProvider: ...


@dataclass
class _GlobalCapState:
    limits_by_alias: dict[str, int] = field(default_factory=dict)
    effective_max: int = 0

    def register_alias(self, alias: str, max_parallel: int) -> None:
        self.limits_by_alias[alias] = max(1, max_parallel)
        self.effective_max = min(self.limits_by_alias.values())


class AdaptiveRequestAdmissionController(RequestPressureSnapshotProvider):
    """AIMD-backed request admission controller with exact request leases."""

    def __init__(
        self,
        config: RequestAdmissionConfig | None = None,
        *,
        event_sink: RequestAdmissionEventSink | None = None,
    ) -> None:
        self._config = config or RequestAdmissionConfig()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._generation = uuid.uuid4().hex
        self._global_caps: dict[ProviderModelKey, _GlobalCapState] = {}
        self._domains: dict[RequestResourceKey, AdaptiveRequestLimitState] = {}
        self._active_leases: dict[str, RequestAdmissionLease] = {}
        self._released: set[str] = set()
        self._released_order: deque[str] = deque(maxlen=RELEASED_LEASE_HISTORY_LIMIT)
        self._aggregate_in_flight: Counter[ProviderModelKey] = Counter()
        self._aggregate_active_leases: Counter[ProviderModelKey] = Counter()
        self._sequence = 0
        self._release_diagnostics: Counter[str] = Counter()
        self._queue = RequestFairQueue()
        self._event_sink = event_sink

    @property
    def pressure(self) -> RequestPressureSnapshotProvider:
        return self

    @property
    def config(self) -> RequestAdmissionConfig:
        return self._config

    def register(
        self,
        *,
        provider_name: str,
        model_id: str,
        alias: str,
        max_parallel_requests: int,
    ) -> None:
        events: list[RequestAdmissionEvent] = []
        with self._lock:
            key = ProviderModelKey(provider_name, model_id)
            cap = self._global_caps.setdefault(key, _GlobalCapState())
            before = cap.effective_max
            cap.register_alias(alias, max_parallel_requests)
            self._sequence += 1
            for resource, state in self._domains.items():
                if resource.provider_model_key == key:
                    effective_max = self._effective_max_for_resource(resource)
                    state.current_limit = min(state.current_limit, effective_max)
            events.append(
                self._request_event_locked(
                    "request_resource_registered",
                    request_resource_key=RequestResourceKey(provider_name, model_id, RequestDomain.CHAT),
                    diagnostics={"alias": alias, "provider_model": key, "max_parallel_requests": max_parallel_requests},
                )
            )
            if before != cap.effective_max:
                events.append(
                    self._request_event_locked(
                        "request_effective_cap_changed",
                        request_resource_key=RequestResourceKey(provider_name, model_id, RequestDomain.CHAT),
                        diagnostics={"provider_model": key, "previous": before, "current": cap.effective_max},
                    )
                )
            self._admit_waiters_locked(events)
            self._condition.notify_all()
        self._emit_events(events)

    def try_acquire(self, item: RequestAdmissionItem) -> RequestAdmissionDecision:
        now = time.monotonic()
        events: list[RequestAdmissionEvent] = []
        decision: RequestAdmissionDecision
        with self._lock:
            events.append(self._request_event_locked("request_wait_started", item=item))
            if self._queued_waiter_ahead_locked(item, now):
                decision = RequestAdmissionDenied(
                    item=item,
                    reason="queued_waiters_ahead",
                    snapshot=self._snapshot_locked(item.resource, now),
                )
                events.append(self._request_event_locked("request_acquire_denied", item=item, decision=decision))
            else:
                denied = self._denial_for(item, now)
                if denied is not None:
                    decision = denied
                    events.append(self._request_event_locked("request_acquire_denied", item=item, decision=decision))
                else:
                    decision = self._acquire_locked(item, now)
                    events.append(self._request_event_locked("request_wait_completed", item=item, lease=decision))
                    events.append(self._request_event_locked("request_lease_acquired", item=item, lease=decision))
        self._emit_events(events)
        return decision

    def acquire_sync(self, item: RequestAdmissionItem) -> RequestAdmissionLease:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("acquire_sync would block the running event loop; use acquire_async instead.")

        timeout = (
            item.queue_wait_timeout_seconds
            if item.queue_wait_timeout_seconds is not None
            else self._config.default_queue_wait_timeout_seconds
        )
        now = time.monotonic()
        deadline = now + timeout if timeout is not None else None
        waiter = RequestWaiter(waiter_id=uuid.uuid4().hex, item=item, enqueued_at=now, deadline_monotonic=deadline)
        events: list[RequestAdmissionEvent] = []
        try:
            while True:
                with self._lock:
                    if waiter.assigned_lease is not None:
                        return waiter.assigned_lease
                    now = time.monotonic()
                    if deadline is not None and now >= deadline:
                        self._remove_waiter_locked(waiter)
                        denied = RequestAdmissionDenied(
                            item=item,
                            reason="queue_timeout",
                            snapshot=self._snapshot_locked(item.resource, now),
                        )
                        events.append(self._request_event_locked("request_wait_timeout", item=item, decision=denied))
                        raise RequestAdmissionError(denied)
                    if not self._queue.contains(waiter.waiter_id) and waiter.assigned_lease is None:
                        self._enqueue_waiter_locked(waiter, events)
                    self._admit_waiters_locked(events)
                    if waiter.assigned_lease is not None:
                        return waiter.assigned_lease
                    now = time.monotonic()
                    if (denied := self._terminal_denial_for(item, now)) is not None:
                        self._remove_waiter_locked(waiter)
                        events.append(self._request_event_locked("request_acquire_denied", item=item, decision=denied))
                        self._condition.notify_all()
                        raise RequestAdmissionError(denied)
                    if deadline is not None and now >= deadline:
                        self._remove_waiter_locked(waiter)
                        denied = RequestAdmissionDenied(
                            item=item,
                            reason="queue_timeout",
                            snapshot=self._snapshot_locked(item.resource, now),
                        )
                        events.append(self._request_event_locked("request_wait_timeout", item=item, decision=denied))
                        raise RequestAdmissionError(denied)
                    wait = self._wait_seconds_locked(item, now, deadline)
                    self._condition.wait(timeout=wait)
        finally:
            self._emit_events(events)

    async def acquire_async(self, item: RequestAdmissionItem) -> RequestAdmissionLease:
        loop = asyncio.get_running_loop()
        wakeup = asyncio.Event()
        timeout = (
            item.queue_wait_timeout_seconds
            if item.queue_wait_timeout_seconds is not None
            else self._config.default_queue_wait_timeout_seconds
        )
        now = time.monotonic()
        deadline = now + timeout if timeout is not None else None
        waiter = RequestWaiter(
            waiter_id=uuid.uuid4().hex,
            item=item,
            enqueued_at=now,
            deadline_monotonic=deadline,
            wakeup=lambda: loop.call_soon_threadsafe(wakeup.set),
        )
        events: list[RequestAdmissionEvent] = []
        try:
            while True:
                with self._lock:
                    if waiter.assigned_lease is not None:
                        return waiter.assigned_lease
                    now = time.monotonic()
                    if deadline is not None and now >= deadline:
                        self._remove_waiter_locked(waiter)
                        denied = RequestAdmissionDenied(
                            item=item,
                            reason="queue_timeout",
                            snapshot=self._snapshot_locked(item.resource, now),
                        )
                        events.append(self._request_event_locked("request_wait_timeout", item=item, decision=denied))
                        raise RequestAdmissionError(denied)
                    if not self._queue.contains(waiter.waiter_id) and waiter.assigned_lease is None:
                        self._enqueue_waiter_locked(waiter, events)
                    self._admit_waiters_locked(events)
                    if waiter.assigned_lease is not None:
                        return waiter.assigned_lease
                    now = time.monotonic()
                    if (denied := self._terminal_denial_for(item, now)) is not None:
                        self._remove_waiter_locked(waiter)
                        events.append(self._request_event_locked("request_acquire_denied", item=item, decision=denied))
                        self._condition.notify_all()
                        raise RequestAdmissionError(denied)
                    if deadline is not None and now >= deadline:
                        self._remove_waiter_locked(waiter)
                        denied = RequestAdmissionDenied(
                            item=item,
                            reason="queue_timeout",
                            snapshot=self._snapshot_locked(item.resource, now),
                        )
                        events.append(self._request_event_locked("request_wait_timeout", item=item, decision=denied))
                        raise RequestAdmissionError(denied)
                    wait = self._wait_seconds_locked(item, now, deadline)
                try:
                    await asyncio.wait_for(wakeup.wait(), timeout=wait)
                except asyncio.TimeoutError:
                    pass
                wakeup.clear()
        except asyncio.CancelledError:
            lease_to_release: RequestAdmissionLease | None = None
            with self._lock:
                lease_to_release = waiter.assigned_lease
                if lease_to_release is None:
                    self._remove_waiter_locked(waiter)
                denied = RequestAdmissionDenied(item=item, reason="cancellation")
                events.append(
                    self._request_event_locked(
                        "request_wait_cancelled",
                        item=item,
                        lease=lease_to_release,
                        decision=denied,
                    )
                )
                self._condition.notify_all()
            if lease_to_release is not None:
                self._emit_events(events)
                events.clear()
                self.release(lease_to_release, RequestReleaseOutcome(kind="local_cancelled"))
            raise
        finally:
            self._emit_events(events)

    def release(self, lease: RequestAdmissionLease, outcome: RequestReleaseOutcome) -> ReleaseResult:
        now = time.monotonic()
        events: list[RequestAdmissionEvent] = []
        result: ReleaseResult
        with self._lock:
            if lease.controller_generation != self._generation:
                self._release_diagnostics["wrong_controller_generation"] += 1
                result = ReleaseResult(released=False, reason="wrong_controller_generation")
                events.append(
                    self._request_event_locked(
                        "request_release_diagnostic", item=lease.item, lease=lease, result=result
                    )
                )
            elif (active := self._active_leases.pop(lease.lease_id, None)) is None:
                reason = "duplicate" if lease.lease_id in self._released else "unknown_lease"
                self._release_diagnostics[reason] += 1
                result = ReleaseResult(released=False, reason=reason)
                events.append(
                    self._request_event_locked(
                        "request_release_diagnostic", item=lease.item, lease=lease, result=result
                    )
                )
            elif active != lease:
                self._active_leases[lease.lease_id] = active
                self._release_diagnostics["stale_lease"] += 1
                result = ReleaseResult(released=False, reason="stale_lease")
                events.append(
                    self._request_event_locked(
                        "request_release_diagnostic", item=lease.item, lease=lease, result=result
                    )
                )
            else:
                self._remember_released_locked(lease.lease_id)
                resource = active.item.resource
                provider_model = resource.provider_model_key
                state = self._get_or_create_state(resource)
                state.in_flight = max(0, state.in_flight - 1)
                state.active_lease_count = max(0, state.active_lease_count - 1)
                state.last_outcome = outcome.kind
                self._aggregate_in_flight[provider_model] = max(0, self._aggregate_in_flight[provider_model] - 1)
                self._aggregate_active_leases[provider_model] = max(
                    0,
                    self._aggregate_active_leases[provider_model] - 1,
                )
                self._apply_outcome(state, resource, active.current_adaptive_limit, outcome, now, events)
                self._sequence += 1
                result = ReleaseResult(released=True, reason="released")
                if outcome.kind == "rate_limited":
                    events.append(self._request_event_locked("request_rate_limited", item=active.item, lease=active))
                events.append(
                    self._request_event_locked(
                        "request_lease_released",
                        item=active.item,
                        lease=active,
                        result=result,
                        outcome=outcome,
                    )
                )
                self._admit_waiters_locked(events)
            self._condition.notify_all()
        self._emit_events(events)
        return result

    def snapshot(self, resource: RequestResourceKey) -> RequestPressureSnapshot | None:
        with self._lock:
            if resource not in self._domains:
                return None
            return self._snapshot_locked(resource, time.monotonic())

    def snapshots(self) -> Mapping[RequestResourceKey, RequestPressureSnapshot]:
        with self._lock:
            now = time.monotonic()
            return {resource: self._snapshot_locked(resource, now) for resource in self._domains}

    def global_snapshot(self, provider: str, model: str) -> ProviderModelPressureSnapshot | None:
        with self._lock:
            key = ProviderModelKey(provider, model)
            if key not in self._global_caps:
                return None
            return self._global_snapshot_locked(key, time.monotonic())

    def global_snapshots(self) -> Mapping[ProviderModelKey, ProviderModelPressureSnapshot]:
        with self._lock:
            now = time.monotonic()
            return {key: self._global_snapshot_locked(key, now) for key in self._global_caps}

    def _queued_waiter_ahead_locked(self, item: RequestAdmissionItem, now: float) -> bool:
        if not self._queue.has_waiters:
            return False
        self._expire_waiters_locked(now)
        selection = self._queue.select_next(lambda waiter, _view: self._denial_for(waiter.item, now) is None)
        if selection is None:
            return False
        selected_key = selection.item.resource.provider_model_key
        return selected_key == item.resource.provider_model_key or selection.item.resource == item.resource

    def _enqueue_waiter_locked(self, waiter: RequestWaiter, events: list[RequestAdmissionEvent]) -> None:
        if self._queue.enqueue(waiter):
            self._get_or_create_state(waiter.item.resource).waiters += 1
            self._sequence += 1
            if self._queue.view().queued_total == 1:
                events.append(self._request_event_locked("request_queue_formed", item=waiter.item))
            events.append(self._request_event_locked("request_wait_started", item=waiter.item))

    def _remove_waiter_locked(self, waiter: RequestWaiter) -> None:
        removed = self._queue.remove(waiter.waiter_id)
        if removed is None:
            return
        state = self._get_or_create_state(waiter.item.resource)
        state.waiters = max(0, state.waiters - 1)
        self._sequence += 1

    def _expire_waiters_locked(self, now: float) -> None:
        for waiter in self._queue.waiters():
            if waiter.deadline_monotonic is not None and now >= waiter.deadline_monotonic:
                self._remove_waiter_locked(waiter)
                self._wake_waiter_locked(waiter)

    def _admit_waiters_locked(self, events: list[RequestAdmissionEvent]) -> None:
        while self._queue.has_waiters:
            now = time.monotonic()
            self._expire_waiters_locked(now)
            if not self._queue.has_waiters:
                return
            selection = self._queue.select_next(lambda waiter, _view: self._denial_for(waiter.item, now) is None)
            if selection is None:
                return
            waiter = self._queue.commit(selection)
            if waiter is None:
                return
            state = self._get_or_create_state(waiter.item.resource)
            state.waiters = max(0, state.waiters - 1)
            lease = self._acquire_locked(waiter.item, now)
            waiter.assigned_lease = lease
            self._wake_waiter_locked(waiter)
            events.append(self._request_event_locked("request_wait_completed", item=waiter.item, lease=lease))
            events.append(self._request_event_locked("request_lease_acquired", item=waiter.item, lease=lease))
            if not self._queue.has_waiters:
                events.append(self._request_event_locked("request_queue_drained", item=waiter.item))

    def _wake_waiter_locked(self, waiter: RequestWaiter) -> None:
        if waiter.wakeup is None:
            return
        waiter.wakeup()

    def _wait_seconds_locked(
        self,
        item: RequestAdmissionItem,
        now: float,
        deadline: float | None,
    ) -> float:
        candidates = [0.05]
        if deadline is not None:
            candidates.append(max(0.0, deadline - now))
        state = self._domains.get(item.resource)
        if state is not None and state.blocked_until > now:
            candidates.append(max(0.0, state.blocked_until - now))
        return max(0.0, min(candidates))

    def _denial_for(self, item: RequestAdmissionItem, now: float) -> RequestAdmissionDenied | None:
        resource = item.resource
        provider_model = resource.provider_model_key
        if provider_model not in self._global_caps:
            return RequestAdmissionDenied(item=item, reason="hard_policy_denial", diagnostics={"unregistered": True})
        state = self._get_or_create_state(resource)
        self._apply_startup_ramp_locked(state, resource, now)
        if now < state.blocked_until:
            return RequestAdmissionDenied(
                item=item,
                reason="cooldown",
                retry_after_seconds=state.blocked_until - now,
                available_after_monotonic=state.blocked_until,
                snapshot=self._snapshot_locked(resource, now),
            )
        effective_max = self._effective_max_for_resource(resource)
        aggregate_cap = self._global_caps[provider_model].effective_max
        if state.in_flight >= min(state.current_limit, effective_max):
            return RequestAdmissionDenied(
                item=item, reason="no_capacity", snapshot=self._snapshot_locked(resource, now)
            )
        if self._aggregate_in_flight[provider_model] >= aggregate_cap:
            return RequestAdmissionDenied(
                item=item, reason="no_capacity", snapshot=self._snapshot_locked(resource, now)
            )
        return None

    def _terminal_denial_for(self, item: RequestAdmissionItem, now: float) -> RequestAdmissionDenied | None:
        denied = self._denial_for(item, now)
        if denied is None or denied.reason not in _TERMINAL_DENIAL_REASONS:
            return None
        return denied

    def _remember_released_locked(self, lease_id: str) -> None:
        if lease_id in self._released:
            return
        maxlen = self._released_order.maxlen
        if maxlen is not None and len(self._released_order) >= maxlen:
            self._released.discard(self._released_order[0])
        self._released.add(lease_id)
        self._released_order.append(lease_id)

    def _acquire_locked(self, item: RequestAdmissionItem, now: float) -> RequestAdmissionLease:
        resource = item.resource
        provider_model = resource.provider_model_key
        state = self._get_or_create_state(resource)
        state.in_flight += 1
        state.active_lease_count += 1
        self._aggregate_in_flight[provider_model] += 1
        self._aggregate_active_leases[provider_model] += 1
        lease = RequestAdmissionLease(
            lease_id=uuid.uuid4().hex,
            item=item,
            acquired_at=now,
            current_adaptive_limit=state.current_limit,
            effective_max=self._effective_max_for_resource(resource),
            controller_generation=self._generation,
        )
        self._active_leases[lease.lease_id] = lease
        self._sequence += 1
        return lease

    def _apply_outcome(
        self,
        state: AdaptiveRequestLimitState,
        resource: RequestResourceKey,
        admitted_adaptive_limit: int,
        outcome: RequestReleaseOutcome,
        now: float,
        events: list[RequestAdmissionEvent],
    ) -> None:
        effective_max = self._effective_max_for_resource(resource)
        if outcome.kind == "rate_limited":
            state.startup_ramp_active = False
            prev_limit = state.current_limit
            should_decrease = admitted_adaptive_limit <= prev_limit
            state.consecutive_rate_limits += 1
            cooldown = (
                outcome.retry_after_seconds
                if outcome.retry_after_seconds is not None and outcome.retry_after_seconds > 0
                else self._config.cooldown_seconds
            )
            state.blocked_until = now + cooldown
            state.success_streak = 0
            if should_decrease:
                state.current_limit = max(
                    1, math.floor(state.current_limit * self._config.multiplicative_decrease_factor)
                )
                if state.rate_limit_ceiling == 0:
                    state.rate_limit_ceiling = max(1, admitted_adaptive_limit)
                if state.current_limit != prev_limit:
                    events.append(
                        self._request_event_locked(
                            "request_limit_decreased",
                            request_resource_key=resource,
                            diagnostics={"previous": prev_limit, "current": state.current_limit},
                        )
                    )
            return
        if state.startup_ramp_active:
            self._apply_startup_ramp_locked(state, resource, now)
            if outcome.kind == "success":
                state.success_streak = 0
                return
        if outcome.kind == "success" and now >= state.blocked_until:
            prev_limit = state.current_limit
            state.consecutive_rate_limits = 0
            state.success_streak += 1
            if state.success_streak >= self._config.successes_until_increase:
                state.current_limit = min(effective_max, state.current_limit + self._config.additive_increase_step)
                state.success_streak = 0
                if state.current_limit != prev_limit:
                    events.append(
                        self._request_event_locked(
                            "request_limit_increased",
                            request_resource_key=resource,
                            diagnostics={"previous": prev_limit, "current": state.current_limit},
                        )
                    )
                    if state.rate_limit_ceiling and state.current_limit > state.rate_limit_ceiling:
                        events.append(
                            self._request_event_locked(
                                "request_soft_ceiling_recovered",
                                request_resource_key=resource,
                                diagnostics={"rate_limit_ceiling": state.rate_limit_ceiling},
                            )
                        )
                    if state.current_limit == effective_max and state.blocked_until <= now:
                        events.append(
                            self._request_event_locked("request_fully_recovered", request_resource_key=resource)
                        )
            return
        if state.in_flight == 0 and outcome.kind not in {"local_cancelled", "local_timeout"}:
            state.consecutive_rate_limits = 0

    def _get_or_create_state(self, resource: RequestResourceKey) -> AdaptiveRequestLimitState:
        state = self._domains.get(resource)
        if state is None:
            initial = self._initial_limit_for_resource(resource)
            ramp_active = self._config.startup_ramp_seconds > 0.0 and initial > DEFAULT_MIN_LIMIT
            state = AdaptiveRequestLimitState(
                current_limit=DEFAULT_MIN_LIMIT if ramp_active else initial,
                startup_ramp_started_at=time.monotonic(),
                startup_ramp_active=ramp_active,
            )
            self._domains[resource] = state
        return state

    def _initial_limit_for_resource(self, resource: RequestResourceKey) -> int:
        effective_max = self._effective_max_for_resource(resource)
        initial = self._config.initial_limits.get(resource, effective_max)
        return max(DEFAULT_MIN_LIMIT, min(initial, effective_max))

    def _effective_max_for_resource(self, resource: RequestResourceKey) -> int:
        provider_model_cap = self._global_caps.get(resource.provider_model_key)
        static_cap = provider_model_cap.effective_max if provider_model_cap is not None else DEFAULT_MIN_LIMIT
        clamp = self._config.max_limit_clamps.get(resource)
        return max(DEFAULT_MIN_LIMIT, min(static_cap, clamp if clamp is not None else static_cap))

    def _apply_startup_ramp_locked(
        self,
        state: AdaptiveRequestLimitState,
        resource: RequestResourceKey,
        now: float,
    ) -> None:
        if not state.startup_ramp_active:
            return
        target_limit = self._initial_limit_for_resource(resource)
        if self._config.startup_ramp_seconds <= 0.0 or target_limit <= DEFAULT_MIN_LIMIT:
            changed = state.current_limit != target_limit or state.startup_ramp_active
            state.current_limit = min(state.current_limit, target_limit)
            state.startup_ramp_active = False
            if changed:
                self._sequence += 1
            return

        elapsed = max(0.0, now - state.startup_ramp_started_at)
        previous_limit = state.current_limit
        if elapsed >= self._config.startup_ramp_seconds:
            state.current_limit = target_limit
            state.startup_ramp_active = False
        else:
            fraction = elapsed / self._config.startup_ramp_seconds
            ramp_slots = math.floor((target_limit - DEFAULT_MIN_LIMIT) * fraction)
            state.current_limit = min(target_limit, DEFAULT_MIN_LIMIT + ramp_slots)
        if state.current_limit != previous_limit or not state.startup_ramp_active:
            self._sequence += 1

    def _snapshot_locked(self, resource: RequestResourceKey, now: float) -> RequestPressureSnapshot:
        state = self._get_or_create_state(resource)
        self._apply_startup_ramp_locked(state, resource, now)
        blocked_until = state.blocked_until if state.blocked_until > now else None
        return RequestPressureSnapshot(
            captured_at=now,
            sequence=self._sequence,
            resource=resource,
            effective_max=self._effective_max_for_resource(resource),
            current_limit=state.current_limit,
            in_flight_count=state.in_flight,
            active_lease_count=state.active_lease_count,
            waiters=state.waiters,
            blocked_until_monotonic=blocked_until,
            cooldown_remaining_seconds=max(0.0, state.blocked_until - now),
            rate_limit_ceiling=state.rate_limit_ceiling,
            consecutive_rate_limits=state.consecutive_rate_limits,
            last_outcome=state.last_outcome,
            leak_diagnostics=dict(self._release_diagnostics),
        )

    def _global_snapshot_locked(self, key: ProviderModelKey, now: float) -> ProviderModelPressureSnapshot:
        cap = self._global_caps[key]
        domains = {
            resource.domain: state.current_limit
            for resource, state in self._domains.items()
            if resource.provider_model_key == key
        }
        return ProviderModelPressureSnapshot(
            captured_at=now,
            sequence=self._sequence,
            provider_model=key,
            static_cap=cap.effective_max,
            aggregate_in_flight=self._aggregate_in_flight[key],
            aggregate_active_lease_count=self._aggregate_active_leases[key],
            aliases=tuple(sorted(cap.limits_by_alias)),
            raw_caps=dict(cap.limits_by_alias),
            domains=domains,
        )

    def _request_event_locked(
        self,
        event_kind: str,
        *,
        item: RequestAdmissionItem | None = None,
        lease: RequestAdmissionLease | None = None,
        decision: RequestAdmissionDenied | None = None,
        result: ReleaseResult | None = None,
        outcome: RequestReleaseOutcome | None = None,
        request_resource_key: RequestResourceKey | None = None,
        diagnostics: Mapping[str, object] | None = None,
    ) -> RequestAdmissionEvent:
        self._sequence += 1
        event_context = item.event_context if item is not None else None
        resource = request_resource_key or (item.resource if item is not None else None)
        group_key = item.group.key if item is not None else None
        reason_or_outcome = None
        if decision is not None:
            reason_or_outcome = decision.reason
        elif outcome is not None:
            reason_or_outcome = outcome.kind
        elif result is not None:
            reason_or_outcome = result.reason
        return RequestAdmissionEvent.capture(
            event_kind,  # type: ignore[arg-type]
            sequence=self._sequence,
            correlation=event_context.captured_correlation
            if event_context is not None
            else runtime_correlation_provider.current(),
            request_attempt_id=event_context.request_attempt_id if event_context is not None else None,
            request_lease_id=lease.lease_id if lease is not None else None,
            request_resource_key=resource,
            request_group_key=group_key,
            reason_or_outcome=reason_or_outcome,
            pressure_snapshot=self._snapshot_locked(resource, time.monotonic()) if resource is not None else None,
            diagnostics=dict(diagnostics or {}),
        )

    def _emit_events(self, events: list[RequestAdmissionEvent]) -> None:
        for event in events:
            if self._event_sink is not None:
                try:
                    self._event_sink.emit_request_event(event)
                except Exception:
                    logger.warning("Request admission event sink raised; dropping event.", exc_info=True)
            emit_request_admission_event(event)
