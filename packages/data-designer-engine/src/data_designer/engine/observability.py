# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextvars
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Literal, Protocol


@dataclass(frozen=True)
class RuntimeCorrelation:
    run_id: str
    row_group: int | None
    task_column: str | None
    task_type: str | None
    scheduling_group_kind: str | None
    scheduling_group_identity_hash: str | None
    task_execution_id: str | None


JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


def _json_safe(value: object) -> JsonValue:
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_safe(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {_json_safe_key(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set | frozenset):
        return [_json_safe(item) for item in sorted(value, key=repr)]
    return str(value)


def _json_safe_key(value: object) -> str:
    safe = _json_safe(value)
    if isinstance(safe, str):
        return safe
    return str(safe)


def _json_safe_dict(value: Mapping[str, object] | None) -> dict[str, JsonValue]:
    if value is None:
        return {}
    return {_json_safe_key(key): _json_safe(item) for key, item in value.items()}


class RuntimeCorrelationProvider:
    """Context-variable backed runtime correlation provider."""

    def __init__(self) -> None:
        self._current: contextvars.ContextVar[RuntimeCorrelation | None] = contextvars.ContextVar(
            "data_designer_runtime_correlation",
            default=None,
        )

    def current(self) -> RuntimeCorrelation | None:
        return self._current.get()

    def set(self, correlation: RuntimeCorrelation | None) -> contextvars.Token:
        return self._current.set(correlation)

    def reset(self, token: contextvars.Token) -> None:
        self._current.reset(token)


runtime_correlation_provider = RuntimeCorrelationProvider()

SchedulerAdmissionEventKind = Literal[
    "scheduler_job_started",
    "scheduler_job_completed",
    "scheduler_health_snapshot",
    "dependency_ready",
    "ready_enqueued",
    "row_group_admitted",
    "row_group_admission_blocked",
    "row_group_admission_target_changed",
    "row_group_checkpointed",
    "selected",
    "queue_empty",
    "admission_blocked",
    "group_capped",
    "request_pressure_advisory_skipped",
    "task_lease_acquired",
    "admission_denied",
    "worker_spawned",
    "worker_spawn_failed",
    "stale_selection",
    "retry_deferred",
    "non_retryable_dropped",
    "cancelled",
    "salvage_redispatched",
    "queue_drained",
    "task_completed",
    "task_lease_released",
    "release_diagnostic",
]

RequestAdmissionEventKind = Literal[
    "request_resource_registered",
    "request_effective_cap_changed",
    "request_queue_formed",
    "request_wait_started",
    "request_wait_completed",
    "request_wait_timeout",
    "request_wait_cancelled",
    "request_acquire_denied",
    "request_lease_acquired",
    "model_request_started",
    "model_request_completed",
    "request_queue_drained",
    "request_rate_limited",
    "request_limit_decreased",
    "request_limit_increased",
    "request_soft_ceiling_recovered",
    "request_fully_recovered",
    "request_lease_released",
    "request_release_diagnostic",
]


@dataclass(frozen=True)
class SchedulerAdmissionEvent:
    event_kind: SchedulerAdmissionEventKind
    captured_at_monotonic: float
    sequence: int
    captured_correlation: JsonValue = None
    task_id: str | None = None
    task_execution_id: str | None = None
    task_lease_id: str | None = None
    scheduler_resource_key: str | None = None
    reason_or_result: str | None = None
    snapshot: JsonValue = None
    diagnostics: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "captured_correlation", _json_safe(self.captured_correlation))
        object.__setattr__(self, "snapshot", _json_safe(self.snapshot))
        object.__setattr__(self, "diagnostics", _json_safe_dict(self.diagnostics))

    @classmethod
    def capture(
        cls,
        event_kind: SchedulerAdmissionEventKind,
        *,
        sequence: int,
        correlation: RuntimeCorrelation | None = None,
        **kwargs: object,
    ) -> SchedulerAdmissionEvent:
        return cls(
            event_kind=event_kind,
            captured_at_monotonic=time.monotonic(),
            sequence=sequence,
            captured_correlation=correlation,
            **kwargs,
        )


@dataclass(frozen=True)
class RequestAdmissionEvent:
    event_kind: RequestAdmissionEventKind
    captured_at_monotonic: float
    sequence: int
    captured_correlation: JsonValue = None
    request_attempt_id: str | None = None
    request_lease_id: str | None = None
    request_resource_key: JsonValue = None
    request_group_key: JsonValue = None
    reason_or_outcome: str | None = None
    pressure_snapshot: JsonValue = None
    diagnostics: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "captured_correlation", _json_safe(self.captured_correlation))
        object.__setattr__(self, "request_resource_key", _json_safe(self.request_resource_key))
        object.__setattr__(self, "request_group_key", _json_safe(self.request_group_key))
        object.__setattr__(self, "pressure_snapshot", _json_safe(self.pressure_snapshot))
        object.__setattr__(self, "diagnostics", _json_safe_dict(self.diagnostics))

    @classmethod
    def capture(
        cls,
        event_kind: RequestAdmissionEventKind,
        *,
        sequence: int,
        correlation: RuntimeCorrelation | None = None,
        **kwargs: object,
    ) -> RequestAdmissionEvent:
        return cls(
            event_kind=event_kind,
            captured_at_monotonic=time.monotonic(),
            sequence=sequence,
            captured_correlation=correlation,
            **kwargs,
        )


class SchedulerAdmissionEventSink(Protocol):
    def emit_scheduler_event(self, event: SchedulerAdmissionEvent) -> None: ...


class RequestAdmissionEventSink(Protocol):
    def emit_request_event(self, event: RequestAdmissionEvent) -> None: ...


class InMemoryAdmissionEventSink:
    """Small sink used by tests, diagnostics, and benchmark smoke runs."""

    def __init__(self) -> None:
        self.scheduler_events: list[SchedulerAdmissionEvent] = []
        self.request_events: list[RequestAdmissionEvent] = []

    def emit_scheduler_event(self, event: SchedulerAdmissionEvent) -> None:
        self.scheduler_events.append(event)

    def emit_request_event(self, event: RequestAdmissionEvent) -> None:
        self.request_events.append(event)


@dataclass(frozen=True)
class CorrelatedRuntimeView:
    scheduler_events: tuple[SchedulerAdmissionEvent, ...]
    request_events: tuple[RequestAdmissionEvent, ...]

    @property
    def timeline(self) -> tuple[SchedulerAdmissionEvent | RequestAdmissionEvent, ...]:
        return tuple(
            sorted(
                (*self.scheduler_events, *self.request_events),
                key=lambda event: (event.captured_at_monotonic, event.sequence),
            )
        )
