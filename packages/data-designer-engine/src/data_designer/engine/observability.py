# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextvars
import itertools
import json
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from os import PathLike
from threading import Lock
from typing import Callable, Literal, Protocol, TextIO, runtime_checkable

logger = logging.getLogger(__name__)


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


@runtime_checkable
class SchedulerAdmissionEventSink(Protocol):
    def emit_scheduler_event(self, event: SchedulerAdmissionEvent) -> None: ...


class JsonlSchedulerEventSink:
    """Append scheduler events as newline-delimited JSON from a single writer.

    A write or flush failure disables the sink and is re-raised once for the caller to handle.
    """

    def __init__(self, path: str | PathLike[str]) -> None:
        self._path = path
        self._file: TextIO | None = None

    def __enter__(self) -> JsonlSchedulerEventSink | None:
        try:
            self._file = open(self._path, "a", encoding="utf-8")
        except Exception:
            logger.warning("Failed to open scheduler event file %s", self._path, exc_info=True)
            return None
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def emit_scheduler_event(self, event: SchedulerAdmissionEvent) -> None:
        file = self._file
        if file is None:
            return
        try:
            file.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
            file.flush()
        except Exception:
            self._file = None
            try:
                file.close()
            except Exception:
                pass
            raise

    def close(self) -> None:
        file = self._file
        self._file = None
        if file is None:
            return
        try:
            file.close()
        except Exception:
            logger.warning("Failed to close scheduler event file %s", self._path, exc_info=True)


def scheduler_event_sink_accepts(
    sink: SchedulerAdmissionEventSink | None,
    event_kind: SchedulerAdmissionEventKind,
) -> bool:
    """Return whether a scheduler sink wants an event, defaulting to all events."""
    if sink is None:
        return False
    accepts = getattr(sink, "accepts_scheduler_event", None)
    if accepts is None:
        return True
    try:
        return bool(accepts(event_kind))
    except Exception:
        logger.warning("Scheduler admission event interest check raised; dropping event.", exc_info=True)
        return False


class _FanoutSchedulerEventSink:
    def __init__(self, sinks: tuple[SchedulerAdmissionEventSink, ...]) -> None:
        self._sinks = sinks

    def accepts_scheduler_event(self, event_kind: SchedulerAdmissionEventKind) -> bool:
        return any(scheduler_event_sink_accepts(sink, event_kind) for sink in self._sinks)

    def emit_scheduler_event(self, event: SchedulerAdmissionEvent) -> None:
        for sink in self._sinks:
            if not scheduler_event_sink_accepts(sink, event.event_kind):
                continue
            try:
                sink.emit_scheduler_event(event)
            except Exception:
                logger.warning("Scheduler admission event sink raised; dropping event.", exc_info=True)


def fanout_scheduler_event_sinks(
    *sinks: SchedulerAdmissionEventSink | None,
) -> SchedulerAdmissionEventSink | None:
    active_sinks = tuple(sink for sink in sinks if sink is not None)
    if len(active_sinks) < 2:
        return active_sinks[0] if active_sinks else None
    return _FanoutSchedulerEventSink(active_sinks)


class RequestAdmissionEventSink(Protocol):
    def emit_request_event(self, event: RequestAdmissionEvent) -> None: ...


RequestAdmissionEventCallback = Callable[[RequestAdmissionEvent], None]

_request_event_callback_lock = Lock()
_request_event_callback_ids = itertools.count()
_request_event_callbacks: dict[int, RequestAdmissionEventCallback] = {}


def subscribe_request_admission_events(callback: RequestAdmissionEventCallback) -> Callable[[], None]:
    callback_id = next(_request_event_callback_ids)
    with _request_event_callback_lock:
        _request_event_callbacks[callback_id] = callback

    def unsubscribe() -> None:
        with _request_event_callback_lock:
            _request_event_callbacks.pop(callback_id, None)

    return unsubscribe


def emit_request_admission_event(event: RequestAdmissionEvent) -> None:
    with _request_event_callback_lock:
        callbacks = tuple(_request_event_callbacks.values())

    for callback in callbacks:
        try:
            callback(event)
        except Exception:
            logger.debug("Request admission event callback failed", exc_info=True)
