# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum

from data_designer.engine.observability import (
    RequestAdmissionEvent,
    RuntimeCorrelation,
    RuntimeCorrelationProvider,
    SchedulerAdmissionEvent,
)
from data_designer.engine.testing import CorrelatedRuntimeView, InMemoryAdmissionEventSink


class _DiagnosticMode(Enum):
    TEST = "test"


@dataclass(frozen=True)
class _DiagnosticPayload:
    label: str
    mode: _DiagnosticMode


def _correlation() -> RuntimeCorrelation:
    return RuntimeCorrelation(
        run_id="run",
        row_group=1,
        task_column="answer",
        task_type="cell",
        scheduling_group_kind="model",
        scheduling_group_identity_hash="hash",
        task_execution_id="task-exec",
    )


def test_runtime_correlation_provider_sets_and_resets_context() -> None:
    provider = RuntimeCorrelationProvider()
    correlation = _correlation()

    token = provider.set(correlation)
    assert provider.current() == correlation

    provider.reset(token)
    assert provider.current() is None


def test_admission_events_capture_correlation_and_diagnostics() -> None:
    correlation = _correlation()

    scheduler_event = SchedulerAdmissionEvent.capture(
        "task_lease_acquired",
        sequence=1,
        correlation=correlation,
        task_id="task-1",
        task_lease_id="lease-1",
        diagnostics={"resource": "submission"},
    )
    request_event = RequestAdmissionEvent.capture(
        "request_lease_acquired",
        sequence=2,
        correlation=correlation,
        request_attempt_id="request-1",
        request_lease_id="lease-2",
        diagnostics={"resource": "chat"},
    )

    assert scheduler_event.captured_correlation == asdict(correlation)
    assert scheduler_event.task_id == "task-1"
    assert scheduler_event.diagnostics == {"resource": "submission"}
    assert request_event.captured_correlation == asdict(correlation)
    assert request_event.request_attempt_id == "request-1"
    assert request_event.diagnostics == {"resource": "chat"}


def test_admission_events_are_json_safe_at_construction() -> None:
    correlation = _correlation()
    payload = _DiagnosticPayload(label="payload", mode=_DiagnosticMode.TEST)

    scheduler_event = SchedulerAdmissionEvent.capture(
        "admission_blocked",
        sequence=1,
        correlation=correlation,
        snapshot=payload,
        diagnostics={"payload": payload, "values": {"b", "a"}, "pair": ("x", _DiagnosticMode.TEST)},
    )
    request_event = RequestAdmissionEvent.capture(
        "request_wait_started",
        sequence=2,
        correlation=correlation,
        request_resource_key=payload,
        request_group_key=("group", _DiagnosticMode.TEST),
        pressure_snapshot={"payload": payload},
        diagnostics={"payload": payload},
    )

    json.dumps(asdict(scheduler_event), sort_keys=True)
    json.dumps(asdict(request_event), sort_keys=True)
    assert scheduler_event.snapshot == {"label": "payload", "mode": "test"}
    assert scheduler_event.diagnostics["values"] == ["a", "b"]
    assert request_event.request_resource_key == {"label": "payload", "mode": "test"}


def test_in_memory_admission_event_sink_collects_scheduler_and_request_events() -> None:
    sink = InMemoryAdmissionEventSink()
    scheduler_event = SchedulerAdmissionEvent.capture("selected", sequence=1)
    request_event = RequestAdmissionEvent.capture("request_wait_started", sequence=2)

    sink.emit_scheduler_event(scheduler_event)
    sink.emit_request_event(request_event)

    assert sink.scheduler_events == [scheduler_event]
    assert sink.request_events == [request_event]


def test_correlated_runtime_view_timeline_sorts_events() -> None:
    scheduler_event = SchedulerAdmissionEvent(event_kind="selected", captured_at_monotonic=2.0, sequence=1)
    first_request_event = RequestAdmissionEvent(
        event_kind="request_wait_started",
        captured_at_monotonic=1.0,
        sequence=3,
    )
    second_request_event = RequestAdmissionEvent(
        event_kind="request_lease_acquired",
        captured_at_monotonic=2.0,
        sequence=0,
    )
    view = CorrelatedRuntimeView(
        scheduler_events=(scheduler_event,),
        request_events=(first_request_event, second_request_event),
    )

    assert view.timeline == (first_request_event, second_request_event, scheduler_event)
