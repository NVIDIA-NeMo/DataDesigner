# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from unittest.mock import Mock

import pytest

from data_designer.engine.observability import (
    JsonlSchedulerEventSink,
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


def test_jsonl_scheduler_event_sink_flushes_closes_and_appends(tmp_path: Path) -> None:
    path = tmp_path / "scheduler_events.jsonl"
    events = [
        SchedulerAdmissionEvent.capture("selected", sequence=1, diagnostics={"label": "café"}),
        SchedulerAdmissionEvent.capture("task_completed", sequence=2),
    ]

    for event in events:
        with JsonlSchedulerEventSink(path) as sink:
            assert sink is not None
            sink.emit_scheduler_event(event)
            contents = path.read_text(encoding="utf-8")
            assert json.loads(contents.splitlines()[-1]) == asdict(event)

    assert "café" in contents
    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == [
        asdict(event) for event in events
    ]


def test_jsonl_scheduler_event_sink_open_failure_is_inactive(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("builtins.open", Mock(side_effect=OSError("open failed")))

    with caplog.at_level(logging.WARNING):
        with JsonlSchedulerEventSink("events.jsonl") as sink:
            assert sink is None

    assert [record.message for record in caplog.records] == ["Failed to open scheduler event file events.jsonl"]


def test_jsonl_scheduler_event_sink_close_failure_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    file = Mock()
    file.close.side_effect = OSError("close failed")
    monkeypatch.setattr("builtins.open", Mock(return_value=file))

    with caplog.at_level(logging.WARNING):
        with JsonlSchedulerEventSink("events.jsonl") as sink:
            assert sink is not None
        sink.close()

    file.close.assert_called_once_with()
    assert [record.message for record in caplog.records] == ["Failed to close scheduler event file events.jsonl"]


def test_jsonl_scheduler_event_sink_disables_after_write_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    file = Mock()
    file.write.side_effect = OSError("write failed")
    monkeypatch.setattr("builtins.open", Mock(return_value=file))
    event = SchedulerAdmissionEvent.capture("selected", sequence=1)

    with JsonlSchedulerEventSink("events.jsonl") as sink:
        assert sink is not None
        with pytest.raises(OSError, match="write failed"):
            sink.emit_scheduler_event(event)
        sink.emit_scheduler_event(event)

    file.write.assert_called_once()
    file.close.assert_called_once_with()


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
