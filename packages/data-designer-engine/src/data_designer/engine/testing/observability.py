# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from data_designer.engine.observability import RequestAdmissionEvent, SchedulerAdmissionEvent


class InMemoryAdmissionEventSink:
    """In-memory admission-event sink for tests and benchmark smoke runs."""

    def __init__(self) -> None:
        self.scheduler_events: list[SchedulerAdmissionEvent] = []
        self.request_events: list[RequestAdmissionEvent] = []

    def emit_scheduler_event(self, event: SchedulerAdmissionEvent) -> None:
        self.scheduler_events.append(event)

    def emit_request_event(self, event: RequestAdmissionEvent) -> None:
        self.request_events.append(event)


@dataclass(frozen=True)
class CorrelatedRuntimeView:
    """Combined chronological view of scheduler and request events for tests."""

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
