# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from data_designer.engine.dataset_builders.scheduling.task_model import Task
from data_designer.engine.models.request_admission.resources import RequestResourceKey

SchedulerResourceKey = Literal["submission", "llm_wait", "local"]


@dataclass(frozen=True, order=True)
class TaskGroupKey:
    """Stable identity for a stream of related scheduler tasks."""

    kind: Literal["model", "custom_model", "local"]
    identity: tuple[str, ...]


@dataclass(frozen=True)
class TaskGroupSpec:
    """Scheduler-internal task group metadata."""

    key: TaskGroupKey
    weight: float = 1.0
    admitted_limit: int | None = None


@dataclass(frozen=True)
class SchedulerResourceRequest:
    """Scheduler task-stage resource request."""

    amounts: Mapping[SchedulerResourceKey, int] = field(default_factory=lambda: {"submission": 1})

    def __post_init__(self) -> None:
        for resource, amount in self.amounts.items():
            if resource not in {"submission", "llm_wait", "local"}:
                raise ValueError(f"Unknown scheduler resource key: {resource!r}")
            if not isinstance(amount, int) or amount <= 0:
                raise ValueError(f"Scheduler resource amount for {resource!r} must be a positive integer.")


@dataclass(frozen=True)
class SchedulableTask:
    """Ready task plus scheduler-owned grouping and resource request."""

    task_id: str
    payload: Task
    group: TaskGroupSpec
    resource_request: SchedulerResourceRequest
    request_resource_key: RequestResourceKey | None = None


def stable_task_id(task: Task) -> str:
    """Return a stable scheduler task id for queue/admission membership."""
    raw = f"{task.column}\0{task.row_group}\0{task.row_index}\0{task.task_type}".encode()
    digest = hashlib.sha1(raw).hexdigest()[:16]
    return f"task-{digest}"
