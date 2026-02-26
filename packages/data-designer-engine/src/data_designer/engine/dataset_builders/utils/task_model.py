# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class Task:
    """A unit of work for the async scheduler."""

    column: str
    row_group: int
    row_index: int | None  # None for batch/full-column tasks
    task_type: Literal["from_scratch", "cell", "batch", "pre_batch_processor", "post_batch_processor"]


@dataclass
class TaskResult:
    """Outcome of a completed task."""

    task: Task
    status: Literal["success", "error"]
    output: Any = None
    error: Exception | None = None
    retryable: bool = False


@dataclass
class TaskTrace:
    """Timing trace for a single task. Only created when tracing is enabled."""

    column: str
    row_group: int
    row_index: int | None
    task_type: str
    dispatched_at: float = 0.0
    slot_acquired_at: float = 0.0
    completed_at: float = 0.0
    status: str = ""
    error: str | None = None

    @staticmethod
    def from_task(task: Task) -> TaskTrace:
        return TaskTrace(
            column=task.column,
            row_group=task.row_group,
            row_index=task.row_index,
            task_type=task.task_type,
        )
