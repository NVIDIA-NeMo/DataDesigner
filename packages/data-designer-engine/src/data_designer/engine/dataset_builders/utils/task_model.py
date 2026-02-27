# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

ColumnName: TypeAlias = str
RowGroup: TypeAlias = int
RowIndex: TypeAlias = int


@dataclass(frozen=True)
class Task:
    """A unit of work for the async scheduler."""

    column: ColumnName
    row_group: RowGroup
    row_index: RowIndex | None  # None for batch/full-column tasks
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

    column: ColumnName
    row_group: RowGroup
    row_index: RowIndex | None
    task_type: str
    dispatched_at: float = 0.0
    slot_acquired_at: float = 0.0
    completed_at: float = 0.0
    status: str = ""
    error: str | None = None

    @classmethod
    def from_task(cls, task: Task) -> TaskTrace:
        return cls(
            column=task.column,
            row_group=task.row_group,
            row_index=task.row_index,
            task_type=task.task_type,
        )
