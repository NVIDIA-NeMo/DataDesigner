# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from data_designer.engine.dataset_builders.utils.task_model import Task, TaskResult, TaskTrace


def test_task_is_frozen() -> None:
    task = Task(column="col_a", row_group=0, row_index=1, task_type="cell")
    with pytest.raises(AttributeError):
        task.column = "col_b"  # type: ignore[misc]


def test_task_hashable_and_in_set() -> None:
    t1 = Task(column="col_a", row_group=0, row_index=1, task_type="cell")
    t2 = Task(column="col_a", row_group=0, row_index=1, task_type="cell")
    t3 = Task(column="col_a", row_group=0, row_index=2, task_type="cell")

    assert t1 == t2
    assert t1 != t3
    assert hash(t1) == hash(t2)

    s: set[Task] = {t1, t2, t3}
    assert len(s) == 2


def test_task_batch_has_none_row_index() -> None:
    task = Task(column="col_a", row_group=0, row_index=None, task_type="batch")
    assert task.row_index is None


@pytest.mark.parametrize(
    "task_type",
    ["from_scratch", "cell", "batch", "pre_batch_processor", "post_batch_processor"],
)
def test_task_types(task_type: str) -> None:
    task = Task(column="col", row_group=0, row_index=0, task_type=task_type)
    assert task.task_type == task_type


def test_task_result_success() -> None:
    task = Task(column="col_a", row_group=0, row_index=0, task_type="cell")
    result = TaskResult(task=task, status="success", output={"col_a": "value"})
    assert result.status == "success"
    assert result.error is None
    assert result.retryable is False


def test_task_result_error() -> None:
    task = Task(column="col_a", row_group=0, row_index=0, task_type="cell")
    exc = ValueError("bad input")
    result = TaskResult(task=task, status="error", error=exc, retryable=True)
    assert result.status == "error"
    assert result.error is exc
    assert result.retryable is True


def test_task_trace_from_task() -> None:
    task = Task(column="col_a", row_group=1, row_index=2, task_type="cell")
    trace = TaskTrace.from_task(task)

    assert trace.column == "col_a"
    assert trace.row_group == 1
    assert trace.row_index == 2
    assert trace.task_type == "cell"
    assert trace.dispatched_at == 0.0
    assert trace.status == ""


def test_task_trace_mutable() -> None:
    task = Task(column="col_a", row_group=0, row_index=None, task_type="batch")
    trace = TaskTrace.from_task(task)

    trace.dispatched_at = 1.0
    trace.slot_acquired_at = 1.5
    trace.completed_at = 2.0
    trace.status = "ok"

    assert trace.dispatched_at == 1.0
    assert trace.completed_at - trace.slot_acquired_at == pytest.approx(0.5)


def test_task_equality_differs_by_type() -> None:
    t1 = Task(column="col_a", row_group=0, row_index=None, task_type="batch")
    t2 = Task(column="col_a", row_group=0, row_index=None, task_type="from_scratch")
    assert t1 != t2
