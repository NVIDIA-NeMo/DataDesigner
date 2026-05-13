# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import Counter

from data_designer.engine.dataset_builders.utils.fair_task_queue import (
    FairTaskQueue,
    TaskGroupKey,
    TaskGroupSpec,
)
from data_designer.engine.dataset_builders.utils.task_model import Task


def _task(column: str, row_index: int) -> Task:
    return Task(column=column, row_group=0, row_index=row_index, task_type="cell")


def _group(name: str, *, weight: float = 1.0, admitted_limit: int | None = None) -> TaskGroupSpec:
    return TaskGroupSpec(
        key=TaskGroupKey(kind="local", identity=(name,)),
        weight=weight,
        admitted_limit=admitted_limit,
    )


def _enqueue(queue: FairTaskQueue, items: list[tuple[Task, TaskGroupSpec]]) -> None:
    for task, group in items:
        queue.enqueue(task, group)


def test_fair_task_queue_equal_groups_round_robins() -> None:
    queue = FairTaskQueue()
    _enqueue(
        queue,
        [
            (task, _group(task.column))
            for task in [
                _task("a", 0),
                _task("a", 1),
                _task("b", 0),
                _task("b", 1),
                _task("c", 0),
                _task("c", 1),
            ]
        ],
    )

    selected = [queue.admit_next() for _ in range(6)]

    assert [selection.task.column for selection in selected if selection is not None] == ["a", "b", "c", "a", "b", "c"]


def test_fair_task_queue_weighted_groups() -> None:
    queue = FairTaskQueue()
    _enqueue(
        queue,
        [
            (task, _group(task.column, weight=2 if task.column == "a" else 1))
            for task in [_task("a", i) for i in range(6)]
        ]
        + [(_task("b", i), _group("b", weight=1)) for i in range(6)],
    )

    selected = [queue.admit_next() for _ in range(6)]
    counts = Counter(selection.task.column for selection in selected if selection is not None)

    assert counts == {"a": 4, "b": 2}


def test_fair_task_queue_discards_queued_tasks() -> None:
    queue = FairTaskQueue()
    stale = _task("a", 0)
    fresh = _task("a", 1)

    _enqueue(queue, [(stale, _group("a")), (fresh, _group("a"))])
    queue.discard(stale)

    selected = queue.admit_next()

    assert selected is not None
    assert selected.task == fresh
    assert queue.admit_next() is None


def test_fair_task_queue_admitted_cap_skips_saturated_group() -> None:
    queue = FairTaskQueue()
    capped = _group("a", admitted_limit=1)
    open_group = _group("b", admitted_limit=1)
    _enqueue(queue, [(_task("a", 0), capped), (_task("a", 1), capped), (_task("b", 0), open_group)])

    first = queue.admit_next()
    selected = queue.admit_next()

    assert first is not None
    assert first.task.column == "a"
    assert selected is not None
    assert selected.task.column == "b"


def test_fair_task_queue_returns_none_when_all_groups_capped() -> None:
    queue = FairTaskQueue()
    group = _group("a", admitted_limit=1)
    first_task = _task("a", 0)
    second_task = _task("a", 1)
    queue.enqueue(first_task, group)
    queue.enqueue(second_task, group)

    first = queue.admit_next()

    assert first is not None
    assert first.task == first_task
    assert queue.admit_next() is None
    assert queue.has_queued_tasks is True


def test_fair_task_queue_release_reopens_saturated_group() -> None:
    queue = FairTaskQueue()
    group = _group("a", admitted_limit=1)
    first_task = _task("a", 0)
    second_task = _task("a", 1)
    queue.enqueue(first_task, group)
    queue.enqueue(second_task, group)
    first = queue.admit_next()

    assert first is not None
    assert queue.admit_next() is None

    queue.release(first.task)
    second = queue.admit_next()

    assert second is not None
    assert second.task == second_task


def test_fair_task_queue_no_duplicate_on_repeated_enqueue() -> None:
    queue = FairTaskQueue()
    task = _task("a", 0)

    queue.enqueue(task, _group("a"))
    queue.enqueue(task, _group("a"))
    first = queue.admit_next()

    assert first is not None
    assert first.task == task
    assert queue.admit_next() is None


def test_fair_task_queue_discard_where_removes_matching_tasks() -> None:
    queue = FairTaskQueue()
    _enqueue(
        queue,
        [(_task(column, i), _group(column)) for column in ["a", "b"] for i in range(2)],
    )

    queue.discard_where(lambda task: task.column == "a")
    selected = [queue.admit_next() for _ in range(2)]

    assert [selection.task.column for selection in selected if selection is not None] == ["b", "b"]
    assert queue.admit_next() is None
