# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import Counter

from data_designer.engine.dataset_builders.scheduling.queue import FairTaskQueue, QueueView
from data_designer.engine.dataset_builders.scheduling.resources import (
    SchedulableTask,
    SchedulerResourceRequest,
    TaskGroupKey,
    TaskGroupSpec,
    stable_task_id,
)
from data_designer.engine.dataset_builders.scheduling.task_model import Task


def _task(column: str, row_index: int) -> Task:
    return Task(column=column, row_group=0, row_index=row_index, task_type="cell")


def _group(name: str, *, weight: float = 1.0, admitted_limit: int | None = None) -> TaskGroupSpec:
    return TaskGroupSpec(
        key=TaskGroupKey(kind="local", identity=(name,)),
        weight=weight,
        admitted_limit=admitted_limit,
    )


def _item(column: str, row_index: int, group: TaskGroupSpec | None = None) -> SchedulableTask:
    task = _task(column, row_index)
    group = group or _group(column)
    return SchedulableTask(
        task_id=stable_task_id(task),
        payload=task,
        group=group,
        resource_request=SchedulerResourceRequest({"submission": 1}),
    )


def _select_and_commit(queue: FairTaskQueue) -> SchedulableTask | None:
    selection = queue.select_next(lambda _item, _view: True)
    if selection is None:
        return None
    return queue.commit(selection)


def test_fair_task_queue_equal_groups_round_robins() -> None:
    queue = FairTaskQueue()
    queue.enqueue(
        [
            _item("a", 0),
            _item("a", 1),
            _item("b", 0),
            _item("b", 1),
            _item("c", 0),
            _item("c", 1),
        ]
    )

    selected = [_select_and_commit(queue) for _ in range(6)]

    assert [item.payload.column for item in selected if item is not None] == ["a", "b", "c", "a", "b", "c"]


def test_fair_task_queue_weighted_groups() -> None:
    queue = FairTaskQueue()
    queue.enqueue(
        [_item("a", i, _group("a", weight=2)) for i in range(6)]
        + [_item("b", i, _group("b", weight=1)) for i in range(6)]
    )

    selected = [_select_and_commit(queue) for _ in range(6)]
    counts = Counter(item.payload.column for item in selected if item is not None)

    assert counts == {"a": 4, "b": 2}


def test_select_next_is_non_mutating_until_commit() -> None:
    queue = FairTaskQueue()
    first = _item("a", 0)
    second = _item("b", 0)
    queue.enqueue([first, second])

    selection = queue.select_next(lambda _item, _view: True)

    assert selection is not None
    assert queue.view().queued_total == 2
    committed = queue.commit(selection)
    assert committed == first
    assert queue.view().queued_total == 1


def test_commit_rejects_stale_selection() -> None:
    queue = FairTaskQueue()
    first = _item("a", 0)
    queue.enqueue([first])

    selection = queue.select_next(lambda _item, _view: True)
    assert selection is not None
    queue.enqueue([_item("b", 0)])

    assert queue.commit(selection) is None
    assert queue.view().queued_total == 2


def test_select_next_uses_scheduler_eligibility_callback() -> None:
    queue = FairTaskQueue()
    queue.enqueue([_item("a", 0), _item("b", 0)])

    selection = queue.select_next(lambda item, _view: item.payload.column == "b")

    assert selection is not None
    assert selection.item.payload.column == "b"
    assert queue.commit(selection) == selection.item


def test_enqueue_is_idempotent_by_task_id() -> None:
    queue = FairTaskQueue()
    item = _item("a", 0)

    first = queue.enqueue([item])
    second = queue.enqueue([item])

    assert first == (item.task_id,)
    assert second == ()
    assert queue.view().queued_total == 1


def test_discard_where_removes_matching_tasks() -> None:
    queue = FairTaskQueue()
    queue.enqueue([_item(column, i) for column in ["a", "b"] for i in range(2)])

    queue.discard_where(lambda item: item.payload.column == "a")
    selected = [_select_and_commit(queue) for _ in range(2)]

    assert [item.payload.column for item in selected if item is not None] == ["b", "b"]
    assert _select_and_commit(queue) is None


def test_queue_view_exposes_group_and_resource_demand() -> None:
    queue = FairTaskQueue()
    group = _group("a")
    task = _task("a", 0)
    item = SchedulableTask(
        task_id=stable_task_id(task),
        payload=task,
        group=group,
        resource_request=SchedulerResourceRequest({"submission": 1, "llm_wait": 1}),
    )

    queue.enqueue([item])
    view: QueueView = queue.view()

    assert view.queued_total == 1
    assert view.queued_by_group[group.key] == 1
    assert view.queued_resource_demand_by_group[group.key]["llm_wait"] == 1
    assert view.first_candidate_resources_by_group[group.key]["submission"] == 1
