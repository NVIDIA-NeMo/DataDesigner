# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.engine.dataset_builders.utils.fair_task_queue import FairTaskQueue
from data_designer.engine.dataset_builders.utils.task_admission import (
    BoundedBorrowTaskAdmissionPolicyConfig,
    TaskAdmissionConfig,
    TaskAdmissionController,
    TaskAdmissionDenied,
    TaskAdmissionLease,
)
from data_designer.engine.dataset_builders.utils.task_model import Task
from data_designer.engine.dataset_builders.utils.task_scheduling import (
    SchedulableTask,
    SchedulerResourceRequest,
    TaskGroupKey,
    TaskGroupSpec,
    stable_task_id,
)


def _item(column: str, row: int = 0, *, group: TaskGroupSpec | None = None) -> SchedulableTask:
    task = Task(column=column, row_group=0, row_index=row, task_type="cell")
    group = group or TaskGroupSpec(TaskGroupKey(kind="local", identity=(column,)))
    return SchedulableTask(
        task_id=stable_task_id(task),
        payload=task,
        group=group,
        resource_request=SchedulerResourceRequest({"submission": 1}),
    )


def _queue_view(*items: SchedulableTask):
    queue = FairTaskQueue()
    queue.enqueue(items)
    return queue.view()


def test_task_admission_acquires_and_releases_exact_lease() -> None:
    controller = TaskAdmissionController(TaskAdmissionConfig(submission_capacity=1))
    item = _item("a")

    decision = controller.try_acquire(item, _queue_view(item))

    assert isinstance(decision, TaskAdmissionLease)
    assert controller.view().resources_available["submission"] == 0
    result = controller.release(decision)
    assert result.released is True
    assert controller.view().resources_available["submission"] == 1


def test_task_admission_denies_when_resource_full() -> None:
    controller = TaskAdmissionController(TaskAdmissionConfig(submission_capacity=1))
    first = _item("a")
    second = _item("b")
    lease = controller.try_acquire(first, _queue_view(first, second))

    assert isinstance(lease, TaskAdmissionLease)
    decision = controller.try_acquire(second, _queue_view(second))

    assert isinstance(decision, TaskAdmissionDenied)
    assert decision.reason == "no_capacity"


def test_task_admission_duplicate_release_does_not_increase_capacity() -> None:
    controller = TaskAdmissionController(TaskAdmissionConfig(submission_capacity=1))
    item = _item("a")
    lease = controller.try_acquire(item, _queue_view(item))
    assert isinstance(lease, TaskAdmissionLease)

    first = controller.release(lease)
    second = controller.release(lease)

    assert first.released is True
    assert second.released is False
    assert second.reason == "duplicate"
    assert controller.view().resources_available["submission"] == 1


def test_task_admission_group_cap_yields_to_peer_pressure() -> None:
    group = TaskGroupSpec(TaskGroupKey(kind="model", identity=("provider", "model")), admitted_limit=1)
    controller = TaskAdmissionController(TaskAdmissionConfig(submission_capacity=2))
    first = _item("a", 0, group=group)
    second = _item("a", 1, group=group)
    peer = _item("b")
    lease = controller.try_acquire(first, _queue_view(first, second, peer))
    assert isinstance(lease, TaskAdmissionLease)

    decision = controller.try_acquire(second, _queue_view(second, peer))

    assert isinstance(decision, TaskAdmissionDenied)
    assert decision.reason == "group_cap"


def test_explain_blocked_reports_group_cap_denials() -> None:
    first_group = TaskGroupSpec(TaskGroupKey(kind="model", identity=("provider", "first")), admitted_limit=1)
    second_group = TaskGroupSpec(TaskGroupKey(kind="model", identity=("provider", "second")), admitted_limit=1)
    controller = TaskAdmissionController(TaskAdmissionConfig(submission_capacity=4))
    first_active = _item("a", 0, group=first_group)
    second_active = _item("b", 0, group=second_group)
    first_queued = _item("a", 1, group=first_group)
    second_queued = _item("b", 1, group=second_group)
    first_lease = controller.try_acquire(first_active, _queue_view(first_active, second_active))
    second_lease = controller.try_acquire(second_active, _queue_view(second_active, first_queued))
    assert isinstance(first_lease, TaskAdmissionLease)
    assert isinstance(second_lease, TaskAdmissionLease)
    queue = FairTaskQueue()
    queue.enqueue((first_queued, second_queued))

    assert queue.select_next(controller.is_eligible) is None
    summary = controller.explain_blocked(queue.view())

    assert summary.dominant_denial_reasons == {"group_cap": 2}


def test_task_admission_group_cap_does_not_block_solo_group() -> None:
    group = TaskGroupSpec(TaskGroupKey(kind="model", identity=("provider", "model")), admitted_limit=1)
    controller = TaskAdmissionController(TaskAdmissionConfig(submission_capacity=2))
    first = _item("a", 0, group=group)
    second = _item("a", 1, group=group)
    lease = controller.try_acquire(first, _queue_view(first, second))
    assert isinstance(lease, TaskAdmissionLease)

    decision = controller.try_acquire(second, _queue_view(second))

    assert isinstance(decision, TaskAdmissionLease)


def test_bounded_borrow_limits_solo_group_borrow_debt() -> None:
    group = TaskGroupSpec(TaskGroupKey(kind="model", identity=("provider", "model")), admitted_limit=1)
    controller = TaskAdmissionController(
        TaskAdmissionConfig(
            submission_capacity=3,
            bounded_borrow=BoundedBorrowTaskAdmissionPolicyConfig(default_borrow_ceiling=1),
        )
    )
    first = _item("a", 0, group=group)
    second = _item("a", 1, group=group)
    third = _item("a", 2, group=group)
    first_lease = controller.try_acquire(first, _queue_view(first, second, third))
    assert isinstance(first_lease, TaskAdmissionLease)
    borrowed = controller.try_acquire(second, _queue_view(second, third))
    assert isinstance(borrowed, TaskAdmissionLease)

    denied = controller.try_acquire(third, _queue_view(third))

    assert isinstance(denied, TaskAdmissionDenied)
    assert denied.reason == "borrow_debt"
    assert controller.view().policy_debt_by_group_resource[(group.key, "submission")] == 1


def test_bounded_borrow_debt_blocks_under_peer_pressure_and_releases() -> None:
    group = TaskGroupSpec(TaskGroupKey(kind="model", identity=("provider", "model")), admitted_limit=1)
    controller = TaskAdmissionController(
        TaskAdmissionConfig(
            submission_capacity=3,
            bounded_borrow=BoundedBorrowTaskAdmissionPolicyConfig(default_borrow_ceiling=1),
        )
    )
    first = _item("a", 0, group=group)
    borrowed_item = _item("a", 1, group=group)
    blocked_item = _item("a", 2, group=group)
    peer = _item("b")
    first_lease = controller.try_acquire(first, _queue_view(first, borrowed_item))
    borrowed = controller.try_acquire(borrowed_item, _queue_view(borrowed_item))
    assert isinstance(first_lease, TaskAdmissionLease)
    assert isinstance(borrowed, TaskAdmissionLease)

    denied = controller.try_acquire(blocked_item, _queue_view(blocked_item, peer))

    assert isinstance(denied, TaskAdmissionDenied)
    assert denied.reason == "borrow_debt"
    controller.release(borrowed)
    assert (group.key, "submission") not in controller.view().policy_debt_by_group_resource
