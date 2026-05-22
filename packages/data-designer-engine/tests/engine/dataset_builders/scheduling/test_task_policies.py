# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.engine.dataset_builders.scheduling.queue import FairTaskQueue, QueueView
from data_designer.engine.dataset_builders.scheduling.resources import (
    SchedulableTask,
    SchedulerResourceRequest,
    TaskGroupKey,
    TaskGroupSpec,
    stable_task_id,
)
from data_designer.engine.dataset_builders.scheduling.task_admission import TaskAdmissionLease, TaskAdmissionView
from data_designer.engine.dataset_builders.scheduling.task_model import Task
from data_designer.engine.dataset_builders.scheduling.task_policies import (
    BoundedBorrowTaskAdmissionPolicy,
    BoundedBorrowTaskAdmissionPolicyConfig,
    StrictFairTaskAdmissionPolicy,
)


def _item(column: str, group: TaskGroupSpec) -> SchedulableTask:
    task = Task(column=column, row_group=0, row_index=0, task_type="cell")
    return SchedulableTask(
        task_id=stable_task_id(task),
        payload=task,
        group=group,
        resource_request=SchedulerResourceRequest({"submission": 1}),
    )


def _queue_view(*items: SchedulableTask) -> QueueView:
    queue = FairTaskQueue()
    queue.enqueue(items)
    return queue.view()


def _admission_view(
    *,
    running_group: TaskGroupKey,
    running_count: int = 1,
    debt: int = 0,
) -> TaskAdmissionView:
    return TaskAdmissionView(
        resource_limits={"submission": 4},
        resources_available={"submission": 3},
        leased_resources={"submission": running_count},
        leased_resources_by_group={running_group: {"submission": running_count}},
        running_counts_by_group={running_group: running_count},
        policy_debt_by_group_resource={(running_group, "submission"): debt} if debt else {},
    )


def _lease(item: SchedulableTask) -> TaskAdmissionLease:
    return TaskAdmissionLease(
        lease_id="lease",
        item=item,
        resources={"submission": 1},
        acquired_at=0.0,
        controller_generation="generation",
    )


def test_strict_fair_policy_allows_group_without_peer_pressure() -> None:
    group = TaskGroupSpec(TaskGroupKey(kind="model", identity=("provider", "model")), admitted_limit=1)
    item = _item("a", group)
    policy = StrictFairTaskAdmissionPolicy()

    decision = policy.evaluate(item, _queue_view(item), _admission_view(running_group=group.key))

    assert decision.allowed is True


def test_strict_fair_policy_denies_capped_group_with_peer_pressure() -> None:
    group = TaskGroupSpec(TaskGroupKey(kind="model", identity=("provider", "model")), admitted_limit=1)
    peer_group = TaskGroupSpec(TaskGroupKey(kind="local", identity=("peer",)))
    item = _item("a", group)
    peer = _item("b", peer_group)
    policy = StrictFairTaskAdmissionPolicy()

    decision = policy.evaluate(item, _queue_view(item, peer), _admission_view(running_group=group.key))

    assert decision.allowed is False
    assert decision.reason == "group_cap"


def test_bounded_borrow_policy_records_borrow_without_peer_pressure() -> None:
    group = TaskGroupSpec(TaskGroupKey(kind="model", identity=("provider", "model")), admitted_limit=1)
    item = _item("a", group)
    policy = BoundedBorrowTaskAdmissionPolicy(
        BoundedBorrowTaskAdmissionPolicyConfig(
            default_borrow_ceiling=1,
            strict_share_rounding="floor",
        )
    )

    decision = policy.evaluate(item, _queue_view(item), _admission_view(running_group=group.key))
    delta = policy.on_acquire(_lease(item), decision)

    assert decision.allowed is True
    assert delta.debt_changes == {(group.key, "submission"): 1}


def test_bounded_borrow_policy_defaults_to_ceil_strict_share_rounding() -> None:
    config = BoundedBorrowTaskAdmissionPolicyConfig()

    assert config.strict_share_rounding == "ceil"
    assert config.default_borrow_ceiling is None
    assert config.dynamic_borrow_reserve_fraction == 0.125
    assert config.dynamic_borrow_max_reserved_slots == 8


def test_bounded_borrow_policy_zero_resource_limit_has_no_strict_share() -> None:
    group = TaskGroupSpec(TaskGroupKey(kind="model", identity=("provider", "model")), admitted_limit=1)
    item = _item("a", group)
    policy = BoundedBorrowTaskAdmissionPolicy(BoundedBorrowTaskAdmissionPolicyConfig())
    view = TaskAdmissionView(
        resource_limits={"submission": 0},
        resources_available={"submission": 0},
        leased_resources={},
        leased_resources_by_group={},
        running_counts_by_group={},
        policy_debt_by_group_resource={},
    )

    decision = policy.evaluate(item, _queue_view(item), view)

    assert decision.allowed is False
    assert decision.reason == "borrow_debt"
    assert decision.diagnostics["strict_share"] == 0


def test_bounded_borrow_policy_denies_existing_debt_under_peer_pressure() -> None:
    group = TaskGroupSpec(TaskGroupKey(kind="model", identity=("provider", "model")), admitted_limit=1)
    peer_group = TaskGroupSpec(TaskGroupKey(kind="local", identity=("peer",)))
    item = _item("a", group)
    peer = _item("b", peer_group)
    policy = BoundedBorrowTaskAdmissionPolicy(BoundedBorrowTaskAdmissionPolicyConfig(default_borrow_ceiling=1))

    decision = policy.evaluate(item, _queue_view(item, peer), _admission_view(running_group=group.key, debt=1))

    assert decision.allowed is False
    assert decision.reason == "borrow_debt"


def test_bounded_borrow_policy_releases_debt() -> None:
    group = TaskGroupSpec(TaskGroupKey(kind="model", identity=("provider", "model")), admitted_limit=1)
    item = _item("a", group)
    policy = BoundedBorrowTaskAdmissionPolicy(BoundedBorrowTaskAdmissionPolicyConfig(default_borrow_ceiling=1))

    delta = policy.on_release(_lease(item))

    assert delta.debt_changes == {(group.key, "submission"): -1}
