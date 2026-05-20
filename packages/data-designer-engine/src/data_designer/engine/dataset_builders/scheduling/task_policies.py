# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

from data_designer.engine.dataset_builders.scheduling.queue import QueueView
from data_designer.engine.dataset_builders.scheduling.resources import (
    SchedulableTask,
    SchedulerResourceKey,
    TaskGroupKey,
)

if TYPE_CHECKING:
    from data_designer.engine.dataset_builders.scheduling.task_admission import (
        TaskAdmissionLease,
        TaskAdmissionView,
    )

TaskAdmissionDenyReason = Literal[
    "no_capacity",
    "group_cap",
    "borrow_debt",
    "shutdown",
    "policy_denial",
]


@dataclass(frozen=True)
class BoundedBorrowTaskAdmissionPolicyConfig:
    """Engine-internal bounded-borrow policy configuration.

    Borrow debt is tracked by task group and scheduler resource. Any completed
    lease in the same group repays debt for the released resources; repayment is
    not tied to the specific lease that originally borrowed.
    """

    borrow_ceiling_by_group_resource: Mapping[tuple[TaskGroupKey, SchedulerResourceKey], int] = field(
        default_factory=dict
    )
    default_borrow_ceiling: int = 0
    strict_share_rounding: Literal["floor", "ceil"] = "floor"
    repay_on_withheld_peer_pressure: bool = True


@dataclass(frozen=True)
class TaskAdmissionPolicyDecision:
    allowed: bool
    reason: TaskAdmissionDenyReason | None = None
    available_after: float | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyStateDelta:
    debt_changes: Mapping[tuple[TaskGroupKey, SchedulerResourceKey], int] = field(default_factory=dict)
    diagnostic_counters: Mapping[str, int] = field(default_factory=dict)


class TaskAdmissionPolicy(Protocol):
    def evaluate(
        self,
        item: SchedulableTask,
        queue_view: QueueView,
        admission_view: TaskAdmissionView,
    ) -> TaskAdmissionPolicyDecision: ...

    def on_acquire(
        self,
        lease: TaskAdmissionLease,
        decision: TaskAdmissionPolicyDecision,
    ) -> PolicyStateDelta: ...

    def on_release(self, lease: TaskAdmissionLease) -> PolicyStateDelta: ...


class StrictFairTaskAdmissionPolicy:
    """Behavior-preserving policy that enforces per-group admitted caps."""

    def evaluate(
        self,
        item: SchedulableTask,
        queue_view: QueueView,
        admission_view: TaskAdmissionView,
    ) -> TaskAdmissionPolicyDecision:
        if item.group.admitted_limit is None:
            return TaskAdmissionPolicyDecision(allowed=True)
        leased_count = admission_view.running_counts_by_group.get(item.group.key, 0)
        if leased_count < item.group.admitted_limit:
            return TaskAdmissionPolicyDecision(allowed=True)
        pressure_resources = _queued_peer_pressure_resources(item, queue_view, admission_view)
        if not pressure_resources:
            return TaskAdmissionPolicyDecision(allowed=True)
        return TaskAdmissionPolicyDecision(
            allowed=False,
            reason="group_cap",
            diagnostics={
                "admitted_limit": item.group.admitted_limit,
                "leased_count": leased_count,
                "pressure_resources": pressure_resources,
            },
        )

    def on_acquire(
        self,
        lease: TaskAdmissionLease,
        decision: TaskAdmissionPolicyDecision,
    ) -> PolicyStateDelta:
        return PolicyStateDelta()

    def on_release(self, lease: TaskAdmissionLease) -> PolicyStateDelta:
        return PolicyStateDelta()


class BoundedBorrowTaskAdmissionPolicy(StrictFairTaskAdmissionPolicy):
    """Strict policy with optional bounded borrow debt over peer pressure."""

    def __init__(self, config: BoundedBorrowTaskAdmissionPolicyConfig) -> None:
        self._config = config

    def evaluate(
        self,
        item: SchedulableTask,
        queue_view: QueueView,
        admission_view: TaskAdmissionView,
    ) -> TaskAdmissionPolicyDecision:
        limit = item.group.admitted_limit
        if limit is None:
            return TaskAdmissionPolicyDecision(allowed=True)

        leased_count = admission_view.running_counts_by_group.get(item.group.key, 0)
        if leased_count < limit:
            return TaskAdmissionPolicyDecision(allowed=True)

        pressure_resources = _queued_peer_pressure_resources(item, queue_view, admission_view)
        if pressure_resources:
            for resource in pressure_resources:
                debt_key = (item.group.key, resource)
                debt = admission_view.policy_debt_by_group_resource.get(debt_key, 0)
                if debt > 0:
                    return TaskAdmissionPolicyDecision(
                        allowed=False,
                        reason="borrow_debt",
                        diagnostics={"resource": resource, "debt": debt},
                    )
            return TaskAdmissionPolicyDecision(
                allowed=False,
                reason="group_cap",
                diagnostics={
                    "admitted_limit": limit,
                    "leased_count": leased_count,
                    "pressure_resources": pressure_resources,
                },
            )

        borrow_resources: list[tuple[SchedulerResourceKey, int]] = []
        for resource, amount in item.resource_request.amounts.items():
            debt_key = (item.group.key, resource)
            debt = admission_view.policy_debt_by_group_resource.get(debt_key, 0)
            ceiling = self._config.borrow_ceiling_by_group_resource.get(
                debt_key,
                self._config.default_borrow_ceiling,
            )
            if debt + amount > ceiling:
                return TaskAdmissionPolicyDecision(
                    allowed=False,
                    reason="borrow_debt",
                    diagnostics={"resource": resource, "debt": debt, "requested": amount, "ceiling": ceiling},
                )
            borrow_resources.append((resource, amount))
        return TaskAdmissionPolicyDecision(allowed=True, diagnostics={"borrow_resources": tuple(borrow_resources)})

    def on_acquire(
        self,
        lease: TaskAdmissionLease,
        decision: TaskAdmissionPolicyDecision,
    ) -> PolicyStateDelta:
        borrow_resources = decision.diagnostics.get("borrow_resources")
        if borrow_resources:
            changes = {
                (lease.item.group.key, resource): amount
                for resource, amount in borrow_resources
                if isinstance(resource, str) and isinstance(amount, int)
            }
            return PolicyStateDelta(debt_changes=changes)
        return PolicyStateDelta()

    def on_release(self, lease: TaskAdmissionLease) -> PolicyStateDelta:
        if not self._config.repay_on_withheld_peer_pressure:
            return PolicyStateDelta()
        # Borrow debt is group-level: any completed lease in the group repays it, clamped to zero by the controller.
        return PolicyStateDelta(
            debt_changes={(lease.item.group.key, resource): -amount for resource, amount in lease.resources.items()}
        )


def _queued_peer_pressure_resources(
    item: SchedulableTask,
    queue_view: QueueView,
    admission_view: TaskAdmissionView,
) -> tuple[SchedulerResourceKey, ...]:
    candidate_resources = _fair_pressure_resources(item.resource_request.amounts)
    pressure_resources: list[SchedulerResourceKey] = []
    for group_key, peer_resources in queue_view.first_candidate_resources_by_group.items():
        if group_key == item.group.key:
            continue
        if not _is_hard_resource_eligible(peer_resources, admission_view):
            continue
        for resource in candidate_resources:
            if peer_resources.get(resource, 0) > 0 and resource not in pressure_resources:
                pressure_resources.append(resource)
    return tuple(pressure_resources)


def _fair_pressure_resources(
    resources: Mapping[SchedulerResourceKey, int],
) -> tuple[SchedulerResourceKey, ...]:
    typed_resources = tuple(resource for resource in resources if resource != "submission")
    if typed_resources:
        return typed_resources
    return tuple(resources)


def _is_hard_resource_eligible(
    resources: Mapping[SchedulerResourceKey, int],
    admission_view: TaskAdmissionView,
) -> bool:
    return all(admission_view.resources_available.get(resource, 0) >= amount for resource, amount in resources.items())
