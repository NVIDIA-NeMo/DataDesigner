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
    """Engine-internal bounded-borrow policy configuration."""

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
        if not _has_queued_peer_group(item.group.key, queue_view):
            return TaskAdmissionPolicyDecision(allowed=True)
        return TaskAdmissionPolicyDecision(
            allowed=False,
            reason="group_cap",
            diagnostics={"admitted_limit": item.group.admitted_limit, "leased_count": leased_count},
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

        if _has_queued_peer_group(item.group.key, queue_view):
            for resource in item.resource_request.amounts:
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
                diagnostics={"admitted_limit": limit, "leased_count": leased_count},
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
        return PolicyStateDelta(
            debt_changes={(lease.item.group.key, resource): -amount for resource, amount in lease.resources.items()}
        )


def _has_queued_peer_group(group_key: TaskGroupKey, queue_view: QueueView) -> bool:
    return any(key != group_key and count > 0 for key, count in queue_view.queued_by_group.items())
