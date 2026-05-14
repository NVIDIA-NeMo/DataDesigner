# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from data_designer.engine.dataset_builders.utils.fair_task_queue import QueueView
from data_designer.engine.dataset_builders.utils.task_model import Task
from data_designer.engine.dataset_builders.utils.task_scheduling import (
    SchedulableTask,
    SchedulerResourceKey,
    SchedulerResourceRequest,
    TaskGroupKey,
)

TaskAdmissionDenyReason = Literal[
    "no_capacity",
    "group_cap",
    "borrow_debt",
    "shutdown",
    "policy_denial",
]

ReleaseReason = Literal[
    "released",
    "duplicate",
    "stale_lease",
    "wrong_controller_generation",
    "unknown_lease",
]


@dataclass(frozen=True)
class TaskAdmissionConfig:
    """Engine-internal scheduler task-stage admission configuration."""

    submission_capacity: int = 256
    resource_limits: Mapping[SchedulerResourceKey, int] = field(default_factory=dict)
    bounded_borrow: BoundedBorrowTaskAdmissionPolicyConfig | None = None

    def __post_init__(self) -> None:
        if self.submission_capacity <= 0:
            raise ValueError("submission_capacity must be positive.")
        merged = {"submission": self.submission_capacity, **self.resource_limits}
        for resource, limit in merged.items():
            if limit <= 0:
                raise ValueError(f"Task admission limit for {resource!r} must be positive.")
        object.__setattr__(self, "resource_limits", merged)


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
class TaskAdmissionView:
    resource_limits: Mapping[SchedulerResourceKey, int]
    resources_available: Mapping[SchedulerResourceKey, int]
    leased_resources: Mapping[SchedulerResourceKey, int]
    leased_resources_by_group: Mapping[TaskGroupKey, Mapping[SchedulerResourceKey, int]]
    running_counts_by_group: Mapping[TaskGroupKey, int]
    policy_debt_by_group_resource: Mapping[tuple[TaskGroupKey, SchedulerResourceKey], int]


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


@dataclass(frozen=True)
class TaskAdmissionLease:
    lease_id: str
    item: SchedulableTask
    resources: Mapping[SchedulerResourceKey, int]
    acquired_at: float
    controller_generation: str


@dataclass(frozen=True)
class TaskAdmissionDenied:
    item: SchedulableTask
    reason: TaskAdmissionDenyReason
    available_after: float | None = None
    snapshot: TaskAdmissionView | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)


TaskAdmissionDecision = TaskAdmissionLease | TaskAdmissionDenied


@dataclass(frozen=True)
class ReleaseResult:
    released: bool
    reason: ReleaseReason
    diagnostics: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskAdmissionBlockSummary:
    queued_count: int
    dominant_denial_reasons: Mapping[TaskAdmissionDenyReason, int]
    available_after: float | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)


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


class TaskAdmissionController:
    """Owns scheduler-level task leases and resource accounting."""

    def __init__(
        self,
        config: TaskAdmissionConfig | None = None,
        policy: TaskAdmissionPolicy | None = None,
    ) -> None:
        self._config = config or TaskAdmissionConfig()
        self._generation = uuid.uuid4().hex
        self._leases: dict[str, TaskAdmissionLease] = {}
        self._released: set[str] = set()
        self._leased_by_resource: Counter[SchedulerResourceKey] = Counter()
        self._leased_by_group: dict[TaskGroupKey, Counter[SchedulerResourceKey]] = defaultdict(Counter)
        self._running_by_group: Counter[TaskGroupKey] = Counter()
        self._policy_debt: Counter[tuple[TaskGroupKey, SchedulerResourceKey]] = Counter()
        self._release_diagnostics: Counter[str] = Counter()
        if policy is not None:
            self._policy = policy
        elif self._config.bounded_borrow is not None:
            self._policy = BoundedBorrowTaskAdmissionPolicy(self._config.bounded_borrow)
        else:
            self._policy = StrictFairTaskAdmissionPolicy()

    def is_eligible(self, item: SchedulableTask, queue_view: QueueView) -> bool:
        return not isinstance(self.try_evaluate(item, queue_view), TaskAdmissionDenied)

    def try_evaluate(
        self, item: SchedulableTask, queue_view: QueueView
    ) -> TaskAdmissionPolicyDecision | TaskAdmissionDenied:
        view = self.view()
        missing = self._missing_resources(item, view)
        if missing:
            return TaskAdmissionDenied(
                item=item,
                reason="no_capacity",
                snapshot=view,
                diagnostics={"missing_resources": missing},
            )
        decision = self._policy.evaluate(item, queue_view, view)
        if not decision.allowed:
            return TaskAdmissionDenied(
                item=item,
                reason=decision.reason or "policy_denial",
                available_after=decision.available_after,
                snapshot=view,
                diagnostics=decision.diagnostics,
            )
        return decision

    def try_acquire(self, item: SchedulableTask, queue_view: QueueView) -> TaskAdmissionDecision:
        evaluated = self.try_evaluate(item, queue_view)
        if isinstance(evaluated, TaskAdmissionDenied):
            return evaluated
        lease = TaskAdmissionLease(
            lease_id=uuid.uuid4().hex,
            item=item,
            resources=dict(item.resource_request.amounts),
            acquired_at=time.monotonic(),
            controller_generation=self._generation,
        )
        for resource, amount in lease.resources.items():
            self._leased_by_resource[resource] += amount
            self._leased_by_group[item.group.key][resource] += amount
        self._running_by_group[item.group.key] += 1
        self._apply_delta(self._policy.on_acquire(lease, evaluated))
        self._leases[lease.lease_id] = lease
        return lease

    def release(self, lease: TaskAdmissionLease) -> ReleaseResult:
        if lease.controller_generation != self._generation:
            self._release_diagnostics["wrong_controller_generation"] += 1
            return ReleaseResult(released=False, reason="wrong_controller_generation")
        active = self._leases.pop(lease.lease_id, None)
        if active is None:
            reason: ReleaseReason = "duplicate" if lease.lease_id in self._released else "unknown_lease"
            self._release_diagnostics[reason] += 1
            return ReleaseResult(released=False, reason=reason)
        if active.item.task_id != lease.item.task_id:
            self._leases[lease.lease_id] = active
            self._release_diagnostics["stale_lease"] += 1
            return ReleaseResult(released=False, reason="stale_lease")

        self._released.add(lease.lease_id)
        for resource, amount in active.resources.items():
            self._leased_by_resource[resource] = max(0, self._leased_by_resource[resource] - amount)
            self._leased_by_group[active.item.group.key][resource] = max(
                0,
                self._leased_by_group[active.item.group.key][resource] - amount,
            )
        self._running_by_group[active.item.group.key] = max(0, self._running_by_group[active.item.group.key] - 1)
        self._apply_delta(self._policy.on_release(active))
        return ReleaseResult(released=True, reason="released")

    def view(self) -> TaskAdmissionView:
        limits = dict(self._config.resource_limits)
        leased = {resource: count for resource, count in self._leased_by_resource.items() if count > 0}
        available = {
            resource: max(0, limit - self._leased_by_resource.get(resource, 0)) for resource, limit in limits.items()
        }
        return TaskAdmissionView(
            resource_limits=limits,
            resources_available=available,
            leased_resources=leased,
            leased_resources_by_group={
                group: {resource: count for resource, count in counts.items() if count > 0}
                for group, counts in self._leased_by_group.items()
            },
            running_counts_by_group={group: count for group, count in self._running_by_group.items() if count > 0},
            policy_debt_by_group_resource={key: count for key, count in self._policy_debt.items() if count > 0},
        )

    def explain_blocked(self, queue_view: QueueView) -> TaskAdmissionBlockSummary:
        reasons: Counter[TaskAdmissionDenyReason] = Counter()
        available_after_values: list[float] = []
        view = self.view()
        for group_key, resources in queue_view.first_candidate_resources_by_group.items():
            for resource, amount in resources.items():
                if view.resources_available.get(resource, 0) < amount:
                    reasons["no_capacity"] += 1
                    break
            else:
                group = queue_view.first_candidate_group_specs_by_group.get(group_key)
                if group is None:
                    continue
                task = SchedulableTask(
                    task_id=f"blocked-{group_key.kind}-{'-'.join(group_key.identity)}",
                    payload=Task(column="", row_group=-1, row_index=None, task_type="batch"),
                    group=group,
                    resource_request=SchedulerResourceRequest(dict(resources)),
                )
                decision = self._policy.evaluate(task, queue_view, view)
                if not decision.allowed:
                    reasons[decision.reason or "policy_denial"] += 1
                    if decision.available_after is not None:
                        available_after_values.append(decision.available_after)
        return TaskAdmissionBlockSummary(
            queued_count=queue_view.queued_total,
            dominant_denial_reasons=dict(reasons),
            available_after=min(available_after_values) if available_after_values else None,
            diagnostics={"snapshot": self.view()},
        )

    def _missing_resources(
        self,
        item: SchedulableTask,
        view: TaskAdmissionView,
    ) -> dict[SchedulerResourceKey, dict[str, int]]:
        missing: dict[SchedulerResourceKey, dict[str, int]] = {}
        for resource, amount in item.resource_request.amounts.items():
            available = view.resources_available.get(resource, 0)
            if available < amount:
                missing[resource] = {"requested": amount, "available": available}
        return missing

    def _apply_delta(self, delta: PolicyStateDelta) -> None:
        for key, change in delta.debt_changes.items():
            self._policy_debt[key] = max(0, self._policy_debt[key] + change)


def _has_queued_peer_group(group_key: TaskGroupKey, queue_view: QueueView) -> bool:
    return any(key != group_key and count > 0 for key, count in queue_view.queued_by_group.items())
