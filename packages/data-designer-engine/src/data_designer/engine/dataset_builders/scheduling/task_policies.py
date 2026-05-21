# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

from data_designer.engine.dataset_builders.scheduling.queue import QueueView
from data_designer.engine.dataset_builders.scheduling.resources import (
    SchedulableTask,
    SchedulerResourceKey,
    TaskGroupKey,
    TaskGroupSpec,
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
DEFAULT_DYNAMIC_BORROW_RESERVE_FRACTION = 0.125
DEFAULT_DYNAMIC_BORROW_MAX_RESERVED_SLOTS = 8


@dataclass(frozen=True)
class BoundedBorrowTaskAdmissionPolicyConfig:
    """Engine-internal bounded-borrow policy configuration.

    Borrow debt is tracked by task group and scheduler resource. Any completed
    lease in the same group repays debt for the released resources; repayment is
    not tied to the specific lease that originally borrowed. When no explicit
    borrow ceiling is configured, the policy reserves one slot per eight
    resource slots, capped at eight reserved slots, and lets solo groups borrow
    up to the remaining capacity.
    """

    borrow_ceiling_by_group_resource: Mapping[tuple[TaskGroupKey, SchedulerResourceKey], int] = field(
        default_factory=dict
    )
    default_borrow_ceiling: int | None = None
    dynamic_borrow_reserve_fraction: float = DEFAULT_DYNAMIC_BORROW_RESERVE_FRACTION
    dynamic_borrow_max_reserved_slots: int = DEFAULT_DYNAMIC_BORROW_MAX_RESERVED_SLOTS
    strict_share_rounding: Literal["floor", "ceil"] = "ceil"
    repay_on_withheld_peer_pressure: bool = True

    def __post_init__(self) -> None:
        if self.default_borrow_ceiling is not None and self.default_borrow_ceiling < 0:
            raise ValueError("default_borrow_ceiling must be non-negative.")
        if not 0 <= self.dynamic_borrow_reserve_fraction <= 1:
            raise ValueError("dynamic_borrow_reserve_fraction must be between 0 and 1.")
        if self.dynamic_borrow_max_reserved_slots <= 0:
            raise ValueError("dynamic_borrow_max_reserved_slots must be positive.")
        for key, ceiling in self.borrow_ceiling_by_group_resource.items():
            if ceiling < 0:
                raise ValueError(f"Borrow ceiling for {key!r} must be non-negative.")


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
        if item.group.admitted_limit is None:
            return TaskAdmissionPolicyDecision(allowed=True)

        pressure_resources = _queued_peer_pressure_resources(item, queue_view, admission_view)
        borrow_resources: list[tuple[SchedulerResourceKey, int]] = []
        diagnostics_by_resource: dict[SchedulerResourceKey, dict[str, int | str]] = {}
        for resource, amount in item.resource_request.amounts.items():
            admitted = admission_view.leased_resources_by_group.get(item.group.key, {}).get(resource, 0)
            strict_share = _strict_share(item, resource, queue_view, admission_view, self._config.strict_share_rounding)
            projected = admitted + amount
            debt_key = (item.group.key, resource)
            debt = admission_view.policy_debt_by_group_resource.get(debt_key, 0)
            diagnostics_by_resource[resource] = {
                "admitted": admitted,
                "requested": amount,
                "strict_share": strict_share,
                "debt": debt,
            }

            if resource in pressure_resources:
                if debt > 0:
                    return TaskAdmissionPolicyDecision(
                        allowed=False,
                        reason="borrow_debt",
                        diagnostics={"resource": resource, "debt": debt, "strict_share": strict_share},
                    )
                if projected > strict_share:
                    return TaskAdmissionPolicyDecision(
                        allowed=False,
                        reason="group_cap",
                        diagnostics={
                            "resource": resource,
                            "admitted": admitted,
                            "requested": amount,
                            "strict_share": strict_share,
                            "pressure_resources": pressure_resources,
                        },
                    )
                continue

            if projected <= strict_share:
                continue

            new_debt = projected - strict_share
            ceiling, ceiling_diagnostics = self._borrow_ceiling(
                debt_key,
                resource_limit=admission_view.resource_limits.get(resource, 0),
                strict_share=strict_share,
            )
            diagnostics_by_resource[resource].update(ceiling_diagnostics)
            if debt + new_debt > ceiling:
                return TaskAdmissionPolicyDecision(
                    allowed=False,
                    reason="borrow_debt",
                    diagnostics={
                        "resource": resource,
                        "debt": debt,
                        "requested": amount,
                        "new_debt": new_debt,
                        "ceiling": ceiling,
                        "strict_share": strict_share,
                    },
                )
            borrow_resources.append((resource, new_debt))
        return TaskAdmissionPolicyDecision(
            allowed=True,
            diagnostics={"borrow_resources": tuple(borrow_resources), "strict_share": diagnostics_by_resource},
        )

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

    def _borrow_ceiling(
        self,
        debt_key: tuple[TaskGroupKey, SchedulerResourceKey],
        *,
        resource_limit: int,
        strict_share: int,
    ) -> tuple[int, dict[str, int | str]]:
        explicit_ceiling = self._config.borrow_ceiling_by_group_resource.get(debt_key)
        if explicit_ceiling is not None:
            return explicit_ceiling, {"ceiling": explicit_ceiling, "ceiling_source": "group_resource"}
        if self._config.default_borrow_ceiling is not None:
            return self._config.default_borrow_ceiling, {
                "ceiling": self._config.default_borrow_ceiling,
                "ceiling_source": "default",
            }
        reserved_slots = _dynamic_reserved_slots(
            resource_limit,
            reserve_fraction=self._config.dynamic_borrow_reserve_fraction,
            max_reserved_slots=self._config.dynamic_borrow_max_reserved_slots,
        )
        target_solo_cap = max(0, resource_limit - reserved_slots)
        borrow_slots = max(0, target_solo_cap - strict_share)
        ceiling = _triangular_number(borrow_slots)
        return ceiling, {
            "ceiling": ceiling,
            "ceiling_source": "dynamic",
            "reserved_slots": reserved_slots,
            "borrow_slots": borrow_slots,
        }


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


def _strict_share(
    item: SchedulableTask,
    resource: SchedulerResourceKey,
    queue_view: QueueView,
    admission_view: TaskAdmissionView,
    rounding: Literal["floor", "ceil"],
) -> int:
    resource_limit = admission_view.resource_limits.get(resource, 0)
    if resource_limit <= 1:
        return 1

    candidate_groups = _competing_group_specs(item, resource, queue_view, admission_view)
    group_weight = max(1.0, item.group.weight)
    if len(candidate_groups) <= 1:
        total_weight = group_weight * 2
    else:
        total_weight = sum(max(1.0, group.weight) for group in candidate_groups.values())
    raw_share = resource_limit * group_weight / total_weight
    if rounding == "ceil":
        rounded_share = math.ceil(raw_share)
    else:
        rounded_share = math.floor(raw_share)
    strict_share = max(1, rounded_share)
    if item.group.admitted_limit is not None:
        strict_share = min(strict_share, item.group.admitted_limit)
    return min(resource_limit, strict_share)


def _dynamic_reserved_slots(resource_limit: int, *, reserve_fraction: float, max_reserved_slots: int) -> int:
    return min(max_reserved_slots, max(1, math.ceil(resource_limit * reserve_fraction)))


def _triangular_number(value: int) -> int:
    return value * (value + 1) // 2


def _competing_group_specs(
    item: SchedulableTask,
    resource: SchedulerResourceKey,
    queue_view: QueueView,
    admission_view: TaskAdmissionView,
) -> dict[TaskGroupKey, TaskGroupSpec]:
    groups: dict[TaskGroupKey, TaskGroupSpec] = {item.group.key: item.group}
    for group_key, peer_resources in queue_view.first_candidate_resources_by_group.items():
        if peer_resources.get(resource, 0) <= 0:
            continue
        if not _is_hard_resource_eligible(peer_resources, admission_view):
            continue
        group = queue_view.first_candidate_group_specs_by_group.get(group_key)
        if group is not None:
            groups[group_key] = group
    return groups


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
