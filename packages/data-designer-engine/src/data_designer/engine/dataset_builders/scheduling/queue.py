# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from data_designer.engine.dataset_builders.scheduling.resources import (
    SchedulableTask,
    SchedulerResourceKey,
    TaskGroupKey,
    TaskGroupSpec,
)


@dataclass(frozen=True)
class QueueView:
    """Read-only queue facts supplied to task admission policies."""

    queued_total: int
    queued_by_group: Mapping[TaskGroupKey, int]
    queued_resource_demand_by_group: Mapping[TaskGroupKey, Mapping[SchedulerResourceKey, int]]
    first_candidate_resources_by_group: Mapping[TaskGroupKey, Mapping[SchedulerResourceKey, int]]
    first_candidate_tasks_by_group: Mapping[TaskGroupKey, SchedulableTask]
    first_candidate_group_specs_by_group: Mapping[TaskGroupKey, TaskGroupSpec]
    queued_peer_demand_by_resource: Mapping[SchedulerResourceKey, int]


@dataclass(frozen=True)
class QueueSelection:
    """Non-mutating fair-queue selection returned to the scheduler."""

    item: SchedulableTask
    queue_view: QueueView
    sequence_version: int


class FairTaskQueue:
    """Virtual-time fair queue that owns ready membership and ordering only."""

    def __init__(self) -> None:
        self._queues: dict[TaskGroupKey, deque[SchedulableTask]] = {}
        self._queued: dict[str, SchedulableTask] = {}
        self._task_groups: dict[str, TaskGroupKey] = {}
        self._group_specs: dict[TaskGroupKey, TaskGroupSpec] = {}
        self._queued_by_group: Counter[TaskGroupKey] = Counter()
        self._queued_resource_demand_by_group: dict[TaskGroupKey, Counter[SchedulerResourceKey]] = defaultdict(Counter)
        self._queued_peer_demand_by_resource: Counter[SchedulerResourceKey] = Counter()
        self._group_finish: dict[TaskGroupKey, float] = {}
        self._active_entries: dict[TaskGroupKey, tuple[float, int]] = {}
        self._sequence = 0
        self._sequence_version = 0
        self._virtual_time = 0.0

    @property
    def has_queued_tasks(self) -> bool:
        return bool(self._queued)

    def enqueue(self, items: Iterable[SchedulableTask]) -> tuple[str, ...]:
        """Add ready tasks idempotently and return newly accepted task ids."""
        accepted: list[str] = []
        for item in items:
            if item.task_id in self._queued:
                continue
            self._group_specs[item.group.key] = item.group
            queue = self._queues.setdefault(item.group.key, deque())
            queue.append(item)
            self._queued[item.task_id] = item
            self._task_groups[item.task_id] = item.group.key
            self._increment_queue_accounting(item)
            self._activate_group(item.group.key)
            accepted.append(item.task_id)
        if accepted:
            self._sequence_version += 1
        return tuple(accepted)

    def discard(self, task_id: str) -> None:
        """Remove a queued task lazily if it is no longer dispatchable."""
        if self._remove_queued_item(task_id) is not None:
            self._sequence_version += 1

    def discard_where(self, predicate: Callable[[SchedulableTask], bool]) -> None:
        """Remove queued tasks matching a predicate."""
        for task_id, item in tuple(self._queued.items()):
            if predicate(item):
                self.discard(task_id)

    def select_next(self, is_eligible: Callable[[SchedulableTask, QueueView], bool]) -> QueueSelection | None:
        """Return the next eligible task without mutating queue state."""
        view = self.view()
        active_entries = sorted((finish, sequence, key) for key, (finish, sequence) in self._active_entries.items())
        for _finish, _sequence, key in active_entries:
            item = self._first_valid_item(key)
            if item is None:
                continue
            if not is_eligible(item, view):
                continue
            return QueueSelection(item=item, queue_view=view, sequence_version=self._sequence_version)
        return None

    def commit(self, selection: QueueSelection) -> SchedulableTask | None:
        """Remove a previously selected task and advance fair-queue state."""
        if selection.sequence_version != self._sequence_version:
            return None
        item = selection.item
        key = self._task_groups.get(item.task_id)
        if key is None or key != item.group.key:
            return None
        queue = self._queues.get(key)
        if queue is None:
            return None
        self._purge_queue_head(key)
        if not queue or queue[0].task_id != item.task_id:
            return None

        queue.popleft()
        self._remove_queued_item(item.task_id)
        self._active_entries.pop(key, None)
        group = self._group_specs[key]
        finish = self._group_finish.get(key, self._virtual_time)
        self._virtual_time = max(self._virtual_time, finish)
        self._group_finish[key] = self._virtual_time + (1.0 / max(group.weight, 1.0))
        self._sequence_version += 1
        self._purge_queue_head(key)
        if queue:
            self._activate_group(key)
        return item

    def view(self) -> QueueView:
        first_by_group: dict[TaskGroupKey, Mapping[SchedulerResourceKey, int]] = {}
        first_tasks_by_group: dict[TaskGroupKey, SchedulableTask] = {}
        first_group_specs: dict[TaskGroupKey, TaskGroupSpec] = {}

        for key in self._queued_by_group:
            first = self._first_valid_item(key)
            if first is None:
                continue
            first_by_group[key] = dict(first.resource_request.amounts)
            first_tasks_by_group[key] = first
            first_group_specs[key] = first.group

        return QueueView(
            queued_total=len(self._queued),
            queued_by_group=dict(self._queued_by_group),
            queued_resource_demand_by_group={
                key: dict(value) for key, value in self._queued_resource_demand_by_group.items()
            },
            first_candidate_resources_by_group=first_by_group,
            first_candidate_tasks_by_group=first_tasks_by_group,
            first_candidate_group_specs_by_group=first_group_specs,
            queued_peer_demand_by_resource=dict(self._queued_peer_demand_by_resource),
        )

    def _activate_group(self, key: TaskGroupKey) -> None:
        self._purge_queue_head(key)
        queue = self._queues.get(key)
        if not queue or key in self._active_entries:
            return
        self._sequence += 1
        finish = self._group_finish.get(key, self._virtual_time)
        self._active_entries[key] = (finish, self._sequence)

    def _first_valid_item(self, key: TaskGroupKey) -> SchedulableTask | None:
        self._purge_queue_head(key)
        queue = self._queues.get(key)
        if not queue:
            return None
        return queue[0]

    def _purge_queue_head(self, key: TaskGroupKey) -> None:
        queue = self._queues.get(key)
        if queue is None:
            return
        while queue:
            item = queue[0]
            if item.task_id in self._queued and self._task_groups.get(item.task_id) == key:
                break
            queue.popleft()

    def _increment_queue_accounting(self, item: SchedulableTask) -> None:
        key = item.group.key
        self._queued_by_group[key] += 1
        for resource, amount in item.resource_request.amounts.items():
            self._queued_resource_demand_by_group[key][resource] += amount
            self._queued_peer_demand_by_resource[resource] += amount

    def _remove_queued_item(self, task_id: str) -> SchedulableTask | None:
        item = self._queued.pop(task_id, None)
        key = self._task_groups.pop(task_id, None)
        if item is None or key is None:
            return item
        self._decrement_queue_accounting(item, key)
        if key not in self._queued_by_group:
            self._active_entries.pop(key, None)
        return item

    def _decrement_queue_accounting(self, item: SchedulableTask, key: TaskGroupKey) -> None:
        self._queued_by_group[key] -= 1
        if self._queued_by_group[key] <= 0:
            del self._queued_by_group[key]

        group_demand = self._queued_resource_demand_by_group.get(key)
        if group_demand is not None:
            for resource, amount in item.resource_request.amounts.items():
                group_demand[resource] -= amount
                if group_demand[resource] <= 0:
                    del group_demand[resource]
            if not group_demand:
                del self._queued_resource_demand_by_group[key]

        for resource, amount in item.resource_request.amounts.items():
            self._queued_peer_demand_by_resource[resource] -= amount
            if self._queued_peer_demand_by_resource[resource] <= 0:
                del self._queued_peer_demand_by_resource[resource]
