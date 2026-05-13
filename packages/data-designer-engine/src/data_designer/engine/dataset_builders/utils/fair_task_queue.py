# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from data_designer.engine.dataset_builders.utils.task_model import Task


@dataclass(frozen=True, order=True)
class TaskGroupKey:
    """Stable identity for a stream of related scheduler tasks."""

    kind: Literal["model", "custom_model", "local"]
    identity: tuple[str, ...]


@dataclass(frozen=True)
class TaskGroupSpec:
    """Scheduling metadata for a task group."""

    key: TaskGroupKey
    weight: float = 1.0
    admitted_limit: int | None = None


@dataclass(frozen=True)
class TaskSelection:
    """A task selected for dispatch with the group metadata used to choose it."""

    task: Task
    group: TaskGroupSpec


class FairTaskQueue:
    """Virtual-time fair queue with peer-sensitive per-group FIFO admission limits."""

    def __init__(self) -> None:
        self._queues: dict[TaskGroupKey, deque[Task]] = {}
        self._queued: set[Task] = set()
        self._task_groups: dict[Task, TaskGroupKey] = {}
        self._group_specs: dict[TaskGroupKey, TaskGroupSpec] = {}
        self._group_finish: dict[TaskGroupKey, float] = {}
        self._admitted_by_group: dict[TaskGroupKey, int] = {}
        self._admitted_task_groups: dict[Task, TaskGroupKey] = {}
        self._heap: list[tuple[float, int, TaskGroupKey]] = []
        self._active_heap_keys: set[TaskGroupKey] = set()
        self._sequence = 0
        self._virtual_time = 0.0

    @property
    def has_queued_tasks(self) -> bool:
        return bool(self._queued)

    def enqueue(self, task: Task, group: TaskGroupSpec) -> None:
        """Add one ready task to its fair scheduling group."""
        self._group_specs[group.key] = group
        if task in self._queued:
            return
        queue = self._queues.setdefault(group.key, deque())
        queue.append(task)
        self._queued.add(task)
        self._task_groups[task] = group.key
        self._activate_group(group.key)

    def discard(self, task: Task) -> None:
        """Remove a queued task lazily if it is no longer dispatchable."""
        self._queued.discard(task)
        self._task_groups.pop(task, None)

    def discard_where(self, predicate: Callable[[Task], bool]) -> None:
        """Remove queued tasks matching a predicate."""
        for task in tuple(self._queued):
            if predicate(task):
                self.discard(task)

    def admit_next(self) -> TaskSelection | None:
        """Admit the next eligible task, or ``None`` if no queued group can run."""
        blocked: list[TaskGroupKey] = []
        try:
            while self._heap:
                finish, _, key = heapq.heappop(self._heap)
                self._active_heap_keys.discard(key)
                self._purge_queue_head(key)
                queue = self._queues.get(key)
                if not queue:
                    continue
                if not self._can_admit_group(key):
                    blocked.append(key)
                    continue

                task = queue.popleft()
                self._queued.discard(task)
                self._task_groups.pop(task, None)
                self._admitted_task_groups[task] = key
                self._admitted_by_group[key] = self._admitted_by_group.get(key, 0) + 1

                group = self._group_specs[key]
                self._virtual_time = max(self._virtual_time, finish)
                self._group_finish[key] = self._virtual_time + (1.0 / max(group.weight, 1.0))
                self._purge_queue_head(key)
                if queue:
                    self._activate_group(key)
                return TaskSelection(task=task, group=group)
            return None
        finally:
            for key in blocked:
                self._activate_group(key)

    def release(self, task: Task) -> None:
        """Release one previously admitted task from its group limit."""
        key = self._admitted_task_groups.pop(task, None)
        if key is None:
            return
        admitted = self._admitted_by_group.get(key, 0)
        if admitted <= 1:
            self._admitted_by_group.pop(key, None)
        else:
            self._admitted_by_group[key] = admitted - 1
        self._activate_group(key)

    def _activate_group(self, key: TaskGroupKey) -> None:
        self._purge_queue_head(key)
        queue = self._queues.get(key)
        if not queue or key in self._active_heap_keys:
            return
        self._sequence += 1
        finish = self._group_finish.get(key, self._virtual_time)
        heapq.heappush(self._heap, (finish, self._sequence, key))
        self._active_heap_keys.add(key)

    def _purge_queue_head(self, key: TaskGroupKey) -> None:
        queue = self._queues.get(key)
        if queue is None:
            return
        while queue:
            task = queue[0]
            if task in self._queued and self._task_groups.get(task) == key:
                break
            queue.popleft()

    def _can_admit_group(self, key: TaskGroupKey) -> bool:
        group = self._group_specs[key]
        if group.admitted_limit is None:
            return True
        if self._admitted_by_group.get(key, 0) < group.admitted_limit:
            return True
        return not self._has_queued_peer_group(key)

    def _has_queued_peer_group(self, key: TaskGroupKey) -> bool:
        return any(queued_key != key for queued_key in self._task_groups.values())
