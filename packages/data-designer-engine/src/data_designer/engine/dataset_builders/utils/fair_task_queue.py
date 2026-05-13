# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Callable, Iterable
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


class FairTaskSelector:
    """Virtual-time fair selector over per-group FIFO task queues."""

    def __init__(self) -> None:
        self._queues: dict[TaskGroupKey, deque[Task]] = {}
        self._queued: set[Task] = set()
        self._task_groups: dict[Task, TaskGroupKey] = {}
        self._group_specs: dict[TaskGroupKey, TaskGroupSpec] = {}
        self._group_finish: dict[TaskGroupKey, float] = {}
        self._heap: list[tuple[float, int, TaskGroupKey]] = []
        self._active_heap_keys: set[TaskGroupKey] = set()
        self._sequence = 0
        self._virtual_time = 0.0

    @property
    def has_queued_tasks(self) -> bool:
        return bool(self._queued)

    def get_group_spec(self, key: TaskGroupKey) -> TaskGroupSpec:
        return self._group_specs[key]

    def sync_ready(self, ready: Iterable[tuple[Task, TaskGroupSpec]]) -> None:
        """Synchronize queued tasks against the scheduler's current ready set."""
        ready_items = sorted(ready, key=self._ready_sort_key)
        ready_tasks = {task for task, _ in ready_items}

        for task, group in ready_items:
            self._group_specs[group.key] = group
            if task in self._queued:
                continue
            queue = self._queues.setdefault(group.key, deque())
            queue.append(task)
            self._queued.add(task)
            self._task_groups[task] = group.key
            self._activate_group(group.key)

        for stale in self._queued - ready_tasks:
            self._queued.discard(stale)
            self._task_groups.pop(stale, None)

    def pop_next(self, can_admit_group: Callable[[TaskGroupKey], bool]) -> TaskSelection | None:
        """Select the next eligible task, or ``None`` if no queued group can run."""
        blocked: list[TaskGroupKey] = []
        try:
            while self._heap:
                finish, _, key = heapq.heappop(self._heap)
                self._active_heap_keys.discard(key)
                self._purge_queue_head(key)
                queue = self._queues.get(key)
                if not queue:
                    continue
                if not can_admit_group(key):
                    blocked.append(key)
                    continue

                task = queue.popleft()
                self._queued.discard(task)
                self._task_groups.pop(task, None)

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

    @staticmethod
    def _ready_sort_key(item: tuple[Task, TaskGroupSpec]) -> tuple[TaskGroupKey, int, int, str, str]:
        task, group = item
        row_index = task.row_index if task.row_index is not None else -1
        return (group.key, task.row_group, row_index, task.column, task.task_type)
