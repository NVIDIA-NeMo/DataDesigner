# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import heapq
from collections import Counter, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from data_designer.engine.models.request_admission.resources import RequestAdmissionItem, RequestResourceKey
from data_designer.engine.models.resources import ProviderModelKey

if TYPE_CHECKING:
    from data_designer.engine.models.request_admission.controller import RequestAdmissionLease


@dataclass
class RequestWaiter:
    waiter_id: str
    item: RequestAdmissionItem
    enqueued_at: float
    deadline_monotonic: float | None = None
    assigned_lease: RequestAdmissionLease | None = None


@dataclass(frozen=True)
class RequestQueueView:
    queued_total: int
    queued_by_group: Mapping[RequestResourceKey, int]
    queued_demand_by_resource: Mapping[RequestResourceKey, int]
    aggregate_provider_model_waiters: Mapping[ProviderModelKey, int]


@dataclass(frozen=True)
class RequestQueueSelection:
    waiter: RequestWaiter
    item: RequestAdmissionItem
    waiter_id: str
    queue_view: RequestQueueView
    sequence_version: int


class RequestFairQueue:
    """Weighted fair waiter queue used by request admission."""

    def __init__(self) -> None:
        self._queues: dict[RequestResourceKey, deque[RequestWaiter]] = {}
        self._queued: dict[str, RequestWaiter] = {}
        self._waiter_groups: dict[str, RequestResourceKey] = {}
        self._group_finish: dict[RequestResourceKey, float] = {}
        self._heap: list[tuple[float, int, RequestResourceKey]] = []
        self._active_heap_entries: dict[RequestResourceKey, tuple[float, int]] = {}
        self._sequence = 0
        self._sequence_version = 0
        self._virtual_time = 0.0

    @property
    def has_waiters(self) -> bool:
        return bool(self._queued)

    def contains(self, waiter_id: str) -> bool:
        return waiter_id in self._queued

    def enqueue(self, waiter: RequestWaiter) -> bool:
        if waiter.waiter_id in self._queued:
            return False
        key = waiter.item.group.key
        queue = self._queues.setdefault(key, deque())
        queue.append(waiter)
        self._queued[waiter.waiter_id] = waiter
        self._waiter_groups[waiter.waiter_id] = key
        self._activate_group(key)
        self._sequence_version += 1
        return True

    def remove(self, waiter_id: str) -> RequestWaiter | None:
        waiter = self._queued.pop(waiter_id, None)
        if waiter is None:
            return None
        self._waiter_groups.pop(waiter_id, None)
        self._sequence_version += 1
        return waiter

    def select_next(
        self, is_eligible: Callable[[RequestWaiter, RequestQueueView], bool]
    ) -> RequestQueueSelection | None:
        view = self.view()
        heap_copy = list(self._heap)
        heapq.heapify(heap_copy)
        active_seen: set[RequestResourceKey] = set()
        while heap_copy:
            finish, sequence, key = heapq.heappop(heap_copy)
            if key in active_seen:
                continue
            if self._active_heap_entries.get(key) != (finish, sequence):
                continue
            active_seen.add(key)
            waiter = self._first_valid_waiter(key)
            if waiter is None:
                continue
            if not is_eligible(waiter, view):
                continue
            return RequestQueueSelection(
                waiter=waiter,
                item=waiter.item,
                waiter_id=waiter.waiter_id,
                queue_view=view,
                sequence_version=self._sequence_version,
            )
        return None

    def commit(self, selection: RequestQueueSelection) -> RequestWaiter | None:
        if selection.sequence_version != self._sequence_version:
            return None
        key = self._waiter_groups.get(selection.waiter_id)
        if key is None or key != selection.item.group.key:
            return None
        queue = self._queues.get(key)
        if queue is None:
            return None
        self._purge_queue_head(key)
        if not queue or queue[0].waiter_id != selection.waiter_id:
            return None

        waiter = queue.popleft()
        self._queued.pop(waiter.waiter_id, None)
        self._waiter_groups.pop(waiter.waiter_id, None)
        self._active_heap_entries.pop(key, None)
        weight = max(selection.item.group.weight, 1.0)
        finish = self._group_finish.get(key, self._virtual_time)
        self._virtual_time = max(self._virtual_time, finish)
        self._group_finish[key] = self._virtual_time + (1.0 / weight)
        self._sequence_version += 1
        self._purge_queue_head(key)
        if queue:
            self._activate_group(key)
        return waiter

    def view(self) -> RequestQueueView:
        queued_by_group: Counter[RequestResourceKey] = Counter()
        demand_by_resource: Counter[RequestResourceKey] = Counter()
        aggregate_waiters: Counter[ProviderModelKey] = Counter()
        for waiter in self._queued.values():
            resource = waiter.item.resource
            queued_by_group[waiter.item.group.key] += 1
            demand_by_resource[resource] += 1
            aggregate_waiters[resource.provider_model_key] += 1
        return RequestQueueView(
            queued_total=len(self._queued),
            queued_by_group=dict(queued_by_group),
            queued_demand_by_resource=dict(demand_by_resource),
            aggregate_provider_model_waiters=dict(aggregate_waiters),
        )

    def _activate_group(self, key: RequestResourceKey) -> None:
        self._purge_queue_head(key)
        queue = self._queues.get(key)
        if not queue or key in self._active_heap_entries:
            return
        self._sequence += 1
        finish = self._group_finish.get(key, self._virtual_time)
        heapq.heappush(self._heap, (finish, self._sequence, key))
        self._active_heap_entries[key] = (finish, self._sequence)

    def _first_valid_waiter(self, key: RequestResourceKey) -> RequestWaiter | None:
        queue = self._queues.get(key)
        if queue is None:
            return None
        for waiter in queue:
            if waiter.waiter_id in self._queued and self._waiter_groups.get(waiter.waiter_id) == key:
                return waiter
        return None

    def _purge_queue_head(self, key: RequestResourceKey) -> None:
        queue = self._queues.get(key)
        if queue is None:
            return
        while queue:
            waiter = queue[0]
            if waiter.waiter_id in self._queued and self._waiter_groups.get(waiter.waiter_id) == key:
                break
            queue.popleft()
