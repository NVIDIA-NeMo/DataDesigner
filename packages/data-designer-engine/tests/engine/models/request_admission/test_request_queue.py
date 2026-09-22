# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import Counter

import pytest

import data_designer.engine.models.request_admission.queue as queue_module
from data_designer.engine.models.request_admission.queue import RequestFairQueue, RequestWaiter
from data_designer.engine.models.request_admission.resources import (
    RequestAdmissionItem,
    RequestDomain,
    RequestGroupSpec,
    RequestResourceKey,
)


def _waiter(group: str, index: int, *, weight: float = 1.0) -> RequestWaiter:
    resource = RequestResourceKey("nvidia", group, RequestDomain.CHAT)
    return RequestWaiter(
        waiter_id=f"{group}-{index}",
        item=RequestAdmissionItem(resource=resource, group=RequestGroupSpec(resource, weight=weight)),
        enqueued_at=float(index),
    )


def _select_and_commit(queue: RequestFairQueue) -> RequestWaiter | None:
    selection = queue.select_next(lambda _waiter, _view: True)
    if selection is None:
        return None
    return queue.commit(selection)


def test_request_fair_queue_equal_groups_round_robin() -> None:
    queue = RequestFairQueue()
    for group in ["a", "b", "c"]:
        for index in range(2):
            assert queue.enqueue(_waiter(group, index))

    selected = [_select_and_commit(queue) for _ in range(6)]

    assert [waiter.item.group.key.model_id for waiter in selected if waiter is not None] == [
        "a",
        "b",
        "c",
        "a",
        "b",
        "c",
    ]


def test_request_fair_queue_weighted_groups() -> None:
    queue = RequestFairQueue()
    for index in range(6):
        assert queue.enqueue(_waiter("a", index, weight=2.0))
        assert queue.enqueue(_waiter("b", index))

    selected = [_select_and_commit(queue) for _ in range(6)]
    counts = Counter(waiter.item.group.key.model_id for waiter in selected if waiter is not None)

    assert counts == {"a": 4, "b": 2}


@pytest.mark.parametrize(("removed_index", "remaining_indexes"), [(0, [1, 2]), (1, [0, 2])])
def test_remove_keeps_group_active_when_waiters_remain(
    removed_index: int,
    remaining_indexes: list[int],
) -> None:
    queue = RequestFairQueue()
    waiters = [_waiter("group", index) for index in range(3)]
    for waiter in waiters:
        assert queue.enqueue(waiter)

    assert queue.remove(waiters[removed_index].waiter_id) is waiters[removed_index]
    selected = [_select_and_commit(queue) for _ in remaining_indexes]

    assert selected == [waiters[index] for index in remaining_indexes]


def test_repeated_group_activation_does_not_increase_selection_work(monkeypatch: pytest.MonkeyPatch) -> None:
    queue = RequestFairQueue()
    for index in range(100):
        waiter = _waiter("group", index)
        assert queue.enqueue(waiter)
        assert _select_and_commit(queue) is waiter

    waiter = _waiter("group", 100)
    assert queue.enqueue(waiter)
    heappop_calls = 0
    original_heappop = queue_module.heapq.heappop

    def counting_heappop(
        heap: list[tuple[float, int, RequestResourceKey]],
    ) -> tuple[float, int, RequestResourceKey]:
        nonlocal heappop_calls
        heappop_calls += 1
        return original_heappop(heap)

    monkeypatch.setattr(queue_module.heapq, "heappop", counting_heappop)

    selection = queue.select_next(lambda _waiter, _view: True)

    assert selection is not None
    assert selection.waiter is waiter
    assert heappop_calls == 1


def test_removed_groups_do_not_increase_selection_work(monkeypatch: pytest.MonkeyPatch) -> None:
    queue = RequestFairQueue()
    for index in range(100):
        waiter = _waiter(f"removed-{index}", index)
        assert queue.enqueue(waiter)
        assert queue.remove(waiter.waiter_id) is waiter

    waiter = _waiter("active", 100)
    assert queue.enqueue(waiter)
    heappop_calls = 0
    original_heappop = queue_module.heapq.heappop

    def counting_heappop(
        heap: list[tuple[float, int, RequestResourceKey]],
    ) -> tuple[float, int, RequestResourceKey]:
        nonlocal heappop_calls
        heappop_calls += 1
        return original_heappop(heap)

    monkeypatch.setattr(queue_module.heapq, "heappop", counting_heappop)

    selection = queue.select_next(lambda _waiter, _view: True)

    assert selection is not None
    assert selection.waiter is waiter
    assert heappop_calls == 1
