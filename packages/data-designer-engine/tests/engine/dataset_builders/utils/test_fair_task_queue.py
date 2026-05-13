# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import Counter

from data_designer.engine.dataset_builders.utils.fair_task_queue import (
    FairTaskSelector,
    TaskGroupKey,
    TaskGroupSpec,
)
from data_designer.engine.dataset_builders.utils.task_model import Task


def _task(column: str, row_index: int) -> Task:
    return Task(column=column, row_group=0, row_index=row_index, task_type="cell")


def _group(name: str, *, weight: float = 1.0, admitted_limit: int | None = None) -> TaskGroupSpec:
    return TaskGroupSpec(
        key=TaskGroupKey(kind="local", identity=(name,)),
        weight=weight,
        admitted_limit=admitted_limit,
    )


def test_fair_task_selector_equal_groups_round_robins() -> None:
    selector = FairTaskSelector()
    selector.sync_ready(
        (task, _group(task.column))
        for task in [
            _task("a", 0),
            _task("a", 1),
            _task("b", 0),
            _task("b", 1),
            _task("c", 0),
            _task("c", 1),
        ]
    )

    selected = [selector.pop_next(lambda _key: True) for _ in range(6)]

    assert [selection.task.column for selection in selected if selection is not None] == ["a", "b", "c", "a", "b", "c"]


def test_fair_task_selector_weighted_groups() -> None:
    selector = FairTaskSelector()
    selector.sync_ready(
        [
            (task, _group(task.column, weight=2 if task.column == "a" else 1))
            for task in [_task("a", i) for i in range(6)]
        ]
        + [(_task("b", i), _group("b", weight=1)) for i in range(6)]
    )

    selected = [selector.pop_next(lambda _key: True) for _ in range(6)]
    counts = Counter(selection.task.column for selection in selected if selection is not None)

    assert counts == {"a": 4, "b": 2}


def test_fair_task_selector_prunes_stale_ready_tasks() -> None:
    selector = FairTaskSelector()
    stale = _task("a", 0)
    fresh = _task("a", 1)

    selector.sync_ready([(stale, _group("a")), (fresh, _group("a"))])
    selector.sync_ready([(fresh, _group("a"))])

    selected = selector.pop_next(lambda _key: True)

    assert selected is not None
    assert selected.task == fresh
    assert selector.pop_next(lambda _key: True) is None


def test_fair_task_selector_admitted_cap_skips_capped_group() -> None:
    selector = FairTaskSelector()
    capped = _group("a", admitted_limit=1)
    open_group = _group("b", admitted_limit=1)
    selector.sync_ready([(_task("a", 0), capped), (_task("b", 0), open_group)])

    selected = selector.pop_next(lambda key: key != capped.key)

    assert selected is not None
    assert selected.task.column == "b"


def test_fair_task_selector_returns_none_when_all_groups_capped() -> None:
    selector = FairTaskSelector()
    group = _group("a", admitted_limit=1)
    selector.sync_ready([(_task("a", 0), group)])

    assert selector.pop_next(lambda _key: False) is None
    assert selector.has_queued_tasks is True


def test_fair_task_selector_no_duplicate_on_repeated_sync_ready() -> None:
    selector = FairTaskSelector()
    task = _task("a", 0)

    selector.sync_ready([(task, _group("a"))])
    selector.sync_ready([(task, _group("a"))])
    first = selector.pop_next(lambda _key: True)

    assert first is not None
    assert first.task == task
    assert selector.pop_next(lambda _key: True) is None
