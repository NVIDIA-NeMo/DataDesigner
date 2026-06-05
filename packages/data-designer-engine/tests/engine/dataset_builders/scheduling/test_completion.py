# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
from dataclasses import dataclass

import pytest

from data_designer.config.column_configs import (
    ExpressionColumnConfig,
    GenerationStrategy,
    LLMTextColumnConfig,
    SamplerColumnConfig,
)
from data_designer.config.sampler_params import SamplerType
from data_designer.engine.dataset_builders.scheduling.completion import (
    MAX_RELEASED_ROW_GROUP_SUMMARY_RANGES,
    CompletionTracker,
)
from data_designer.engine.dataset_builders.scheduling.resources import stable_task_id
from data_designer.engine.dataset_builders.scheduling.task_model import SliceRef, Task
from data_designer.engine.dataset_builders.utils.execution_graph import ExecutionGraph

MODEL_ALIAS = "stub"


def _build_simple_graph() -> ExecutionGraph:
    """topic (full-column) → question (cell-by-cell) → score (full-column)."""
    configs = [
        SamplerColumnConfig(name="topic", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="question", prompt="{{ topic }}", model_alias=MODEL_ALIAS),
        ExpressionColumnConfig(name="score", expr="{{ question }}"),
    ]
    strategies = {
        "topic": GenerationStrategy.FULL_COLUMN,
        "question": GenerationStrategy.CELL_BY_CELL,
        "score": GenerationStrategy.FULL_COLUMN,
    }
    return ExecutionGraph.create(configs, strategies)


@dataclass
class ReadyTasksFixture:
    tracker: CompletionTracker
    dispatched: set[Task]


@pytest.fixture()
def ready_ctx() -> ReadyTasksFixture:
    """CompletionTracker wired to the simple 3-column graph with one row group of size 3."""
    graph = _build_simple_graph()
    return ReadyTasksFixture(
        tracker=CompletionTracker.with_graph(graph, [(0, 3)]),
        dispatched=set(),
    )


# -- mark_cell_complete / is_complete --------------------------------------


def test_mark_and_check_complete() -> None:
    tracker = CompletionTracker()
    tracker.mark_cell_complete("col_a", row_group=0, row_index=0)

    assert tracker.is_complete(SliceRef("col_a", 0, 0))
    assert not tracker.is_complete(SliceRef("col_a", 0, 1))
    assert not tracker.is_complete(SliceRef("col_a", 1, 0))
    assert not tracker.is_complete(SliceRef("col_b", 0, 0))


def test_mark_row_range_complete() -> None:
    tracker = CompletionTracker()
    tracker.mark_row_range_complete("col_a", row_group=0, row_group_size=3)

    assert tracker.is_complete(SliceRef("col_a", 0, 0))
    assert tracker.is_complete(SliceRef("col_a", 0, 1))
    assert tracker.is_complete(SliceRef("col_a", 0, 2))
    assert not tracker.is_complete(SliceRef("col_a", 0, 3))


def test_mark_row_range_complete_raises_on_size_mismatch(ready_ctx: ReadyTasksFixture) -> None:
    with pytest.raises(ValueError, match="Row-group size mismatch"):
        ready_ctx.tracker.mark_row_range_complete("topic", row_group=0, row_group_size=2)


def test_mark_cell_complete_raises_on_unknown_row_group(ready_ctx: ReadyTasksFixture) -> None:
    with pytest.raises(ValueError, match="Unknown row_group"):
        ready_ctx.tracker.mark_cell_complete("question", row_group=999, row_index=0)


def test_mark_cell_complete_raises_on_out_of_range_row_index(ready_ctx: ReadyTasksFixture) -> None:
    with pytest.raises(ValueError, match="row_index out of range"):
        ready_ctx.tracker.mark_cell_complete("question", row_group=0, row_index=3)


def test_drop_row_raises_on_out_of_range_row_index(ready_ctx: ReadyTasksFixture) -> None:
    with pytest.raises(ValueError, match="row_index out of range"):
        ready_ctx.tracker.drop_row(row_group=0, row_index=3)


# -- is_all_complete -----------------------------------------------------------


def test_all_complete_cell_level() -> None:
    tracker = CompletionTracker()
    tracker.mark_cell_complete("col_a", 0, 0)
    tracker.mark_cell_complete("col_a", 0, 1)

    assert tracker.is_all_complete([SliceRef("col_a", 0, 0), SliceRef("col_a", 0, 1)])
    assert not tracker.is_all_complete([SliceRef("col_a", 0, 0), SliceRef("col_a", 0, 2)])


def test_all_complete_batch_level() -> None:
    tracker = CompletionTracker()
    tracker.mark_row_range_complete("col_a", 0, 3)

    assert tracker.is_all_complete([SliceRef("col_a", 0, None)])


def test_all_complete_batch_single_cell_not_sufficient() -> None:
    """mark_cell_complete on one row must NOT make is_all_complete return True for batch check."""
    tracker = CompletionTracker()
    tracker.mark_cell_complete("col_a", 0, 0)

    assert not tracker.is_all_complete([SliceRef("col_a", 0, None)])


def test_all_complete_batch_not_present() -> None:
    tracker = CompletionTracker()
    assert not tracker.is_all_complete([SliceRef("col_a", 0, None)])


def test_all_complete_empty_list() -> None:
    tracker = CompletionTracker()
    assert tracker.is_all_complete([])


# -- drop_row / is_dropped -------------------------------------------------


def test_drop_row() -> None:
    tracker = CompletionTracker()
    tracker.drop_row(row_group=0, row_index=2)

    assert tracker.is_dropped(0, 2)
    assert not tracker.is_dropped(0, 0)
    assert not tracker.is_dropped(1, 2)
    assert tracker.dropped_row_count(0, 3) == 1
    assert tracker.dropped_row_count(0, 2) == 0


# -- is_row_group_complete --------------------------------------------------


def test_row_group_complete() -> None:
    tracker = CompletionTracker()
    tracker.mark_row_range_complete("col_a", 0, 3)
    tracker.mark_row_range_complete("col_b", 0, 3)

    assert tracker.is_row_group_complete(0, 3, ["col_a", "col_b"])


def test_row_group_incomplete() -> None:
    tracker = CompletionTracker()
    tracker.mark_row_range_complete("col_a", 0, 3)

    assert not tracker.is_row_group_complete(0, 3, ["col_a", "col_b"])


def test_row_group_incomplete_with_out_of_range_cell_completion() -> None:
    tracker = CompletionTracker()
    tracker.mark_cell_complete("col_a", 0, 0)
    tracker.mark_cell_complete("col_a", 0, 1)
    tracker.mark_cell_complete("col_a", 0, 99)

    assert not tracker.is_column_complete_for_rg("col_a", 0)
    assert not tracker.is_row_group_complete(0, 3, ["col_a"])


def test_row_group_incomplete_when_batch_marker_size_mismatches() -> None:
    tracker = CompletionTracker()
    tracker.mark_row_range_complete("col_a", 0, 2)

    assert tracker.is_row_group_complete(0, 2, ["col_a"])
    assert not tracker.is_row_group_complete(0, 3, ["col_a"])


def test_row_group_complete_with_dropped_rows() -> None:
    tracker = CompletionTracker()
    tracker.mark_cell_complete("col_a", 0, 0)
    tracker.mark_cell_complete("col_a", 0, 2)
    tracker.mark_cell_complete("col_b", 0, 0)
    tracker.mark_cell_complete("col_b", 0, 2)
    tracker.drop_row(0, 1)  # row 1 is dropped

    assert tracker.is_row_group_complete(0, 3, ["col_a", "col_b"])


def test_row_group_not_complete_missing_non_dropped() -> None:
    tracker = CompletionTracker()
    tracker.mark_cell_complete("col_a", 0, 0)
    tracker.mark_cell_complete("col_b", 0, 0)
    tracker.drop_row(0, 1)
    # row 2 is not dropped and not complete

    assert not tracker.is_row_group_complete(0, 3, ["col_a", "col_b"])


def test_release_row_group_clears_heavy_state_and_preserves_summary(ready_ctx: ReadyTasksFixture) -> None:
    tracker = ready_ctx.tracker
    tracker.mark_row_range_complete("topic", 0, 3)
    tracker.mark_cell_complete("question", 0, 0)
    tracker.mark_cell_complete("question", 0, 2)
    tracker.drop_row(0, 1)
    tracker.mark_row_range_complete("score", 0, 3)

    tracker.release_row_group(0, 3, ["topic", "question", "score"])

    assert tracker._completed == {}
    assert tracker._dropped == {}
    assert tracker._batch_complete == {}
    assert tracker.ready_frontier() == ()
    assert tracker.is_row_group_complete(0, 3, ["topic", "question", "score"])
    assert tracker.is_dropped(0, 1)
    assert tracker.dropped_row_count(0, 3) == 1
    assert tracker.is_complete(SliceRef("question", 0, 0))
    assert not tracker.is_complete(SliceRef("question", 0, 1))
    assert tracker.is_complete(SliceRef("score", 0, 1))
    assert tracker.is_complete(SliceRef("score", 0, None))
    assert not tracker.is_complete(SliceRef("question", 0, None))


def test_release_row_group_merges_clean_summaries_into_one_range() -> None:
    graph = _build_simple_graph()
    tracker = CompletionTracker.with_graph(graph, [(rg_id, 3) for rg_id in range(4)])

    for row_group in range(4):
        tracker.mark_row_range_complete("topic", row_group, 3)
        for row_index in range(3):
            tracker.mark_cell_complete("question", row_group, row_index)
        tracker.mark_row_range_complete("score", row_group, 3)
        tracker.release_row_group(row_group, 3, ["topic", "question", "score"])

    assert tracker._completed == {}
    assert tracker._dropped == {}
    assert tracker._batch_complete == {}
    assert tracker._released_row_groups == {}
    assert len(tracker._released_range_summaries) == 1
    assert tracker.dropped_row_count(2, 3) == 0
    assert tracker.is_row_group_complete(2, 3, ["topic", "question", "score"])
    assert tracker.is_complete(SliceRef("question", 2, 1))
    assert tracker.is_complete(SliceRef("score", 2, None))

    delta = tracker.add_root_tasks(1, 3)

    assert [task.row_group for task in delta.added] == [1]
    assert [(start, end) for start, end, _released in tracker._released_range_summaries] == [(0, 0), (2, 3)]
    assert not tracker.is_row_group_complete(1, 3, ["topic", "question", "score"])
    assert tracker.is_row_group_complete(2, 3, ["topic", "question", "score"])


def test_release_row_group_preserves_fragmented_clean_ranges_compactly() -> None:
    graph = _build_simple_graph()
    tracker = CompletionTracker.with_graph(graph, [(rg_id, 3) for rg_id in range(8)])

    for row_group in (0, 2, 4, 6):
        tracker.mark_row_range_complete("topic", row_group, 3)
        tracker.mark_row_range_complete("score", row_group, 3)
        tracker.release_row_group(row_group, 3, ["topic", "score"])

    assert [(start, end) for start, end, _released in tracker._released_range_summaries] == [
        (0, 0),
        (2, 2),
        (4, 4),
        (6, 6),
    ]

    tracker.mark_row_range_complete("topic", 7, 3)

    assert [(start, end) for start, end, _released in tracker._released_range_summaries] == [
        (0, 0),
        (2, 2),
        (4, 4),
        (6, 6),
    ]

    tracker.mark_row_range_complete("topic", 1, 3)
    tracker.mark_row_range_complete("score", 1, 3)
    tracker.release_row_group(1, 3, ["topic", "score"])

    assert [(start, end) for start, end, _released in tracker._released_range_summaries] == [
        (0, 2),
        (4, 4),
        (6, 6),
    ]


def test_release_row_group_stores_dropped_rows_as_survivor_ranges_when_smaller() -> None:
    graph = _build_simple_graph()
    tracker = CompletionTracker.with_graph(graph, [(0, 10)])
    tracker.mark_row_range_complete("topic", 0, 10)
    tracker.mark_row_range_complete("score", 0, 10)
    for row_index in range(9):
        tracker.drop_row(0, row_index)

    tracker.release_row_group(0, 10, ["topic", "score"])

    released = tracker._released_row_group(0)
    assert released is not None
    assert released.rows.stores_survivors
    assert released.rows.intervals == ((9, 9),)
    assert tracker._released_row_groups == {}
    assert len(tracker._released_range_summaries) == 1
    assert tracker.dropped_row_count(0, 10) == 9
    assert tracker.is_dropped(0, 0)
    assert not tracker.is_dropped(0, 9)


def test_release_row_group_merges_dropped_summaries_into_one_range() -> None:
    graph = _build_simple_graph()
    tracker = CompletionTracker.with_graph(graph, [(row_group, 3) for row_group in range(4)])

    for row_group in range(4):
        tracker.mark_row_range_complete("topic", row_group, 3)
        tracker.mark_row_range_complete("score", row_group, 3)
        tracker.drop_row(row_group, 1)
        tracker.release_row_group(row_group, 3, ["topic", "score"])

    assert tracker._released_row_groups == {}
    assert [(start, end) for start, end, _released in tracker._released_range_summaries] == [(0, 3)]
    assert tracker.dropped_row_count(2, 3) == 1
    assert tracker.is_dropped(2, 1)
    assert not tracker.is_dropped(2, 0)
    assert tracker.is_row_group_complete(2, 3, ["topic", "score"])

    delta = tracker.add_root_tasks(1, 3)

    assert [task.row_group for task in delta.added] == [1]
    assert [(start, end) for start, end, _released in tracker._released_range_summaries] == [(0, 0), (2, 3)]
    assert not tracker.is_row_group_complete(1, 3, ["topic", "score"])


def test_release_row_group_bounds_fragmented_dropped_row_summary() -> None:
    graph = _build_simple_graph()
    row_group_size = 10_000
    tracker = CompletionTracker.with_graph(graph, [(0, row_group_size)])

    tracker.mark_row_range_complete("topic", 0, row_group_size)
    tracker.mark_row_range_complete("score", 0, row_group_size)
    for row_index in range(0, row_group_size, 2):
        tracker.drop_row(0, row_index)

    tracker.release_row_group(0, row_group_size, ["topic", "score"])

    released = tracker._released_row_group(0)
    assert released is not None
    assert not released.rows.exact
    assert released.rows.intervals == ()
    assert tracker._released_row_groups == {}
    assert len(tracker._released_range_summaries) == 1
    assert tracker.dropped_row_count(0, row_group_size) == row_group_size // 2
    assert tracker.is_row_group_complete(0, row_group_size, ["topic", "score"])
    assert tracker.is_complete(SliceRef("topic", 0, 0))


def test_release_row_group_aggregates_first_fragmented_drop_without_sort(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = _build_simple_graph()
    row_group_size = 1_000_000
    tracker = CompletionTracker.with_graph(graph, [(0, row_group_size)])

    tracker.mark_row_range_complete("topic", 0, row_group_size)
    tracker.mark_row_range_complete("score", 0, row_group_size)
    tracker._dropped[0] = set(range(0, row_group_size, 2))

    def fail_sorted(*_args: object, **_kwargs: object) -> list[object]:
        raise AssertionError("fragmented released rows should aggregate without sorting every dropped row")

    monkeypatch.setattr(builtins, "sorted", fail_sorted)

    tracker.release_row_group(0, row_group_size, ["topic", "score"])

    released = tracker._released_row_group(0)
    assert released is not None
    assert not released.rows.exact
    assert released.rows.intervals == ()
    assert tracker._released_exact_row_interval_count == 0
    assert tracker.dropped_row_count(0, row_group_size) == row_group_size // 2


def test_release_row_group_keeps_large_contiguous_dropped_split_exact() -> None:
    graph = _build_simple_graph()
    row_group_size = 10_000
    tracker = CompletionTracker.with_graph(graph, [(0, row_group_size)])

    tracker.mark_row_range_complete("topic", 0, row_group_size)
    tracker.mark_row_range_complete("score", 0, row_group_size)
    for row_index in range(row_group_size // 2):
        tracker.drop_row(0, row_index)

    tracker.release_row_group(0, row_group_size, ["topic", "score"])

    released = tracker._released_row_group(0)
    assert released is not None
    assert released.rows.exact
    assert released.rows.intervals == ((0, 4_999),)
    assert tracker.dropped_row_count(0, row_group_size) == row_group_size // 2
    assert tracker.is_dropped(0, 0)
    assert tracker.is_dropped(0, 4_999)
    assert not tracker.is_dropped(0, 5_000)


def test_release_row_group_bounds_exact_dropped_row_summaries_across_run() -> None:
    graph = _build_simple_graph()
    row_group_size = 8_192
    tracker = CompletionTracker.with_graph(graph, [(0, row_group_size), (1, row_group_size)])

    for row_group in (0, 1):
        tracker.mark_row_range_complete("topic", row_group, row_group_size)
        tracker.mark_row_range_complete("score", row_group, row_group_size)
        for row_index in range(0, row_group_size, 2):
            tracker.drop_row(row_group, row_index)
        for row_index in range(1, row_group_size, 2):
            tracker.mark_cell_complete("question", row_group, row_index)
        tracker.release_row_group(row_group, row_group_size, ["topic", "question", "score"])

    first = tracker._released_row_group(0)
    second = tracker._released_row_group(1)
    assert first is not None
    assert second is not None
    assert first.rows.exact
    assert len(first.rows.intervals) == 4_096
    assert not second.rows.exact
    assert second.rows.intervals == ()
    assert tracker.dropped_row_count(0, row_group_size) == row_group_size // 2
    assert tracker.dropped_row_count(1, row_group_size) == row_group_size // 2
    assert tracker.is_row_group_complete(1, row_group_size, ["topic", "question", "score"])
    assert tracker.is_complete(SliceRef("question", 1, 1))


def test_release_row_group_bounds_alternating_summary_ranges() -> None:
    graph = _build_simple_graph()
    row_group_count = MAX_RELEASED_ROW_GROUP_SUMMARY_RANGES + 16
    tracker = CompletionTracker.with_graph(graph, [(row_group, 8) for row_group in range(row_group_count)])

    for row_group in range(row_group_count):
        tracker.mark_row_range_complete("topic", row_group, 8)
        tracker.mark_row_range_complete("score", row_group, 8)
        tracker.drop_row(row_group, row_group % 8)
        tracker.release_row_group(row_group, 8, ["topic", "score"])

    assert len(tracker._released_range_summaries) == MAX_RELEASED_ROW_GROUP_SUMMARY_RANGES
    assert tracker._released_row_group(0) is None
    assert tracker.is_row_group_complete(row_group_count - 1, 8, ["topic", "score"])
    assert tracker.dropped_row_count(row_group_count - 1, 8) == 1


def test_reopened_released_row_groups_keep_summary_ranges_bounded() -> None:
    graph = _build_simple_graph()
    row_group_count = MAX_RELEASED_ROW_GROUP_SUMMARY_RANGES * 2 + 1
    tracker = CompletionTracker.with_graph(graph, [(row_group, 3) for row_group in range(row_group_count)])

    for row_group in range(row_group_count):
        tracker.mark_row_range_complete("topic", row_group, 3)
        tracker.mark_row_range_complete("score", row_group, 3)
        tracker.release_row_group(row_group, 3, ["topic", "score"])

    assert len(tracker._released_range_summaries) == 1

    for row_group in range(1, row_group_count, 2):
        tracker.add_root_tasks(row_group, 3)

    assert len(tracker._released_range_summaries) <= MAX_RELEASED_ROW_GROUP_SUMMARY_RANGES


# -- get_ready_tasks --------------------------------------------------------


def test_get_ready_tasks_frontier_empty_without_seed(ready_ctx: ReadyTasksFixture) -> None:
    """Frontier starts empty - seed_frontier() must be called explicitly."""
    ready = ready_ctx.tracker.get_ready_tasks(ready_ctx.dispatched)
    assert len(ready) == 0


def test_get_ready_tasks_seed_frontier(ready_ctx: ReadyTasksFixture) -> None:
    """seed_frontier() populates the frontier with root tasks."""
    ready_ctx.tracker.seed_frontier()
    ready = ready_ctx.tracker.get_ready_tasks(ready_ctx.dispatched)

    assert len(ready) == 1
    assert ready[0].column == "topic"
    assert ready[0].task_type == "from_scratch"


def test_mark_enqueued_uses_scheduler_stable_task_id(ready_ctx: ReadyTasksFixture) -> None:
    ready_ctx.tracker.seed_frontier()
    task = ready_ctx.tracker.ready_frontier()[0]

    ready_ctx.tracker.mark_enqueued({stable_task_id(task)})

    assert ready_ctx.tracker.ready_frontier() == ()


def test_get_ready_tasks_after_seed_complete(ready_ctx: ReadyTasksFixture) -> None:
    delta = ready_ctx.tracker.mark_row_range_complete("topic", 0, 3)

    ready = ready_ctx.tracker.get_ready_tasks(ready_ctx.dispatched)

    question_tasks = [t for t in ready if t.column == "question"]
    assert len(question_tasks) == 3
    assert all(t.task_type == "cell" for t in question_tasks)
    assert {t.row_index for t in question_tasks} == {0, 1, 2}
    assert set(delta.added) == set(question_tasks)
    assert delta.removed == ()


def test_fan_out_cell_completion_readies_all_children_for_same_row() -> None:
    configs = [
        SamplerColumnConfig(name="topic", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="heavy", prompt="{{ topic }}", model_alias=MODEL_ALIAS),
        LLMTextColumnConfig(name="child_a", prompt="{{ heavy }}", model_alias=MODEL_ALIAS),
        LLMTextColumnConfig(name="child_b", prompt="{{ heavy }}", model_alias=MODEL_ALIAS),
        LLMTextColumnConfig(name="child_c", prompt="{{ heavy }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {config.name: GenerationStrategy.CELL_BY_CELL for config in configs[1:]}
    strategies["topic"] = GenerationStrategy.FULL_COLUMN
    graph = ExecutionGraph.create(configs, strategies)
    tracker = CompletionTracker.with_graph(graph, [(0, 2)])
    tracker.mark_row_range_complete("topic", 0, 2)

    delta = tracker.mark_cell_complete("heavy", 0, 0)

    assert {task.column for task in delta.added} == {"child_a", "child_b", "child_c"}
    assert {task.row_index for task in delta.added} == {0}
    ready = tracker.get_ready_tasks(set())
    assert not any(task.column.startswith("child_") and task.row_index == 1 for task in ready)


def test_fan_in_cell_downstream_waits_for_all_same_row_upstreams() -> None:
    configs = [
        SamplerColumnConfig(name="topic", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="up_a", prompt="{{ topic }}", model_alias=MODEL_ALIAS),
        LLMTextColumnConfig(name="up_b", prompt="{{ topic }}", model_alias=MODEL_ALIAS),
        LLMTextColumnConfig(name="up_c", prompt="{{ topic }}", model_alias=MODEL_ALIAS),
        LLMTextColumnConfig(name="judge", prompt="{{ up_a }} {{ up_b }} {{ up_c }}", model_alias=MODEL_ALIAS),
    ]
    strategies = {config.name: GenerationStrategy.CELL_BY_CELL for config in configs[1:]}
    strategies["topic"] = GenerationStrategy.FULL_COLUMN
    graph = ExecutionGraph.create(configs, strategies)
    tracker = CompletionTracker.with_graph(graph, [(0, 2)])
    tracker.mark_row_range_complete("topic", 0, 2)

    first_delta = tracker.mark_cell_complete("up_a", 0, 0)
    second_delta = tracker.mark_cell_complete("up_b", 0, 0)
    final_delta = tracker.mark_cell_complete("up_c", 0, 0)

    assert not any(task.column == "judge" for task in first_delta.added)
    assert not any(task.column == "judge" for task in second_delta.added)
    assert final_delta.added == (Task(column="judge", row_group=0, row_index=0, task_type="cell"),)
    ready = tracker.get_ready_tasks(set())
    assert not any(task.column == "judge" and task.row_index == 1 for task in ready)


def test_get_ready_tasks_skips_dispatched(ready_ctx: ReadyTasksFixture) -> None:
    ready_ctx.tracker.mark_row_range_complete("topic", 0, 3)

    ready1 = ready_ctx.tracker.get_ready_tasks(ready_ctx.dispatched)
    ready_ctx.dispatched.update(ready1)

    ready2 = ready_ctx.tracker.get_ready_tasks(ready_ctx.dispatched)
    assert len(ready2) == 0


def test_get_ready_tasks_skips_dropped_rows(ready_ctx: ReadyTasksFixture) -> None:
    ready_ctx.tracker.mark_row_range_complete("topic", 0, 3)
    removed = Task(column="question", row_group=0, row_index=1, task_type="cell")
    delta = ready_ctx.tracker.drop_row(0, 1)

    ready = ready_ctx.tracker.get_ready_tasks(ready_ctx.dispatched)

    question_tasks = [t for t in ready if t.column == "question"]
    assert len(question_tasks) == 2
    assert {t.row_index for t in question_tasks} == {0, 2}
    assert delta.added == ()
    assert delta.removed == (removed,)


def test_drop_row_unblocks_full_column_downstream(ready_ctx: ReadyTasksFixture) -> None:
    """Dropping the last incomplete CELL_BY_CELL row should make downstream FULL_COLUMN ready."""
    ready_ctx.tracker.mark_row_range_complete("topic", 0, 3)
    ready_ctx.tracker.mark_cell_complete("question", 0, 0)
    ready_ctx.tracker.mark_cell_complete("question", 0, 1)
    # question[2] never completes -- drop it instead
    delta = ready_ctx.tracker.drop_row(0, 2)

    ready = ready_ctx.tracker.get_ready_tasks(ready_ctx.dispatched)
    score_tasks = [t for t in ready if t.column == "score"]
    assert len(score_tasks) == 1
    assert score_tasks[0].task_type == "batch"
    assert score_tasks[0] in delta.added


def test_cached_remaining_cell_count_updates_after_drop(ready_ctx: ReadyTasksFixture) -> None:
    ready_ctx.tracker.mark_row_range_complete("topic", 0, 3)
    assert not ready_ctx.tracker.is_row_group_complete(0, 3, ["topic", "question", "score"])
    assert ready_ctx.tracker._remaining_cell_rows[0]["question"] == 3

    ready_ctx.tracker.mark_cell_complete("question", 0, 0)
    ready_ctx.tracker.mark_cell_complete("question", 0, 1)
    delta = ready_ctx.tracker.drop_row(0, 2)

    assert ready_ctx.tracker._remaining_cell_rows[0]["question"] == 0
    assert Task(column="score", row_group=0, row_index=None, task_type="batch") in delta.added


def test_get_ready_tasks_full_column_waits_for_all_cells(ready_ctx: ReadyTasksFixture) -> None:
    ready_ctx.tracker.mark_row_range_complete("topic", 0, 3)
    ready_ctx.tracker.mark_cell_complete("question", 0, 0)
    ready_ctx.tracker.mark_cell_complete("question", 0, 1)
    # question[0,2] not done yet

    ready = ready_ctx.tracker.get_ready_tasks(ready_ctx.dispatched)

    score_tasks = [t for t in ready if t.column == "score"]
    assert len(score_tasks) == 0


def test_get_ready_tasks_full_column_ready_when_all_cells_done(ready_ctx: ReadyTasksFixture) -> None:
    ready_ctx.tracker.mark_row_range_complete("topic", 0, 3)
    delta = None
    for ri in range(3):
        delta = ready_ctx.tracker.mark_cell_complete("question", 0, ri)

    ready = ready_ctx.tracker.get_ready_tasks(ready_ctx.dispatched)

    score_tasks = [t for t in ready if t.column == "score"]
    assert len(score_tasks) == 1
    assert score_tasks[0].task_type == "batch"
    assert delta is not None
    assert delta.added == (score_tasks[0],)


def test_cached_remaining_cell_count_updates_after_completion(ready_ctx: ReadyTasksFixture) -> None:
    ready_ctx.tracker.mark_row_range_complete("topic", 0, 3)
    assert not ready_ctx.tracker.is_column_complete_for_rg("question", 0)
    assert ready_ctx.tracker._remaining_cell_rows[0]["question"] == 3

    delta = None
    for ri in range(3):
        delta = ready_ctx.tracker.mark_cell_complete("question", 0, ri)

    assert ready_ctx.tracker._remaining_cell_rows[0]["question"] == 0
    assert delta is not None
    assert delta.added == (Task(column="score", row_group=0, row_index=None, task_type="batch"),)


def test_get_ready_tasks_multiple_row_groups() -> None:
    graph = _build_simple_graph()
    tracker = CompletionTracker.with_graph(graph, [(0, 3), (1, 2)])
    dispatched: set[Task] = set()

    tracker.mark_row_range_complete("topic", 0, 3)
    tracker.mark_row_range_complete("topic", 1, 2)

    ready = tracker.get_ready_tasks(dispatched)

    question_tasks = [t for t in ready if t.column == "question"]
    assert len(question_tasks) == 5  # 3 from rg0 + 2 from rg1


def test_frontier_delta_return_is_empty_when_frontier_does_not_change(ready_ctx: ReadyTasksFixture) -> None:
    ready_ctx.tracker.mark_row_range_complete("topic", 0, 3)

    delta = ready_ctx.tracker.mark_row_range_complete("topic", 0, 3)

    assert delta.empty


def test_get_ready_tasks_skips_already_complete_batch(ready_ctx: ReadyTasksFixture) -> None:
    ready_ctx.tracker.mark_row_range_complete("topic", 0, 3)

    ready = ready_ctx.tracker.get_ready_tasks(ready_ctx.dispatched)

    topic_tasks = [t for t in ready if t.column == "topic"]
    assert len(topic_tasks) == 0


# -- Strategy-safe completion API ------------------------------------------


def test_mark_cell_complete_raises_for_full_column_strategy(ready_ctx: ReadyTasksFixture) -> None:
    with pytest.raises(ValueError, match="mark_cell_complete.*requires cell_by_cell.*full_column"):
        ready_ctx.tracker.mark_cell_complete("topic", row_group=0, row_index=0)


def test_mark_row_range_complete_raises_for_cell_by_cell_strategy(ready_ctx: ReadyTasksFixture) -> None:
    ready_ctx.tracker.mark_row_range_complete("topic", 0, 3)
    with pytest.raises(ValueError, match="mark_row_range_complete.*requires full_column.*cell_by_cell"):
        ready_ctx.tracker.mark_row_range_complete("question", row_group=0, row_group_size=3)


# -- Re-enqueue regression tests -------------------------------------------


def test_completed_cell_not_reenqueued_after_later_upstream() -> None:
    """A → B → C chain: completing C then firing a late upstream event must not re-enqueue C."""
    graph = _build_simple_graph()
    tracker = CompletionTracker.with_graph(graph, [(0, 2)])
    dispatched: set[Task] = set()

    # Complete the full pipeline
    tracker.mark_row_range_complete("topic", 0, 2)
    tracker.mark_cell_complete("question", 0, 0)
    tracker.mark_cell_complete("question", 0, 1)
    tracker.mark_row_range_complete("score", 0, 2)

    # Fire a late upstream cell event after score is already done
    tracker.mark_cell_complete("question", 0, 0)

    ready = tracker.get_ready_tasks(dispatched)
    score_tasks = [t for t in ready if t.column == "score"]
    assert len(score_tasks) == 0


def test_completed_batch_not_reenqueued_by_upstream_cell() -> None:
    """After a FULL_COLUMN downstream is completed, a late cell upstream event must not re-add it."""
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="gen", prompt="{{ seed }}", model_alias=MODEL_ALIAS),
        ExpressionColumnConfig(name="agg", expr="{{ gen }}"),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "gen": GenerationStrategy.CELL_BY_CELL,
        "agg": GenerationStrategy.FULL_COLUMN,
    }
    graph = ExecutionGraph.create(configs, strategies)
    tracker = CompletionTracker.with_graph(graph, [(0, 2)])
    dispatched: set[Task] = set()

    # Complete seed and gen[0] — agg not ready yet
    tracker.mark_row_range_complete("seed", 0, 2)
    tracker.mark_cell_complete("gen", 0, 0)

    ready = tracker.get_ready_tasks(dispatched)
    assert not any(t.column == "agg" for t in ready)

    # Complete gen[1] — agg becomes ready
    tracker.mark_cell_complete("gen", 0, 1)
    ready = tracker.get_ready_tasks(dispatched)
    assert any(t.column == "agg" for t in ready)

    # Complete agg, then verify it doesn't reappear
    tracker.mark_row_range_complete("agg", 0, 2)
    ready = tracker.get_ready_tasks(dispatched)
    assert not any(t.column == "agg" for t in ready)
