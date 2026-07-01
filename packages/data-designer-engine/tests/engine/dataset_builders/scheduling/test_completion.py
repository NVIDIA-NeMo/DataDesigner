# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import pytest

from data_designer.config.column_configs import (
    ExpressionColumnConfig,
    GenerationStrategy,
    LLMTextColumnConfig,
    SamplerColumnConfig,
)
from data_designer.config.sampler_params import SamplerType
from data_designer.engine.dataset_builders.scheduling.completion import CompletionTracker
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


@pytest.fixture()
def ready_ctx() -> ReadyTasksFixture:
    """CompletionTracker wired to the simple 3-column graph with one row group of size 3."""
    graph = _build_simple_graph()
    return ReadyTasksFixture(
        tracker=CompletionTracker.with_graph(graph, [(0, 3)]),
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


# -- drop_row / is_dropped -------------------------------------------------


def test_drop_row() -> None:
    tracker = CompletionTracker()
    tracker.drop_row(row_group=0, row_index=2)

    assert tracker.is_dropped(0, 2)
    assert not tracker.is_dropped(0, 0)
    assert not tracker.is_dropped(1, 2)


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


# -- ready frontier ---------------------------------------------------------


def test_ready_frontier_starts_empty(ready_ctx: ReadyTasksFixture) -> None:
    ready = ready_ctx.tracker.ready_frontier()
    assert len(ready) == 0


def test_add_root_tasks_populates_frontier(ready_ctx: ReadyTasksFixture) -> None:
    ready_ctx.tracker.add_root_tasks(0, 3, columns=("topic",))
    ready = ready_ctx.tracker.ready_frontier()

    assert len(ready) == 1
    assert ready[0].column == "topic"
    assert ready[0].task_type == "from_scratch"


def test_mark_enqueued_uses_scheduler_stable_task_id(ready_ctx: ReadyTasksFixture) -> None:
    ready_ctx.tracker.add_root_tasks(0, 3, columns=("topic",))
    task = ready_ctx.tracker.ready_frontier()[0]

    ready_ctx.tracker.mark_enqueued({stable_task_id(task)})

    assert ready_ctx.tracker.ready_frontier() == ()


def test_ready_frontier_after_seed_complete(ready_ctx: ReadyTasksFixture) -> None:
    delta = ready_ctx.tracker.mark_row_range_complete("topic", 0, 3)

    ready = ready_ctx.tracker.ready_frontier()

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
    ready = tracker.ready_frontier()
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
    ready = tracker.ready_frontier()
    assert not any(task.column == "judge" and task.row_index == 1 for task in ready)


def test_ready_frontier_skips_dropped_rows(ready_ctx: ReadyTasksFixture) -> None:
    ready_ctx.tracker.mark_row_range_complete("topic", 0, 3)
    removed = Task(column="question", row_group=0, row_index=1, task_type="cell")
    delta = ready_ctx.tracker.drop_row(0, 1)

    ready = ready_ctx.tracker.ready_frontier()

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

    ready = ready_ctx.tracker.ready_frontier()
    score_tasks = [t for t in ready if t.column == "score"]
    assert len(score_tasks) == 1
    assert score_tasks[0].task_type == "batch"
    assert score_tasks[0] in delta.added


def test_ready_frontier_full_column_waits_for_all_cells(ready_ctx: ReadyTasksFixture) -> None:
    ready_ctx.tracker.mark_row_range_complete("topic", 0, 3)
    ready_ctx.tracker.mark_cell_complete("question", 0, 0)
    ready_ctx.tracker.mark_cell_complete("question", 0, 1)
    # question[0,2] not done yet

    ready = ready_ctx.tracker.ready_frontier()

    score_tasks = [t for t in ready if t.column == "score"]
    assert len(score_tasks) == 0


def test_ready_frontier_full_column_ready_when_all_cells_done(ready_ctx: ReadyTasksFixture) -> None:
    ready_ctx.tracker.mark_row_range_complete("topic", 0, 3)
    delta = None
    for ri in range(3):
        delta = ready_ctx.tracker.mark_cell_complete("question", 0, ri)

    ready = ready_ctx.tracker.ready_frontier()

    score_tasks = [t for t in ready if t.column == "score"]
    assert len(score_tasks) == 1
    assert score_tasks[0].task_type == "batch"
    assert delta is not None
    assert delta.added == (score_tasks[0],)


def test_ready_frontier_multiple_row_groups() -> None:
    graph = _build_simple_graph()
    tracker = CompletionTracker.with_graph(graph, [(0, 3), (1, 2)])
    tracker.mark_row_range_complete("topic", 0, 3)
    tracker.mark_row_range_complete("topic", 1, 2)

    ready = tracker.ready_frontier()

    question_tasks = [t for t in ready if t.column == "question"]
    assert len(question_tasks) == 5  # 3 from rg0 + 2 from rg1


def test_frontier_delta_return_is_empty_when_frontier_does_not_change(ready_ctx: ReadyTasksFixture) -> None:
    ready_ctx.tracker.mark_row_range_complete("topic", 0, 3)

    delta = ready_ctx.tracker.mark_row_range_complete("topic", 0, 3)

    assert delta.empty


def test_ready_frontier_skips_already_complete_batch(ready_ctx: ReadyTasksFixture) -> None:
    ready_ctx.tracker.mark_row_range_complete("topic", 0, 3)

    ready = ready_ctx.tracker.ready_frontier()

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
    # Complete the full pipeline
    tracker.mark_row_range_complete("topic", 0, 2)
    tracker.mark_cell_complete("question", 0, 0)
    tracker.mark_cell_complete("question", 0, 1)
    tracker.mark_row_range_complete("score", 0, 2)

    # Fire a late upstream cell event after score is already done
    tracker.mark_cell_complete("question", 0, 0)

    ready = tracker.ready_frontier()
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
    # Complete seed and gen[0] — agg not ready yet
    tracker.mark_row_range_complete("seed", 0, 2)
    tracker.mark_cell_complete("gen", 0, 0)

    ready = tracker.ready_frontier()
    assert not any(t.column == "agg" for t in ready)

    # Complete gen[1] — agg becomes ready
    tracker.mark_cell_complete("gen", 0, 1)
    ready = tracker.ready_frontier()
    assert any(t.column == "agg" for t in ready)

    # Complete agg, then verify it doesn't reappear
    tracker.mark_row_range_complete("agg", 0, 2)
    ready = tracker.ready_frontier()
    assert not any(t.column == "agg" for t in ready)
