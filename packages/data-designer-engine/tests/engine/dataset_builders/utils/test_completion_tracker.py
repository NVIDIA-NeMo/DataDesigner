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
from data_designer.engine.dataset_builders.utils.completion_tracker import CompletionTracker
from data_designer.engine.dataset_builders.utils.execution_graph import ExecutionGraph
from data_designer.engine.dataset_builders.utils.task_model import CellRef, Task

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
        tracker=CompletionTracker(graph, [(0, 3)]),
        dispatched=set(),
    )


def test_tracker_requires_row_groups_with_graph() -> None:
    graph = _build_simple_graph()
    with pytest.raises(ValueError, match="provided together"):
        CompletionTracker(graph=graph, row_groups=None)


def test_tracker_requires_graph_with_row_groups() -> None:
    with pytest.raises(ValueError, match="provided together"):
        CompletionTracker(graph=None, row_groups=[(0, 3)])


# -- mark_cell_complete / is_complete --------------------------------------


def test_mark_and_check_complete() -> None:
    tracker = CompletionTracker()
    tracker.mark_cell_complete("col_a", row_group=0, row_index=0)

    assert tracker.is_complete(CellRef("col_a", 0, 0))
    assert not tracker.is_complete(CellRef("col_a", 0, 1))
    assert not tracker.is_complete(CellRef("col_a", 1, 0))
    assert not tracker.is_complete(CellRef("col_b", 0, 0))


def test_mark_row_range_complete() -> None:
    tracker = CompletionTracker()
    tracker.mark_row_range_complete("col_a", row_group=0, row_group_size=3)

    assert tracker.is_complete(CellRef("col_a", 0, 0))
    assert tracker.is_complete(CellRef("col_a", 0, 1))
    assert tracker.is_complete(CellRef("col_a", 0, 2))
    assert not tracker.is_complete(CellRef("col_a", 0, 3))


def test_mark_row_range_complete_raises_on_size_mismatch(ready_ctx: ReadyTasksFixture) -> None:
    with pytest.raises(ValueError, match="Row-group size mismatch"):
        ready_ctx.tracker.mark_row_range_complete("topic", row_group=0, row_group_size=2)


def test_mark_cell_complete_raises_on_unknown_row_group(ready_ctx: ReadyTasksFixture) -> None:
    with pytest.raises(ValueError, match="Unknown row_group"):
        ready_ctx.tracker.mark_cell_complete("question", row_group=999, row_index=0)


# -- is_all_complete -----------------------------------------------------------


def test_all_complete_cell_level() -> None:
    tracker = CompletionTracker()
    tracker.mark_cell_complete("col_a", 0, 0)
    tracker.mark_cell_complete("col_a", 0, 1)

    assert tracker.is_all_complete([CellRef("col_a", 0, 0), CellRef("col_a", 0, 1)])
    assert not tracker.is_all_complete([CellRef("col_a", 0, 0), CellRef("col_a", 0, 2)])


def test_all_complete_batch_level() -> None:
    tracker = CompletionTracker()
    tracker.mark_row_range_complete("col_a", 0, 3)

    assert tracker.is_all_complete([CellRef("col_a", 0, None)])


def test_all_complete_batch_single_cell_not_sufficient() -> None:
    """mark_cell_complete on one row must NOT make is_all_complete return True for batch check."""
    tracker = CompletionTracker()
    tracker.mark_cell_complete("col_a", 0, 0)

    assert not tracker.is_all_complete([CellRef("col_a", 0, None)])


def test_all_complete_batch_not_present() -> None:
    tracker = CompletionTracker()
    assert not tracker.is_all_complete([CellRef("col_a", 0, None)])


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


# -- get_ready_tasks --------------------------------------------------------


def test_get_ready_tasks_seeds_first(ready_ctx: ReadyTasksFixture) -> None:
    ready = ready_ctx.tracker.get_ready_tasks(ready_ctx.dispatched)

    assert len(ready) == 1
    assert ready[0].column == "topic"
    assert ready[0].task_type == "batch"


def test_get_ready_tasks_after_seed_complete(ready_ctx: ReadyTasksFixture) -> None:
    ready_ctx.tracker.mark_row_range_complete("topic", 0, 3)

    ready = ready_ctx.tracker.get_ready_tasks(ready_ctx.dispatched)

    question_tasks = [t for t in ready if t.column == "question"]
    assert len(question_tasks) == 3
    assert all(t.task_type == "cell" for t in question_tasks)
    assert {t.row_index for t in question_tasks} == {0, 1, 2}


def test_get_ready_tasks_skips_dispatched(ready_ctx: ReadyTasksFixture) -> None:
    ready_ctx.tracker.mark_row_range_complete("topic", 0, 3)

    ready1 = ready_ctx.tracker.get_ready_tasks(ready_ctx.dispatched)
    ready_ctx.dispatched.update(ready1)

    ready2 = ready_ctx.tracker.get_ready_tasks(ready_ctx.dispatched)
    assert len(ready2) == 0


def test_get_ready_tasks_skips_dropped_rows(ready_ctx: ReadyTasksFixture) -> None:
    ready_ctx.tracker.mark_row_range_complete("topic", 0, 3)
    ready_ctx.tracker.drop_row(0, 1)

    ready = ready_ctx.tracker.get_ready_tasks(ready_ctx.dispatched)

    question_tasks = [t for t in ready if t.column == "question"]
    assert len(question_tasks) == 2
    assert {t.row_index for t in question_tasks} == {0, 2}


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
    for ri in range(3):
        ready_ctx.tracker.mark_cell_complete("question", 0, ri)

    ready = ready_ctx.tracker.get_ready_tasks(ready_ctx.dispatched)

    score_tasks = [t for t in ready if t.column == "score"]
    assert len(score_tasks) == 1
    assert score_tasks[0].task_type == "batch"


def test_get_ready_tasks_multiple_row_groups() -> None:
    graph = _build_simple_graph()
    tracker = CompletionTracker(graph, [(0, 3), (1, 2)])
    dispatched: set[Task] = set()

    tracker.mark_row_range_complete("topic", 0, 3)
    tracker.mark_row_range_complete("topic", 1, 2)

    ready = tracker.get_ready_tasks(dispatched)

    question_tasks = [t for t in ready if t.column == "question"]
    assert len(question_tasks) == 5  # 3 from rg0 + 2 from rg1


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


def test_completed_cell_not_reenqueued_after_later_upstream(ready_ctx: ReadyTasksFixture) -> None:
    """A → B → C chain: completing C[row=0] then completing another A upstream must not re-enqueue C[row=0]."""
    graph = _build_simple_graph()
    tracker = CompletionTracker(graph, [(0, 2)])
    dispatched: set[Task] = set()

    # Complete the full pipeline for row 0
    tracker.mark_row_range_complete("topic", 0, 2)
    tracker.mark_cell_complete("question", 0, 0)
    tracker.mark_cell_complete("question", 0, 1)

    # score should now be ready
    ready = tracker.get_ready_tasks(dispatched)
    score_tasks = [t for t in ready if t.column == "score"]
    assert len(score_tasks) == 1

    # Complete score, then re-complete an upstream cell — score must not reappear
    tracker.mark_row_range_complete("score", 0, 2)

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
    tracker = CompletionTracker(graph, [(0, 2)])
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
