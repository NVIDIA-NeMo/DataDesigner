# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# Helpers to build minimal graphs without real column configs
from data_designer.config.column_configs import (
    ExpressionColumnConfig,
    GenerationStrategy,
    LLMTextColumnConfig,
    SamplerColumnConfig,
)
from data_designer.config.sampler_params import SamplerType
from data_designer.engine.dataset_builders.utils.completion_tracker import CompletionTracker
from data_designer.engine.dataset_builders.utils.execution_graph import ExecutionGraph, build_execution_graph
from data_designer.engine.dataset_builders.utils.task_model import Task

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
    return build_execution_graph(configs, strategies)


# -- mark_complete / is_complete -------------------------------------------


def test_mark_and_check_complete() -> None:
    tracker = CompletionTracker()
    tracker.mark_complete("col_a", row_group=0, row_index=0)

    assert tracker.is_complete("col_a", 0, 0)
    assert not tracker.is_complete("col_a", 0, 1)
    assert not tracker.is_complete("col_a", 1, 0)
    assert not tracker.is_complete("col_b", 0, 0)


def test_mark_batch_complete() -> None:
    tracker = CompletionTracker()
    tracker.mark_batch_complete("col_a", row_group=0, row_group_size=3)

    assert tracker.is_complete("col_a", 0, 0)
    assert tracker.is_complete("col_a", 0, 1)
    assert tracker.is_complete("col_a", 0, 2)
    assert not tracker.is_complete("col_a", 0, 3)


# -- all_complete -----------------------------------------------------------


def test_all_complete_cell_level() -> None:
    tracker = CompletionTracker()
    tracker.mark_complete("col_a", 0, 0)
    tracker.mark_complete("col_a", 0, 1)

    assert tracker.all_complete([("col_a", 0, 0), ("col_a", 0, 1)])
    assert not tracker.all_complete([("col_a", 0, 0), ("col_a", 0, 2)])


def test_all_complete_batch_level() -> None:
    tracker = CompletionTracker()
    tracker.mark_batch_complete("col_a", 0, 3)

    assert tracker.all_complete([("col_a", 0, None)])


def test_all_complete_batch_not_present() -> None:
    tracker = CompletionTracker()
    assert not tracker.all_complete([("col_a", 0, None)])


def test_all_complete_empty_list() -> None:
    tracker = CompletionTracker()
    assert tracker.all_complete([])


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
    tracker.mark_batch_complete("col_a", 0, 3)
    tracker.mark_batch_complete("col_b", 0, 3)

    assert tracker.is_row_group_complete(0, 3, ["col_a", "col_b"])


def test_row_group_incomplete() -> None:
    tracker = CompletionTracker()
    tracker.mark_batch_complete("col_a", 0, 3)

    assert not tracker.is_row_group_complete(0, 3, ["col_a", "col_b"])


def test_row_group_complete_with_dropped_rows() -> None:
    tracker = CompletionTracker()
    tracker.mark_complete("col_a", 0, 0)
    tracker.mark_complete("col_a", 0, 2)
    tracker.mark_complete("col_b", 0, 0)
    tracker.mark_complete("col_b", 0, 2)
    tracker.drop_row(0, 1)  # row 1 is dropped

    assert tracker.is_row_group_complete(0, 3, ["col_a", "col_b"])


def test_row_group_not_complete_missing_non_dropped() -> None:
    tracker = CompletionTracker()
    tracker.mark_complete("col_a", 0, 0)
    tracker.mark_complete("col_b", 0, 0)
    tracker.drop_row(0, 1)
    # row 2 is not dropped and not complete

    assert not tracker.is_row_group_complete(0, 3, ["col_a", "col_b"])


# -- get_ready_tasks --------------------------------------------------------


def test_get_ready_tasks_seeds_first() -> None:
    graph = _build_simple_graph()
    tracker = CompletionTracker()
    dispatched: set[Task] = set()

    ready = tracker.get_ready_tasks(graph, [(0, 3)], dispatched)

    # Only the seed column should be ready (no upstream)
    assert len(ready) == 1
    assert ready[0].column == "topic"
    assert ready[0].task_type == "batch"


def test_get_ready_tasks_after_seed_complete() -> None:
    graph = _build_simple_graph()
    tracker = CompletionTracker()
    dispatched: set[Task] = set()

    tracker.mark_batch_complete("topic", 0, 3)

    ready = tracker.get_ready_tasks(graph, [(0, 3)], dispatched)

    # All question cells should be ready (topic is done)
    question_tasks = [t for t in ready if t.column == "question"]
    assert len(question_tasks) == 3
    assert all(t.task_type == "cell" for t in question_tasks)
    assert {t.row_index for t in question_tasks} == {0, 1, 2}


def test_get_ready_tasks_skips_dispatched() -> None:
    graph = _build_simple_graph()
    tracker = CompletionTracker()
    dispatched: set[Task] = set()

    tracker.mark_batch_complete("topic", 0, 3)

    ready1 = tracker.get_ready_tasks(graph, [(0, 3)], dispatched)
    dispatched.update(ready1)

    ready2 = tracker.get_ready_tasks(graph, [(0, 3)], dispatched)
    assert len(ready2) == 0


def test_get_ready_tasks_skips_dropped_rows() -> None:
    graph = _build_simple_graph()
    tracker = CompletionTracker()
    dispatched: set[Task] = set()

    tracker.mark_batch_complete("topic", 0, 3)
    tracker.drop_row(0, 1)

    ready = tracker.get_ready_tasks(graph, [(0, 3)], dispatched)

    question_tasks = [t for t in ready if t.column == "question"]
    assert len(question_tasks) == 2
    assert {t.row_index for t in question_tasks} == {0, 2}


def test_get_ready_tasks_full_column_waits_for_all_cells() -> None:
    graph = _build_simple_graph()
    tracker = CompletionTracker()
    dispatched: set[Task] = set()

    tracker.mark_batch_complete("topic", 0, 3)
    tracker.mark_complete("question", 0, 0)
    tracker.mark_complete("question", 0, 1)
    # question[0,2] not done yet

    ready = tracker.get_ready_tasks(graph, [(0, 3)], dispatched)

    score_tasks = [t for t in ready if t.column == "score"]
    assert len(score_tasks) == 0  # score waits for all question rows


def test_get_ready_tasks_full_column_ready_when_all_cells_done() -> None:
    graph = _build_simple_graph()
    tracker = CompletionTracker()
    dispatched: set[Task] = set()

    tracker.mark_batch_complete("topic", 0, 3)
    for ri in range(3):
        tracker.mark_complete("question", 0, ri)

    ready = tracker.get_ready_tasks(graph, [(0, 3)], dispatched)

    score_tasks = [t for t in ready if t.column == "score"]
    assert len(score_tasks) == 1
    assert score_tasks[0].task_type == "batch"


def test_get_ready_tasks_multiple_row_groups() -> None:
    graph = _build_simple_graph()
    tracker = CompletionTracker()
    dispatched: set[Task] = set()

    # Both row groups have topic done
    tracker.mark_batch_complete("topic", 0, 3)
    tracker.mark_batch_complete("topic", 1, 2)

    ready = tracker.get_ready_tasks(graph, [(0, 3), (1, 2)], dispatched)

    question_tasks = [t for t in ready if t.column == "question"]
    assert len(question_tasks) == 5  # 3 from rg0 + 2 from rg1


def test_get_ready_tasks_skips_already_complete_batch() -> None:
    graph = _build_simple_graph()
    tracker = CompletionTracker()
    dispatched: set[Task] = set()

    tracker.mark_batch_complete("topic", 0, 3)

    ready = tracker.get_ready_tasks(graph, [(0, 3)], dispatched)

    # topic is already complete, should not be in ready tasks
    topic_tasks = [t for t in ready if t.column == "topic"]
    assert len(topic_tasks) == 0
