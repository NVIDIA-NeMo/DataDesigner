# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for completion tracking."""

import threading

import pytest

from data_designer.engine.execution_graph.completion import (
    CHECKPOINT_VERSION,
    CompletionTracker,
    ThreadSafeCompletionTracker,
)
from data_designer.engine.execution_graph.graph import ExecutionGraph
from data_designer.engine.execution_graph.node_id import BarrierNodeId, CellNodeId
from data_designer.engine.execution_graph.traits import ExecutionTraits

from .conftest import create_descriptor  # noqa: TID252

# --- CompletionTracker tests ---


def test_completion_tracker_initial_state() -> None:
    tracker = CompletionTracker(num_records=100)
    assert tracker.num_records == 100
    assert len(tracker) == 0


def test_completion_tracker_mark_cell_complete() -> None:
    tracker = CompletionTracker(num_records=10)
    node = CellNodeId(5, "col_a")

    assert not tracker.is_complete(node)
    tracker.mark_complete(node)
    assert tracker.is_complete(node)


def test_completion_tracker_mark_barrier_complete() -> None:
    tracker = CompletionTracker(num_records=10)
    node = BarrierNodeId("col_a")

    assert not tracker.is_barrier_complete("col_a")
    tracker.mark_complete(node)
    assert tracker.is_barrier_complete("col_a")


def test_completion_tracker_contains_operator() -> None:
    tracker = CompletionTracker(num_records=10)
    node = CellNodeId(0, "col_a")

    assert node not in tracker
    tracker.mark_complete(node)
    assert node in tracker


def test_completion_tracker_column_completion_tracking() -> None:
    tracker = CompletionTracker(num_records=3)

    # Complete some cells
    tracker.mark_complete(CellNodeId(0, "col_a"))
    assert tracker.column_completion_count("col_a") == 1
    assert not tracker.is_column_complete("col_a")

    tracker.mark_complete(CellNodeId(1, "col_a"))
    assert tracker.column_completion_count("col_a") == 2
    assert not tracker.is_column_complete("col_a")

    tracker.mark_complete(CellNodeId(2, "col_a"))
    assert tracker.column_completion_count("col_a") == 3
    assert tracker.is_column_complete("col_a")


def test_completion_tracker_memory_compaction() -> None:
    tracker = CompletionTracker(num_records=3)

    # Complete all cells of a column
    for r in range(3):
        tracker.mark_complete(CellNodeId(r, "col_a"))

    # Column should be in completed_columns set, not in partial tracking
    assert "col_a" in tracker._completed_columns
    assert "col_a" not in tracker._column_completion

    # Individual cells should still be queryable
    assert tracker.is_complete(CellNodeId(0, "col_a"))
    assert tracker.is_complete(CellNodeId(1, "col_a"))
    assert tracker.is_complete(CellNodeId(2, "col_a"))


def test_completion_tracker_len_counts_all_completed() -> None:
    tracker = CompletionTracker(num_records=5)

    tracker.mark_complete(CellNodeId(0, "col_a"))
    tracker.mark_complete(CellNodeId(1, "col_a"))
    tracker.mark_complete(BarrierNodeId("col_b"))

    assert len(tracker) == 3


def test_completion_tracker_len_with_compacted_column() -> None:
    tracker = CompletionTracker(num_records=3)

    # Complete entire column
    for r in range(3):
        tracker.mark_complete(CellNodeId(r, "col_a"))

    # Add one cell from another column
    tracker.mark_complete(CellNodeId(0, "col_b"))

    # 3 from col_a + 1 from col_b = 4
    assert len(tracker) == 4


def test_completion_tracker_reset() -> None:
    tracker = CompletionTracker(num_records=3)

    tracker.mark_complete(CellNodeId(0, "col_a"))
    tracker.mark_complete(BarrierNodeId("col_b"))

    tracker.reset()

    assert len(tracker) == 0
    assert not tracker.is_complete(CellNodeId(0, "col_a"))
    assert not tracker.is_barrier_complete("col_b")


def test_completion_tracker_duplicate_mark_complete() -> None:
    tracker = CompletionTracker(num_records=3)
    node = CellNodeId(0, "col_a")

    tracker.mark_complete(node)
    tracker.mark_complete(node)  # Duplicate should be safe

    assert tracker.column_completion_count("col_a") == 1


def test_completion_tracker_mark_complete_after_column_compacted() -> None:
    tracker = CompletionTracker(num_records=3)

    # Complete entire column
    for r in range(3):
        tracker.mark_complete(CellNodeId(r, "col_a"))

    # Try to mark a cell that's already complete
    tracker.mark_complete(CellNodeId(0, "col_a"))  # Should be no-op

    # Column should still be complete
    assert tracker.is_column_complete("col_a")
    assert len(tracker) == 3


# --- ThreadSafeCompletionTracker tests ---


def test_thread_safe_tracker_basic_operations() -> None:
    tracker = ThreadSafeCompletionTracker(num_records=10)

    assert tracker.num_records == 10
    assert len(tracker) == 0

    node = CellNodeId(0, "col_a")
    tracker.mark_complete(node)
    assert tracker.is_complete(node)
    assert node in tracker


def test_thread_safe_tracker_thread_safety() -> None:
    tracker = ThreadSafeCompletionTracker(num_records=1000)
    errors: list[Exception] = []

    def mark_cells(start: int, end: int, column: str) -> None:
        try:
            for r in range(start, end):
                tracker.mark_complete(CellNodeId(r, column))
        except Exception as e:
            errors.append(e)

    # Create threads that mark different ranges
    threads = [
        threading.Thread(target=mark_cells, args=(0, 500, "col_a")),
        threading.Thread(target=mark_cells, args=(500, 1000, "col_a")),
        threading.Thread(target=mark_cells, args=(0, 500, "col_b")),
        threading.Thread(target=mark_cells, args=(500, 1000, "col_b")),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert tracker.is_column_complete("col_a")
    assert tracker.is_column_complete("col_b")
    assert len(tracker) == 2000


def test_thread_safe_tracker_barrier_operations() -> None:
    tracker = ThreadSafeCompletionTracker(num_records=10)

    tracker.mark_complete(BarrierNodeId("barrier_col"))
    assert tracker.is_barrier_complete("barrier_col")


def test_thread_safe_tracker_reset() -> None:
    tracker = ThreadSafeCompletionTracker(num_records=10)

    tracker.mark_complete(CellNodeId(0, "col_a"))
    tracker.reset()

    assert len(tracker) == 0
    assert not tracker.is_complete(CellNodeId(0, "col_a"))


# --- Checkpoint tests ---


def _create_simple_graph(num_records: int = 10) -> ExecutionGraph:
    """Create a simple graph with two columns for checkpoint testing."""
    descriptors = [
        create_descriptor("a", ExecutionTraits.START | ExecutionTraits.ROW_STREAMABLE),
        create_descriptor(
            "b",
            ExecutionTraits.CELL_BY_CELL | ExecutionTraits.ROW_STREAMABLE,
            ["a"],
        ),
    ]
    return ExecutionGraph(num_records=num_records, column_descriptors=descriptors)


def _create_barrier_graph(num_records: int = 10) -> ExecutionGraph:
    """Create a graph with a barrier column for checkpoint testing."""
    descriptors = [
        create_descriptor("a", ExecutionTraits.START | ExecutionTraits.ROW_STREAMABLE),
        create_descriptor("b", ExecutionTraits.BARRIER, ["a"]),
    ]
    return ExecutionGraph(num_records=num_records, column_descriptors=descriptors)


def test_checkpoint_no_completed_rows() -> None:
    graph = _create_simple_graph(num_records=10)
    tracker = CompletionTracker(num_records=10)

    checkpoint = tracker.to_checkpoint(graph)

    assert checkpoint["version"] == CHECKPOINT_VERSION
    assert checkpoint["completed_rows"] == 0


def test_checkpoint_some_completed_rows() -> None:
    graph = _create_simple_graph(num_records=10)
    tracker = CompletionTracker(num_records=10)

    # Complete first 5 rows across all columns
    for row in range(5):
        tracker.mark_complete(CellNodeId(row, "a"))
        tracker.mark_complete(CellNodeId(row, "b"))

    checkpoint = tracker.to_checkpoint(graph)

    assert checkpoint["version"] == CHECKPOINT_VERSION
    assert checkpoint["completed_rows"] == 5


def test_checkpoint_all_completed_rows() -> None:
    graph = _create_simple_graph(num_records=5)
    tracker = CompletionTracker(num_records=5)

    # Complete all rows
    for row in range(5):
        tracker.mark_complete(CellNodeId(row, "a"))
        tracker.mark_complete(CellNodeId(row, "b"))

    checkpoint = tracker.to_checkpoint(graph)

    assert checkpoint["version"] == CHECKPOINT_VERSION
    assert checkpoint["completed_rows"] == 5


def test_checkpoint_partial_row_not_counted() -> None:
    graph = _create_simple_graph(num_records=10)
    tracker = CompletionTracker(num_records=10)

    # Complete first 3 rows fully
    for row in range(3):
        tracker.mark_complete(CellNodeId(row, "a"))
        tracker.mark_complete(CellNodeId(row, "b"))

    # Complete row 3 only in column a (partial row)
    tracker.mark_complete(CellNodeId(3, "a"))

    checkpoint = tracker.to_checkpoint(graph)

    # Only 3 complete rows (row 3 is incomplete)
    assert checkpoint["completed_rows"] == 3


def test_checkpoint_non_contiguous_rows_only_counts_contiguous() -> None:
    graph = _create_simple_graph(num_records=10)
    tracker = CompletionTracker(num_records=10)

    # Complete rows 0, 1, 2, but skip row 3, complete row 4
    for row in [0, 1, 2, 4]:
        tracker.mark_complete(CellNodeId(row, "a"))
        tracker.mark_complete(CellNodeId(row, "b"))

    checkpoint = tracker.to_checkpoint(graph)

    # Only count contiguous rows from 0 (0, 1, 2)
    assert checkpoint["completed_rows"] == 3


def test_from_checkpoint_restores_state() -> None:
    graph = _create_simple_graph(num_records=10)
    checkpoint = {"version": CHECKPOINT_VERSION, "completed_rows": 5}

    tracker = CompletionTracker.from_checkpoint(checkpoint, graph)

    # All cells in rows 0-4 should be complete
    for row in range(5):
        assert tracker.is_complete(CellNodeId(row, "a"))
        assert tracker.is_complete(CellNodeId(row, "b"))

    # Cells in rows 5-9 should not be complete
    for row in range(5, 10):
        assert not tracker.is_complete(CellNodeId(row, "a"))
        assert not tracker.is_complete(CellNodeId(row, "b"))


def test_from_checkpoint_restores_all_complete() -> None:
    graph = _create_simple_graph(num_records=5)
    checkpoint = {"version": CHECKPOINT_VERSION, "completed_rows": 5}

    tracker = CompletionTracker.from_checkpoint(checkpoint, graph)

    # All cells should be complete
    for row in range(5):
        assert tracker.is_complete(CellNodeId(row, "a"))
        assert tracker.is_complete(CellNodeId(row, "b"))

    # Both columns should be marked as fully complete
    assert tracker.is_column_complete("a")
    assert tracker.is_column_complete("b")


def test_from_checkpoint_empty_checkpoint() -> None:
    graph = _create_simple_graph(num_records=10)
    checkpoint = {"version": CHECKPOINT_VERSION, "completed_rows": 0}

    tracker = CompletionTracker.from_checkpoint(checkpoint, graph)

    # No cells should be complete
    for row in range(10):
        assert not tracker.is_complete(CellNodeId(row, "a"))
        assert not tracker.is_complete(CellNodeId(row, "b"))


def test_from_checkpoint_invalid_version() -> None:
    graph = _create_simple_graph(num_records=10)
    checkpoint = {"version": 999, "completed_rows": 5}

    with pytest.raises(ValueError, match="Incompatible checkpoint version"):
        CompletionTracker.from_checkpoint(checkpoint, graph)


def test_checkpoint_roundtrip() -> None:
    graph = _create_simple_graph(num_records=10)
    tracker = CompletionTracker(num_records=10)

    # Complete first 7 rows
    for row in range(7):
        tracker.mark_complete(CellNodeId(row, "a"))
        tracker.mark_complete(CellNodeId(row, "b"))

    # Create checkpoint and restore
    checkpoint = tracker.to_checkpoint(graph)
    restored = CompletionTracker.from_checkpoint(checkpoint, graph)

    # Verify restored state matches original
    for row in range(10):
        for col in ["a", "b"]:
            assert tracker.is_complete(CellNodeId(row, col)) == restored.is_complete(CellNodeId(row, col))


def test_checkpoint_with_barrier_column() -> None:
    graph = _create_barrier_graph(num_records=5)
    tracker = CompletionTracker(num_records=5)

    # Complete all cells in column a
    for row in range(5):
        tracker.mark_complete(CellNodeId(row, "a"))

    # Complete barrier and all cells in column b
    tracker.mark_complete(BarrierNodeId("b"))
    for row in range(5):
        tracker.mark_complete(CellNodeId(row, "b"))

    checkpoint = tracker.to_checkpoint(graph)
    assert checkpoint["completed_rows"] == 5


def test_from_checkpoint_with_barrier_marks_barrier_complete() -> None:
    graph = _create_barrier_graph(num_records=5)
    checkpoint = {"version": CHECKPOINT_VERSION, "completed_rows": 5}

    tracker = CompletionTracker.from_checkpoint(checkpoint, graph)

    # Barrier should be marked complete
    assert tracker.is_barrier_complete("b")

    # All cells should be complete
    for row in range(5):
        assert tracker.is_complete(CellNodeId(row, "a"))
        assert tracker.is_complete(CellNodeId(row, "b"))


def test_checkpoint_barrier_incomplete_means_no_rows_complete() -> None:
    graph = _create_barrier_graph(num_records=5)
    tracker = CompletionTracker(num_records=5)

    # Complete all cells in column a
    for row in range(5):
        tracker.mark_complete(CellNodeId(row, "a"))

    # Barrier not complete, so no rows are "complete" even though column a is done
    checkpoint = tracker.to_checkpoint(graph)
    assert checkpoint["completed_rows"] == 0


# --- ThreadSafeCompletionTracker checkpoint tests ---


def test_thread_safe_checkpoint_no_completed_rows() -> None:
    graph = _create_simple_graph(num_records=10)
    tracker = ThreadSafeCompletionTracker(num_records=10)

    checkpoint = tracker.to_checkpoint(graph)

    assert checkpoint["version"] == CHECKPOINT_VERSION
    assert checkpoint["completed_rows"] == 0


def test_thread_safe_checkpoint_some_completed_rows() -> None:
    graph = _create_simple_graph(num_records=10)
    tracker = ThreadSafeCompletionTracker(num_records=10)

    for row in range(5):
        tracker.mark_complete(CellNodeId(row, "a"))
        tracker.mark_complete(CellNodeId(row, "b"))

    checkpoint = tracker.to_checkpoint(graph)

    assert checkpoint["completed_rows"] == 5


def test_thread_safe_from_checkpoint() -> None:
    graph = _create_simple_graph(num_records=10)
    checkpoint = {"version": CHECKPOINT_VERSION, "completed_rows": 5}

    tracker = ThreadSafeCompletionTracker.from_checkpoint(checkpoint, graph)

    for row in range(5):
        assert tracker.is_complete(CellNodeId(row, "a"))
        assert tracker.is_complete(CellNodeId(row, "b"))

    for row in range(5, 10):
        assert not tracker.is_complete(CellNodeId(row, "a"))
        assert not tracker.is_complete(CellNodeId(row, "b"))


def test_thread_safe_checkpoint_roundtrip() -> None:
    graph = _create_simple_graph(num_records=10)
    tracker = ThreadSafeCompletionTracker(num_records=10)

    for row in range(7):
        tracker.mark_complete(CellNodeId(row, "a"))
        tracker.mark_complete(CellNodeId(row, "b"))

    checkpoint = tracker.to_checkpoint(graph)
    restored = ThreadSafeCompletionTracker.from_checkpoint(checkpoint, graph)

    for row in range(10):
        for col in ["a", "b"]:
            assert tracker.is_complete(CellNodeId(row, col)) == restored.is_complete(CellNodeId(row, col))
