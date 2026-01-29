# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for completion tracking."""

import threading

from data_designer.engine.execution_graph.completion import (
    CompletionTracker,
    ThreadSafeCompletionTracker,
)
from data_designer.engine.execution_graph.node_id import BarrierNodeId, CellNodeId

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
