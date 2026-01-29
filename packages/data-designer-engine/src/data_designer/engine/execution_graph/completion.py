# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Completion tracking for execution graph nodes.

This module provides memory-efficient completion trackers for tracking which
nodes have completed in an execution graph. Two variants are provided:

- CompletionTracker: Simple, no locks, for asyncio/single-threaded use
- ThreadSafeCompletionTracker: Thread-safe variant with internal locking

Both trackers use O(C) memory instead of O(C×R) by tracking fully completed
columns as sets and only storing partial progress for in-progress columns.
"""

from __future__ import annotations

import threading

from data_designer.engine.execution_graph.node_id import BarrierNodeId, CellNodeId, NodeId


class CompletionTracker:
    """Memory-efficient completion tracking for large datasets.

    Instead of storing every completed NodeId (O(C×R) memory), this tracker:
    - Tracks fully completed columns as a set of names (O(C))
    - Only stores partial completion progress for in-progress columns
    - Automatically compacts when columns fully complete

    This tracker is NOT thread-safe. Use ThreadSafeCompletionTracker for
    concurrent access from multiple threads.

    Example:
        >>> tracker = CompletionTracker(num_records=1000)
        >>> tracker.mark_complete(CellNodeId(0, "col_a"))
        >>> CellNodeId(0, "col_a") in tracker
        True
        >>> tracker.is_column_complete("col_a")
        False
    """

    def __init__(self, num_records: int) -> None:
        """Initialize the completion tracker.

        Args:
            num_records: The total number of records in the dataset.
        """
        self._num_records = num_records
        self._completed_columns: set[str] = set()
        self._completed_barriers: set[str] = set()
        self._column_completion: dict[str, set[int]] = {}

    @property
    def num_records(self) -> int:
        """The total number of records in the dataset."""
        return self._num_records

    def mark_complete(self, node: NodeId) -> None:
        """Mark a node as completed.

        Args:
            node: The node to mark as complete.
        """
        if isinstance(node, BarrierNodeId):
            self._completed_barriers.add(node.column)
        elif isinstance(node, CellNodeId):
            # Skip if column is already fully complete
            if node.column in self._completed_columns:
                return

            progress = self._column_completion.setdefault(node.column, set())
            progress.add(node.row)

            # Check if column is now fully complete
            if len(progress) == self._num_records:
                self._completed_columns.add(node.column)
                # Remove partial progress to save memory
                del self._column_completion[node.column]

    def is_complete(self, node: NodeId) -> bool:
        """Check if a node is completed.

        Args:
            node: The node to check.

        Returns:
            True if the node is completed, False otherwise.
        """
        if isinstance(node, BarrierNodeId):
            return node.column in self._completed_barriers
        elif isinstance(node, CellNodeId):
            if node.column in self._completed_columns:
                return True
            progress = self._column_completion.get(node.column, set())
            return node.row in progress
        return False

    def is_column_complete(self, column: str) -> bool:
        """Check if all cells of a column are complete.

        Args:
            column: The column name to check.

        Returns:
            True if all cells of the column are complete, False otherwise.
        """
        return column in self._completed_columns

    def is_barrier_complete(self, column: str) -> bool:
        """Check if a barrier is complete.

        Args:
            column: The column name of the barrier.

        Returns:
            True if the barrier is complete, False otherwise.
        """
        return column in self._completed_barriers

    def column_completion_count(self, column: str) -> int:
        """Get the number of completed cells for a column.

        Args:
            column: The column name.

        Returns:
            The number of completed cells.
        """
        if column in self._completed_columns:
            return self._num_records
        return len(self._column_completion.get(column, set()))

    def reset(self) -> None:
        """Reset the tracker to its initial state."""
        self._completed_columns.clear()
        self._completed_barriers.clear()
        self._column_completion.clear()

    def __contains__(self, node: NodeId) -> bool:
        """Support `node in tracker` syntax.

        Args:
            node: The node to check.

        Returns:
            True if the node is completed.
        """
        return self.is_complete(node)

    def __len__(self) -> int:
        """Return the total number of completed nodes.

        Note: This is O(C) where C is the number of in-progress columns,
        not O(C×R) since we compact fully completed columns.
        """
        completed_cells = len(self._completed_columns) * self._num_records + sum(
            len(progress) for progress in self._column_completion.values()
        )
        completed_barriers = len(self._completed_barriers)
        return completed_cells + completed_barriers


class ThreadSafeCompletionTracker:
    """Thread-safe completion tracker for concurrent access.

    This tracker wraps a CompletionTracker with a lock to ensure thread-safe
    access when marking nodes complete from multiple threads concurrently.

    Use this tracker when using thread pool executors or other multi-threaded
    execution environments.

    Example:
        >>> tracker = ThreadSafeCompletionTracker(num_records=1000)
        >>> # Safe to call from multiple threads
        >>> tracker.mark_complete(CellNodeId(0, "col_a"))
    """

    def __init__(self, num_records: int) -> None:
        """Initialize the thread-safe completion tracker.

        Args:
            num_records: The total number of records in the dataset.
        """
        self._tracker = CompletionTracker(num_records)
        self._lock = threading.Lock()

    @property
    def num_records(self) -> int:
        """The total number of records in the dataset."""
        return self._tracker.num_records

    def mark_complete(self, node: NodeId) -> None:
        """Mark a node as completed (thread-safe).

        Args:
            node: The node to mark as complete.
        """
        with self._lock:
            self._tracker.mark_complete(node)

    def is_complete(self, node: NodeId) -> bool:
        """Check if a node is completed (thread-safe).

        Args:
            node: The node to check.

        Returns:
            True if the node is completed, False otherwise.
        """
        with self._lock:
            return self._tracker.is_complete(node)

    def is_column_complete(self, column: str) -> bool:
        """Check if all cells of a column are complete (thread-safe).

        Args:
            column: The column name to check.

        Returns:
            True if all cells of the column are complete, False otherwise.
        """
        with self._lock:
            return self._tracker.is_column_complete(column)

    def is_barrier_complete(self, column: str) -> bool:
        """Check if a barrier is complete (thread-safe).

        Args:
            column: The column name of the barrier.

        Returns:
            True if the barrier is complete, False otherwise.
        """
        with self._lock:
            return self._tracker.is_barrier_complete(column)

    def column_completion_count(self, column: str) -> int:
        """Get the number of completed cells for a column (thread-safe).

        Args:
            column: The column name.

        Returns:
            The number of completed cells.
        """
        with self._lock:
            return self._tracker.column_completion_count(column)

    def reset(self) -> None:
        """Reset the tracker to its initial state (thread-safe)."""
        with self._lock:
            self._tracker.reset()

    def __contains__(self, node: NodeId) -> bool:
        """Support `node in tracker` syntax (thread-safe).

        Args:
            node: The node to check.

        Returns:
            True if the node is completed.
        """
        return self.is_complete(node)

    def __len__(self) -> int:
        """Return the total number of completed nodes (thread-safe)."""
        with self._lock:
            return len(self._tracker)
