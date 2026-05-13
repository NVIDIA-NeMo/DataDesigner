# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from data_designer.config.column_configs import GenerationStrategy
from data_designer.engine.dataset_builders.utils.task_model import SliceRef, Task

if TYPE_CHECKING:
    from data_designer.engine.dataset_builders.utils.execution_graph import ExecutionGraph


@dataclass(frozen=True)
class FrontierDelta:
    """Tasks added to or removed from the ready frontier by a tracker mutation."""

    added: tuple[Task, ...] = ()
    removed: tuple[Task, ...] = ()

    @property
    def empty(self) -> bool:
        return not self.added and not self.removed


class CompletionTracker:
    """Tracks which cells (column, row_group, row_index) are done.

    Row indices are local to their row group (0-based).

    Use ``with_graph`` to create a frontier-enabled tracker where
    ``get_ready_tasks`` returns in O(frontier) instead of scanning all
    columns x rows x row groups.
    """

    def __init__(self) -> None:
        # row_group → column → set of completed local row indices
        self._completed: dict[int, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
        # row_group → set of dropped row indices
        self._dropped: dict[int, set[int]] = defaultdict(set)

        self._graph: ExecutionGraph | None = None
        self._row_group_sizes: dict[int, int] = {}
        self._batch_complete: dict[int, set[str]] = defaultdict(set)
        self._frontier: set[Task] = set()
        self._pending_delta_added: list[Task] = []
        self._pending_delta_removed: list[Task] = []

    @classmethod
    def with_graph(cls, graph: ExecutionGraph, row_groups: list[tuple[int, int]]) -> CompletionTracker:
        """Create a frontier-enabled tracker backed by an execution graph."""
        tracker = cls()
        tracker._graph = graph
        tracker._row_group_sizes = dict(row_groups)
        return tracker

    def mark_cell_complete(self, column: str, row_group: int, row_index: int) -> FrontierDelta:
        self._validate_row_group(row_group)
        self._validate_strategy(column, GenerationStrategy.CELL_BY_CELL, "mark_cell_complete")
        self._completed[row_group][column].add(row_index)
        removed: list[Task] = []
        added: list[Task] = []
        if self._graph is not None:
            task = Task(column=column, row_group=row_group, row_index=row_index, task_type="cell")
            if self._discard_frontier_task(task):
                removed.append(task)
            added.extend(self._enqueue_downstream(column, row_group, row_index=row_index))
        return self._record_delta(added=added, removed=removed)

    def mark_row_range_complete(self, column: str, row_group: int, row_group_size: int) -> FrontierDelta:
        expected = self._validate_row_group(row_group)
        self._validate_strategy(column, GenerationStrategy.FULL_COLUMN, "mark_row_range_complete")
        if expected is not None and row_group_size != expected:
            raise ValueError(f"Row-group size mismatch for rg={row_group}: got {row_group_size}, expected {expected}")
        self._completed[row_group][column] = set(range(row_group_size))
        self._batch_complete[row_group].add(column)
        removed: list[Task] = []
        added: list[Task] = []
        if self._graph is not None:
            task = Task(column=column, row_group=row_group, row_index=None, task_type="batch")
            if self._discard_frontier_task(task):
                removed.append(task)
            added.extend(self._enqueue_downstream(column, row_group, row_index=None))
        return self._record_delta(added=added, removed=removed)

    def is_complete(self, ref: SliceRef) -> bool:
        return ref.row_index in self._completed.get(ref.row_group, {}).get(ref.column, set())

    def is_all_complete(self, cells: list[SliceRef]) -> bool:
        """Check whether all the given cells are done.

        A ``row_index`` of ``None`` means the entire batch for that column must
        have been completed via ``mark_row_range_complete``.
        """
        for ref in cells:
            if ref.row_index is None:
                if ref.column not in self._batch_complete.get(ref.row_group, set()):
                    return False
            elif not self.is_complete(ref):
                return False
        return True

    def is_column_complete_for_rg(self, column: str, row_group_index: int) -> bool:
        """Check if *column* has been fully completed for *row_group_index*."""
        if column in self._batch_complete.get(row_group_index, set()):
            return True
        rg_size = self._row_group_sizes.get(row_group_index, 0)
        if rg_size == 0:
            return False
        completed = self._completed.get(row_group_index, {}).get(column, set())
        dropped = self._dropped.get(row_group_index, set())
        return all(ri in completed or ri in dropped for ri in range(rg_size))

    def drop_row(self, row_group: int, row_index: int) -> FrontierDelta:
        self._validate_row_group(row_group)
        self._dropped[row_group].add(row_index)
        removed: list[Task] = []
        added: list[Task] = []
        if self._graph is not None:
            # Remove cell tasks for this row from the frontier
            for col in self._graph.columns:
                task = Task(column=col, row_group=row_group, row_index=row_index, task_type="cell")
                if self._discard_frontier_task(task):
                    removed.append(task)
            # Dropping a row may unblock batch downstream tasks
            added.extend(self._reevaluate_batch_tasks(row_group))
        return self._record_delta(added=added, removed=removed)

    def is_dropped(self, row_group: int, row_index: int) -> bool:
        return row_index in self._dropped.get(row_group, set())

    def is_row_group_complete(
        self,
        row_group: int,
        row_group_size: int,
        all_columns: list[str],
    ) -> bool:
        """All non-dropped rows have all columns done."""
        dropped = self._dropped.get(row_group, set())
        completed = self._completed.get(row_group, {})
        for ri in range(row_group_size):
            if ri in dropped:
                continue
            for col in all_columns:
                if ri not in completed.get(col, set()):
                    return False
        return True

    def get_ready_tasks(self, dispatched: set[Task], admitted_rgs: set[int] | None = None) -> list[Task]:
        """Return all currently dispatchable tasks from the frontier.

        Excludes already-dispatched/in-flight tasks and tasks for row groups
        not yet admitted (if ``admitted_rgs`` is provided).
        """
        return [
            t for t in self._frontier if t not in dispatched and (admitted_rgs is None or t.row_group in admitted_rgs)
        ]

    def consume_frontier_delta(self) -> FrontierDelta:
        """Return and clear frontier changes recorded since the previous consume."""
        delta = FrontierDelta(
            added=tuple(self._pending_delta_added),
            removed=tuple(self._pending_delta_removed),
        )
        self._pending_delta_added.clear()
        self._pending_delta_removed.clear()
        return delta

    def is_frontier_task(self, task: Task) -> bool:
        """Return whether *task* is still in the ready frontier."""
        return task in self._frontier

    def seed_frontier(self) -> None:
        """Populate the frontier with root tasks (columns with no upstream deps).

        Not called automatically - the scheduler manages root dispatch directly
        to handle stateful locks and multi-column dedup. Call this explicitly
        for static introspection (e.g., capacity planning, task enumeration).
        """
        if self._graph is None:
            raise RuntimeError("This method requires a graph to be set.")
        for col in self._graph.get_root_columns():
            strategy = self._graph.get_strategy(col)
            for rg_id, rg_size in self._row_group_sizes.items():
                if strategy == GenerationStrategy.CELL_BY_CELL:
                    for ri in range(rg_size):
                        self._frontier.add(Task(column=col, row_group=rg_id, row_index=ri, task_type="cell"))
                else:
                    self._frontier.add(Task(column=col, row_group=rg_id, row_index=None, task_type="batch"))

    def _record_delta(self, *, added: list[Task], removed: list[Task]) -> FrontierDelta:
        self._pending_delta_added.extend(added)
        self._pending_delta_removed.extend(removed)
        return FrontierDelta(added=tuple(added), removed=tuple(removed))

    def _add_frontier_task(self, task: Task) -> bool:
        if task in self._frontier:
            return False
        self._frontier.add(task)
        return True

    def _discard_frontier_task(self, task: Task) -> bool:
        if task not in self._frontier:
            return False
        self._frontier.remove(task)
        return True

    def _enqueue_downstream(self, column: str, row_group: int, row_index: int | None) -> list[Task]:
        """Add newly-ready downstream tasks to the frontier."""
        if self._graph is None:
            raise RuntimeError("This method requires a graph to be set.")
        added: list[Task] = []
        rg_completed = self._completed.get(row_group, {})
        rg_dropped = self._dropped.get(row_group, set())
        rg_batch_complete = self._batch_complete.get(row_group, set())
        rg_size = self._row_group_sizes[row_group]

        for down in self._graph.get_downstream_columns(column):
            batch_ups, cell_ups = self._graph.split_upstream_by_strategy(down)

            if any(up not in rg_batch_complete for up in batch_ups):
                continue

            down_strategy = self._graph.get_strategy(down)

            if down_strategy == GenerationStrategy.CELL_BY_CELL:
                cell_up_completed = [rg_completed.get(up, set()) for up in cell_ups]
                if row_index is not None:
                    # Cell completion: only check the same row
                    down_completed = rg_completed.get(down, set())
                    if (
                        row_index not in rg_dropped
                        and row_index not in down_completed
                        and all(row_index in s for s in cell_up_completed)
                    ):
                        task = Task(column=down, row_group=row_group, row_index=row_index, task_type="cell")
                        if self._add_frontier_task(task):
                            added.append(task)
                else:
                    # Batch completion: check all non-dropped, non-complete rows
                    down_completed = rg_completed.get(down, set())
                    for ri in range(rg_size):
                        if ri in rg_dropped or ri in down_completed:
                            continue
                        if all(ri in s for s in cell_up_completed):
                            task = Task(column=down, row_group=row_group, row_index=ri, task_type="cell")
                            if self._add_frontier_task(task):
                                added.append(task)
            else:
                # FULL_COLUMN downstream: ready when all cell upstreams are fully complete
                if down not in rg_batch_complete and self._are_cell_ups_complete(
                    cell_ups, rg_completed, rg_size, rg_dropped
                ):
                    task = Task(column=down, row_group=row_group, row_index=None, task_type="batch")
                    if self._add_frontier_task(task):
                        added.append(task)
        return added

    def _reevaluate_batch_tasks(self, row_group: int) -> list[Task]:
        """Check if any batch tasks became ready after a row was dropped."""
        if self._graph is None:
            raise RuntimeError("This method requires a graph to be set.")
        added: list[Task] = []
        rg_completed = self._completed.get(row_group, {})
        rg_dropped = self._dropped.get(row_group, set())
        rg_batch_complete = self._batch_complete.get(row_group, set())
        rg_size = self._row_group_sizes[row_group]

        for col in self._graph.get_topological_order():
            if self._graph.get_strategy(col) != GenerationStrategy.FULL_COLUMN:
                continue
            if col in rg_batch_complete:
                continue
            batch_ups, cell_ups = self._graph.split_upstream_by_strategy(col)
            if any(up not in rg_batch_complete for up in batch_ups):
                continue
            if self._are_cell_ups_complete(cell_ups, rg_completed, rg_size, rg_dropped):
                task = Task(column=col, row_group=row_group, row_index=None, task_type="batch")
                if self._add_frontier_task(task):
                    added.append(task)
        return added

    def _are_cell_ups_complete(
        self,
        cell_ups: list[str],
        rg_completed: dict[str, set[int]],
        rg_size: int,
        rg_dropped: set[int],
    ) -> bool:
        """Check all non-dropped rows are complete for each cell-by-cell upstream column."""
        for up in cell_ups:
            up_completed = rg_completed.get(up, set())
            for ri in range(rg_size):
                if ri not in rg_dropped and ri not in up_completed:
                    return False
        return True

    def _validate_strategy(self, column: str, expected: GenerationStrategy, method: str) -> None:
        """Validate that *column* matches the expected strategy in graph-enabled mode."""
        if self._graph is None:
            return
        actual = self._graph.get_strategy(column)
        if actual != expected:
            raise ValueError(f"{method}() requires {expected.value} strategy, but column '{column}' has {actual.value}")

    def _validate_row_group(self, row_group: int) -> int | None:
        """Validate row-group id in graph-enabled mode and return its expected size."""
        if self._graph is None:
            return None
        expected = self._row_group_sizes.get(row_group)
        if expected is None:
            known = sorted(self._row_group_sizes)
            raise ValueError(f"Unknown row_group {row_group}. Known row_groups: {known}")
        return expected
