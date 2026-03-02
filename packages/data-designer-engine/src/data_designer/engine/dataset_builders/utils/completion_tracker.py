# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from data_designer.config.column_configs import GenerationStrategy
from data_designer.engine.dataset_builders.utils.task_model import ColumnName, RowGroupIndex, RowIndex, Task

if TYPE_CHECKING:
    from data_designer.engine.dataset_builders.utils.execution_graph import ExecutionGraph


class CompletionTracker:
    """Tracks which (column, row_group, row_index) tuples are done.

    All access is from the single asyncio event loop thread — no locks needed.
    Row indices are local to their row group (0-based).

    When *graph* and *row_groups* are provided, an event-driven frontier is
    maintained so that ``get_ready_tasks`` returns in O(frontier) instead of
    scanning all columns × rows × row groups.
    """

    def __init__(
        self,
        graph: ExecutionGraph | None = None,
        row_groups: list[tuple[RowGroupIndex, int]] | None = None,
    ) -> None:
        # row_group → column → set of completed local row indices
        self._completed: dict[RowGroupIndex, dict[ColumnName, set[RowIndex]]] = defaultdict(lambda: defaultdict(set))
        # row_group → set of dropped row indices
        self._dropped: dict[RowGroupIndex, set[RowIndex]] = defaultdict(set)

        self._graph = graph
        self._row_group_sizes: dict[RowGroupIndex, int] = {}
        self._frontier: set[Task] = set()

        if graph is not None and row_groups is not None:
            self._row_group_sizes = {rg_id: size for rg_id, size in row_groups}
            self._seed_frontier()

    def _seed_frontier(self) -> None:
        """Populate the frontier with root tasks (columns with no upstream deps)."""
        assert self._graph is not None
        for col in self._graph.topological_order():
            if self._graph.upstream(col):
                continue
            strategy = self._graph.strategy(col)
            for rg_id, rg_size in self._row_group_sizes.items():
                if strategy == GenerationStrategy.CELL_BY_CELL:
                    for ri in range(rg_size):
                        self._frontier.add(Task(column=col, row_group=rg_id, row_index=ri, task_type="cell"))
                else:
                    self._frontier.add(Task(column=col, row_group=rg_id, row_index=None, task_type="batch"))

    def mark_complete(self, column: ColumnName, row_group: RowGroupIndex, row_index: RowIndex) -> None:
        self._completed[row_group][column].add(row_index)
        if self._graph is not None:
            self._frontier.discard(Task(column=column, row_group=row_group, row_index=row_index, task_type="cell"))
            self._enqueue_downstream(column, row_group, row_index=row_index)

    def mark_batch_complete(self, column: ColumnName, row_group: RowGroupIndex, row_group_size: int) -> None:
        self._completed[row_group][column] = set(range(row_group_size))
        if self._graph is not None:
            self._frontier.discard(Task(column=column, row_group=row_group, row_index=None, task_type="batch"))
            self._enqueue_downstream(column, row_group, row_index=None)

    def _enqueue_downstream(self, column: ColumnName, row_group: RowGroupIndex, row_index: RowIndex | None) -> None:
        """Add newly-ready downstream tasks to the frontier."""
        assert self._graph is not None
        rg_completed = self._completed.get(row_group, {})
        rg_dropped = self._dropped.get(row_group, set())
        rg_size = self._row_group_sizes[row_group]

        for down in self._graph.downstream(column):
            batch_ups, cell_ups = self._graph.upstream_by_strategy(down)

            # All batch upstreams must be present in completed dict
            if any(up not in rg_completed for up in batch_ups):
                continue

            down_strategy = self._graph.strategy(down)

            if down_strategy == GenerationStrategy.CELL_BY_CELL:
                cell_up_completed = [rg_completed.get(up, set()) for up in cell_ups]
                if row_index is not None:
                    # Cell completion: only check the same row
                    if row_index not in rg_dropped and all(row_index in s for s in cell_up_completed):
                        task = Task(column=down, row_group=row_group, row_index=row_index, task_type="cell")
                        self._frontier.add(task)
                else:
                    # Batch completion: check all non-dropped, non-complete rows
                    down_completed = rg_completed.get(down, set())
                    for ri in range(rg_size):
                        if ri in rg_dropped or ri in down_completed:
                            continue
                        if all(ri in s for s in cell_up_completed):
                            task = Task(column=down, row_group=row_group, row_index=ri, task_type="cell")
                            self._frontier.add(task)
            else:
                # FULL_COLUMN downstream: ready when all cell upstreams are fully complete
                if self._are_cell_ups_complete(cell_ups, rg_completed, rg_size, rg_dropped):
                    task = Task(column=down, row_group=row_group, row_index=None, task_type="batch")
                    self._frontier.add(task)

    def is_complete(self, column: ColumnName, row_group: RowGroupIndex, row_index: RowIndex) -> bool:
        return row_index in self._completed.get(row_group, {}).get(column, set())

    def is_all_complete(self, cells: list[tuple[ColumnName, RowGroupIndex, RowIndex | None]]) -> bool:
        """Check whether all the given (column, row_group, row_index) tuples are done.

        A ``row_index`` of ``None`` means the entire batch for that column must
        be complete (i.e., that column key must exist in the row group's dict).
        """
        for col, rg, ri in cells:
            if ri is None:
                if col not in self._completed.get(rg, {}):
                    return False
            elif not self.is_complete(col, rg, ri):
                return False
        return True

    def drop_row(self, row_group: RowGroupIndex, row_index: RowIndex) -> None:
        self._dropped[row_group].add(row_index)
        if self._graph is not None:
            # Remove cell tasks for this row from the frontier
            for col in self._graph.columns:
                self._frontier.discard(Task(column=col, row_group=row_group, row_index=row_index, task_type="cell"))
            # Dropping a row may unblock batch downstream tasks
            self._reevaluate_batch_tasks(row_group)

    def _reevaluate_batch_tasks(self, row_group: RowGroupIndex) -> None:
        """Check if any batch tasks became ready after a row was dropped."""
        assert self._graph is not None
        rg_completed = self._completed.get(row_group, {})
        rg_dropped = self._dropped.get(row_group, set())
        rg_size = self._row_group_sizes[row_group]

        for col in self._graph.topological_order():
            if self._graph.strategy(col) != GenerationStrategy.FULL_COLUMN:
                continue
            if col in rg_completed:
                continue
            batch_ups, cell_ups = self._graph.upstream_by_strategy(col)
            if any(up not in rg_completed for up in batch_ups):
                continue
            if self._are_cell_ups_complete(cell_ups, rg_completed, rg_size, rg_dropped):
                task = Task(column=col, row_group=row_group, row_index=None, task_type="batch")
                self._frontier.add(task)

    def is_dropped(self, row_group: RowGroupIndex, row_index: RowIndex) -> bool:
        return row_index in self._dropped.get(row_group, set())

    def is_row_group_complete(
        self,
        row_group: RowGroupIndex,
        row_group_size: int,
        all_columns: list[ColumnName],
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

    def get_ready_tasks(self, dispatched: set[Task]) -> list[Task]:
        """Return all currently dispatchable tasks from the frontier.

        Excludes already-dispatched/in-flight tasks.
        """
        return [t for t in self._frontier if t not in dispatched]

    def _are_cell_ups_complete(
        self,
        cell_ups: list[ColumnName],
        rg_completed: dict[ColumnName, set[RowIndex]],
        rg_size: int,
        rg_dropped: set[RowIndex],
    ) -> bool:
        """Check all non-dropped rows are complete for each cell-by-cell upstream column."""
        for up in cell_ups:
            up_completed = rg_completed.get(up, set())
            for ri in range(rg_size):
                if ri not in rg_dropped and ri not in up_completed:
                    return False
        return True
