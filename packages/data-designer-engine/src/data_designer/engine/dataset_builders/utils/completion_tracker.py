# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from data_designer.config.column_configs import GenerationStrategy
from data_designer.engine.dataset_builders.utils.task_model import Task

if TYPE_CHECKING:
    from data_designer.engine.dataset_builders.utils.execution_graph import ExecutionGraph


class CompletionTracker:
    """Tracks which (column, row_group, row_index) tuples are done.

    All access is from the single asyncio event loop thread — no locks needed.
    Row indices are local to their row group (0-based).
    """

    def __init__(self) -> None:
        # row_group → column → set of completed local row indices
        self._completed: dict[int, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
        # row_group → set of dropped row indices
        self._dropped: dict[int, set[int]] = defaultdict(set)

    def mark_complete(self, column: str, row_group: int, row_index: int) -> None:
        self._completed[row_group][column].add(row_index)

    def mark_batch_complete(self, column: str, row_group: int, row_group_size: int) -> None:
        self._completed[row_group][column] = set(range(row_group_size))

    def is_complete(self, column: str, row_group: int, row_index: int) -> bool:
        return row_index in self._completed.get(row_group, {}).get(column, set())

    def all_complete(self, cells: list[tuple[str, int, int | None]]) -> bool:
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

    def is_ready(
        self,
        column: str,
        row_group: int,
        row_index: int,
        graph: ExecutionGraph,
        row_group_size: int,
    ) -> bool:
        """Check if all upstream columns are done for this (column, row_group, row_index)."""
        deps = graph.cell_dependencies(column, row_group, row_index, row_group_size)
        return self.all_complete(deps)

    def is_batch_ready(
        self,
        column: str,
        row_group: int,
        row_group_size: int,
        graph: ExecutionGraph,
    ) -> bool:
        """Check if all upstream columns are done for all non-dropped rows in the row group."""
        deps = graph.cell_dependencies(column, row_group, None, row_group_size)
        # Dropped rows don't need their upstream cells complete
        deps = [(c, rg, ri) for c, rg, ri in deps if ri is None or not self.is_dropped(rg, ri)]
        return self.all_complete(deps)

    def drop_row(self, row_group: int, row_index: int) -> None:
        self._dropped[row_group].add(row_index)

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

    def get_ready_tasks(
        self,
        graph: ExecutionGraph,
        row_groups: list[tuple[int, int]],
        dispatched: set[Task],
    ) -> list[Task]:
        """Return all currently dispatchable tasks.

        Excludes dropped rows and already-dispatched/in-flight tasks.
        """
        ready: list[Task] = []
        for rg_id, rg_size in row_groups:
            for col in graph.topological_order():
                strategy = graph.strategy(col)
                if strategy == GenerationStrategy.CELL_BY_CELL:
                    for ri in range(rg_size):
                        if self.is_dropped(rg_id, ri):
                            continue
                        if self.is_complete(col, rg_id, ri):
                            continue
                        task = Task(column=col, row_group=rg_id, row_index=ri, task_type="cell")
                        if task in dispatched:
                            continue
                        if self.is_ready(col, rg_id, ri, graph, rg_size):
                            ready.append(task)
                else:
                    task = Task(column=col, row_group=rg_id, row_index=None, task_type="batch")
                    if task in dispatched:
                        continue
                    if col in self._completed.get(rg_id, {}):
                        continue
                    if self.is_batch_ready(col, rg_id, rg_size, graph):
                        ready.append(task)
        return ready
