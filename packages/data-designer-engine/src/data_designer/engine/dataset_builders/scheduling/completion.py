# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from data_designer.config.column_configs import GenerationStrategy
from data_designer.engine.dataset_builders.row_group_plan import (
    RowGroupInput,
    RowGroupPlanLike,
    normalize_row_group_plan,
)
from data_designer.engine.dataset_builders.scheduling.resources import stable_task_id
from data_designer.engine.dataset_builders.scheduling.task_model import SliceRef, Task

if TYPE_CHECKING:
    from data_designer.engine.dataset_builders.utils.execution_graph import ExecutionGraph


MAX_EXACT_RELEASED_ROW_INTERVALS = 4_096
MAX_RELEASED_ROW_GROUP_SUMMARY_RANGES = 4_096


def _range_index(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    value: int,
) -> int | None:
    if not ranges or value < ranges[0][0] or value > ranges[-1][1]:
        return None
    index = bisect_right(ranges, (value, sys.maxsize)) - 1
    if index < 0:
        return None
    start, end = ranges[index]
    if start <= value <= end:
        return index
    return None


def _ranges_from_sorted_values(
    values: list[int],
    *,
    max_ranges: int | None = None,
) -> tuple[tuple[int, int], ...] | None:
    if not values:
        return ()
    ranges: list[tuple[int, int]] = []
    start = end = values[0]
    for value in values[1:]:
        if value == end + 1:
            end = value
            continue
        ranges.append((start, end))
        if max_ranges is not None and len(ranges) > max_ranges:
            return None
        start = end = value
    ranges.append((start, end))
    if max_ranges is not None and len(ranges) > max_ranges:
        return None
    return tuple(ranges)


def _survivor_ranges(
    row_group_size: int,
    dropped_rows: set[int],
    *,
    max_ranges: int | None = None,
) -> tuple[tuple[int, int], ...] | None:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    end: int | None = None
    for row_index in range(row_group_size):
        if row_index in dropped_rows:
            if start is not None and end is not None:
                ranges.append((start, end))
                if max_ranges is not None and len(ranges) > max_ranges:
                    return None
                start = end = None
            continue
        if start is None:
            start = end = row_index
        else:
            end = row_index
    if start is not None and end is not None:
        ranges.append((start, end))
    if max_ranges is not None and len(ranges) > max_ranges:
        return None
    return tuple(ranges)


def _dropped_ranges(
    row_group_size: int,
    dropped_rows: set[int],
    *,
    max_ranges: int | None = None,
) -> tuple[tuple[int, int], ...] | None:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    end: int | None = None
    for row_index in range(row_group_size):
        if row_index not in dropped_rows:
            if start is not None and end is not None:
                ranges.append((start, end))
                if max_ranges is not None and len(ranges) > max_ranges:
                    return None
                start = end = None
            continue
        if start is None:
            start = end = row_index
        else:
            end = row_index
    if start is not None and end is not None:
        ranges.append((start, end))
    if max_ranges is not None and len(ranges) > max_ranges:
        return None
    return tuple(ranges)


@dataclass(frozen=True)
class FrontierDelta:
    """Tasks added to or removed from the ready frontier by a tracker mutation."""

    added: tuple[Task, ...] = ()
    removed: tuple[Task, ...] = ()

    @property
    def empty(self) -> bool:
        return not self.added and not self.removed


@dataclass(frozen=True)
class _ReleasedColumns:
    complete_columns: frozenset[str]
    batch_columns: frozenset[str]


@dataclass(frozen=True)
class _ReleasedRows:
    row_group_size: int
    intervals: tuple[tuple[int, int], ...]
    stores_survivors: bool
    exact: bool = True
    dropped_count_value: int | None = None

    def is_dropped(self, row_index: int) -> bool:
        if not self.exact or not 0 <= row_index < self.row_group_size:
            return False
        in_intervals = _range_index(self.intervals, row_index) is not None
        return not in_intervals if self.stores_survivors else in_intervals

    @property
    def dropped_count(self) -> int:
        if not self.exact:
            if self.dropped_count_value is None:
                return 0
            return self.dropped_count_value
        interval_count = sum(end - start + 1 for start, end in self.intervals)
        if self.stores_survivors:
            return self.row_group_size - interval_count
        return interval_count


@dataclass(frozen=True)
class _ReleasedRowGroup:
    row_group_size: int
    columns: _ReleasedColumns
    rows: _ReleasedRows

    def is_dropped(self, row_index: int) -> bool:
        return self.rows.is_dropped(row_index)


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
        self._row_group_plan: RowGroupPlanLike | None = None
        self._batch_complete: dict[int, dict[str, int]] = defaultdict(dict)
        self._released_row_groups: dict[int, _ReleasedRowGroup] = {}
        self._released_range_summaries: list[tuple[int, int, _ReleasedRowGroup]] = []
        self._remaining_cell_rows: dict[int, dict[str, int]] = defaultdict(dict)
        # Exact post-checkpoint dropped-row identity is diagnostic-only. Keep it
        # globally bounded so released summaries cannot grow with total rows.
        self._released_exact_row_interval_count = 0
        self._frontier: set[Task] = set()

    @classmethod
    def with_graph(cls, graph: ExecutionGraph, row_groups: RowGroupInput) -> CompletionTracker:
        """Create a frontier-enabled tracker backed by an execution graph."""
        tracker = cls()
        tracker._graph = graph
        tracker._row_group_plan = normalize_row_group_plan(row_groups)
        return tracker

    def mark_cell_complete(self, column: str, row_group: int, row_index: int) -> FrontierDelta:
        row_group_size = self._validate_row_group(row_group)
        self._validate_strategy(column, GenerationStrategy.CELL_BY_CELL, "mark_cell_complete")
        if row_group_size is not None and not 0 <= row_index < row_group_size:
            raise ValueError(f"row_index out of range for rg={row_group}: got {row_index}, size {row_group_size}")
        self._forget_released_row_group(row_group)
        completed = self._completed[row_group][column]
        was_complete = row_index in completed
        completed.add(row_index)
        if not was_complete and row_index not in self._dropped.get(row_group, set()):
            self._decrement_remaining_cell_rows(row_group, column)
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
        self._forget_released_row_group(row_group)
        if row_group in self._completed:
            self._completed[row_group].pop(column, None)
        self._remaining_cell_rows.get(row_group, {}).pop(column, None)
        self._batch_complete[row_group][column] = row_group_size
        removed: list[Task] = []
        added: list[Task] = []
        if self._graph is not None:
            task = Task(column=column, row_group=row_group, row_index=None, task_type="batch")
            if self._discard_frontier_task(task):
                removed.append(task)
            added.extend(self._enqueue_downstream(column, row_group, row_index=None))
        return self._record_delta(added=added, removed=removed)

    def is_complete(self, ref: SliceRef) -> bool:
        if released := self._released_row_group(ref.row_group):
            if ref.column not in released.columns.complete_columns:
                return False
            if ref.row_index is None:
                return ref.column in released.columns.batch_columns
            if not 0 <= ref.row_index < released.row_group_size:
                return False
            if ref.column in released.columns.batch_columns:
                return True
            if not released.rows.exact:
                return True
            return not released.is_dropped(ref.row_index)
        batch_complete = self._batch_complete.get(ref.row_group, {})
        if ref.column in batch_complete:
            if ref.row_index is None:
                return True
            return 0 <= ref.row_index < batch_complete[ref.column]
        if ref.row_index is None:
            return False
        return ref.row_index in self._completed.get(ref.row_group, {}).get(ref.column, set())

    def is_all_complete(self, cells: list[SliceRef]) -> bool:
        """Check whether all the given cells are done.

        A ``row_index`` of ``None`` means the entire batch for that column must
        have been completed via ``mark_row_range_complete``.
        """
        return all(self.is_complete(ref) for ref in cells)

    def is_column_complete_for_rg(self, column: str, row_group_index: int) -> bool:
        """Check if *column* has been fully completed for *row_group_index*."""
        if released := self._released_row_group(row_group_index):
            return column in released.columns.complete_columns
        if column in self._batch_complete.get(row_group_index, {}):
            return True
        rg_size = self._row_group_size_or_default(row_group_index, default=0)
        if rg_size == 0:
            return False
        return self._is_cell_column_complete(row_group_index, column, rg_size)

    def drop_row(self, row_group: int, row_index: int) -> FrontierDelta:
        row_group_size = self._validate_row_group(row_group)
        if row_group_size is not None and not 0 <= row_index < row_group_size:
            raise ValueError(f"row_index out of range for rg={row_group}: got {row_index}, size {row_group_size}")
        self._forget_released_row_group(row_group)
        dropped = self._dropped[row_group]
        was_dropped = row_index in dropped
        dropped.add(row_index)
        if not was_dropped:
            self._decrement_remaining_for_dropped_row(row_group, row_index)
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
        if released := self._released_row_group(row_group):
            return released.is_dropped(row_index)
        return row_index in self._dropped.get(row_group, set())

    def dropped_row_count(self, row_group: int, row_group_size: int) -> int:
        if released := self._released_row_group(row_group):
            return released.rows.dropped_count
        return sum(1 for row_index in self._dropped.get(row_group, set()) if 0 <= row_index < row_group_size)

    def is_row_group_complete(
        self,
        row_group: int,
        row_group_size: int,
        all_columns: list[str],
    ) -> bool:
        """All non-dropped rows have all columns done."""
        if released := self._released_row_group(row_group):
            return row_group_size == released.row_group_size and set(all_columns).issubset(
                released.columns.complete_columns
            )
        batch_complete = self._batch_complete.get(row_group, {})
        for col in all_columns:
            if col in batch_complete and batch_complete[col] == row_group_size:
                continue
            if not self._is_cell_column_complete(row_group, col, row_group_size):
                return False
        return True

    def ready_frontier(self) -> tuple[Task, ...]:
        """Return dependency-ready tasks not yet acknowledged as enqueued."""
        return tuple(self._frontier)

    def mark_enqueued(self, task_ids: set[str] | list[str] | tuple[str, ...]) -> None:
        """Acknowledge tasks accepted by the ready queue."""
        wanted = set(task_ids)
        self._frontier = {task for task in self._frontier if stable_task_id(task) not in wanted}

    def mark_complete(self, task: Task) -> None:
        """Compatibility hook for scheduler terminal accounting."""

    def release_row_group(self, row_group: int, row_group_size: int, all_columns: list[str]) -> None:
        """Release completion state for a row group after durable checkpointing."""
        self._forget_released_row_group(row_group)
        columns = _ReleasedColumns(
            complete_columns=frozenset(all_columns),
            batch_columns=frozenset(self._batch_complete.get(row_group, {})),
        )
        dropped_rows = self._dropped.get(row_group, set())
        released = _ReleasedRowGroup(
            row_group_size=row_group_size,
            columns=columns,
            rows=self._released_rows(row_group_size, dropped_rows),
        )
        if self._row_group_plan is not None:
            self._released_row_groups.pop(row_group, None)
            self._add_released_summary_range(row_group, released)
        else:
            self._released_row_groups[row_group] = released
        self._completed.pop(row_group, None)
        self._dropped.pop(row_group, None)
        self._batch_complete.pop(row_group, None)
        self._remaining_cell_rows.pop(row_group, None)
        self._frontier = {task for task in self._frontier if task.row_group != row_group}

    def add_ready_tasks(self, tasks: list[Task] | tuple[Task, ...]) -> FrontierDelta:
        """Add ready tasks to the frontier idempotently."""
        added: list[Task] = []
        for task in tasks:
            if self._add_frontier_task(task):
                added.append(task)
        return self._record_delta(added=added, removed=[])

    def get_ready_tasks(self, dispatched: set[Task], admitted_rgs: set[int] | None = None) -> list[Task]:
        """Return all currently dispatchable tasks from the frontier."""
        return [
            t
            for t in self.ready_frontier()
            if t not in dispatched and (admitted_rgs is None or t.row_group in admitted_rgs)
        ]

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
            if self._row_group_plan is None:
                raise RuntimeError("This method requires row groups to be set.")
            for rg_id, rg_size in self._row_group_plan:
                self.add_root_tasks(rg_id, rg_size, columns=(col,))

    def add_root_tasks(
        self,
        row_group: int,
        row_group_size: int,
        *,
        columns: tuple[str, ...] | None = None,
    ) -> FrontierDelta:
        """Add root/from-scratch tasks for one admitted row group."""
        if self._graph is None:
            raise RuntimeError("This method requires a graph to be set.")
        expected = self._validate_row_group(row_group)
        if expected is not None and expected != row_group_size:
            raise ValueError(f"Row-group size mismatch for rg={row_group}: got {row_group_size}, expected {expected}")
        self._forget_released_row_group(row_group)
        root_columns = columns or tuple(self._graph.get_root_columns())
        added: list[Task] = []
        for col in root_columns:
            strategy = self._graph.get_strategy(col)
            if strategy == GenerationStrategy.CELL_BY_CELL:
                for ri in range(row_group_size):
                    task = Task(column=col, row_group=row_group, row_index=ri, task_type="cell")
                    if self._add_frontier_task(task):
                        added.append(task)
            else:
                task = Task(column=col, row_group=row_group, row_index=None, task_type="from_scratch")
                if self._add_frontier_task(task):
                    added.append(task)
        return self._record_delta(added=added, removed=[])

    def _released_row_group(self, row_group: int) -> _ReleasedRowGroup | None:
        if released := self._released_row_groups.get(row_group):
            return released
        index = bisect_right(self._released_range_summaries, (row_group, sys.maxsize)) - 1
        if index >= 0:
            start, end, released = self._released_range_summaries[index]
            if start <= row_group <= end:
                return released
        return None

    def _released_rows(self, row_group_size: int, dropped_rows: set[int]) -> _ReleasedRows:
        valid_dropped_count = sum(1 for row_index in dropped_rows if 0 <= row_index < row_group_size)
        survivor_count = row_group_size - valid_dropped_count
        exact_budget = MAX_EXACT_RELEASED_ROW_INTERVALS - self._released_exact_row_interval_count
        if exact_budget <= 0:
            return self._released_rows_aggregate(row_group_size, valid_dropped_count)

        if survivor_count < valid_dropped_count:
            intervals = _survivor_ranges(row_group_size, dropped_rows, max_ranges=exact_budget)
            if intervals is None:
                return self._released_rows_aggregate(row_group_size, valid_dropped_count)
            self._released_exact_row_interval_count += len(intervals)
            return _ReleasedRows(
                row_group_size=row_group_size,
                intervals=intervals,
                stores_survivors=True,
            )
        if valid_dropped_count <= exact_budget:
            valid_dropped = sorted(row_index for row_index in dropped_rows if 0 <= row_index < row_group_size)
            intervals = _ranges_from_sorted_values(valid_dropped, max_ranges=exact_budget)
        else:
            intervals = _dropped_ranges(row_group_size, dropped_rows, max_ranges=exact_budget)
        if intervals is None:
            return self._released_rows_aggregate(row_group_size, valid_dropped_count)
        self._released_exact_row_interval_count += len(intervals)
        return _ReleasedRows(
            row_group_size=row_group_size,
            intervals=intervals,
            stores_survivors=False,
        )

    def _released_rows_aggregate(self, row_group_size: int, dropped_count: int) -> _ReleasedRows:
        return _ReleasedRows(
            row_group_size=row_group_size,
            intervals=(),
            stores_survivors=False,
            exact=False,
            dropped_count_value=dropped_count,
        )

    def _add_released_summary_range(self, row_group: int, released: _ReleasedRowGroup) -> None:
        index = (
            bisect_right(
                self._released_range_summaries,
                (row_group, sys.maxsize),
            )
            - 1
        )
        if index >= 0:
            start, end, existing = self._released_range_summaries[index]
            if start <= row_group <= end:
                return
            if existing == released and end + 1 == row_group:
                right_index = index + 1
                if (
                    right_index < len(self._released_range_summaries)
                    and self._released_range_summaries[right_index][0] == row_group + 1
                    and self._released_range_summaries[right_index][2] == released
                ):
                    self._released_range_summaries[index] = (
                        start,
                        self._released_range_summaries[right_index][1],
                        released,
                    )
                    del self._released_range_summaries[right_index]
                else:
                    self._released_range_summaries[index] = (start, row_group, released)
                return
        right_index = index + 1
        if (
            right_index < len(self._released_range_summaries)
            and self._released_range_summaries[right_index][0] == row_group + 1
            and self._released_range_summaries[right_index][2] == released
        ):
            _right_start, right_end, _right_released = self._released_range_summaries[right_index]
            self._released_range_summaries[right_index] = (row_group, right_end, released)
        else:
            self._released_range_summaries.insert(right_index, (row_group, row_group, released))
        self._trim_released_summary_ranges()

    def _trim_released_summary_ranges(self) -> None:
        excess = len(self._released_range_summaries) - MAX_RELEASED_ROW_GROUP_SUMMARY_RANGES
        if excess > 0:
            del self._released_range_summaries[:excess]

    def _forget_released_row_group(self, row_group: int) -> None:
        self._released_row_groups.pop(row_group, None)
        self._remove_released_summary_range(row_group)

    def _remove_released_summary_range(self, row_group: int) -> None:
        index = (
            bisect_right(
                self._released_range_summaries,
                (row_group, sys.maxsize),
            )
            - 1
        )
        if index < 0:
            return
        start, end, released = self._released_range_summaries[index]
        if not start <= row_group <= end:
            return
        if start == end:
            del self._released_range_summaries[index]
        elif row_group == start:
            self._released_range_summaries[index] = (start + 1, end, released)
        elif row_group == end:
            self._released_range_summaries[index] = (start, end - 1, released)
        else:
            self._released_range_summaries[index : index + 1] = [
                (start, row_group - 1, released),
                (row_group + 1, end, released),
            ]
        self._trim_released_summary_ranges()

    def _record_delta(self, *, added: list[Task], removed: list[Task]) -> FrontierDelta:
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
        rg_batch_complete = self._batch_complete.get(row_group, {})
        rg_size = self._row_group_size(row_group)

        for down in sorted(self._graph.get_downstream_columns(column)):
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
                if down not in rg_batch_complete and self._are_cell_ups_complete(row_group, cell_ups, rg_size):
                    task = Task(column=down, row_group=row_group, row_index=None, task_type="batch")
                    if self._add_frontier_task(task):
                        added.append(task)
        return added

    def _reevaluate_batch_tasks(self, row_group: int) -> list[Task]:
        """Check if any batch tasks became ready after a row was dropped."""
        if self._graph is None:
            raise RuntimeError("This method requires a graph to be set.")
        added: list[Task] = []
        rg_batch_complete = self._batch_complete.get(row_group, {})
        rg_size = self._row_group_size(row_group)

        for col in self._graph.get_topological_order():
            if self._graph.get_strategy(col) != GenerationStrategy.FULL_COLUMN:
                continue
            if col in rg_batch_complete:
                continue
            batch_ups, cell_ups = self._graph.split_upstream_by_strategy(col)
            if any(up not in rg_batch_complete for up in batch_ups):
                continue
            if self._are_cell_ups_complete(row_group, cell_ups, rg_size):
                task = Task(column=col, row_group=row_group, row_index=None, task_type="batch")
                if self._add_frontier_task(task):
                    added.append(task)
        return added

    def _are_cell_ups_complete(
        self,
        row_group: int,
        cell_ups: list[str],
        rg_size: int,
    ) -> bool:
        """Check all non-dropped rows are complete for each cell-by-cell upstream column."""
        for up in cell_ups:
            if not self._is_cell_column_complete(row_group, up, rg_size):
                return False
        return True

    def _is_cell_column_complete(self, row_group: int, column: str, row_group_size: int) -> bool:
        completed = self._completed.get(row_group, {}).get(column, set())
        dropped = self._dropped.get(row_group, set())
        if self._row_group_plan is not None and self._row_group_plan.has_row_group(row_group):
            return self._remaining_cell_row_count(row_group, column, row_group_size, completed, dropped) == 0
        return self._row_indices_complete_with_dropped(row_group_size, completed, dropped)

    def _remaining_cell_row_count(
        self,
        row_group: int,
        column: str,
        row_group_size: int,
        completed: set[int],
        dropped: set[int],
    ) -> int:
        remaining_by_column = self._remaining_cell_rows[row_group]
        if column in remaining_by_column:
            return remaining_by_column[column]
        if dropped:
            completed_non_dropped = len(completed - dropped)
            remaining = row_group_size - len(dropped) - completed_non_dropped
        else:
            remaining = row_group_size - len(completed)
        remaining_by_column[column] = max(0, remaining)
        return remaining_by_column[column]

    def _decrement_remaining_cell_rows(self, row_group: int, column: str) -> None:
        remaining_by_column = self._remaining_cell_rows.get(row_group)
        if remaining_by_column is None or column not in remaining_by_column:
            return
        remaining_by_column[column] = max(0, remaining_by_column[column] - 1)

    def _decrement_remaining_for_dropped_row(self, row_group: int, row_index: int) -> None:
        remaining_by_column = self._remaining_cell_rows.get(row_group)
        if not remaining_by_column:
            return
        completed_by_column = self._completed.get(row_group, {})
        for column in list(remaining_by_column):
            if row_index not in completed_by_column.get(column, set()):
                remaining_by_column[column] = max(0, remaining_by_column[column] - 1)

    def _row_indices_complete_with_dropped(
        self,
        row_group_size: int,
        completed: set[int],
        dropped: set[int],
    ) -> bool:
        if not dropped:
            if len(completed) < row_group_size:
                return False
            if row_group_size == 0:
                return True
            if len(completed) == row_group_size:
                return min(completed) == 0 and max(completed) == row_group_size - 1
            return all(ri in completed for ri in range(row_group_size))
        if len(completed) + len(dropped) < row_group_size:
            return False
        return all(ri in completed or ri in dropped for ri in range(row_group_size))

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
        if self._row_group_plan is None:
            raise RuntimeError("This method requires row groups to be set.")
        if not self._row_group_plan.has_row_group(row_group):
            raise ValueError(
                f"Unknown row_group {row_group}. Known row_groups: {self._row_group_plan.describe_known_row_groups()}"
            )
        return self._row_group_plan.row_group_size(row_group)

    def _row_group_size(self, row_group: int) -> int:
        if self._row_group_plan is None:
            raise RuntimeError("This method requires row groups to be set.")
        return self._row_group_plan.row_group_size(row_group)

    def _row_group_size_or_default(self, row_group: int, *, default: int) -> int:
        if self._row_group_plan is None or not self._row_group_plan.has_row_group(row_group):
            return default
        return self._row_group_plan.row_group_size(row_group)
