# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any, Callable

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.dataset_builders.utils.completion_tracker import CompletionTracker
from data_designer.engine.dataset_builders.utils.task_model import Task, TaskTrace

if TYPE_CHECKING:
    from data_designer.engine.column_generators.generators.base import ColumnGenerator
    from data_designer.engine.dataset_builders.utils.execution_graph import ExecutionGraph
    from data_designer.engine.dataset_builders.utils.row_group_buffer import RowGroupBufferManager

logger = logging.getLogger(__name__)


class AsyncTaskScheduler:
    """Dependency-aware async task scheduler for the dataset builder.

    Replaces sequential column-by-column processing with parallel dispatch
    based on the ``ExecutionGraph`` and ``CompletionTracker``.
    """

    def __init__(
        self,
        generators: dict[str, ColumnGenerator],
        graph: ExecutionGraph,
        tracker: CompletionTracker,
        row_groups: list[tuple[int, int]],
        buffer_manager: RowGroupBufferManager | None = None,
        *,
        max_concurrent_row_groups: int = 3,
        max_submitted_tasks: int = 256,
        salvage_max_rounds: int = 2,
        on_row_group_complete: Callable[[int], None] | None = None,
        trace: bool = False,
    ) -> None:
        self._generators = generators
        self._graph = graph
        self._tracker = tracker
        self._row_groups = row_groups
        self._buffer_manager = buffer_manager

        self._rg_semaphore = asyncio.Semaphore(max_concurrent_row_groups)
        self._submission_semaphore = asyncio.Semaphore(max_submitted_tasks)

        self._dispatched: set[Task] = set()
        self._in_flight: set[Task] = set()
        self._wake_event = asyncio.Event()
        self._salvage_max_rounds = salvage_max_rounds
        self._on_row_group_complete = on_row_group_complete

        # Multi-column dedup: group output columns by generator identity
        instance_to_columns: dict[int, list[str]] = {}
        for col, gen in generators.items():
            instance_to_columns.setdefault(id(gen), []).append(col)
        self._instance_to_columns = instance_to_columns

        # Stateful generator tracking: instance_id → asyncio.Lock
        self._stateful_locks: dict[int, asyncio.Lock] = {}
        for col, gen in generators.items():
            if gen.is_order_dependent and id(gen) not in self._stateful_locks:
                self._stateful_locks[id(gen)] = asyncio.Lock()

        # Deferred retryable failures: list of (task, attempt_count)
        self._deferred: list[tuple[Task, int]] = []

        # Active row groups (admitted but not yet checkpointed)
        self._active_rgs: list[tuple[int, int]] = []
        self._admitted_rg_ids: set[int] = set()

        # Tracing
        self._trace = trace
        self.traces: list[TaskTrace] = []

        # Stats
        self._success_count = 0
        self._error_count = 0
        self._all_rgs_admitted = False

        # Pre-compute row-group sizes for O(1) lookup
        self._rg_size_map: dict[int, int] = {rg_id: size for rg_id, size in row_groups}

    async def _admit_row_groups(self) -> None:
        """Admit row groups as semaphore slots become available."""
        for rg_id, rg_size in self._row_groups:
            await self._rg_semaphore.acquire()
            self._active_rgs.append((rg_id, rg_size))
            self._admitted_rg_ids.add(rg_id)

            if self._buffer_manager is not None:
                self._buffer_manager.init_row_group(rg_id, rg_size)

            await self._dispatch_seeds(rg_id, rg_size)
            self._wake_event.set()
        self._all_rgs_admitted = True
        self._wake_event.set()

    async def run(self) -> None:
        """Main scheduler loop."""
        all_columns = self._graph.columns

        # Launch admission as a background task so it interleaves with dispatch.
        admission_task = asyncio.create_task(self._admit_row_groups())

        # Main dispatch loop
        while True:
            self._wake_event.clear()

            ready = self._tracker.get_ready_tasks(self._dispatched, self._admitted_rg_ids)
            for task in ready:
                await self._submission_semaphore.acquire()
                self._dispatched.add(task)
                self._in_flight.add(task)
                asyncio.create_task(self._execute_task(task))

            # Check for completed row groups
            completed_rgs: list[tuple[int, int]] = []
            for rg_id, rg_size in self._active_rgs:
                if self._tracker.is_row_group_complete(rg_id, rg_size, all_columns):
                    completed_rgs.append((rg_id, rg_size))

            for rg_id, rg_size in completed_rgs:
                self._active_rgs.remove((rg_id, rg_size))
                if self._buffer_manager is not None:
                    self._buffer_manager.checkpoint_row_group(rg_id)
                if self._on_row_group_complete:
                    self._on_row_group_complete(rg_id)
                self._rg_semaphore.release()

            # Are we done?
            all_done = self._all_rgs_admitted and not self._active_rgs and not self._in_flight
            if all_done:
                break

            if self._all_rgs_admitted and not ready and not self._in_flight:
                break

            if not ready:
                await self._wake_event.wait()

        # Cancel admission if still running
        if not admission_task.done():
            admission_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await admission_task

        # Phase 3: Salvage rounds for retryable failures
        for round_num in range(self._salvage_max_rounds):
            if not self._deferred:
                break
            logger.info(f"Salvage round {round_num + 1}/{self._salvage_max_rounds}: {len(self._deferred)} tasks")
            to_retry = self._deferred
            self._deferred = []
            for task, _attempt in to_retry:
                if task.task_type == "from_scratch":
                    # from_scratch tasks are not in the frontier; re-dispatch directly
                    gid = id(self._generators[task.column])
                    self._dispatched.discard(task)
                    # Also clear the batch alias so completion tracking works
                    self._dispatched.discard(
                        Task(column=task.column, row_group=task.row_group, row_index=None, task_type="batch")
                    )
                    for sibling in self._instance_to_columns.get(gid, []):
                        if sibling != task.column:
                            self._dispatched.discard(
                                Task(column=sibling, row_group=task.row_group, row_index=None, task_type="from_scratch")
                            )
                            self._dispatched.discard(
                                Task(column=sibling, row_group=task.row_group, row_index=None, task_type="batch")
                            )
                    await self._submission_semaphore.acquire()
                    self._dispatched.add(task)
                    self._in_flight.add(task)
                    asyncio.create_task(self._execute_seed_task(task, gid))
                else:
                    self._dispatched.discard(task)
            # Re-enter main loop for one pass (frontier-based tasks)
            ready = self._tracker.get_ready_tasks(self._dispatched, self._admitted_rg_ids)
            for task in ready:
                await self._submission_semaphore.acquire()
                self._dispatched.add(task)
                self._in_flight.add(task)
                asyncio.create_task(self._execute_task(task))
            # Wait for all in-flight to finish
            while self._in_flight:
                self._wake_event.clear()
                await self._wake_event.wait()

        # Warn if any row groups were not checkpointed
        if self._active_rgs:
            incomplete = [rg_id for rg_id, _ in self._active_rgs]
            logger.error(
                f"Scheduler exited with {len(self._active_rgs)} unfinished row group(s): {incomplete}. "
                "These row groups were not checkpointed."
            )

    async def _dispatch_seeds(self, rg_id: int, rg_size: int) -> None:
        """Dispatch from_scratch tasks for a row group."""
        seed_cols = [col for col in self._graph.get_topological_order() if not self._graph.get_upstream_columns(col)]
        seen_instances: set[int] = set()

        for col in seed_cols:
            gen = self._generators[col]
            gid = id(gen)
            if gid in seen_instances:
                continue
            seen_instances.add(gid)

            task = Task(column=col, row_group=rg_id, row_index=None, task_type="from_scratch")
            # Also mark the "batch" variant as dispatched to prevent get_ready_tasks
            # from generating a duplicate for this column
            batch_alias = Task(column=col, row_group=rg_id, row_index=None, task_type="batch")
            if task in self._dispatched or batch_alias in self._dispatched:
                continue

            # Acquire stateful lock *before* submission semaphore to preserve
            # row-group ordering. Held until generation completes (_execute_seed_task).
            if gid in self._stateful_locks:
                await self._stateful_locks[gid].acquire()

            await self._submission_semaphore.acquire()
            self._dispatched.add(task)
            self._dispatched.add(batch_alias)
            # Also mark all sibling output columns as dispatched (multi-column dedup)
            for sibling_col in self._instance_to_columns.get(gid, []):
                if sibling_col != col:
                    self._dispatched.add(
                        Task(column=sibling_col, row_group=rg_id, row_index=None, task_type="from_scratch")
                    )
                    self._dispatched.add(Task(column=sibling_col, row_group=rg_id, row_index=None, task_type="batch"))
            self._in_flight.add(task)
            asyncio.create_task(self._execute_seed_task(task, gid))

    async def _execute_seed_task(self, task: Task, generator_id: int) -> None:
        """Execute a from_scratch task and release stateful lock if held."""
        try:
            await self._execute_task_inner(task)
        finally:
            if generator_id in self._stateful_locks:
                self._stateful_locks[generator_id].release()

    async def _execute_task(self, task: Task) -> None:
        """Execute a single task (cell or batch)."""
        await self._execute_task_inner(task)

    async def _execute_task_inner(self, task: Task) -> None:
        """Core task execution logic."""
        trace: TaskTrace | None = None
        if self._trace:
            trace = TaskTrace.from_task(task)
            trace.dispatched_at = time.perf_counter()

        generator = self._generators[task.column]
        output_cols = self._instance_to_columns.get(id(generator), [task.column])

        try:
            if self._trace and trace:
                trace.slot_acquired_at = time.perf_counter()

            if task.task_type == "from_scratch":
                await self._run_from_scratch(task, generator)
            elif task.task_type == "cell":
                await self._run_cell(task, generator)
            elif task.task_type == "batch":
                await self._run_batch(task, generator)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")

            # Mark all output columns complete
            for col in output_cols:
                if task.row_index is None:
                    rg_size = self._get_rg_size(task.row_group)
                    self._tracker.mark_row_range_complete(col, task.row_group, rg_size)
                else:
                    self._tracker.mark_cell_complete(col, task.row_group, task.row_index)

            self._success_count += 1
            if self._trace and trace:
                trace.status = "ok"

        except Exception as exc:
            self._error_count += 1
            if self._trace and trace:
                trace.status = "error"
                trace.error = str(exc)

            retryable = self._is_retryable(exc)
            if retryable:
                self._deferred.append((task, 1))
            else:
                # Non-retryable: drop the affected row(s)
                if task.row_index is not None:
                    self._tracker.drop_row(task.row_group, task.row_index)
                    if self._buffer_manager:
                        self._buffer_manager.drop_row(task.row_group, task.row_index)
                else:
                    # Batch/from_scratch failure: drop all rows in the row group
                    rg_size = self._get_rg_size(task.row_group)
                    for ri in range(rg_size):
                        self._tracker.drop_row(task.row_group, ri)
                        if self._buffer_manager:
                            self._buffer_manager.drop_row(task.row_group, ri)
                logger.warning(
                    f"Non-retryable failure on {task.column}[rg={task.row_group}, row={task.row_index}]: {exc}"
                )

        finally:
            if self._trace and trace:
                trace.completed_at = time.perf_counter()
                self.traces.append(trace)

            self._in_flight.discard(task)
            self._submission_semaphore.release()
            self._wake_event.set()

    async def _run_from_scratch(self, task: Task, generator: ColumnGenerator) -> Any:
        """Execute a from_scratch task."""
        rg_size = self._get_rg_size(task.row_group)
        from data_designer.engine.column_generators.generators.base import FromScratchColumnGenerator

        if isinstance(generator, FromScratchColumnGenerator):
            result_df = await generator.agenerate_from_scratch(rg_size)
        else:
            result_df = await generator.agenerate({})

        # Write results to buffer
        if self._buffer_manager is not None:
            output_cols = self._instance_to_columns.get(id(generator), [task.column])
            for col in output_cols:
                if col in result_df.columns:
                    values = result_df[col].tolist()
                    self._buffer_manager.update_batch(task.row_group, col, values)

        return result_df

    async def _run_cell(self, task: Task, generator: ColumnGenerator) -> Any:
        """Execute a cell-by-cell task."""
        assert task.row_index is not None

        if self._tracker.is_dropped(task.row_group, task.row_index):
            return None

        # Read row from buffer
        if self._buffer_manager is not None:
            row_data = dict(self._buffer_manager.get_row(task.row_group, task.row_index))
        else:
            row_data = {}

        result = await generator.agenerate(row_data)

        # Write back to buffer
        if self._buffer_manager is not None and not self._tracker.is_dropped(task.row_group, task.row_index):
            output_cols = self._instance_to_columns.get(id(generator), [task.column])
            for col in output_cols:
                if col in result:
                    self._buffer_manager.update_cell(task.row_group, task.row_index, col, result[col])

        return result

    async def _run_batch(self, task: Task, generator: ColumnGenerator) -> Any:
        """Execute a full-column/batch task."""
        if self._buffer_manager is not None:
            batch_df = self._buffer_manager.get_dataframe(task.row_group)
        else:
            batch_df = lazy.pd.DataFrame()

        result_df = await generator.agenerate(batch_df)

        # Merge result columns back to buffer
        if self._buffer_manager is not None:
            output_cols = self._instance_to_columns.get(id(generator), [task.column])
            rg_size = self._get_rg_size(task.row_group)
            dropped = set()
            for ri in range(rg_size):
                if self._buffer_manager.is_dropped(task.row_group, ri):
                    dropped.add(ri)

            # Map result rows (which exclude dropped) back to buffer indices
            active_rows = rg_size - len(dropped)
            if len(result_df) != active_rows:
                logger.warning(
                    f"Batch generator for '{task.column}' returned {len(result_df)} rows "
                    f"but {active_rows} were expected (rg={task.row_group})."
                )
            result_idx = 0
            for ri in range(rg_size):
                if ri in dropped:
                    continue
                if result_idx < len(result_df):
                    for col in output_cols:
                        if col in result_df.columns:
                            self._buffer_manager.update_cell(task.row_group, ri, col, result_df.iloc[result_idx][col])
                result_idx += 1

        return result_df

    def _get_rg_size(self, row_group: int) -> int:
        try:
            return self._rg_size_map[row_group]
        except KeyError:
            raise ValueError(f"Unknown row group: {row_group}") from None

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Classify whether an exception is retryable."""
        # HTTP-level transient errors
        exc_str = str(exc).lower()
        for pattern in ("429", "500", "502", "503", "504", "timeout", "timed out"):
            if pattern in exc_str:
                return True
        return False
