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
from data_designer.engine.models.errors import (
    ModelAPIConnectionError,
    ModelInternalServerError,
    ModelRateLimitError,
    ModelTimeoutError,
)

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
        on_seeds_complete: Callable[[int, int], None] | None = None,
        on_before_checkpoint: Callable[[int, int], None] | None = None,
        shutdown_error_rate: float = 0.5,
        shutdown_error_window: int = 10,
        disable_early_shutdown: bool = False,
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
        self._on_seeds_complete = on_seeds_complete
        self._on_before_checkpoint = on_before_checkpoint

        # Error rate shutdown (caller passes pre-normalized values via RunConfig)
        self._shutdown_error_rate = shutdown_error_rate
        self._shutdown_error_window = shutdown_error_window
        self._disable_early_shutdown = disable_early_shutdown
        self._early_shutdown = False

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

        # Per-RG in-flight counters for O(1) lookup
        self._in_flight_counts: dict[int, int] = {}

        # Deferred retryable failures (retried in salvage rounds)
        self._deferred: list[Task] = []

        # Active row groups (admitted but not yet checkpointed)
        self._active_rgs: list[tuple[int, int]] = []
        self._admitted_rg_ids: set[int] = set()
        self._seeds_dispatched_rgs: set[int] = set()
        self._pre_batch_done_rgs: set[int] = set()

        # Tracing
        self._trace = trace
        self.traces: list[TaskTrace] = []

        # Stats
        self._success_count = 0
        self._error_count = 0
        self._all_rgs_admitted = False

        # Pre-compute row-group sizes for O(1) lookup
        self._rg_size_map: dict[int, int] = dict(row_groups)

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
        seed_cols = frozenset(c for c in all_columns if not self._graph.get_upstream_columns(c))
        has_pre_batch = self._on_seeds_complete is not None

        # Launch admission as a background task so it interleaves with dispatch.
        admission_task = asyncio.create_task(self._admit_row_groups())

        # Main dispatch loop
        while True:
            if self._early_shutdown:
                logger.warning("Early shutdown triggered - error rate exceeded threshold")
                self._checkpoint_completed_row_groups(all_columns)
                break

            self._wake_event.clear()

            self._run_seeds_complete_check(seed_cols)

            ready = self._tracker.get_ready_tasks(self._dispatched, self._admitted_rg_ids)
            # Gate non-seed tasks on pre-batch completion when a pre-batch callback is configured
            if has_pre_batch:
                ready = [t for t in ready if t.row_group in self._pre_batch_done_rgs or t.column in seed_cols]
            for task in ready:
                await self._submission_semaphore.acquire()
                self._dispatched.add(task)
                self._in_flight.add(task)
                self._in_flight_counts[task.row_group] = self._in_flight_counts.get(task.row_group, 0) + 1
                asyncio.create_task(self._execute_task(task))

            self._checkpoint_completed_row_groups(all_columns)

            # Are we done?
            all_done = self._all_rgs_admitted and not self._active_rgs and not self._in_flight
            if all_done:
                break

            # All admitted RGs finished their non-deferred work but may not be
            # "complete" yet (deferred tasks remain for salvage). Exit the main
            # loop so salvage rounds can handle them.
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
            for task in to_retry:
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
                    # Acquire stateful lock (mirrors _dispatch_seeds) so
                    # _execute_seed_task can safely release it in finally.
                    if gid in self._stateful_locks:
                        await self._stateful_locks[gid].acquire()
                    await self._submission_semaphore.acquire()
                    self._dispatched.add(task)
                    # Re-register batch alias to mirror _dispatch_seeds and prevent
                    # duplicate dispatch if the frontier contains a stale batch task.
                    self._dispatched.add(
                        Task(column=task.column, row_group=task.row_group, row_index=None, task_type="batch")
                    )
                    self._in_flight.add(task)
                    self._in_flight_counts[task.row_group] = self._in_flight_counts.get(task.row_group, 0) + 1
                    asyncio.create_task(self._execute_seed_task(task, gid))
                else:
                    self._dispatched.discard(task)
            # Drain: dispatch frontier tasks and any newly-ready downstream tasks
            # until nothing remains in-flight or in the frontier.
            await self._drain_frontier(seed_cols, has_pre_batch, all_columns)
            self._checkpoint_completed_row_groups(all_columns)

        if self._active_rgs:
            incomplete = [rg_id for rg_id, _ in self._active_rgs]
            logger.error(
                f"Scheduler exited with {len(self._active_rgs)} unfinished row group(s): {incomplete}. "
                "These row groups were not checkpointed."
            )

    async def _drain_frontier(self, seed_cols: frozenset[str], has_pre_batch: bool, all_columns: list[str]) -> None:
        """Dispatch all frontier tasks and their downstream until quiescent."""
        while True:
            self._run_seeds_complete_check(seed_cols)
            ready = self._tracker.get_ready_tasks(self._dispatched, self._admitted_rg_ids)
            if has_pre_batch:
                ready = [t for t in ready if t.row_group in self._pre_batch_done_rgs or t.column in seed_cols]
            for task in ready:
                await self._submission_semaphore.acquire()
                self._dispatched.add(task)
                self._in_flight.add(task)
                self._in_flight_counts[task.row_group] = self._in_flight_counts.get(task.row_group, 0) + 1
                asyncio.create_task(self._execute_task(task))
            if not self._in_flight:
                break
            self._wake_event.clear()
            await self._wake_event.wait()

    def _checkpoint_completed_row_groups(self, all_columns: list[str]) -> None:
        """Checkpoint any row groups that reached completion."""
        completed = [
            (rg_id, rg_size)
            for rg_id, rg_size in self._active_rgs
            if self._tracker.is_row_group_complete(rg_id, rg_size, all_columns)
        ]
        for rg_id, rg_size in completed:
            self._active_rgs.remove((rg_id, rg_size))
            self._admitted_rg_ids.discard(rg_id)
            try:
                if self._on_before_checkpoint:
                    try:
                        self._on_before_checkpoint(rg_id, rg_size)
                    except Exception:
                        logger.error(
                            f"on_before_checkpoint failed for row group {rg_id}, checkpointing un-processed data.",
                            exc_info=True,
                        )
                if self._buffer_manager is not None:
                    self._buffer_manager.checkpoint_row_group(rg_id)
                if self._on_row_group_complete:
                    self._on_row_group_complete(rg_id)
            except Exception:
                logger.error(f"Failed to checkpoint row group {rg_id}.", exc_info=True)
            finally:
                self._rg_semaphore.release()

    def _run_seeds_complete_check(self, seed_cols: frozenset[str]) -> None:
        """Run pre-batch callbacks for row groups whose seeds just completed."""
        for rg_id, rg_size in self._active_rgs:
            if rg_id in self._seeds_dispatched_rgs and rg_id not in self._pre_batch_done_rgs:
                all_seeds_done = all(self._tracker.is_column_complete_for_rg(col, rg_id) for col in seed_cols)
                if all_seeds_done and not self._in_flight_for_rg(rg_id):
                    self._pre_batch_done_rgs.add(rg_id)
                    if self._on_seeds_complete:
                        try:
                            self._on_seeds_complete(rg_id, rg_size)
                        except Exception as exc:
                            logger.warning(f"Pre-batch processor failed for row group {rg_id}, skipping: {exc}")
                            for ri in range(rg_size):
                                self._tracker.drop_row(rg_id, ri)
                                if self._buffer_manager:
                                    self._buffer_manager.drop_row(rg_id, ri)

    def _in_flight_for_rg(self, rg_id: int) -> bool:
        """Check if any tasks are in-flight for a given row group."""
        return self._in_flight_counts.get(rg_id, 0) > 0

    def _check_error_rate(self) -> None:
        """Trigger early shutdown if error rate exceeds threshold."""
        if self._disable_early_shutdown:
            return
        completed = self._success_count + self._error_count
        if completed < self._shutdown_error_window:
            return
        error_rate = self._error_count / max(1, completed)
        if error_rate > self._shutdown_error_rate:
            self._early_shutdown = True

    async def _dispatch_seeds(self, rg_id: int, rg_size: int) -> None:
        """Dispatch from_scratch tasks for a row group."""
        self._seeds_dispatched_rgs.add(rg_id)
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
            self._in_flight_counts[task.row_group] = self._in_flight_counts.get(task.row_group, 0) + 1
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
        retryable = False
        # When True, skip removing from _dispatched so the task isn't re-dispatched
        # from the frontier (it was never completed, so it stays in the frontier).
        skipped = False

        try:
            # Skip tasks whose row group was already checkpointed (can happen
            # when a vacuously-ready downstream is dispatched via create_task
            # in the same loop iteration that checkpoints the row group).
            if not any(rg_id == task.row_group for rg_id, _ in self._active_rgs):
                skipped = True
                return

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
            self._check_error_rate()
            if self._trace and trace:
                trace.status = "error"
                trace.error = str(exc)

            retryable = self._is_retryable(exc)
            if retryable:
                self._deferred.append(task)
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
            self._in_flight_counts[task.row_group] = self._in_flight_counts.get(task.row_group, 0) - 1
            if not retryable and not skipped:
                self._dispatched.discard(task)
            self._submission_semaphore.release()
            self._wake_event.set()

    async def _run_from_scratch(self, task: Task, generator: ColumnGenerator) -> Any:
        """Execute a from_scratch task."""
        rg_size = self._get_rg_size(task.row_group)
        # Runtime import: needed for isinstance check; module-level would cause circular import
        from data_designer.engine.column_generators.generators.base import FromScratchColumnGenerator

        if isinstance(generator, FromScratchColumnGenerator):
            result_df = await generator.agenerate_from_scratch(rg_size)
        else:
            result_df = await generator.agenerate(lazy.pd.DataFrame())

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
        if task.row_index is None:
            raise ValueError(f"Cell task requires a row_index, got None for column '{task.column}'")

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
                raise ValueError(
                    f"Batch generator for '{task.column}' returned {len(result_df)} rows "
                    f"but {active_rows} were expected (rg={task.row_group})."
                )
            result_idx = 0
            for ri in range(rg_size):
                if ri in dropped:
                    continue
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

    _RETRYABLE_MODEL_ERRORS = (
        ModelRateLimitError,
        ModelTimeoutError,
        ModelInternalServerError,
        ModelAPIConnectionError,
    )

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Classify whether an exception is retryable."""
        return isinstance(exc, AsyncTaskScheduler._RETRYABLE_MODEL_ERRORS)
