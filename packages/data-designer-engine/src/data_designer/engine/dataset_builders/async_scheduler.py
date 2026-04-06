# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import GenerationStrategy
from data_designer.engine.context import current_row_group
from data_designer.engine.dataset_builders.utils.async_progress_reporter import (
    DEFAULT_REPORT_INTERVAL,
    AsyncProgressReporter,
)
from data_designer.engine.dataset_builders.utils.completion_tracker import CompletionTracker
from data_designer.engine.dataset_builders.utils.progress_tracker import ProgressTracker
from data_designer.engine.dataset_builders.utils.skip_evaluator import evaluate_skip_when, should_skip_by_propagation
from data_designer.engine.dataset_builders.utils.skip_provenance import (
    apply_skip_to_record,
    get_skipped_column_names,
    strip_skip_metadata_from_records,
)
from data_designer.engine.dataset_builders.utils.sticky_progress_bar import StickyProgressBar
from data_designer.engine.dataset_builders.utils.task_model import SliceRef, Task, TaskTrace
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

DEFAULT_TASK_POOL_SIZE: int = 256
LLM_WAIT_POOL_MULTIPLIER: int = 2

_RETRYABLE_MODEL_ERRORS = (
    ModelRateLimitError,
    ModelTimeoutError,
    ModelInternalServerError,
    ModelAPIConnectionError,
)


class TrackingSemaphore(asyncio.Semaphore):
    """``asyncio.Semaphore`` subclass that exposes available permits publicly."""

    @property
    def available_permits(self) -> int:
        return self._value  # type: ignore[attr-defined]


@dataclass
class _RowGroupState:
    """Lifecycle state for a single admitted row group."""

    size: int
    seeds_dispatched: bool = False
    pre_batch_done: bool = False
    in_flight_count: int = 0


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
        max_submitted_tasks: int = DEFAULT_TASK_POOL_SIZE,
        max_llm_wait_tasks: int = DEFAULT_TASK_POOL_SIZE,
        salvage_max_rounds: int = 2,
        on_finalize_row_group: Callable[[int], None] | None = None,
        on_seeds_complete: Callable[[int, int], None] | None = None,
        on_before_checkpoint: Callable[[int, int], None] | None = None,
        shutdown_error_rate: float = 0.5,
        shutdown_error_window: int = 10,
        disable_early_shutdown: bool = False,
        trace: bool = False,
        num_records: int = 0,
        buffer_size: int = 0,
        progress_interval: float | None = None,
        progress_bar: bool = False,
    ) -> None:
        self._generators = generators
        self._graph = graph
        self._tracker = tracker
        self._row_groups = row_groups
        self._buffer_manager = buffer_manager

        self._rg_semaphore = asyncio.Semaphore(max_concurrent_row_groups)
        self._submission_semaphore = TrackingSemaphore(max_submitted_tasks)
        self._llm_wait_semaphore = TrackingSemaphore(max_llm_wait_tasks)

        self._llm_bound_lookup = build_llm_bound_lookup(generators)

        self._dispatched: set[Task] = set()
        self._in_flight: set[Task] = set()
        self._worker_tasks: set[asyncio.Task] = set()
        self._wake_event = asyncio.Event()
        self._salvage_max_rounds = salvage_max_rounds
        self._on_finalize_row_group = on_finalize_row_group
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

        # Per-RG lifecycle state (admitted but not yet checkpointed)
        self._rg_states: dict[int, _RowGroupState] = {}

        # Deferred retryable failures (retried in salvage rounds)
        self._deferred: list[Task] = []

        # Tracing
        self._trace = trace
        self.traces: list[TaskTrace] = []

        # Sliding window for error rate shutdown
        self._recent_outcomes: deque[bool] = deque(maxlen=shutdown_error_window)
        self._all_rgs_admitted = False

        # Pre-compute row-group sizes for O(1) lookup
        self._rg_size_map: dict[int, int] = dict(row_groups)

        # Pre-compute seed columns (graph is static)
        self._seed_cols: frozenset[str] = frozenset(c for c in graph.columns if not graph.get_upstream_columns(c))

        # Per-column progress tracking (cell-by-cell only; full-column tasks are instant)
        self._progress_bar = StickyProgressBar() if progress_bar else None
        self._reporter = self._setup_async_progress_reporter(num_records, buffer_size, progress_interval)

    def _setup_async_progress_reporter(
        self,
        num_records: int,
        buffer_size: int,
        progress_interval: float | None,
    ) -> AsyncProgressReporter | None:
        if num_records <= 0 or buffer_size <= 0:
            return None

        task_counts = self._graph.compute_task_count(num_records, buffer_size)
        trackers: dict[str, ProgressTracker] = {}
        for col in self._graph.columns:
            if self._graph.get_strategy(col) != GenerationStrategy.CELL_BY_CELL:
                continue
            trackers[col] = ProgressTracker(
                total_records=task_counts[col],
                label=f"column '{col}'",
                quiet=True,
            )

        if not trackers:
            return None

        interval = progress_interval if progress_interval is not None else DEFAULT_REPORT_INTERVAL
        return AsyncProgressReporter(
            trackers,
            report_interval=interval,
            progress_bar=self._progress_bar,
        )

    def _spawn_worker(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task:
        """Create a tracked worker task that auto-removes itself on completion."""
        task = asyncio.create_task(coro)
        self._worker_tasks.add(task)
        task.add_done_callback(self._worker_tasks.discard)
        return task

    async def _cancel_workers(self) -> None:
        """Cancel all tracked worker tasks and wait for them to finish."""
        for t in self._worker_tasks:
            t.cancel()
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()

    async def _admit_row_groups(self) -> None:
        """Admit row groups as semaphore slots become available."""
        for rg_id, rg_size in self._row_groups:
            await self._rg_semaphore.acquire()
            self._rg_states[rg_id] = _RowGroupState(size=rg_size)

            if self._buffer_manager is not None:
                self._buffer_manager.init_row_group(rg_id, rg_size)

            await self._dispatch_seeds(rg_id, rg_size)
            self._wake_event.set()
        self._all_rgs_admitted = True
        self._wake_event.set()

    async def run(self) -> None:
        """Main scheduler loop.

        On cancellation (``CancelledError``), all tracked worker tasks are
        cancelled and awaited so that held semaphore permits are released
        before the error propagates.
        """
        all_columns = self._graph.columns
        seed_cols = self._seed_cols
        has_pre_batch = self._on_seeds_complete is not None

        num_rgs = len(self._row_groups)

        with self._progress_bar or contextlib.nullcontext():
            if self._reporter:
                self._reporter.log_start(num_row_groups=num_rgs)

            # Launch admission as a background task so it interleaves with dispatch.
            admission_task = asyncio.create_task(self._admit_row_groups())

            try:
                # Main dispatch loop
                await self._main_dispatch_loop(seed_cols, has_pre_batch, all_columns)

                # Cancel admission if still running
                if not admission_task.done():
                    admission_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await admission_task

                if self._reporter:
                    self._reporter.log_final()

                if self._rg_states:
                    incomplete = list(self._rg_states)
                    logger.error(
                        f"Scheduler exited with {len(self._rg_states)} unfinished row group(s): {incomplete}. "
                        "These row groups were not checkpointed."
                    )

            except asyncio.CancelledError:
                if not admission_task.done():
                    admission_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await admission_task
                await asyncio.shield(self._cancel_workers())
                raise

    async def _main_dispatch_loop(
        self,
        seed_cols: frozenset[str],
        has_pre_batch: bool,
        all_columns: list[str],
    ) -> None:
        """Core dispatch loop extracted from ``run()``."""
        while True:
            if self._early_shutdown:
                logger.warning("Early shutdown triggered - error rate exceeded threshold")
                if self._deferred:
                    await self._salvage_stalled_row_groups(seed_cols, has_pre_batch, all_columns)
                self._checkpoint_completed_row_groups(all_columns)
                break

            self._wake_event.clear()

            if has_pre_batch:
                self._run_seeds_complete_check(seed_cols)

            admitted_ids = set(self._rg_states)
            ready = self._tracker.get_ready_tasks(self._dispatched, admitted_ids)
            # Gate non-seed tasks on pre-batch completion when a pre-batch callback is configured
            if has_pre_batch:
                ready = [
                    t
                    for t in ready
                    if (s := self._rg_states.get(t.row_group)) is not None and s.pre_batch_done or t.column in seed_cols
                ]
            for task in ready:
                await self._submission_semaphore.acquire()
                self._dispatched.add(task)
                self._in_flight.add(task)
                if (s := self._rg_states.get(task.row_group)) is not None:
                    s.in_flight_count += 1
                self._spawn_worker(self._execute_task(task))

            self._checkpoint_completed_row_groups(all_columns)

            # Eagerly salvage any row groups that have only deferred tasks,
            # even if other row groups are still in-flight.  This frees
            # semaphore slots so admission doesn't lose capacity.
            if self._deferred:
                await self._salvage_stalled_row_groups(seed_cols, has_pre_batch, all_columns)

            # Are we done?
            all_done = self._all_rgs_admitted and not self._rg_states and not self._in_flight
            if all_done:
                break

            if not ready and not self._in_flight:
                if self._all_rgs_admitted:
                    break

            if not ready:
                await self._wake_event.wait()

    async def _salvage_rounds(
        self,
        seed_cols: frozenset[str],
        has_pre_batch: bool,
        all_columns: list[str],
    ) -> None:
        """Phase 3: retry deferred (transient-failure) tasks."""
        for round_num in range(self._salvage_max_rounds):
            if not self._deferred:
                break
            logger.debug(f"Salvage round {round_num + 1}/{self._salvage_max_rounds}: {len(self._deferred)} tasks")
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
                    # Re-mark sibling columns as dispatched to mirror _dispatch_seeds
                    # and prevent _drain_frontier from re-dispatching them.
                    for sibling in self._instance_to_columns.get(gid, []):
                        if sibling != task.column:
                            self._dispatched.add(
                                Task(column=sibling, row_group=task.row_group, row_index=None, task_type="from_scratch")
                            )
                            self._dispatched.add(
                                Task(column=sibling, row_group=task.row_group, row_index=None, task_type="batch")
                            )
                    self._in_flight.add(task)
                    if (s := self._rg_states.get(task.row_group)) is not None:
                        s.in_flight_count += 1
                    self._spawn_worker(self._execute_seed_task(task, gid))
                else:
                    self._dispatched.discard(task)
            # Drain: dispatch frontier tasks and any newly-ready downstream tasks
            # until nothing remains in-flight or in the frontier.
            await self._drain_frontier(seed_cols, has_pre_batch, all_columns)
            self._checkpoint_completed_row_groups(all_columns)

    async def _drain_frontier(self, seed_cols: frozenset[str], has_pre_batch: bool, all_columns: list[str]) -> None:
        """Dispatch all frontier tasks and their downstream until quiescent."""
        while True:
            if has_pre_batch:
                self._run_seeds_complete_check(seed_cols)
            admitted_ids = set(self._rg_states)
            ready = self._tracker.get_ready_tasks(self._dispatched, admitted_ids)
            if has_pre_batch:
                ready = [
                    t
                    for t in ready
                    if (s := self._rg_states.get(t.row_group)) is not None and s.pre_batch_done or t.column in seed_cols
                ]
            for task in ready:
                await self._submission_semaphore.acquire()
                self._dispatched.add(task)
                self._in_flight.add(task)
                if (s := self._rg_states.get(task.row_group)) is not None:
                    s.in_flight_count += 1
                self._spawn_worker(self._execute_task(task))
            if not ready and not self._in_flight:
                break
            if not self._in_flight:
                continue
            self._wake_event.clear()
            await self._wake_event.wait()

    async def _salvage_stalled_row_groups(
        self,
        seed_cols: frozenset[str],
        has_pre_batch: bool,
        all_columns: list[str],
    ) -> None:
        """Salvage row groups whose tasks are all deferred (0 in-flight).

        Retries deferred tasks inline so the row groups can be checkpointed
        and their semaphore slots freed, preventing deadlock when admission
        is blocked.
        """
        stalled_rgs = {
            t.row_group
            for t in self._deferred
            if (s := self._rg_states.get(t.row_group)) is not None and s.in_flight_count == 0
        }
        if not stalled_rgs:
            return

        num_rgs = len(self._row_groups)
        width = len(str(num_rgs))
        for rg_id in sorted(stalled_rgs):
            rg_deferred = [t for t in self._deferred if t.row_group == rg_id]
            logger.info(f"🔄 ({rg_id + 1:0{width}d}/{num_rgs}) Salvaging {len(rg_deferred)} deferred task(s)")

        # Partition deferred into stalled (retry now) and other (keep for later).
        stalled_deferred = [t for t in self._deferred if t.row_group in stalled_rgs]
        other_deferred = [t for t in self._deferred if t.row_group not in stalled_rgs]
        self._deferred = stalled_deferred
        await self._salvage_rounds(seed_cols, has_pre_batch, all_columns)
        # Separate stalled tasks that exhausted retries from any new failures
        # that _drain_frontier may have appended for non-stalled row groups.
        exhausted = [t for t in self._deferred if t.row_group in stalled_rgs]
        newly_deferred = [t for t in self._deferred if t.row_group not in stalled_rgs]
        for task in exhausted:
            # If the row was already dropped by an earlier task in this loop,
            # the skip was already counted; don't also record a failure.
            already_dropped = task.row_index is not None and self._tracker.is_dropped(task.row_group, task.row_index)
            if not already_dropped and self._reporter:
                self._reporter.record_failure(task.column)
            if task.row_index is not None:
                self._drop_row(task.row_group, task.row_index, exclude_columns={task.column})
            else:
                rg_size = self._get_rg_size(task.row_group)
                self._drop_row_group(task.row_group, rg_size, exclude_columns={task.column})
        self._checkpoint_completed_row_groups(all_columns)
        self._deferred = other_deferred + newly_deferred

    def _checkpoint_completed_row_groups(self, all_columns: list[str]) -> None:
        """Checkpoint any row groups that reached completion."""
        completed = [
            (rg_id, state.size)
            for rg_id, state in self._rg_states.items()
            if self._tracker.is_row_group_complete(rg_id, state.size, all_columns)
        ]
        for rg_id, rg_size in completed:
            dropped = False
            try:
                del self._rg_states[rg_id]
                if self._on_before_checkpoint:
                    try:
                        self._on_before_checkpoint(rg_id, rg_size)
                    except Exception:
                        # Post-batch is mandatory; drop rather than checkpoint unprocessed data.
                        logger.error(
                            f"on_before_checkpoint failed for row group {rg_id}, dropping row group.",
                            exc_info=True,
                        )
                        self._drop_row_group(rg_id, rg_size)
                        if self._buffer_manager:
                            self._buffer_manager.free_row_group(rg_id)
                        dropped = True
                # If all rows were dropped (e.g. seed failure), free instead of finalizing
                if not dropped and all(self._tracker.is_dropped(rg_id, ri) for ri in range(rg_size)):
                    if self._buffer_manager:
                        self._buffer_manager.free_row_group(rg_id)
                    dropped = True
                if not dropped and self._on_finalize_row_group is not None:
                    self._on_finalize_row_group(rg_id)
            except Exception:
                logger.error(f"Failed to checkpoint row group {rg_id}.", exc_info=True)
            finally:
                self._rg_semaphore.release()

        # Clean up deferred tasks for checkpointed row groups
        if completed:
            checkpointed = {rg_id for rg_id, _ in completed}
            self._deferred = [t for t in self._deferred if t.row_group not in checkpointed]

    def _run_seeds_complete_check(self, seed_cols: frozenset[str]) -> None:
        """Run pre-batch callbacks for row groups whose seeds just completed."""
        for rg_id, state in list(self._rg_states.items()):
            if state.seeds_dispatched and not state.pre_batch_done:
                all_seeds_done = all(self._tracker.is_column_complete_for_rg(col, rg_id) for col in seed_cols)
                if all_seeds_done and state.in_flight_count == 0:
                    state.pre_batch_done = True
                    if self._on_seeds_complete:
                        try:
                            self._on_seeds_complete(rg_id, state.size)
                            # The callback may drop rows (e.g. pre-batch filtering).
                            # Record skipped tasks for any newly-dropped rows so
                            # progress reporting stays accurate.
                            if self._reporter:
                                for ri in range(state.size):
                                    if self._tracker.is_dropped(rg_id, ri):
                                        self._record_skipped_tasks_for_row(rg_id, ri)
                        except Exception:
                            logger.warning(
                                f"Pre-batch processor failed for row group {rg_id}, skipping.",
                                exc_info=True,
                            )
                            self._drop_row_group(rg_id, state.size)

    def _drop_row(self, row_group: int, row_index: int, *, exclude_columns: set[str] | None = None) -> None:
        if self._tracker.is_dropped(row_group, row_index):
            return

        self._record_skipped_tasks_for_row(row_group, row_index, exclude_columns=exclude_columns)
        self._tracker.drop_row(row_group, row_index)
        if self._buffer_manager:
            self._buffer_manager.drop_row(row_group, row_index)

    def _drop_row_group(self, row_group: int, row_group_size: int, *, exclude_columns: set[str] | None = None) -> None:
        for row_index in range(row_group_size):
            self._drop_row(row_group, row_index, exclude_columns=exclude_columns)

    def _record_skipped_tasks_for_row(
        self,
        row_group: int,
        row_index: int,
        *,
        exclude_columns: set[str] | None = None,
    ) -> None:
        if self._reporter is None:
            return

        excluded = exclude_columns or set()
        in_flight_columns = {
            task.column for task in self._in_flight if task.row_group == row_group and task.row_index == row_index
        }

        for column in self._graph.columns:
            if column in excluded or self._graph.get_strategy(column) != GenerationStrategy.CELL_BY_CELL:
                continue
            if column in in_flight_columns:
                continue
            if self._tracker.is_complete(SliceRef(column=column, row_group=row_group, row_index=row_index)):
                continue
            self._reporter.record_skipped(column)

    def _check_error_rate(self, *, success: bool) -> None:
        """Trigger early shutdown if recent error rate exceeds threshold."""
        if self._disable_early_shutdown or self._early_shutdown:
            return
        self._recent_outcomes.append(success)
        if len(self._recent_outcomes) < self._shutdown_error_window:
            return
        errors = sum(1 for ok in self._recent_outcomes if not ok)
        if errors / self._shutdown_error_window >= self._shutdown_error_rate:
            self._early_shutdown = True

    async def _dispatch_seeds(self, rg_id: int, rg_size: int) -> None:
        """Dispatch from_scratch tasks for a row group."""
        self._rg_states[rg_id].seeds_dispatched = True
        seed_cols = self._seed_cols
        if not seed_cols:
            return
        num_rgs = len(self._rg_size_map)
        width = len(str(num_rgs))
        logger.info(f"🚀 ({rg_id + 1:0{width}d}/{num_rgs}) Dispatching with {rg_size} records")
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
            if (s := self._rg_states.get(task.row_group)) is not None:
                s.in_flight_count += 1
            self._spawn_worker(self._execute_seed_task(task, gid))

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
        """Core task execution logic.

        For LLM-bound tasks, uses a one-way semaphore handoff: acquires the
        LLM-wait slot while still holding the submission slot, then releases
        the submission slot (never reacquired).  This prevents cross-key
        starvation while bounding live coroutines.
        """
        num_rgs = len(self._row_groups)
        token = current_row_group.set((task.row_group, num_rgs))
        try:
            await self._execute_task_inner_impl(task)
        finally:
            current_row_group.reset(token)

    async def _execute_task_inner_impl(self, task: Task) -> None:
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
        is_llm = self._llm_bound_lookup.get(task.column, False)
        holds_submission = True
        holds_llm_wait = False

        try:
            # Skip tasks whose row group was already checkpointed (can happen
            # when a vacuously-ready downstream is dispatched via create_task
            # in the same loop iteration that checkpoints the row group).
            if task.row_group not in self._rg_states:
                skipped = True
                return

            if is_llm:
                await self._llm_wait_semaphore.acquire()
                holds_llm_wait = True
                self._submission_semaphore.release()
                holds_submission = False

            if self._trace and trace:
                trace.slot_acquired_at = time.perf_counter()

            cell_skipped = False
            if task.task_type == "from_scratch":
                await self._run_from_scratch(task, generator)
            elif task.task_type == "cell":
                _result, cell_skipped = await self._run_cell(task, generator)
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

            self._check_error_rate(success=True)
            if self._reporter:
                if cell_skipped:
                    self._reporter.record_skipped(task.column)
                else:
                    self._reporter.record_success(task.column)
            if self._trace and trace:
                trace.status = "ok"

        except Exception as exc:
            if not isinstance(exc, ModelRateLimitError):
                self._check_error_rate(success=False)
            retryable = self._is_retryable(exc)
            if not retryable and self._reporter:
                self._reporter.record_failure(task.column)
            if self._trace and trace:
                trace.status = "error"
                trace.error = str(exc)

            if retryable:
                self._deferred.append(task)
            else:
                # Non-retryable: drop the affected row(s)
                if task.row_index is not None:
                    self._drop_row(task.row_group, task.row_index, exclude_columns={task.column})
                else:
                    # Batch/from_scratch failure: drop all rows in the row group
                    rg_size = self._get_rg_size(task.row_group)
                    self._drop_row_group(task.row_group, rg_size, exclude_columns={task.column})
                logger.warning(
                    f"Non-retryable failure on {task.column}[rg={task.row_group}, row={task.row_index}]: {exc}"
                )

        finally:
            if self._trace and trace:
                trace.completed_at = time.perf_counter()
                self.traces.append(trace)

            self._in_flight.discard(task)
            if (s := self._rg_states.get(task.row_group)) is not None:
                s.in_flight_count = max(0, s.in_flight_count - 1)
            if not retryable and not skipped:
                self._dispatched.discard(task)
            if holds_llm_wait:
                self._llm_wait_semaphore.release()
            if holds_submission:
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

    async def _run_cell(self, task: Task, generator: ColumnGenerator) -> tuple[Any, bool]:
        """Execute a cell-by-cell task. Returns ``(result, skipped)``."""
        if task.row_index is None:
            raise ValueError(f"Cell task requires a row_index, got None for column '{task.column}'")

        if self._tracker.is_dropped(task.row_group, task.row_index):
            return None, True

        # Evaluate skip against the live buffer record (no copy needed —
        # there is no `await` between the read and the provenance write).
        if self._buffer_manager is not None:
            record = self._buffer_manager.get_row(task.row_group, task.row_index)
        else:
            record = {}

        # Skip evaluation: propagation first, then expression gate
        skipped_cols = get_skipped_column_names(record)
        if self._graph.should_propagate_skip(task.column):
            required = self._graph.get_required_columns(task.column)
            if should_skip_by_propagation(required, skipped_cols, propagate_skip=True):
                self._apply_skip_to_record(task, record)
                return None, True

        skip_config = self._graph.get_skip_config(task.column)
        if skip_config is not None:
            if evaluate_skip_when(skip_config.when, record):
                self._apply_skip_to_record(task, record)
                return None, True

        # Copy for generation: agenerate crosses an await boundary, so the
        # generator must not hold a mutable reference to the live record.
        result = await generator.agenerate(dict(record))

        # Write back to buffer
        if self._buffer_manager is not None and not self._tracker.is_dropped(task.row_group, task.row_index):
            output_cols = self._instance_to_columns.get(id(generator), [task.column])
            for col in output_cols:
                if col in result:
                    self._buffer_manager.update_cell(task.row_group, task.row_index, col, result[col])

        return result, False

    def _apply_skip_to_record(self, task: Task, record: dict) -> None:
        """Write skip provenance directly into *record* (the live buffer row)."""
        skip_config = self._graph.get_skip_config(task.column)
        skip_value = skip_config.value if skip_config is not None else None
        apply_skip_to_record(
            record,
            column_name=task.column,
            cell_value=skip_value,
            side_effect_columns=self._graph.get_side_effect_columns(task.column),
        )

    async def _run_batch(self, task: Task, generator: ColumnGenerator) -> Any:
        """Execute a full-column/batch task."""
        from data_designer.engine.dataset_builders.multi_column_configs import MultiColumnConfig

        rg_size = self._get_rg_size(task.row_group)

        if self._buffer_manager is not None:
            pre_dropped: set[int] = {ri for ri in range(rg_size) if self._buffer_manager.is_dropped(task.row_group, ri)}

            # Skip evaluation only applies to single-column configs.
            # Multi-column configs (sampler/seed) are rejected by the SkipConfig
            # model validator, so they never carry skip metadata.
            pre_skipped: set[int] = set()
            is_multi = isinstance(generator.config, MultiColumnConfig)
            if not is_multi:
                for ri in range(rg_size):
                    if ri in pre_dropped:
                        continue
                    row_data = dict(self._buffer_manager.get_row(task.row_group, ri))
                    skipped_cols = get_skipped_column_names(row_data)

                    should_skip = False
                    if self._graph.should_propagate_skip(task.column):
                        required = self._graph.get_required_columns(task.column)
                        if should_skip_by_propagation(required, skipped_cols, propagate_skip=True):
                            should_skip = True

                    if not should_skip:
                        skip_config = self._graph.get_skip_config(task.column)
                        if skip_config is not None:
                            should_skip = evaluate_skip_when(skip_config.when, row_data)

                    if should_skip:
                        record = self._buffer_manager.get_row(task.row_group, ri)
                        sc = self._graph.get_skip_config(task.column)
                        apply_skip_to_record(
                            record,
                            column_name=task.column,
                            cell_value=sc.value if sc is not None else None,
                            side_effect_columns=self._graph.get_side_effect_columns(task.column),
                        )
                        pre_skipped.add(ri)

            # Build DataFrame excluding dropped and skipped rows
            active_rows_data = [
                self._buffer_manager.get_row(task.row_group, ri)
                for ri in range(rg_size)
                if ri not in pre_dropped and ri not in pre_skipped
            ]
            batch_df = (
                lazy.pd.DataFrame(strip_skip_metadata_from_records(active_rows_data))
                if active_rows_data
                else lazy.pd.DataFrame()
            )
        else:
            batch_df = lazy.pd.DataFrame()
            pre_dropped = set()
            pre_skipped = set()

        if len(batch_df) == 0:
            return batch_df

        result_df = await generator.agenerate(batch_df)

        # Merge result columns back to buffer
        if self._buffer_manager is not None:
            output_cols = self._instance_to_columns.get(id(generator), [task.column])
            active_rows = rg_size - len(pre_dropped) - len(pre_skipped)
            if len(result_df) != active_rows:
                raise ValueError(
                    f"Batch generator for '{task.column}' returned {len(result_df)} rows "
                    f"but {active_rows} were expected (rg={task.row_group})."
                )
            result_idx = 0
            for ri in range(rg_size):
                if ri in pre_dropped or ri in pre_skipped:
                    continue
                if not self._buffer_manager.is_dropped(task.row_group, ri):
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

    def get_semaphore_permits(self) -> tuple[int, int]:
        """Return ``(submission_available, llm_wait_available)`` for diagnostics."""
        return (
            self._submission_semaphore.available_permits,
            self._llm_wait_semaphore.available_permits,
        )

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Classify whether an exception is retryable."""
        return isinstance(exc, _RETRYABLE_MODEL_ERRORS)


def build_llm_bound_lookup(generators: dict[str, ColumnGenerator]) -> dict[str, bool]:
    return {col: gen.is_llm_bound for col, gen in generators.items()}
