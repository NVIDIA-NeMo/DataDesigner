# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import GenerationStrategy
from data_designer.engine.capacity import (
    AsyncCapacityConfigured,
    AsyncCapacityObservedMaxima,
    AsyncCapacityPlan,
    AsyncCapacityRuntimeSnapshot,
    CapacityValue,
    RowGroupAdmission,
)
from data_designer.engine.context import current_row_group
from data_designer.engine.dataset_builders.errors import DatasetGenerationError
from data_designer.engine.dataset_builders.multi_column_configs import MultiColumnConfig
from data_designer.engine.dataset_builders.scheduling.completion import CompletionTracker, FrontierDelta
from data_designer.engine.dataset_builders.scheduling.queue import (
    FairTaskQueue,
)
from data_designer.engine.dataset_builders.scheduling.resolver import TaskSchedulingResolver
from data_designer.engine.dataset_builders.scheduling.resources import (
    SchedulableTask,
    stable_task_id,
)
from data_designer.engine.dataset_builders.scheduling.task_admission import (
    TaskAdmissionConfig,
    TaskAdmissionController,
    TaskAdmissionDenied,
    TaskAdmissionLease,
)
from data_designer.engine.dataset_builders.scheduling.task_model import SliceRef, Task, TaskTrace
from data_designer.engine.dataset_builders.utils.async_progress_reporter import (
    DEFAULT_REPORT_INTERVAL,
    AsyncProgressReporter,
)
from data_designer.engine.dataset_builders.utils.progress_tracker import ProgressTracker
from data_designer.engine.dataset_builders.utils.skip_evaluator import should_skip_column_for_record
from data_designer.engine.dataset_builders.utils.skip_tracker import (
    apply_skip_to_record,
    strip_skip_metadata_from_records,
)
from data_designer.engine.dataset_builders.utils.sticky_progress_bar import StickyProgressBar
from data_designer.engine.models.errors import RETRYABLE_MODEL_ERRORS
from data_designer.engine.observability import (
    RuntimeCorrelation,
    SchedulerAdmissionEvent,
    SchedulerAdmissionEventSink,
    runtime_correlation_provider,
)

if TYPE_CHECKING:
    from data_designer.engine.column_generators.generators.base import ColumnGenerator
    from data_designer.engine.dataset_builders.utils.execution_graph import ExecutionGraph
    from data_designer.engine.dataset_builders.utils.row_group_buffer import RowGroupBufferManager

logger = logging.getLogger(__name__)

DEFAULT_TASK_POOL_SIZE: int = 256
MODEL_TASK_ADMISSION_HEADROOM_MULTIPLIER: int = 2
MODEL_GROUP_ADMISSION_BACKLOG_MULTIPLIER: int = 2

# Degraded-provider WARN: emit at most one warning per interval when the
# rolling fraction of retryable errors exceeds the threshold. Distinct from
# the early-shutdown gate (which fires on non-retryable errors).
# TODO: thread these through RunConfig so users can tune them per run.
DEGRADED_WARN_RATE: float = 0.5
DEGRADED_WARN_WINDOW: int = 20
DEGRADED_WARN_INTERVAL_S: float = 60.0


@dataclass
class _RowGroupState:
    """Lifecycle state for a single admitted row group."""

    size: int
    seeds_dispatched: bool = False
    pre_batch_done: bool = False
    in_flight_count: int = 0


@dataclass(frozen=True)
class _DispatchOutcome:
    """Result of one fair-dispatch pass over the persistent ready queue."""

    dispatched: bool = False
    admission_blocked: bool = False


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
        max_model_task_admission: int = DEFAULT_TASK_POOL_SIZE,
        task_admission_config: TaskAdmissionConfig | None = None,
        salvage_max_rounds: int = 2,
        on_finalize_row_group: Callable[[int], None] | None = None,
        on_seeds_complete: Callable[[int, int], FrontierDelta | None] | None = None,
        on_before_checkpoint: Callable[[int, int], None] | None = None,
        shutdown_error_rate: float = 0.5,
        shutdown_error_window: int = 10,
        disable_early_shutdown: bool = False,
        degraded_warn_rate: float = DEGRADED_WARN_RATE,
        degraded_warn_window: int = DEGRADED_WARN_WINDOW,
        degraded_warn_interval_s: float = DEGRADED_WARN_INTERVAL_S,
        trace: bool = False,
        num_records: int = 0,
        buffer_size: int = 0,
        progress_interval: float | None = None,
        progress_bar: bool = False,
        scheduler_event_sink: SchedulerAdmissionEventSink | None = None,
        run_id: str | None = None,
    ) -> None:
        self._generators = generators
        self._graph = graph
        self._tracker = tracker
        self._row_groups = row_groups
        self._buffer_manager = buffer_manager

        self._rg_semaphore = asyncio.Semaphore(max_concurrent_row_groups)

        self._task_scheduling = TaskSchedulingResolver(
            generators,
            model_group_limit_multiplier=MODEL_GROUP_ADMISSION_BACKLOG_MULTIPLIER,
            model_group_limit_cap=max_model_task_admission,
        )
        admission_config = task_admission_config or TaskAdmissionConfig(
            submission_capacity=max_submitted_tasks,
            resource_limits={"llm_wait": max_model_task_admission, "local": max_submitted_tasks},
        )
        self._task_admission = TaskAdmissionController(admission_config)
        self._task_admission_config = admission_config
        self._fair_queue = FairTaskQueue()
        self._pending_pre_batch_ready: defaultdict[int, list[Task]] = defaultdict(list)
        self._pending_pre_batch_ready_tasks: set[Task] = set()

        self._dispatched: set[Task] = set()
        self._in_flight: set[Task] = set()
        self._worker_tasks: set[asyncio.Task] = set()
        self._wake_event = asyncio.Event()
        self._run_id = run_id or f"run-{uuid.uuid4().hex}"
        self._scheduler_event_sink = scheduler_event_sink
        self._scheduler_event_sequence = 0
        self._salvage_max_rounds = salvage_max_rounds
        self._on_finalize_row_group = on_finalize_row_group
        self._on_seeds_complete = on_seeds_complete
        self._on_before_checkpoint = on_before_checkpoint

        # Error rate shutdown (caller passes pre-normalized values via RunConfig)
        self._shutdown_error_rate = shutdown_error_rate
        self._shutdown_error_window = shutdown_error_window
        self._disable_early_shutdown = disable_early_shutdown
        self._early_shutdown = False

        # Multi-column dedup: group output columns by generator identity.
        # _gen_instance_to_columns holds only real (graph-registered) columns
        # and is used for completion tracking.
        # _gen_instance_to_columns_including_side_effects extends that with
        # side-effect columns for buffer writes only.
        gen_instance_to_columns: dict[int, list[str]] = {}
        for col, gen in generators.items():
            gen_instance_to_columns.setdefault(id(gen), []).append(col)
        self._gen_instance_to_columns = gen_instance_to_columns

        seen_cols: set[str] = {col for col in generators}
        gen_instance_to_columns_incl_se: dict[int, list[str]] = {k: list(v) for k, v in gen_instance_to_columns.items()}
        for col, gen in generators.items():
            for side_effect_col in getattr(gen.config, "side_effect_columns", []):
                if side_effect_col not in seen_cols:
                    gen_instance_to_columns_incl_se.setdefault(id(gen), []).append(side_effect_col)
                    seen_cols.add(side_effect_col)
        self._gen_instance_to_columns_including_side_effects = gen_instance_to_columns_incl_se

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

        # Degraded-provider WARN: separate window tracking retryable-vs-not for
        # every outcome (success or failure), rate-limited to one log per interval.
        self._degraded_warn_rate = degraded_warn_rate
        self._degraded_warn_window = degraded_warn_window
        self._degraded_warn_interval_s = degraded_warn_interval_s
        self._recent_retryable: deque[bool] = deque(maxlen=degraded_warn_window)
        # Initialize to -inf so the first WARN is always emitted regardless of
        # the monotonic clock's absolute value (which can be near-zero on freshly
        # booted CI runners).
        self._last_degraded_warn_at: float = float("-inf")

        # Row groups that were partially salvaged after early shutdown
        # (i.e., some rows complete, some incomplete-then-dropped). Surfaced
        # via the partial_row_groups property as a structured signal.
        self._partial_row_groups: list[int] = []

        # First non-retryable error encountered, if any. Surfaced via the
        # ``first_non_retryable_error`` property so the interface can include
        # the original cause in user-facing errors when a run produces 0 records
        # (e.g. a deterministic seed-source failure). Sync engine preserved this
        # context naturally because the from_scratch task raised; the async
        # engine drops rows and continues, losing the cause unless we capture it.
        self._first_non_retryable_error: Exception | None = None

        # Pre-compute row-group sizes for O(1) lookup
        self._rg_size_map: dict[int, int] = dict(row_groups)
        self._max_concurrent_row_groups = max_concurrent_row_groups
        self._max_submitted_tasks = max_submitted_tasks
        self._max_model_task_admission = max_model_task_admission
        self._num_records = num_records
        self._buffer_size = buffer_size
        self._observed_max_row_groups_in_flight = 0
        self._observed_max_task_leases_by_resource: dict[str, int] = {}
        self._observed_max_queued_by_group: dict[str, int] = {}

        # Pre-compute seed columns (graph is static)
        self._seed_cols: tuple[str, ...] = tuple(c for c in graph.columns if not graph.get_upstream_columns(c))

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

    @property
    def active_worker_count(self) -> int:
        return sum(1 for t in self._worker_tasks if not t.done())

    @property
    def early_shutdown(self) -> bool:
        """True if the run terminated via the early-shutdown gate."""
        return self._early_shutdown

    @property
    def partial_row_groups(self) -> tuple[int, ...]:
        """Row group ids that were partially salvaged after early shutdown.

        Empty unless ``early_shutdown`` is True. Each id had some rows
        complete and the rest dropped before checkpointing.
        """
        return tuple(self._partial_row_groups)

    @property
    def first_non_retryable_error(self) -> Exception | None:
        """The first non-retryable error captured by the scheduler, if any.

        Surfaced so callers can preserve the original cause when a run produces
        0 records due to deterministic failures (e.g. invalid seed sources).
        Returns ``None`` for runs that completed without non-retryable errors.
        """
        return self._first_non_retryable_error

    def _spawn_worker(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task:
        """Create a tracked worker task that auto-removes itself on completion."""
        task = asyncio.create_task(coro)
        self._worker_tasks.add(task)
        task.add_done_callback(self._worker_tasks.discard)
        return task

    def _emit_scheduler_event(
        self,
        event_kind: str,
        *,
        task: Task | None = None,
        lease: TaskAdmissionLease | None = None,
        task_execution_id: str | None = None,
        scheduler_resource_key: str | None = None,
        reason_or_result: str | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> None:
        if self._scheduler_event_sink is None:
            return
        self._scheduler_event_sequence += 1
        correlation = None
        if task is not None:
            group = lease.item.group if lease is not None else self._schedulable_task(task).group
            identity_hash = hashlib.sha1("\0".join(group.key.identity).encode()).hexdigest()[:16]
            correlation = RuntimeCorrelation(
                run_id=self._run_id,
                row_group=task.row_group,
                task_column=task.column,
                task_type=task.task_type,
                scheduling_group_kind=group.key.kind,
                scheduling_group_identity_hash=identity_hash,
                task_execution_id=task_execution_id,
            )
        try:
            self._scheduler_event_sink.emit_scheduler_event(
                SchedulerAdmissionEvent.capture(
                    event_kind,  # type: ignore[arg-type]
                    sequence=self._scheduler_event_sequence,
                    correlation=correlation,
                    task_id=stable_task_id(task) if task is not None else None,
                    task_execution_id=task_execution_id,
                    task_lease_id=lease.lease_id if lease is not None else None,
                    scheduler_resource_key=scheduler_resource_key,
                    reason_or_result=reason_or_result,
                    snapshot=self.task_admission_snapshot(),
                    diagnostics=diagnostics or {},
                )
            )
        except Exception:
            return

    def _record_observed_task_state(self) -> None:
        self._observed_max_row_groups_in_flight = max(self._observed_max_row_groups_in_flight, len(self._rg_states))
        view = self._task_admission.view()
        for resource, count in view.leased_resources.items():
            self._observed_max_task_leases_by_resource[resource] = max(
                self._observed_max_task_leases_by_resource.get(resource, 0),
                count,
            )
        queue_view = self._fair_queue.view()
        for group, count in queue_view.queued_by_group.items():
            label = f"{group.kind}:{'/'.join(group.identity)}"
            self._observed_max_queued_by_group[label] = max(self._observed_max_queued_by_group.get(label, 0), count)

    async def _cancel_workers(self) -> None:
        """Cancel all tracked worker tasks and wait for them to finish."""
        for t in self._worker_tasks:
            t.cancel()
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()

    def _apply_frontier_delta(self, delta: FrontierDelta) -> None:
        if delta.empty:
            return
        for task in delta.removed:
            self._discard_ready_task(task)
        for task in delta.added:
            self._enqueue_ready_task(task)

    def _enqueue_ready_task(self, task: Task) -> None:
        if task in self._dispatched or task.row_group not in self._rg_states:
            return
        if not self._tracker.is_frontier_task(task):
            return
        self._emit_scheduler_event("dependency_ready", task=task)
        state = self._rg_states[task.row_group]
        if self._on_seeds_complete is not None and not state.pre_batch_done and task.column not in self._seed_cols:
            if task not in self._pending_pre_batch_ready_tasks:
                self._pending_pre_batch_ready[task.row_group].append(task)
                self._pending_pre_batch_ready_tasks.add(task)
            return
        schedulable = self._schedulable_task(task)
        accepted = self._fair_queue.enqueue((schedulable,))
        if accepted:
            self._tracker.mark_enqueued(accepted)
            self._emit_scheduler_event("ready_enqueued", task=task)
            self._record_observed_task_state()
            self._wake_event.set()

    def _discard_ready_task(self, task: Task) -> None:
        self._fair_queue.discard(stable_task_id(task))
        self._pending_pre_batch_ready_tasks.discard(task)

    def _flush_pre_batch_ready(self, row_group: int) -> None:
        pending = self._pending_pre_batch_ready.pop(row_group, [])
        for task in pending:
            if task not in self._pending_pre_batch_ready_tasks:
                continue
            self._pending_pre_batch_ready_tasks.discard(task)
            self._enqueue_ready_task(task)

    def _drop_pending_ready_for_row_group(self, row_group: int) -> None:
        pending = self._pending_pre_batch_ready.pop(row_group, [])
        for task in pending:
            self._pending_pre_batch_ready_tasks.discard(task)
        self._fair_queue.discard_where(lambda item: item.payload.row_group == row_group)

    def _dispatch_queued_tasks(self) -> _DispatchOutcome:
        dispatched = False

        while self._fair_queue.has_queued_tasks:
            selection = self._fair_queue.select_next(lambda item, view: self._task_admission.is_eligible(item, view))
            if selection is None:
                summary = self._task_admission.explain_blocked(self._fair_queue.view())
                if "group_cap" in summary.dominant_denial_reasons:
                    event_kind = "group_capped"
                elif summary.dominant_denial_reasons:
                    event_kind = "admission_blocked"
                else:
                    event_kind = "queue_empty"
                self._emit_scheduler_event(
                    event_kind,
                    diagnostics={
                        "queued_count": summary.queued_count,
                        "reasons": dict(summary.dominant_denial_reasons),
                    },
                )
                return _DispatchOutcome(dispatched=dispatched, admission_blocked=True)

            self._emit_scheduler_event("selected", task=selection.item.payload)
            decision = self._task_admission.try_acquire(selection.item, selection.queue_view)
            if isinstance(decision, TaskAdmissionDenied):
                self._emit_scheduler_event(
                    "admission_denied",
                    task=selection.item.payload,
                    reason_or_result=decision.reason,
                    diagnostics=dict(decision.diagnostics),
                )
                return _DispatchOutcome(dispatched=dispatched, admission_blocked=True)
            self._emit_scheduler_event("task_lease_acquired", task=selection.item.payload, lease=decision)

            committed = self._fair_queue.commit(selection)
            if committed is None:
                result = self._task_admission.release(decision)
                self._emit_scheduler_event(
                    "stale_selection",
                    task=selection.item.payload,
                    lease=decision,
                    reason_or_result=result.reason,
                )
                return _DispatchOutcome(dispatched=dispatched, admission_blocked=True)

            self._dispatch_selected_task(committed, decision)
            dispatched = True
            self._record_observed_task_state()

        if dispatched:
            self._emit_scheduler_event("queue_drained")
        return _DispatchOutcome(dispatched=dispatched)

    def _dispatch_selected_task(self, item: SchedulableTask, lease: TaskAdmissionLease) -> None:
        task = item.payload
        task_execution_id = f"task-exec-{uuid.uuid4().hex}"
        self._dispatched.add(task)
        self._in_flight.add(task)
        if (s := self._rg_states.get(task.row_group)) is not None:
            s.in_flight_count += 1
        try:
            self._spawn_worker(self._execute_task(task, lease, task_execution_id))
            self._emit_scheduler_event("worker_spawned", task=task, lease=lease, task_execution_id=task_execution_id)
        except Exception:
            result = self._task_admission.release(lease)
            self._emit_scheduler_event(
                "worker_spawn_failed",
                task=task,
                lease=lease,
                task_execution_id=task_execution_id,
                reason_or_result=result.reason,
            )
            self._in_flight.discard(task)
            raise

    def _schedulable_task(self, task: Task) -> SchedulableTask:
        return self._task_scheduling.schedulable_task(task, self._task_flow_identity(task))

    def _task_flow_identity(self, task: Task) -> tuple[str, ...]:
        generator = self._generators[task.column]
        output_columns = self._gen_instance_to_columns.get(id(generator), [task.column])
        return tuple(output_columns)

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
            finally:
                # Always cancel admission + drain in-flight workers, regardless
                # of how the dispatch loop exited (normal, early shutdown,
                # CancelledError, or processor failure).
                if not admission_task.done():
                    admission_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await admission_task
                await asyncio.shield(self._cancel_workers())
                # Salvage partially-complete row groups left over from early
                # shutdown. Must run AFTER _cancel_workers - in-flight tasks
                # could otherwise write into a buffer that's being finalized.
                if self._early_shutdown and self._rg_states:
                    self._finalize_after_shutdown(all_columns)

            # Reached only on the clean-exit path; an exception in the
            # dispatch loop or the finally block propagates and skips this.
            if self._reporter:
                self._reporter.log_final()

            if self._rg_states:
                incomplete = list(self._rg_states)
                logger.error(
                    f"Scheduler exited with {len(self._rg_states)} unfinished row group(s): {incomplete}. "
                    "These row groups were not checkpointed."
                )

    async def _main_dispatch_loop(
        self,
        seed_cols: tuple[str, ...],
        has_pre_batch: bool,
        all_columns: list[str],
    ) -> None:
        """Core dispatch loop extracted from ``run()``."""
        while True:
            if self._early_shutdown:
                logger.warning("Early shutdown triggered - non-retryable error rate exceeded threshold")
                if self._deferred:
                    await self._salvage_stalled_row_groups(seed_cols, has_pre_batch, all_columns)
                self._checkpoint_completed_row_groups(all_columns)
                break

            self._wake_event.clear()

            if has_pre_batch:
                self._run_seeds_complete_check(seed_cols)

            dispatch_outcome = self._dispatch_queued_tasks()

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

            pending_pre_batch = has_pre_batch and any(
                state.seeds_dispatched and not state.pre_batch_done for state in self._rg_states.values()
            )
            if not self._fair_queue.has_queued_tasks and not self._in_flight:
                if self._all_rgs_admitted and not pending_pre_batch:
                    break
                if pending_pre_batch:
                    continue

            if not self._fair_queue.has_queued_tasks or dispatch_outcome.admission_blocked:
                await self._wake_event.wait()

    async def _salvage_rounds(
        self,
        seed_cols: tuple[str, ...],
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
                    for sibling in self._gen_instance_to_columns.get(gid, []):
                        if sibling != task.column:
                            self._dispatched.discard(
                                Task(column=sibling, row_group=task.row_group, row_index=None, task_type="from_scratch")
                            )
                            self._dispatched.discard(
                                Task(column=sibling, row_group=task.row_group, row_index=None, task_type="batch")
                            )
                    self._apply_frontier_delta(self._tracker.add_ready_tasks((task,)))
                else:
                    self._dispatched.discard(task)
                    self._apply_frontier_delta(self._tracker.add_ready_tasks((task,)))
            # Drain: dispatch frontier tasks and any newly-ready downstream tasks
            # until nothing remains in-flight or in the frontier.
            await self._drain_frontier(seed_cols, has_pre_batch)
            self._checkpoint_completed_row_groups(all_columns)

    async def _drain_frontier(self, seed_cols: tuple[str, ...], has_pre_batch: bool) -> None:
        """Dispatch all frontier tasks and their downstream until quiescent."""
        while True:
            if has_pre_batch:
                self._run_seeds_complete_check(seed_cols)
            dispatch_outcome = self._dispatch_queued_tasks()
            has_queued = self._fair_queue.has_queued_tasks
            if not has_queued and not self._in_flight:
                break
            if has_queued and not dispatch_outcome.dispatched and not self._in_flight:
                raise RuntimeError(
                    "Ready frontier is admission-blocked with no in-flight task to release scheduler capacity."
                )
            if not self._in_flight:
                continue
            self._wake_event.clear()
            await self._wake_event.wait()

    async def _salvage_stalled_row_groups(
        self,
        seed_cols: tuple[str, ...],
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
            try:
                if self._on_before_checkpoint:
                    try:
                        self._on_before_checkpoint(rg_id, rg_size)
                    except DatasetGenerationError:
                        raise
                    except Exception as exc:
                        raise DatasetGenerationError(
                            f"Post-batch processor failed for row group {rg_id}: {exc}"
                        ) from exc
                # Remove from tracking only after the callback succeeds.
                del self._rg_states[rg_id]
                # If all rows were dropped (e.g. seed failure), free instead of finalizing
                if all(self._tracker.is_dropped(rg_id, ri) for ri in range(rg_size)):
                    if self._buffer_manager:
                        self._buffer_manager.free_row_group(rg_id)
                elif self._on_finalize_row_group is not None:
                    self._on_finalize_row_group(rg_id)
            except DatasetGenerationError:
                raise
            except Exception:
                logger.error(f"Failed to checkpoint row group {rg_id}.", exc_info=True)
            finally:
                self._rg_semaphore.release()

        # Clean up deferred tasks for checkpointed row groups
        if completed:
            checkpointed = {rg_id for rg_id, _ in completed}
            self._deferred = [t for t in self._deferred if t.row_group not in checkpointed]
            for rg_id in checkpointed:
                self._drop_pending_ready_for_row_group(rg_id)

    def _finalize_after_shutdown(self, all_columns: list[str]) -> None:
        """Salvage row groups left in flight when early shutdown fired.

        For each remaining row group, drop rows that aren't fully complete
        (and weren't already dropped); after that, ``is_row_group_complete``
        is true by construction over the surviving rows, so delegating to
        ``_checkpoint_completed_row_groups`` writes survivors and frees
        zero-survivor groups via the buffer manager's existing logic.

        Note on processors: ``_checkpoint_completed_row_groups`` calls
        ``on_before_checkpoint`` (post-batch) but never ``on_seeds_complete``
        (pre-batch). If the gate fires before seeds completed for a row
        group, that row group's pre-batch processor never ran. Survivors
        are checkpointed without it. This is the existing contract for
        partial-row-group salvage.
        """
        for rg_id in list(self._rg_states.keys()):
            rg_size = self._rg_states[rg_id].size
            had_incomplete = False
            for ri in range(rg_size):
                if self._tracker.is_dropped(rg_id, ri):
                    continue
                if all(
                    self._tracker.is_complete(SliceRef(column=col, row_group=rg_id, row_index=ri))
                    for col in all_columns
                ):
                    continue
                had_incomplete = True
                self._drop_row(rg_id, ri)
            if had_incomplete:
                survivors = sum(1 for ri in range(rg_size) if not self._tracker.is_dropped(rg_id, ri))
                if survivors > 0:
                    self._partial_row_groups.append(rg_id)
                    logger.warning(f"Row group {rg_id}: salvaging {survivors} of {rg_size} rows after early shutdown.")
                else:
                    logger.warning(f"Row group {rg_id}: 0 of {rg_size} rows survived early shutdown - skipping write.")
        self._checkpoint_completed_row_groups(all_columns)

    def _run_seeds_complete_check(self, seed_cols: tuple[str, ...]) -> None:
        """Run pre-batch callbacks for row groups whose seeds just completed."""
        for rg_id, state in list(self._rg_states.items()):
            if state.seeds_dispatched and not state.pre_batch_done:
                all_seeds_done = all(self._tracker.is_column_complete_for_rg(col, rg_id) for col in seed_cols)
                if all_seeds_done and state.in_flight_count == 0:
                    state.pre_batch_done = True
                    if self._on_seeds_complete:
                        try:
                            delta = self._on_seeds_complete(rg_id, state.size)
                        except DatasetGenerationError:
                            raise
                        except Exception as exc:
                            raise DatasetGenerationError(
                                f"Pre-batch processor failed for row group {rg_id}: {exc}"
                            ) from exc
                        # The callback may drop rows (e.g. pre-batch filtering).
                        # Record skipped tasks for any newly-dropped rows so
                        # progress reporting stays accurate.
                        if self._reporter:
                            for ri in range(state.size):
                                if self._tracker.is_dropped(rg_id, ri):
                                    self._record_skipped_tasks_for_row(rg_id, ri)
                        if delta is not None:
                            self._apply_frontier_delta(delta)
                    self._flush_pre_batch_ready(rg_id)

    def _drop_row(self, row_group: int, row_index: int, *, exclude_columns: set[str] | None = None) -> None:
        if self._tracker.is_dropped(row_group, row_index):
            return

        self._record_skipped_tasks_for_row(row_group, row_index, exclude_columns=exclude_columns)
        self._apply_frontier_delta(self._tracker.drop_row(row_group, row_index))
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

    def _record_retryable_outcome(self, *, retryable: bool) -> None:
        """Track retryable-error rate and emit a rate-limited WARN under provider degradation.

        Distinct from ``_check_error_rate``: every LLM-bound task outcome (success
        or failure) feeds this window so the rate reflects the provider's overall
        health, not just the error mix. The call site filters on ``is_llm`` so
        non-LLM tasks (samplers, expressions, non-LLM customs) don't dilute the
        rate. Only retryable errors (rate-limit, timeout, 5xx, connection) count
        toward the rate; non-retryable failures register as 0.
        """
        if self._degraded_warn_window <= 0:
            return
        self._recent_retryable.append(retryable)
        if len(self._recent_retryable) < self._degraded_warn_window:
            return
        rate = sum(self._recent_retryable) / self._degraded_warn_window
        if rate < self._degraded_warn_rate:
            return
        now = time.monotonic()
        if now - self._last_degraded_warn_at < self._degraded_warn_interval_s:
            return
        self._last_degraded_warn_at = now
        pct = int(round(rate * 100))
        logger.warning(
            f"Provider showing degraded performance: {pct}% of last {self._degraded_warn_window} "
            "task outcomes were retryable errors (rate-limit, timeout, 5xx, connection). "
            "Run may take longer than expected; salvage will retry these."
        )

    async def _dispatch_seeds(self, rg_id: int, rg_size: int) -> None:
        """Make from-scratch/root tasks ready for a row group."""
        self._rg_states[rg_id].seeds_dispatched = True
        seed_cols = self._seed_cols
        if not seed_cols:
            return
        num_rgs = len(self._rg_size_map)
        width = len(str(num_rgs))
        logger.info(f"🚀 ({rg_id + 1:0{width}d}/{num_rgs}) Dispatching with {rg_size} records")
        seen_instances: set[int] = set()
        root_columns: list[str] = []

        for col in seed_cols:
            gen = self._generators[col]
            gid = id(gen)
            if gid in seen_instances:
                continue
            seen_instances.add(gid)
            root_columns.append(col)

        self._apply_frontier_delta(self._tracker.add_root_tasks(rg_id, rg_size, columns=tuple(root_columns)))

    async def _execute_task(self, task: Task, lease: TaskAdmissionLease, task_execution_id: str) -> None:
        """Execute a single task (cell or batch)."""
        await self._execute_task_inner(task, lease, task_execution_id)

    async def _execute_task_inner(self, task: Task, lease: TaskAdmissionLease, task_execution_id: str) -> None:
        """Core task execution logic."""
        num_rgs = len(self._row_groups)
        token = current_row_group.set((task.row_group, num_rgs))
        group = lease.item.group
        identity_hash = hashlib.sha1("\0".join(group.key.identity).encode()).hexdigest()[:16]
        correlation_token = runtime_correlation_provider.set(
            RuntimeCorrelation(
                run_id=self._run_id,
                row_group=task.row_group,
                task_column=task.column,
                task_type=task.task_type,
                scheduling_group_kind=group.key.kind,
                scheduling_group_identity_hash=identity_hash,
                task_execution_id=task_execution_id,
            )
        )
        try:
            await self._execute_task_inner_impl(task, lease, task_execution_id)
        finally:
            runtime_correlation_provider.reset(correlation_token)
            current_row_group.reset(token)

    async def _execute_task_inner_impl(self, task: Task, lease: TaskAdmissionLease, task_execution_id: str) -> None:
        trace: TaskTrace | None = None
        if self._trace:
            trace = TaskTrace.from_task(task)
            trace.dispatched_at = time.perf_counter()

        generator = self._generators[task.column]
        output_cols = self._gen_instance_to_columns.get(id(generator), [task.column])
        retryable = False
        cancelled = False
        # When True, skip removing from _dispatched so the task isn't re-dispatched
        # from the frontier (it was never completed, so it stays in the frontier).
        skipped = False
        uses_model_stage_resource = "llm_wait" in lease.resources
        stateful_lock_acquired = False

        try:
            # Skip tasks whose row group was already checkpointed (can happen
            # when a vacuously-ready downstream is dispatched via create_task
            # in the same loop iteration that checkpoints the row group).
            if task.row_group not in self._rg_states:
                skipped = True
                return

            if task.task_type == "from_scratch" and id(generator) in self._stateful_locks:
                await self._stateful_locks[id(generator)].acquire()
                stateful_lock_acquired = True

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
                    delta = self._tracker.mark_row_range_complete(col, task.row_group, rg_size)
                else:
                    delta = self._tracker.mark_cell_complete(col, task.row_group, task.row_index)
                self._apply_frontier_delta(delta)

            self._check_error_rate(success=True)
            # The degraded-provider WARN is provider-scoped: only feed the
            # window from LLM-bound tasks so a healthy non-model task mix
            # (samplers, expressions, non-LLM customs) doesn't dilute the
            # rate and silence the WARN under genuine provider stress.
            if uses_model_stage_resource:
                self._record_retryable_outcome(retryable=False)
            if self._reporter:
                if cell_skipped:
                    self._reporter.record_skipped(task.column)
                else:
                    self._reporter.record_success(task.column)
            if self._trace and trace:
                trace.status = "ok"

        except asyncio.CancelledError:
            cancelled = True
            if self._trace and trace:
                trace.status = "cancelled"
            self._emit_scheduler_event("cancelled", task=task, lease=lease, task_execution_id=task_execution_id)
            raise

        except Exception as exc:
            retryable = self._is_retryable(exc)
            # Only non-retryable errors (auth, schema, code bugs) count toward
            # the early-shutdown gate. Retryable errors (rate-limit, timeout,
            # transient 5xx, connection blips) cluster under provider degradation
            # and would otherwise trip the gate even when salvage could recover.
            if not retryable:
                self._check_error_rate(success=False)
            if uses_model_stage_resource:
                self._record_retryable_outcome(retryable=retryable)
            if not retryable and self._reporter:
                self._reporter.record_failure(task.column)
            if self._trace and trace:
                trace.status = "error"
                trace.error = str(exc)

            if retryable:
                self._deferred.append(task)
                self._emit_scheduler_event(
                    "retry_deferred", task=task, lease=lease, task_execution_id=task_execution_id
                )
            else:
                # Capture the first non-retryable error for the interface to surface
                # as the root cause when the run produces 0 records (e.g. deterministic
                # seed failures). Subsequent failures are still logged below.
                if self._first_non_retryable_error is None:
                    self._first_non_retryable_error = exc
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
                self._emit_scheduler_event(
                    "non_retryable_dropped",
                    task=task,
                    lease=lease,
                    task_execution_id=task_execution_id,
                    diagnostics={"error_type": type(exc).__name__},
                )

        finally:
            if self._trace and trace:
                trace.completed_at = time.perf_counter()
                self.traces.append(trace)

            self._tracker.mark_complete(task)
            if not cancelled:
                self._emit_scheduler_event(
                    "task_completed",
                    task=task,
                    lease=lease,
                    task_execution_id=task_execution_id,
                )
            self._in_flight.discard(task)
            if (s := self._rg_states.get(task.row_group)) is not None:
                s.in_flight_count = max(0, s.in_flight_count - 1)
            if not retryable and not skipped:
                self._dispatched.discard(task)
            if stateful_lock_acquired:
                self._stateful_locks[id(generator)].release()
            release_result = self._task_admission.release(lease)
            self._emit_scheduler_event(
                "task_lease_released",
                task=task,
                lease=lease,
                task_execution_id=task_execution_id,
                reason_or_result=release_result.reason,
            )
            if not release_result.released:
                self._emit_scheduler_event(
                    "release_diagnostic",
                    task=task,
                    lease=lease,
                    task_execution_id=task_execution_id,
                    reason_or_result=release_result.reason,
                )
            self._record_observed_task_state()
            self._wake_event.set()

    async def _run_from_scratch(self, task: Task, generator: ColumnGenerator) -> Any:
        """Execute a from_scratch task."""
        rg_size = self._get_rg_size(task.row_group)
        # Runtime import: needed for isinstance check; module-level would cause circular import
        from data_designer.engine.column_generators.generators.base import FromScratchColumnGenerator

        if isinstance(generator, FromScratchColumnGenerator):
            result_df = await generator.agenerate_from_scratch(rg_size)
        else:
            # Non-FromScratch generators dispatched as seeds (no upstream columns)
            # operate on existing buffer rows — same contract as the sync engine's
            # FULL_COLUMN path. Pass an ``rg_size``-row snapshot so the generator
            # produces ``rg_size`` rows back, instead of an empty DataFrame which
            # would yield zero values and fail ``update_batch``.
            if self._buffer_manager is not None:
                records = [self._buffer_manager.get_row(task.row_group, ri) for ri in range(rg_size)]
                input_df = lazy.pd.DataFrame(records)
            else:
                input_df = lazy.pd.DataFrame(index=range(rg_size))
            result_df = await generator.agenerate(input_df)

        # Write results to buffer (include side-effect columns)
        if self._buffer_manager is not None:
            write_cols = self._gen_instance_to_columns_including_side_effects.get(id(generator), [task.column])
            for col in write_cols:
                if col in result_df.columns:
                    values = result_df[col].tolist()
                    self._buffer_manager.update_batch(task.row_group, col, values)

        return result_df

    async def _run_cell(self, task: Task, generator: ColumnGenerator) -> tuple[Any, bool]:
        """Execute a cell-by-cell task. Returns ``(result, skipped)``."""
        if task.row_index is None:
            raise ValueError(f"Cell task requires a row_index, got None for column '{task.column}'")

        if self._tracker.is_dropped(task.row_group, task.row_index):
            return None, False

        # Evaluate skip against the live buffer record (no copy needed —
        # there is no `await` between the read and the skip-metadata write).
        if self._buffer_manager is not None:
            record = self._buffer_manager.get_row(task.row_group, task.row_index)
        else:
            record = {}

        if self._should_skip_record(task.column, record):
            self._apply_skip_to_record(task, record)
            skip_config = self._graph.get_skip_config(task.column)
            return skip_config.value if skip_config is not None else None, True

        # Copy for generation: agenerate crosses an await boundary, so the
        # generator must not hold a mutable reference to the live record.
        result = await generator.agenerate(dict(record))

        # Write back to buffer (include side-effect columns)
        if self._buffer_manager is not None and not self._tracker.is_dropped(task.row_group, task.row_index):
            write_cols = self._gen_instance_to_columns_including_side_effects.get(id(generator), [task.column])
            for col in write_cols:
                if col in result:
                    self._buffer_manager.update_cell(task.row_group, task.row_index, col, result[col])

        return result, False

    def _should_skip_record(self, column: str, record: dict) -> bool:
        """Decide whether a cell should be skipped (propagation first, then expression gate)."""
        skip_config = self._graph.get_skip_config(column)
        return should_skip_column_for_record(
            record,
            propagate_skip=self._graph.should_propagate_skip(column),
            required_columns=self._graph.get_required_columns(column),
            skip_config_when=skip_config.when if skip_config is not None else None,
        )

    def _apply_skip_to_record(self, task: Task, record: dict) -> None:
        """Write skip metadata directly into *record* (the live buffer row)."""
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
        rg_size = self._get_rg_size(task.row_group)

        if self._buffer_manager is not None:
            pre_dropped: set[int] = {ri for ri in range(rg_size) if self._buffer_manager.is_dropped(task.row_group, ri)}
            active_rows_data: list[dict] = []

            # Skip evaluation only applies to single-column configs.
            # Multi-column configs (sampler/seed) are rejected by the SkipConfig
            # model validator, so they never carry skip metadata.
            pre_skipped: set[int] = set()
            is_multi = isinstance(generator.config, MultiColumnConfig)
            for ri in range(rg_size):
                if ri in pre_dropped:
                    continue

                record = self._buffer_manager.get_row(task.row_group, ri)
                if not is_multi and self._should_skip_record(task.column, record):
                    self._apply_skip_to_record(task, record)
                    pre_skipped.add(ri)
                    continue

                active_rows_data.append(record)

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

        # Merge result columns back to buffer (include side-effect columns)
        if self._buffer_manager is not None:
            write_cols = self._gen_instance_to_columns_including_side_effects.get(id(generator), [task.column])
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
                    for col in write_cols:
                        if col in result_df.columns:
                            self._buffer_manager.update_cell(task.row_group, ri, col, result_df.iloc[result_idx][col])
                result_idx += 1

        return result_df

    def _get_rg_size(self, row_group: int) -> int:
        try:
            return self._rg_size_map[row_group]
        except KeyError:
            raise ValueError(f"Unknown row group: {row_group}") from None

    def task_admission_snapshot(self) -> object:
        """Return the current scheduler task-admission snapshot for diagnostics."""
        return self._task_admission.view()

    def capacity_plan(self) -> AsyncCapacityPlan:
        """Return the scheduler-side async capacity explanation for this run."""
        task_view = self._task_admission.view()
        return AsyncCapacityPlan(
            configured=AsyncCapacityConfigured(
                buffer_size=CapacityValue(value=self._buffer_size, source="run_config"),
                row_group_admission=RowGroupAdmission(
                    row_group_concurrency=CapacityValue(
                        value=self._max_concurrent_row_groups,
                        source="dataset_builder",
                    ),
                    observed_in_flight=len(self._rg_states),
                ),
                submission_capacity=CapacityValue(value=self._max_submitted_tasks, source="dataset_builder"),
                task_resource_limits=CapacityValue(
                    value=dict(self._task_admission_config.resource_limits),
                    source="engine_internal_config",
                ),
                request_resources=CapacityValue(
                    value=(),
                    source="runtime_snapshot",
                    missing_reason="request admission resources are reported by the model registry request controller",
                ),
                provider_model_static_caps=CapacityValue(
                    value={},
                    source="runtime_snapshot",
                    missing_reason="request admission caps are reported by the model registry request controller",
                ),
                request_domain_initial_limits=CapacityValue(
                    value={},
                    source="runtime_snapshot",
                    missing_reason="request admission limits are reported by the model registry request controller",
                ),
                request_admission_config=CapacityValue(
                    value=None,
                    source="runtime_snapshot",
                    missing_reason="request admission config is owned by the model registry request controller",
                ),
                transport_pool_limits=CapacityValue(
                    value={},
                    source="adapter_config",
                    missing_reason="transport pool utilization is adapter-specific",
                ),
            ),
            runtime_snapshot=AsyncCapacityRuntimeSnapshot(),
            observed_maxima=AsyncCapacityObservedMaxima(
                row_groups_in_flight=self._observed_max_row_groups_in_flight,
                queued_tasks_by_group=dict(self._observed_max_queued_by_group),
                task_leases_by_resource=dict(self._observed_max_task_leases_by_resource or task_view.leased_resources),
                transport_pool_utilization=None,
            ),
        )

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Classify whether an exception is retryable."""
        return isinstance(exc, RETRYABLE_MODEL_ERRORS)
