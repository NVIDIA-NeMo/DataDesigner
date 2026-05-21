# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import time
import uuid
from collections import Counter, defaultdict, deque
from collections.abc import Coroutine, Mapping
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
    RequestAdmissionConfigSnapshot,
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
from data_designer.engine.dataset_builders.scheduling.task_policies import BoundedBorrowTaskAdmissionPolicyConfig
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
from data_designer.engine.errors import DataDesignerError
from data_designer.engine.models.clients.errors import ProviderError
from data_designer.engine.models.errors import RETRYABLE_MODEL_ERRORS, GenerationValidationFailureError
from data_designer.engine.models.request_admission.config import RequestAdmissionConfig
from data_designer.engine.models.request_admission.resources import RequestResourceKey
from data_designer.engine.models.resources import ProviderModelKey, ProviderModelStaticCap
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
    from data_designer.engine.models.request_admission.pressure import RequestPressureSnapshotProvider

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
INTERNAL_BUG_EXCEPTIONS = (KeyError, TypeError, AttributeError, AssertionError)


def _identity_hash(identity: tuple[str, ...]) -> str:
    return hashlib.sha1("\0".join(identity).encode()).hexdigest()[:16]


def _request_resource_label(resource: object | None) -> str | None:
    if resource is None:
        return None
    provider = getattr(resource, "provider_name", None)
    model = getattr(resource, "model_id", None)
    domain = getattr(resource, "domain", None)
    domain_value = getattr(domain, "value", domain)
    if provider is None or model is None or domain_value is None:
        return str(resource)
    return f"{provider}/{model}/{domain_value}"


def _string_keyed_counts(values: Mapping[object, int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in values.items()}


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
        adaptive_row_group_admission: bool = False,
        adaptive_row_group_initial_target: int = 1,
        request_pressure_provider: RequestPressureSnapshotProvider | None = None,
        request_pressure_advisory: bool = False,
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
            bounded_borrow=BoundedBorrowTaskAdmissionPolicyConfig(),
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
        self._fatal_worker_error: BaseException | None = None

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
        self._observed_max_request_waiters_by_resource: dict[RequestResourceKey, int] = {}
        self._observed_max_request_in_flight_by_resource: dict[RequestResourceKey, int] = {}
        self._observed_max_provider_model_aggregate_in_flight: dict[ProviderModelKey, int] = {}
        self._observed_max_request_domain_current_limits: dict[RequestResourceKey, int] = {}
        self._adaptive_row_group_admission = adaptive_row_group_admission
        self._row_group_admission_hard_cap = max(1, max_concurrent_row_groups)
        self._row_group_admission_target = (
            max(1, min(self._row_group_admission_hard_cap, adaptive_row_group_initial_target))
            if adaptive_row_group_admission
            else self._row_group_admission_hard_cap
        )
        self._observed_max_row_group_admission_target = self._row_group_admission_target
        self._row_group_admission_event = asyncio.Event()
        self._row_group_admission_event.set()
        self._row_group_admission_pressure_ticks = 0
        self._row_group_admission_blocked_reasons: Counter[str] = Counter()
        self._adaptive_max_admitted_rows = self._max_admitted_rows_guardrail()
        self._request_pressure_provider = request_pressure_provider
        self._request_pressure_advisory = request_pressure_advisory and request_pressure_provider is not None
        self._request_pressure_advisory_skips = 0

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

    def _raise_if_fatal_worker_error(self) -> None:
        if self._fatal_worker_error is None:
            return
        raise DatasetGenerationError(
            "Unexpected internal task failure in async scheduler."
        ) from self._fatal_worker_error

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
        event_diagnostics = dict(diagnostics or {})
        if task is not None:
            schedulable = lease.item if lease is not None else self._schedulable_task(task)
            group = schedulable.group
            identity_hash = _identity_hash(group.key.identity)
            event_diagnostics.setdefault("task_group_key", group.key)
            event_diagnostics.setdefault("resource_request", dict(schedulable.resource_request.amounts))
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
                    diagnostics=event_diagnostics,
                )
            )
        except Exception:
            logger.warning("Scheduler admission event sink raised; dropping event.", exc_info=True)
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
        if self._request_pressure_provider is None:
            return
        for resource, snapshot in self._request_pressure_provider.snapshots().items():
            self._observed_max_request_waiters_by_resource[resource] = max(
                self._observed_max_request_waiters_by_resource.get(resource, 0),
                snapshot.waiters,
            )
            self._observed_max_request_in_flight_by_resource[resource] = max(
                self._observed_max_request_in_flight_by_resource.get(resource, 0),
                snapshot.in_flight_count,
            )
            self._observed_max_request_domain_current_limits[resource] = max(
                self._observed_max_request_domain_current_limits.get(resource, 0),
                snapshot.current_limit,
            )
        for provider_model, snapshot in self._request_pressure_provider.global_snapshots().items():
            self._observed_max_provider_model_aggregate_in_flight[provider_model] = max(
                self._observed_max_provider_model_aggregate_in_flight.get(provider_model, 0),
                snapshot.aggregate_in_flight,
            )

    def _emit_scheduler_health_snapshot(self, reason: str) -> None:
        self._emit_scheduler_event(
            "scheduler_health_snapshot",
            diagnostics=self._scheduler_health_diagnostics(reason=reason),
        )

    def _scheduler_health_diagnostics(self, *, reason: str) -> dict[str, object]:
        queue_view = self._fair_queue.view()
        task_view = self._task_admission.view()
        return {
            "reason": reason,
            "active_row_groups": len(self._rg_states),
            "target_row_groups": self._row_group_admission_target,
            "hard_cap_row_groups": self._row_group_admission_hard_cap,
            "active_admitted_rows": self._active_admitted_row_count(),
            "max_admitted_rows": self._adaptive_max_admitted_rows,
            "all_row_groups_admitted": self._all_rgs_admitted,
            "queued_total": queue_view.queued_total,
            "queued_by_group": _string_keyed_counts(queue_view.queued_by_group),
            "queued_demand_by_resource": dict(queue_view.queued_peer_demand_by_resource),
            "leased_resources": dict(task_view.leased_resources),
            "resource_limits": dict(task_view.resource_limits),
            "resources_available": dict(task_view.resources_available),
            "in_flight_tasks": len(self._in_flight),
            "active_workers": self.active_worker_count,
            "deferred_tasks": len(self._deferred),
            "pending_pre_batch_tasks": len(self._pending_pre_batch_ready_tasks),
            "dispatched_tasks": len(self._dispatched),
            "request_pressure_advisory_enabled": self._request_pressure_advisory,
            "request_pressure_advisory_skips": self._request_pressure_advisory_skips,
            "row_group_admission_blocked_reasons": dict(self._row_group_admission_blocked_reasons),
            "request_pressure": self._request_pressure_diagnostics(),
        }

    def _scheduler_job_diagnostics(self) -> dict[str, object]:
        row_group_sizes = [size for _rg_id, size in self._row_groups]
        strategies = {column: self._graph.get_strategy(column).value for column in self._graph.columns}
        task_count_by_strategy = Counter(strategies.values())
        return {
            "run_id": self._run_id,
            "num_records": self._num_records,
            "buffer_size": self._buffer_size,
            "row_group_count": len(self._row_groups),
            "row_group_total_rows": sum(row_group_sizes),
            "row_group_min_size": min(row_group_sizes, default=0),
            "row_group_max_size": max(row_group_sizes, default=0),
            "graph_column_count": len(self._graph.columns),
            "graph_root_columns": tuple(self._graph.get_root_columns()),
            "graph_depth": len(self._graph.get_longest_dependency_chain()),
            "task_count_by_strategy": dict(task_count_by_strategy),
            "column_scheduling": self._column_scheduling_diagnostics(strategies),
            "resource_limits": dict(self._task_admission_config.resource_limits),
            "submission_capacity": self._task_admission_config.submission_capacity,
            "adaptive_row_group_admission": self._adaptive_row_group_admission,
            "row_group_initial_target": self._row_group_admission_target,
            "row_group_hard_cap": self._row_group_admission_hard_cap,
            "max_admitted_rows": self._adaptive_max_admitted_rows,
            "request_pressure_advisory_enabled": self._request_pressure_advisory,
        }

    def _column_scheduling_diagnostics(self, strategies: dict[str, str]) -> tuple[dict[str, object], ...]:
        diagnostics = []
        for column in self._graph.columns:
            task_type = "batch" if self._graph.get_strategy(column) != GenerationStrategy.CELL_BY_CELL else "cell"
            row_index = None if task_type == "batch" else 0
            task = Task(column=column, row_group=0, row_index=row_index, task_type=task_type)
            resolved = self._task_scheduling.scheduling_for_task(task, self._task_flow_identity(task))
            diagnostics.append(
                {
                    "column": column,
                    "strategy": strategies[column],
                    "group_kind": resolved.group.key.kind,
                    "group_identity_hash": _identity_hash(resolved.group.key.identity),
                    "group_weight": resolved.group.weight,
                    "group_admitted_limit": resolved.group.admitted_limit,
                    "resource_request": dict(resolved.resource_request.amounts),
                    "request_resource": _request_resource_label(resolved.request_resource_key),
                }
            )
        return tuple(diagnostics)

    def _request_pressure_diagnostics(self) -> dict[str, object]:
        if self._request_pressure_provider is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "resources": {
                _request_resource_label(resource): {
                    "effective_max": snapshot.effective_max,
                    "current_limit": snapshot.current_limit,
                    "in_flight_count": snapshot.in_flight_count,
                    "active_lease_count": snapshot.active_lease_count,
                    "waiters": snapshot.waiters,
                    "blocked": snapshot.blocked_until_monotonic is not None,
                    "cooldown_remaining_seconds": snapshot.cooldown_remaining_seconds,
                    "last_outcome": snapshot.last_outcome,
                }
                for resource, snapshot in self._request_pressure_provider.snapshots().items()
            },
            "provider_models": {
                f"{provider_model.provider_name}/{provider_model.model_id}": {
                    "static_cap": snapshot.static_cap,
                    "aggregate_in_flight": snapshot.aggregate_in_flight,
                    "aggregate_active_lease_count": snapshot.aggregate_active_lease_count,
                    "domains": {domain.value: count for domain, count in snapshot.domains.items()},
                }
                for provider_model, snapshot in self._request_pressure_provider.global_snapshots().items()
            },
        }

    def _request_pressure_item_diagnostics(self, item: SchedulableTask) -> dict[str, object]:
        if item.request_resource_key is None or self._request_pressure_provider is None:
            return {"request_resource": None}
        snapshot = self._request_pressure_provider.snapshot(item.request_resource_key)
        global_snapshot = self._request_pressure_provider.global_snapshot(
            item.request_resource_key.provider_name,
            item.request_resource_key.model_id,
        )
        diagnostics: dict[str, object] = {
            "request_resource": _request_resource_label(item.request_resource_key),
            "pressure_reason": self._request_pressure_reason(item),
            "resource_snapshot": None,
            "provider_model_snapshot": None,
        }
        if snapshot is not None:
            diagnostics["resource_snapshot"] = {
                "effective_max": snapshot.effective_max,
                "current_limit": snapshot.current_limit,
                "in_flight_count": snapshot.in_flight_count,
                "waiters": snapshot.waiters,
                "blocked": snapshot.blocked_until_monotonic is not None,
                "cooldown_remaining_seconds": snapshot.cooldown_remaining_seconds,
            }
        if global_snapshot is not None:
            diagnostics["provider_model_snapshot"] = {
                "static_cap": global_snapshot.static_cap,
                "aggregate_in_flight": global_snapshot.aggregate_in_flight,
                "aggregate_active_lease_count": global_snapshot.aggregate_active_lease_count,
            }
        return diagnostics

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
        self._enqueue_ready_tasks(delta.added)

    def _enqueue_ready_task(self, task: Task) -> None:
        self._enqueue_ready_tasks((task,))

    def _enqueue_ready_tasks(self, tasks: tuple[Task, ...]) -> None:
        schedulables: list[SchedulableTask] = []
        accepted_tasks_by_id: dict[str, Task] = {}
        for task in tasks:
            if task in self._dispatched or task.row_group not in self._rg_states:
                continue
            if not self._tracker.is_frontier_task(task):
                continue
            self._emit_scheduler_event("dependency_ready", task=task)
            state = self._rg_states[task.row_group]
            if self._on_seeds_complete is not None and not state.pre_batch_done and task.column not in self._seed_cols:
                if task not in self._pending_pre_batch_ready_tasks:
                    self._pending_pre_batch_ready[task.row_group].append(task)
                    self._pending_pre_batch_ready_tasks.add(task)
                continue
            schedulable = self._schedulable_task(task)
            schedulables.append(schedulable)
            accepted_tasks_by_id[schedulable.task_id] = task

        if not schedulables:
            return
        accepted = self._fair_queue.enqueue(schedulables)
        if accepted:
            self._tracker.mark_enqueued(accepted)
            for task_id in accepted:
                self._emit_scheduler_event("ready_enqueued", task=accepted_tasks_by_id[task_id])
            self._record_observed_task_state()
            self._wake_event.set()

    def _discard_ready_task(self, task: Task) -> None:
        self._fair_queue.discard(stable_task_id(task))
        self._pending_pre_batch_ready_tasks.discard(task)

    def _flush_pre_batch_ready(self, row_group: int) -> None:
        pending = self._pending_pre_batch_ready.pop(row_group, [])
        ready = []
        for task in pending:
            if task not in self._pending_pre_batch_ready_tasks:
                continue
            self._pending_pre_batch_ready_tasks.discard(task)
            ready.append(task)
        self._enqueue_ready_tasks(tuple(ready))

    def _drop_pending_ready_for_row_group(self, row_group: int) -> None:
        pending = self._pending_pre_batch_ready.pop(row_group, [])
        for task in pending:
            self._pending_pre_batch_ready_tasks.discard(task)
        self._fair_queue.discard_where(lambda item: item.payload.row_group == row_group)

    def _dispatch_queued_tasks(self) -> _DispatchOutcome:
        dispatched = False

        while self._fair_queue.has_queued_tasks:
            selection = self._fair_queue.select_next(self._is_dispatch_eligible)
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
                self._emit_scheduler_health_snapshot(event_kind)
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
            self._emit_scheduler_health_snapshot("queue_drained")
        return _DispatchOutcome(dispatched=dispatched)

    def _is_dispatch_eligible(self, item: SchedulableTask, view: Any) -> bool:
        if not self._task_admission.is_eligible(item, view):
            return False
        if not self._request_pressure_advisory:
            return True
        if not self._is_request_pressure_limited(item):
            return True
        open_peer = self._request_pressure_open_peer(item, view)
        if open_peer is not None:
            self._request_pressure_advisory_skips += 1
            self._emit_scheduler_event(
                "request_pressure_advisory_skipped",
                task=item.payload,
                diagnostics=self._request_pressure_item_diagnostics(item)
                | {
                    "open_peer_task_id": open_peer.task_id,
                    "open_peer_column": open_peer.payload.column,
                    "open_peer_request_resource": _request_resource_label(open_peer.request_resource_key),
                    "skip_count": self._request_pressure_advisory_skips,
                },
            )
            return False
        return True

    def _is_request_pressure_limited(self, item: SchedulableTask) -> bool:
        return self._request_pressure_reason(item) is not None

    def _request_pressure_reason(self, item: SchedulableTask) -> str | None:
        if item.request_resource_key is None or self._request_pressure_provider is None:
            return None
        snapshot = self._request_pressure_provider.snapshot(item.request_resource_key)
        global_snapshot = self._request_pressure_provider.global_snapshot(
            item.request_resource_key.provider_name,
            item.request_resource_key.model_id,
        )
        if (
            global_snapshot is not None
            and global_snapshot.static_cap > 0
            and global_snapshot.aggregate_in_flight >= global_snapshot.static_cap
        ):
            return "provider_model_aggregate_cap"
        if snapshot is None:
            return None
        if snapshot.cooldown_remaining_seconds > 0.0 or snapshot.blocked_until_monotonic is not None:
            return "cooldown"
        if snapshot.waiters > 0:
            return "waiters"
        if snapshot.current_limit > 0 and snapshot.in_flight_count >= snapshot.current_limit:
            return "resource_limit"
        return None

    def _has_request_pressure_open_peer(self, item: SchedulableTask, view: Any) -> bool:
        return self._request_pressure_open_peer(item, view) is not None

    def _request_pressure_open_peer(self, item: SchedulableTask, view: Any) -> SchedulableTask | None:
        for peer in view.first_candidate_tasks_by_group.values():
            if peer.task_id == item.task_id:
                continue
            if not self._task_admission.is_eligible(peer, view):
                continue
            if not self._is_request_pressure_limited(peer):
                return peer
        return None

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
            self._in_flight.discard(task)
            self._dispatched.discard(task)
            if (s := self._rg_states.get(task.row_group)) is not None:
                s.in_flight_count = max(0, s.in_flight_count - 1)
            self._emit_scheduler_event(
                "worker_spawn_failed",
                task=task,
                lease=lease,
                task_execution_id=task_execution_id,
                reason_or_result=result.reason,
            )
            raise

    def _schedulable_task(self, task: Task) -> SchedulableTask:
        return self._task_scheduling.schedulable_task(task, self._task_flow_identity(task))

    def _task_flow_identity(self, task: Task) -> tuple[str, ...]:
        generator = self._generators[task.column]
        output_columns = self._gen_instance_to_columns.get(id(generator), [task.column])
        return tuple(output_columns)

    def _max_admitted_rows_guardrail(self) -> int:
        if self._num_records > 0 and self._buffer_size > 0:
            return min(self._num_records, max(3 * self._buffer_size, 8192))
        total_rows = sum(size for _rg_id, size in self._row_groups)
        return max(1, total_rows)

    async def _wait_for_row_group_admission_capacity(self, row_group_size: int) -> None:
        while True:
            target_blocked = len(self._rg_states) >= self._row_group_admission_target
            row_guard_blocked = not self._row_group_row_guard_allows(row_group_size)
            if not target_blocked and not row_guard_blocked:
                return
            self._row_group_admission_event.clear()
            target_blocked = len(self._rg_states) >= self._row_group_admission_target
            row_guard_blocked = not self._row_group_row_guard_allows(row_group_size)
            if not target_blocked and not row_guard_blocked:
                return
            if row_guard_blocked:
                self._row_group_admission_blocked_reasons["max_admitted_rows"] += 1
                self._emit_scheduler_event(
                    "row_group_admission_blocked",
                    diagnostics=self._row_group_admission_diagnostics(reason="max_admitted_rows"),
                )
                self._emit_scheduler_health_snapshot("row_group_admission_blocked")
            await self._row_group_admission_event.wait()
            self._raise_if_fatal_worker_error()

    def _row_group_row_guard_allows(self, row_group_size: int) -> bool:
        if not self._adaptive_row_group_admission:
            return True
        admitted_rows = self._active_admitted_row_count()
        return admitted_rows == 0 or admitted_rows + row_group_size <= self._adaptive_max_admitted_rows

    def _active_admitted_row_count(self) -> int:
        return sum(state.size for state in self._rg_states.values())

    def _maybe_update_adaptive_row_group_target(self) -> None:
        if not self._adaptive_row_group_admission:
            return
        if self._all_rgs_admitted or self._early_shutdown or self._fatal_worker_error is not None:
            return
        if len(self._rg_states) >= self._row_group_admission_hard_cap:
            self._row_group_admission_pressure_ticks = 0
            return
        reason = self._adaptive_row_group_block_reason()
        if reason is not None:
            self._row_group_admission_blocked_reasons[reason] += 1
            self._row_group_admission_pressure_ticks = 0
            self._emit_scheduler_event(
                "row_group_admission_blocked",
                diagnostics=self._row_group_admission_diagnostics(reason=reason),
            )
            self._emit_scheduler_health_snapshot("row_group_admission_blocked")
            return

        self._row_group_admission_pressure_ticks += 1
        if self._fair_queue.view().queued_total > 0 and self._row_group_admission_pressure_ticks < 2:
            return
        old_target = self._row_group_admission_target
        self._row_group_admission_target = min(self._row_group_admission_hard_cap, old_target + 1)
        self._observed_max_row_group_admission_target = max(
            self._observed_max_row_group_admission_target,
            self._row_group_admission_target,
        )
        self._row_group_admission_pressure_ticks = 0
        if self._row_group_admission_target != old_target:
            self._emit_scheduler_event(
                "row_group_admission_target_changed",
                diagnostics=self._row_group_admission_diagnostics(reason="horizon_limited")
                | {"old_target": old_target, "new_target": self._row_group_admission_target},
            )
            self._emit_scheduler_health_snapshot("row_group_admission_target_changed")
            self._row_group_admission_event.set()

    def _adaptive_row_group_block_reason(self) -> str | None:
        if self._deferred:
            return "deferred_tasks"
        next_size = self._next_unadmitted_row_group_size()
        if next_size is None:
            return "no_pending_row_groups"
        if not self._row_group_row_guard_allows(next_size):
            return "max_admitted_rows"
        queue_view = self._fair_queue.view()
        queue_guard = max(self._max_submitted_tasks * 4, self._max_model_task_admission * 2)
        if queue_view.queued_total >= queue_guard:
            return "queued_task_guardrail"
        task_view = self._task_admission.view()
        llm_limit = task_view.resource_limits.get("llm_wait", 0)
        if llm_limit <= 0:
            return "no_llm_wait_resource"
        llm_available = task_view.resources_available.get("llm_wait", 0)
        queued_llm = queue_view.queued_peer_demand_by_resource.get("llm_wait", 0)
        if llm_available <= 0:
            return "llm_wait_saturated"
        if llm_available <= queued_llm and queue_view.queued_total > 0:
            return "queued_llm_demand"
        return None

    def _next_unadmitted_row_group_size(self) -> int | None:
        for rg_id, rg_size in self._row_groups:
            if rg_id not in self._rg_states and not self._tracker.is_row_group_complete(
                rg_id, rg_size, self._graph.columns
            ):
                return rg_size
        return None

    def _row_group_admission_diagnostics(self, *, reason: str) -> dict[str, object]:
        queue_view = self._fair_queue.view()
        task_view = self._task_admission.view()
        admitted_rows = self._active_admitted_row_count()
        return {
            "mode": "adaptive" if self._adaptive_row_group_admission else "fixed",
            "reason": reason,
            "active_row_groups": len(self._rg_states),
            "target_row_groups": self._row_group_admission_target,
            "hard_cap": self._row_group_admission_hard_cap,
            "admitted_rows": admitted_rows,
            "max_admitted_rows": self._adaptive_max_admitted_rows,
            "queued_total": queue_view.queued_total,
            "queued_llm_wait_demand": queue_view.queued_peer_demand_by_resource.get("llm_wait", 0),
            "llm_wait_limit": task_view.resource_limits.get("llm_wait", 0),
            "llm_wait_leased": task_view.leased_resources.get("llm_wait", 0),
            "llm_wait_available": task_view.resources_available.get("llm_wait", 0),
            "blocked_reasons": dict(self._row_group_admission_blocked_reasons),
        }

    async def _admit_row_groups(self) -> None:
        """Admit row groups as semaphore slots become available."""
        all_admitted = True
        for rg_id, rg_size in self._row_groups:
            await self._wait_for_row_group_admission_capacity(rg_size)
            if self._early_shutdown or self._fatal_worker_error is not None:
                all_admitted = False
                break
            await self._rg_semaphore.acquire()
            if self._early_shutdown or self._fatal_worker_error is not None:
                self._rg_semaphore.release()
                all_admitted = False
                break
            if not self._row_group_row_guard_allows(rg_size):
                self._rg_semaphore.release()
                await self._wait_for_row_group_admission_capacity(rg_size)
                await self._rg_semaphore.acquire()
                if self._early_shutdown or self._fatal_worker_error is not None:
                    self._rg_semaphore.release()
                    all_admitted = False
                    break
            self._rg_states[rg_id] = _RowGroupState(size=rg_size)

            if self._buffer_manager is not None:
                self._buffer_manager.init_row_group(rg_id, rg_size)

            await self._dispatch_seeds(rg_id, rg_size)
            self._emit_scheduler_event(
                "row_group_admitted",
                diagnostics=self._row_group_admission_diagnostics(reason="admitted")
                | {"row_group": rg_id, "row_group_size": rg_size},
            )
            self._emit_scheduler_health_snapshot("row_group_admitted")
            self._wake_event.set()
        self._all_rgs_admitted = all_admitted
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

            self._emit_scheduler_event("scheduler_job_started", diagnostics=self._scheduler_job_diagnostics())
            self._emit_scheduler_health_snapshot("start")

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

            self._emit_scheduler_health_snapshot("completed")
            self._emit_scheduler_event(
                "scheduler_job_completed", diagnostics=self._scheduler_health_diagnostics(reason="completed")
            )

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
            self._raise_if_fatal_worker_error()
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
            self._maybe_update_adaptive_row_group_target()

            # Eagerly salvage any row groups that have only deferred tasks,
            # even if other row groups are still in-flight.  This frees
            # semaphore slots so admission doesn't lose capacity.
            if self._deferred:
                await self._salvage_stalled_row_groups(seed_cols, has_pre_batch, all_columns)
                self._maybe_update_adaptive_row_group_target()

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
                    await asyncio.sleep(0)
                    continue

            if not self._fair_queue.has_queued_tasks or dispatch_outcome.admission_blocked:
                if self._fair_queue.has_queued_tasks and not dispatch_outcome.dispatched and not self._in_flight:
                    raise RuntimeError(
                        "Ready frontier is admission-blocked with no in-flight task to release scheduler capacity."
                    )
                await self._wake_event.wait()
                self._raise_if_fatal_worker_error()

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
            self._raise_if_fatal_worker_error()
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
            self._raise_if_fatal_worker_error()

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
            dropped_rows = sum(1 for ri in range(rg_size) if self._tracker.is_dropped(rg_id, ri))
            checkpointed = False
            checkpoint_result = "unknown"
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
                if dropped_rows == rg_size:
                    if self._buffer_manager:
                        self._buffer_manager.free_row_group(rg_id)
                    checkpoint_result = "all_rows_dropped"
                elif self._on_finalize_row_group is not None:
                    self._on_finalize_row_group(rg_id)
                    checkpoint_result = "finalized"
                else:
                    checkpoint_result = "completed"
                checkpointed = True
            except DatasetGenerationError:
                raise
            except Exception:
                logger.error(f"Failed to checkpoint row group {rg_id}.", exc_info=True)
            finally:
                self._rg_semaphore.release()
                self._row_group_admission_event.set()
            if checkpointed:
                self._emit_scheduler_event(
                    "row_group_checkpointed",
                    diagnostics={
                        "row_group": rg_id,
                        "row_group_size": rg_size,
                        "dropped_rows": dropped_rows,
                        "surviving_rows": rg_size - dropped_rows,
                        "result": checkpoint_result,
                        "active_row_groups": len(self._rg_states),
                    },
                )
                self._emit_scheduler_health_snapshot("row_group_checkpointed")

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
                log_message = (
                    f"Non-retryable failure on {task.column}[rg={task.row_group}, row={task.row_index}]: {exc}"
                )
                if self._is_expected_non_retryable(exc):
                    logger.warning(log_message)
                elif self._is_internal_bug(exc):
                    logger.error("Unexpected fatal %s", log_message, exc_info=True)
                    self._fatal_worker_error = exc
                    self._wake_event.set()
                    raise
                else:
                    logger.error("Unexpected %s", log_message, exc_info=True)
                # Non-retryable data/user/provider failures drop the affected row(s);
                # internal bug-shaped failures above abort the run instead.
                if task.row_index is not None:
                    self._drop_row(task.row_group, task.row_index, exclude_columns={task.column})
                else:
                    # Batch/from_scratch failure: drop all rows in the row group
                    rg_size = self._get_rg_size(task.row_group)
                    self._drop_row_group(task.row_group, rg_size, exclude_columns={task.column})
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

    async def _run_generator_call(self, task: Task, operation: str, call: Coroutine[Any, Any, Any]) -> Any:
        """Run user/plugin generator code while preserving scheduler-owned failures."""
        try:
            return await call
        except Exception as exc:
            if self._is_retryable(exc) or self._is_expected_non_retryable(exc):
                raise
            raise DatasetGenerationError(
                f"Generator failed for column '{task.column}' during {operation}: {exc}"
            ) from exc

    def _require_dataframe_result(
        self,
        task: Task,
        operation: str,
        result: Any,
        *,
        expected_rows: int | None = None,
    ) -> Any:
        if not isinstance(result, lazy.pd.DataFrame):
            raise DatasetGenerationError(
                f"{operation} for column '{task.column}' must return a DataFrame, got {type(result).__name__}."
            )
        if expected_rows is not None and len(result) != expected_rows:
            raise DatasetGenerationError(
                f"{operation} for column '{task.column}' returned {len(result)} rows "
                f"but {expected_rows} were expected (rg={task.row_group})."
            )
        return result

    async def _run_from_scratch(self, task: Task, generator: ColumnGenerator) -> Any:
        """Execute a from_scratch task."""
        rg_size = self._get_rg_size(task.row_group)
        # Runtime import: needed for isinstance check; module-level would cause circular import
        from data_designer.engine.column_generators.generators.base import FromScratchColumnGenerator

        if isinstance(generator, FromScratchColumnGenerator):
            result_df = await self._run_generator_call(
                task,
                "from-scratch generation",
                generator.agenerate_from_scratch(rg_size),
            )
            result_operation = "From-scratch generator"
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
            result_df = await self._run_generator_call(
                task,
                "full-column generation",
                generator.agenerate(input_df),
            )
            result_operation = "Full-column generator"
        result_df = self._require_dataframe_result(
            task,
            result_operation,
            result_df,
            expected_rows=rg_size,
        )

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
        result = await self._run_generator_call(
            task,
            "cell generation",
            generator.agenerate(dict(record)),
        )

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

        active_rows = rg_size - len(pre_dropped) - len(pre_skipped) if self._buffer_manager is not None else None
        result_df = await self._run_generator_call(
            task,
            "batch generation",
            generator.agenerate(batch_df),
        )
        result_df = self._require_dataframe_result(
            task,
            "Batch generator",
            result_df,
            expected_rows=active_rows,
        )

        # Merge result columns back to buffer (include side-effect columns)
        if self._buffer_manager is not None:
            write_cols = self._gen_instance_to_columns_including_side_effects.get(id(generator), [task.column])
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
        request_snapshots = (
            dict(self._request_pressure_provider.snapshots()) if self._request_pressure_provider is not None else {}
        )
        provider_snapshots = (
            dict(self._request_pressure_provider.global_snapshots())
            if self._request_pressure_provider is not None
            else {}
        )
        request_resources = tuple(sorted(request_snapshots))
        provider_model_static_caps = {
            provider_model: ProviderModelStaticCap(
                cap=snapshot.static_cap,
                aliases=snapshot.aliases,
                raw_caps=snapshot.raw_caps,
            )
            for provider_model, snapshot in provider_snapshots.items()
        }
        request_config = self._request_pressure_provider.config if self._request_pressure_provider is not None else None
        request_config_snapshot = (
            RequestAdmissionConfigSnapshot.from_config(request_config)
            if isinstance(request_config, RequestAdmissionConfig)
            else None
        )
        request_domain_initial_limits: dict[RequestResourceKey, int] = {}
        if request_config_snapshot is not None:
            request_domain_initial_limits.update(request_config_snapshot.initial_limits)
        for resource, snapshot in request_snapshots.items():
            configured_initial = (
                request_config_snapshot.initial_limits.get(resource) if request_config_snapshot is not None else None
            )
            request_domain_initial_limits[resource] = (
                max(1, min(configured_initial, snapshot.effective_max))
                if configured_initial is not None
                else snapshot.effective_max
            )
        request_domain_current_limits = {
            resource: snapshot.current_limit for resource, snapshot in request_snapshots.items()
        }
        request_domain_effective_max = {
            resource: snapshot.effective_max for resource, snapshot in request_snapshots.items()
        }
        request_domain_blocked_until = {
            resource: snapshot.blocked_until_monotonic for resource, snapshot in request_snapshots.items()
        }
        provider_model_aggregate_in_flight = {
            provider_model: snapshot.aggregate_in_flight for provider_model, snapshot in provider_snapshots.items()
        }
        return AsyncCapacityPlan(
            configured=AsyncCapacityConfigured(
                buffer_size=CapacityValue(value=self._buffer_size, source="run_config"),
                row_group_admission=RowGroupAdmission(
                    row_group_concurrency=CapacityValue(
                        value=self._max_concurrent_row_groups,
                        source="dataset_builder",
                    ),
                    observed_in_flight=len(self._rg_states),
                    mode="adaptive" if self._adaptive_row_group_admission else "fixed",
                    target_in_flight=self._row_group_admission_target,
                    observed_max_target=self._observed_max_row_group_admission_target,
                    max_admitted_rows=self._adaptive_max_admitted_rows,
                    blocked_reasons=dict(self._row_group_admission_blocked_reasons),
                ),
                submission_capacity=CapacityValue(value=self._max_submitted_tasks, source="dataset_builder"),
                task_resource_limits=CapacityValue(
                    value=dict(self._task_admission_config.resource_limits),
                    source="engine_internal_config",
                ),
                request_resources=CapacityValue(
                    value=request_resources,
                    source="runtime_snapshot",
                    missing_reason=None if request_resources else "request admission has not observed any resources",
                ),
                provider_model_static_caps=CapacityValue(
                    value=provider_model_static_caps,
                    source="model_metadata",
                    missing_reason=None if provider_model_static_caps else "request admission has no registered models",
                ),
                request_domain_initial_limits=CapacityValue(
                    value=request_domain_initial_limits,
                    source="engine_internal_config" if request_config_snapshot is not None else "runtime_snapshot",
                    missing_reason=None
                    if request_domain_initial_limits
                    else "request admission has not observed any domain limits",
                ),
                request_admission_config=CapacityValue(
                    value=request_config_snapshot,
                    source="engine_internal_config",
                    missing_reason=None
                    if request_config_snapshot is not None
                    else "request admission config is not exposed by the pressure provider",
                ),
                transport_pool_limits=CapacityValue(
                    value={},
                    source="adapter_config",
                    missing_reason="transport pool utilization is adapter-specific",
                ),
            ),
            runtime_snapshot=AsyncCapacityRuntimeSnapshot(
                request_domain_current_limits=request_domain_current_limits,
                request_domain_effective_max=request_domain_effective_max,
                request_domain_blocked_until=request_domain_blocked_until,
                provider_model_aggregate_in_flight=provider_model_aggregate_in_flight,
            ),
            observed_maxima=AsyncCapacityObservedMaxima(
                row_groups_in_flight=self._observed_max_row_groups_in_flight,
                queued_tasks_by_group=dict(self._observed_max_queued_by_group),
                task_leases_by_resource=dict(self._observed_max_task_leases_by_resource or task_view.leased_resources),
                request_waiters_by_resource=dict(
                    self._observed_max_request_waiters_by_resource
                    or {resource: snapshot.waiters for resource, snapshot in request_snapshots.items()}
                ),
                request_in_flight_by_resource=dict(
                    self._observed_max_request_in_flight_by_resource
                    or {resource: snapshot.in_flight_count for resource, snapshot in request_snapshots.items()}
                ),
                provider_model_aggregate_in_flight=dict(
                    self._observed_max_provider_model_aggregate_in_flight or provider_model_aggregate_in_flight
                ),
                request_domain_current_limits=dict(
                    self._observed_max_request_domain_current_limits or request_domain_current_limits
                ),
                transport_pool_utilization=None,
            ),
        )

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        """Classify whether an exception is retryable."""
        return isinstance(exc, RETRYABLE_MODEL_ERRORS)

    @staticmethod
    def _is_expected_non_retryable(exc: BaseException) -> bool:
        return isinstance(
            exc,
            (
                DataDesignerError,
                DatasetGenerationError,
                GenerationValidationFailureError,
                ProviderError,
            ),
        )

    def _is_internal_bug(self, exc: BaseException) -> bool:
        return isinstance(exc, INTERNAL_BUG_EXCEPTIONS) and not self._is_expected_non_retryable(exc)
