# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic async scheduling benchmark smoke harness."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import (
    ExpressionColumnConfig,
    GenerationStrategy,
    LLMTextColumnConfig,
    SamplerColumnConfig,
)
from data_designer.config.sampler_params import SamplerType
from data_designer.config.scheduling import SchedulingMetadata
from data_designer.engine.capacity import (
    AsyncCapacityConfigured,
    AsyncCapacityObservedMaxima,
    AsyncCapacityPlan,
    AsyncCapacityRuntimeSnapshot,
    CapacityValue,
    RequestAdmissionConfigSnapshot,
    RowGroupAdmission,
)
from data_designer.engine.column_generators.generators.base import ColumnGenerator, FromScratchColumnGenerator
from data_designer.engine.dataset_builders.async_scheduler import AsyncTaskScheduler
from data_designer.engine.dataset_builders.scheduling.completion import CompletionTracker
from data_designer.engine.dataset_builders.scheduling.queue import FairTaskQueue
from data_designer.engine.dataset_builders.scheduling.resources import (
    SchedulableTask,
    SchedulerResourceRequest,
    TaskGroupKey,
    TaskGroupSpec,
)
from data_designer.engine.dataset_builders.scheduling.task_admission import TaskAdmissionConfig, TaskAdmissionController
from data_designer.engine.dataset_builders.scheduling.task_model import Task
from data_designer.engine.dataset_builders.utils.execution_graph import ExecutionGraph
from data_designer.engine.dataset_builders.utils.row_group_buffer import RowGroupBufferManager
from data_designer.engine.models.request_admission.config import RequestAdmissionConfig
from data_designer.engine.models.request_admission.controller import (
    AdaptiveRequestAdmissionController,
    RequestAdmissionLease,
)
from data_designer.engine.models.request_admission.outcomes import RequestReleaseOutcome
from data_designer.engine.models.request_admission.resources import (
    RequestAdmissionItem,
    RequestDomain,
    RequestEventContext,
    RequestGroupSpec,
    RequestResourceKey,
)
from data_designer.engine.models.resources import ProviderModelKey, ProviderModelStaticCap
from data_designer.engine.observability import (
    InMemoryAdmissionEventSink,
    SchedulerAdmissionEvent,
    runtime_correlation_provider,
)

ARTIFACT_SCHEMA_VERSION = "async-scheduling-benchmark-v1"
HARNESS_VERSION = "1.1"
ASYNC_WAKEUP_GATE_SECONDS = 0.025


@dataclass(frozen=True)
class BenchmarkInputs:
    baseline_ref: str
    candidate_ref: str
    scenario: str
    record_count: int
    buffer_size: int
    row_group_concurrency: int
    task_admission_capacity: int
    request_latency_seconds: float
    upstream_latency_seconds: float
    downstream_latency_seconds: float
    fanout_width: int
    model_stage_weight: int
    adaptive_row_group_admission: bool
    request_pressure_advisory: bool
    warmups: int
    iterations: int
    seed: int
    scenario_version: str
    harness_version: str


class _BenchmarkStorage:
    dataset_name = "async-scheduling-benchmark"

    def __init__(self) -> None:
        self.write_count = 0

    def get_file_paths(self) -> dict[str, str]:
        return {}

    def write_batch_to_parquet_file(self, **kwargs: object) -> str:
        self.write_count += 1
        return f"/tmp/async-scheduling-benchmark/partial-{self.write_count}.parquet"

    def move_partial_result_to_final_file_path(self, batch_number: int) -> str:
        return f"/tmp/async-scheduling-benchmark/final-{batch_number}.parquet"


class _BenchmarkSeedGenerator(FromScratchColumnGenerator[ExpressionColumnConfig]):
    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.FULL_COLUMN

    def generate(self, data: lazy.pd.DataFrame) -> lazy.pd.DataFrame:
        return data

    def generate_from_scratch(self, num_records: int) -> lazy.pd.DataFrame:
        return lazy.pd.DataFrame({self.config.name: list(range(num_records))})


class _BenchmarkTimedCellGenerator(ColumnGenerator[ExpressionColumnConfig]):
    def __init__(
        self,
        *args: object,
        delay_seconds: float,
        model_stage: bool = False,
        model_stage_weight: int = 1,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._delay_seconds = delay_seconds
        self._model_stage = model_stage
        self._model_stage_weight = model_stage_weight

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def get_scheduling_metadata(self) -> SchedulingMetadata:
        if self._model_stage:
            return SchedulingMetadata.custom_model(
                "benchmark-provider",
                self.config.name,
                "chat",
                weight=self._model_stage_weight,
            )
        return super().get_scheduling_metadata()

    def generate(self, data: dict) -> dict:
        data[self.config.name] = f"{self.config.name}_{data.get('seed', '?')}"
        return data

    async def agenerate(self, data: dict) -> dict:
        if self._delay_seconds:
            await asyncio.sleep(self._delay_seconds)
        return self.generate(data)


class _BenchmarkRequestPressureCellGenerator(ColumnGenerator[ExpressionColumnConfig]):
    def __init__(
        self,
        *args: object,
        request_admission: AdaptiveRequestAdmissionController,
        provider_name: str,
        model_id: str,
        delay_seconds: float,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._request_admission = request_admission
        self._resource = RequestResourceKey(provider_name, model_id, RequestDomain.CHAT)
        self._delay_seconds = delay_seconds

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def get_scheduling_metadata(self) -> SchedulingMetadata:
        return SchedulingMetadata.model(
            self._resource.provider_name,
            self._resource.model_id,
            self._resource.domain.value,
            weight=1,
        )

    def generate(self, data: dict) -> dict:
        data[self.config.name] = f"{self.config.name}_{data.get('seed', '?')}"
        return data

    async def agenerate(self, data: dict) -> dict:
        item = RequestAdmissionItem(
            self._resource,
            RequestGroupSpec(self._resource),
            queue_wait_timeout_seconds=30.0,
            event_context=RequestEventContext(
                captured_correlation=runtime_correlation_provider.current(),
                request_attempt_id=f"benchmark-request-{uuid.uuid4().hex}",
            ),
        )
        lease = await self._request_admission.acquire_async(item)
        try:
            if self._delay_seconds:
                await asyncio.sleep(self._delay_seconds)
            return self.generate(data)
        finally:
            self._request_admission.release(lease, RequestReleaseOutcome(kind="success"))


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = BenchmarkInputs(
        baseline_ref=args.baseline_ref,
        candidate_ref=args.candidate_ref,
        scenario=args.scenario,
        record_count=args.record_count,
        buffer_size=args.buffer_size,
        row_group_concurrency=args.row_group_concurrency,
        task_admission_capacity=args.task_admission_capacity,
        request_latency_seconds=args.request_latency_seconds,
        upstream_latency_seconds=args.upstream_latency_seconds,
        downstream_latency_seconds=args.downstream_latency_seconds,
        fanout_width=args.fanout_width,
        model_stage_weight=args.model_stage_weight or max(1, args.task_admission_capacity // 2),
        adaptive_row_group_admission=args.adaptive_row_group_admission,
        request_pressure_advisory=args.request_pressure_advisory,
        warmups=args.warmups,
        iterations=args.iterations,
        seed=args.seed,
        scenario_version=args.scenario_version,
        harness_version=args.harness_version,
    )

    for _ in range(args.warmups):
        _run_iteration(inputs, measured=False)

    iterations = [_run_iteration(inputs, measured=True) for _ in range(args.iterations)]
    artifact = _artifact(inputs, iterations)
    _validate_artifact(artifact)
    json_path = output_dir / "async_scheduling_benchmark.json"
    csv_path = output_dir / "async_scheduling_benchmark.csv"
    md_path = output_dir / "async_scheduling_benchmark.md"
    json_path.write_text(json.dumps(_to_jsonable(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, artifact)
    _write_markdown(md_path, artifact)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="origin/main")
    parser.add_argument("--candidate-ref", default="HEAD")
    parser.add_argument("--scenario", default="queue-admission-smoke")
    parser.add_argument("--record-count", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=64)
    parser.add_argument("--row-group-concurrency", type=int, default=2)
    parser.add_argument("--task-admission-capacity", type=int, default=8)
    parser.add_argument("--request-latency-seconds", type=float, default=0.0)
    parser.add_argument("--upstream-latency-seconds", type=float, default=0.01)
    parser.add_argument("--downstream-latency-seconds", type=float, default=0.0)
    parser.add_argument("--fanout-width", type=int, default=3)
    parser.add_argument(
        "--model-stage-weight",
        type=int,
        default=0,
        help="Synthetic custom-model scheduling weight. Defaults to the modeled llm_wait capacity.",
    )
    parser.add_argument("--adaptive-row-group-admission", action="store_true")
    parser.add_argument("--request-pressure-advisory", action="store_true")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=645)
    parser.add_argument("--scenario-version", default="1")
    parser.add_argument("--harness-version", default=HARNESS_VERSION)
    return parser.parse_args()


def _run_iteration(inputs: BenchmarkInputs, *, measured: bool) -> dict[str, Any]:
    if inputs.scenario == "adaptive-request-pressure":
        return _run_adaptive_request_pressure_iteration(inputs, measured=measured)
    if inputs.scenario == "request-pressure-advisory":
        return _run_request_pressure_iteration(inputs, measured=measured)
    if inputs.scenario == "real-pipeline-overlap":
        return _run_real_pipeline_iteration(inputs, measured=measured)
    return _run_queue_admission_iteration(inputs, measured=measured)


def _run_queue_admission_iteration(inputs: BenchmarkInputs, *, measured: bool) -> dict[str, Any]:
    sink = InMemoryAdmissionEventSink()
    task_config = TaskAdmissionConfig(
        submission_capacity=inputs.task_admission_capacity,
        resource_limits={
            "llm_wait": max(1, inputs.task_admission_capacity // 2),
            "local": inputs.task_admission_capacity,
        },
    )
    task_controller = TaskAdmissionController(task_config)
    queue = FairTaskQueue()
    request_config = RequestAdmissionConfig(increase_after_successes=1)
    request_controller = AdaptiveRequestAdmissionController(request_config, event_sink=sink)
    request_controller.register(
        provider_name="mock-provider",
        model_id="mock-model",
        alias="mock-alias",
        max_parallel_requests=max(1, inputs.task_admission_capacity // 2),
    )

    groups = (
        TaskGroupSpec(
            TaskGroupKey("model", ("model", "mock-provider", "mock-model", "chat", "root")),
            weight=2.0,
            admitted_limit=4,
        ),
        TaskGroupSpec(TaskGroupKey("local", ("local", "default", "downstream")), weight=1.0),
    )
    tasks = _schedulable_tasks(inputs.record_count, groups)
    accepted = queue.enqueue(tasks)
    selected: list[str] = []
    request_resource = RequestResourceKey("mock-provider", "mock-model", RequestDomain.CHAT)
    request_item = RequestAdmissionItem(request_resource, RequestGroupSpec(request_resource))
    started = time.monotonic()
    sequence = 0
    accepted_by_id = {task.task_id: task for task in tasks if task.task_id in accepted}
    for task_id in accepted:
        sequence += 1
        task = accepted_by_id[task_id]
        sink.emit_scheduler_event(
            SchedulerAdmissionEvent.capture(
                "ready_enqueued",
                sequence=sequence,
                task_id=task.task_id,
                snapshot=task_controller.view(),
                diagnostics={"resource_request": dict(task.resource_request.amounts)},
            )
        )

    while queue.has_queued_tasks:
        selection = queue.select_next(lambda item, view: task_controller.is_eligible(item, view))
        if selection is None:
            break
        sequence += 1
        sink.emit_scheduler_event(
            SchedulerAdmissionEvent.capture(
                "selected",
                sequence=sequence,
                task_id=selection.item.task_id,
                snapshot=task_controller.view(),
                diagnostics={"resource_request": dict(selection.item.resource_request.amounts)},
            )
        )
        decision = task_controller.try_acquire(selection.item, selection.queue_view)
        if not hasattr(decision, "lease_id"):
            break
        lease = decision
        committed = queue.commit(selection)
        if committed is None:
            task_controller.release(lease)
            continue
        sequence += 1
        sink.emit_scheduler_event(
            SchedulerAdmissionEvent.capture(
                "task_lease_acquired",
                sequence=sequence,
                task_id=committed.task_id,
                task_lease_id=lease.lease_id,
                snapshot=task_controller.view(),
                diagnostics={"resource_request": dict(committed.resource_request.amounts)},
            )
        )
        request_lease = request_controller.acquire_sync(request_item)
        if inputs.request_latency_seconds:
            time.sleep(inputs.request_latency_seconds)
        request_controller.release(request_lease, RequestReleaseOutcome(kind="success"))
        task_controller.release(lease)
        selected.append(committed.task_id)

    wall_time = time.monotonic() - started
    async_request_wakeup_seconds = asyncio.run(_measure_async_request_wakeup_seconds())
    task_snapshot = task_controller.view()
    request_snapshots = request_controller.pressure.snapshots()
    global_snapshots = request_controller.pressure.global_snapshots()
    output_hash = hashlib.sha256("\n".join(selected).encode()).hexdigest()
    max_task_leases = _max_task_leases_by_resource(sink.scheduler_events)
    max_request_in_flight = _max_request_in_flight_by_resource(sink.request_events)
    utilization_metrics = {
        "scheduler_resources": _scheduler_utilization_metrics(sink.scheduler_events),
        "request_resources": _request_utilization_metrics(sink.request_events),
    }
    timeline = [{"stream": "scheduler", **_event_payload(event)} for event in sink.scheduler_events] + [
        {"stream": "request", **_event_payload(event)} for event in sink.request_events
    ]
    timeline.sort(key=lambda event: (event["captured_at_monotonic"], event["sequence"]))
    return {
        "measured": measured,
        "wall_time_seconds": wall_time,
        "async_request_wakeup_seconds": async_request_wakeup_seconds,
        "timeline": timeline,
        "utilization_metrics": utilization_metrics,
        "final_task_snapshot": task_snapshot,
        "final_request_snapshot": {
            "domains": request_snapshots,
            "provider_models": global_snapshots,
            "zero_active_request_leases": all(
                snapshot.active_lease_count == 0 for snapshot in request_snapshots.values()
            ),
            "zero_request_waiters": all(snapshot.waiters == 0 for snapshot in request_snapshots.values()),
        },
        "output_hashes": {"selected_task_ids": output_hash},
        "per_layer_observed_maxima": {
            "selected_tasks": len(selected),
            "accepted_tasks": len(accepted),
            "task_leases_by_resource": max_task_leases,
            "request_in_flight_by_resource": max_request_in_flight,
            "active_task_leases_at_end": sum(task_snapshot.leased_resources.values()),
            "active_request_leases_at_end": sum(snapshot.active_lease_count for snapshot in request_snapshots.values()),
        },
        "accepted_task_count": len(accepted),
        "selected_task_count": len(selected),
    }


def _run_real_pipeline_iteration(inputs: BenchmarkInputs, *, measured: bool) -> dict[str, Any]:
    return asyncio.run(_run_real_pipeline_iteration_async(inputs, measured=measured))


async def _run_real_pipeline_iteration_async(inputs: BenchmarkInputs, *, measured: bool) -> dict[str, Any]:
    sink = InMemoryAdmissionEventSink()
    fanout_width = max(1, inputs.fanout_width)
    upstream_cols = [f"heavy_{index}" for index in range(fanout_width)]
    downstream_cols = [f"fast_{index}" for index in range(fanout_width)]
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        *[LLMTextColumnConfig(name=col, prompt="{{ seed }}", model_alias="benchmark") for col in upstream_cols],
        *[
            LLMTextColumnConfig(name=downstream, prompt=f"{{{{ {upstream} }}}}", model_alias="benchmark")
            for upstream, downstream in zip(upstream_cols, downstream_cols)
        ],
    ]
    strategies: dict[str, GenerationStrategy] = {"seed": GenerationStrategy.FULL_COLUMN}
    strategies.update({col: GenerationStrategy.CELL_BY_CELL for col in (*upstream_cols, *downstream_cols)})
    provider = object()
    generators: dict[str, ColumnGenerator] = {
        "seed": _BenchmarkSeedGenerator(
            config=ExpressionColumnConfig(name="seed", expr="{{ x }}"), resource_provider=provider
        ),
        **{
            col: _BenchmarkTimedCellGenerator(
                config=ExpressionColumnConfig(name=col, expr="{{ x }}"),
                resource_provider=provider,
                delay_seconds=inputs.upstream_latency_seconds,
                model_stage=True,
                model_stage_weight=inputs.model_stage_weight,
            )
            for col in upstream_cols
        },
        **{
            col: _BenchmarkTimedCellGenerator(
                config=ExpressionColumnConfig(name=col, expr="{{ x }}"),
                resource_provider=provider,
                delay_seconds=inputs.downstream_latency_seconds,
                model_stage=True,
                model_stage_weight=inputs.model_stage_weight,
            )
            for col in downstream_cols
        },
    }
    row_groups = _row_groups(inputs.record_count, inputs.buffer_size)
    graph = ExecutionGraph.create(configs, strategies)
    tracker = CompletionTracker.with_graph(graph, row_groups)
    storage = _BenchmarkStorage()
    buffer_manager = RowGroupBufferManager(storage)
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_manager,
        max_concurrent_row_groups=inputs.row_group_concurrency,
        max_submitted_tasks=inputs.task_admission_capacity,
        max_model_task_admission=max(1, inputs.task_admission_capacity // 2),
        scheduler_event_sink=sink,
        trace=True,
        num_records=inputs.record_count,
        buffer_size=inputs.buffer_size,
        adaptive_row_group_admission=inputs.adaptive_row_group_admission,
    )

    started = time.monotonic()
    await scheduler.run()
    wall_time = time.monotonic() - started
    async_request_wakeup_seconds = await _measure_async_request_wakeup_seconds()
    traces = [_trace_payload(trace) for trace in scheduler.traces]
    pipeline_metrics = _pipeline_metrics(
        traces,
        sink.scheduler_events,
        upstream_cols=upstream_cols,
        downstream_cols=downstream_cols,
        submission_capacity=inputs.task_admission_capacity,
        llm_wait_capacity=max(1, inputs.task_admission_capacity // 2),
        row_group_concurrency=inputs.row_group_concurrency,
        capacity_plan=scheduler.capacity_plan(),
        expected_task_count=inputs.record_count * fanout_width,
    )
    utilization_metrics = {
        "scheduler_resources": _scheduler_utilization_metrics(sink.scheduler_events),
        "request_resources": {},
    }
    timeline = [{"stream": "scheduler", **_event_payload(event)} for event in sink.scheduler_events]
    timeline.sort(key=lambda event: (event["captured_at_monotonic"], event["sequence"]))
    task_snapshot = scheduler.task_admission_snapshot()
    return {
        "measured": measured,
        "wall_time_seconds": wall_time,
        "async_request_wakeup_seconds": async_request_wakeup_seconds,
        "timeline": timeline,
        "traces": traces,
        "pipeline_metrics": pipeline_metrics,
        "utilization_metrics": utilization_metrics,
        "capacity_plan": scheduler.capacity_plan(),
        "final_task_snapshot": task_snapshot,
        "final_request_snapshot": {
            "domains": {},
            "provider_models": {},
            "zero_active_request_leases": True,
            "zero_request_waiters": True,
        },
        "output_hashes": {
            "completed_task_trace": hashlib.sha256(
                "\n".join(f"{trace['column']}:{trace['row_group']}:{trace['row_index']}" for trace in traces).encode()
            ).hexdigest()
        },
        "per_layer_observed_maxima": {
            "selected_tasks": len(traces),
            "active_task_leases_at_end": sum(task_snapshot.leased_resources.values()),
            "active_request_leases_at_end": 0,
            **pipeline_metrics["observed_maxima"],
        },
        "accepted_task_count": len(traces),
        "selected_task_count": len(traces),
    }


def _run_request_pressure_iteration(inputs: BenchmarkInputs, *, measured: bool) -> dict[str, Any]:
    return asyncio.run(
        _run_request_pressure_iteration_async(
            inputs,
            measured=measured,
            row_groups=[(0, inputs.record_count)],
            max_concurrent_row_groups=1,
            adaptive_row_group_admission=False,
        )
    )


def _run_adaptive_request_pressure_iteration(inputs: BenchmarkInputs, *, measured: bool) -> dict[str, Any]:
    return asyncio.run(
        _run_request_pressure_iteration_async(
            inputs,
            measured=measured,
            row_groups=_row_groups(inputs.record_count, inputs.buffer_size),
            max_concurrent_row_groups=inputs.row_group_concurrency,
            adaptive_row_group_admission=inputs.adaptive_row_group_admission,
        )
    )


async def _run_request_pressure_iteration_async(
    inputs: BenchmarkInputs,
    *,
    measured: bool,
    row_groups: list[tuple[int, int]],
    max_concurrent_row_groups: int,
    adaptive_row_group_admission: bool,
) -> dict[str, Any]:
    sink = InMemoryAdmissionEventSink()
    request_controller = AdaptiveRequestAdmissionController(
        RequestAdmissionConfig(default_queue_wait_timeout_seconds=30.0),
        event_sink=sink,
    )
    request_controller.register(
        provider_name="aa-pressured-provider",
        model_id="pressured-model",
        alias="pressured",
        max_parallel_requests=1,
    )
    request_controller.register(
        provider_name="zz-open-provider",
        model_id="open-model",
        alias="open",
        max_parallel_requests=1,
    )
    pressured_resource = RequestResourceKey("aa-pressured-provider", "pressured-model", RequestDomain.CHAT)
    holder_item = RequestAdmissionItem(pressured_resource, RequestGroupSpec(pressured_resource))
    holder_lease = request_controller.try_acquire(holder_item)
    if not isinstance(holder_lease, RequestAdmissionLease):
        raise RuntimeError(f"Expected pressure holder request lease, got {holder_lease.reason}")

    async def release_holder() -> None:
        await asyncio.sleep(max(inputs.request_latency_seconds, inputs.upstream_latency_seconds))
        request_controller.release(holder_lease, RequestReleaseOutcome(kind="success"))

    release_task = asyncio.create_task(release_holder())
    configs = [
        SamplerColumnConfig(name="seed", sampler_type=SamplerType.CATEGORY, params={"values": ["A"]}),
        LLMTextColumnConfig(name="a_pressured", prompt="{{ seed }}", model_alias="pressured"),
        LLMTextColumnConfig(name="z_open", prompt="{{ seed }}", model_alias="open"),
    ]
    strategies = {
        "seed": GenerationStrategy.FULL_COLUMN,
        "a_pressured": GenerationStrategy.CELL_BY_CELL,
        "z_open": GenerationStrategy.CELL_BY_CELL,
    }
    provider = object()
    generators: dict[str, ColumnGenerator] = {
        "seed": _BenchmarkSeedGenerator(
            config=ExpressionColumnConfig(name="seed", expr="{{ x }}"), resource_provider=provider
        ),
        "a_pressured": _BenchmarkRequestPressureCellGenerator(
            config=ExpressionColumnConfig(name="a_pressured", expr="{{ x }}"),
            resource_provider=provider,
            request_admission=request_controller,
            provider_name="aa-pressured-provider",
            model_id="pressured-model",
            delay_seconds=inputs.downstream_latency_seconds,
        ),
        "z_open": _BenchmarkRequestPressureCellGenerator(
            config=ExpressionColumnConfig(name="z_open", expr="{{ x }}"),
            resource_provider=provider,
            request_admission=request_controller,
            provider_name="zz-open-provider",
            model_id="open-model",
            delay_seconds=inputs.downstream_latency_seconds,
        ),
    }
    graph = ExecutionGraph.create(configs, strategies)
    tracker = CompletionTracker.with_graph(graph, row_groups)
    storage = _BenchmarkStorage()
    buffer_manager = RowGroupBufferManager(storage)
    scheduler = AsyncTaskScheduler(
        generators=generators,
        graph=graph,
        tracker=tracker,
        row_groups=row_groups,
        buffer_manager=buffer_manager,
        max_concurrent_row_groups=max_concurrent_row_groups,
        max_submitted_tasks=max(1, inputs.task_admission_capacity),
        max_model_task_admission=max(1, inputs.task_admission_capacity),
        scheduler_event_sink=sink,
        trace=True,
        num_records=inputs.record_count,
        buffer_size=inputs.buffer_size,
        adaptive_row_group_admission=adaptive_row_group_admission,
        request_pressure_provider=request_controller.pressure,
        request_pressure_advisory=inputs.request_pressure_advisory,
    )

    started = time.monotonic()
    try:
        await scheduler.run()
    finally:
        await release_task
    wall_time = time.monotonic() - started
    async_request_wakeup_seconds = await _measure_async_request_wakeup_seconds()
    traces = [_trace_payload(trace) for trace in scheduler.traces]
    utilization_metrics = {
        "scheduler_resources": _scheduler_utilization_metrics(sink.scheduler_events),
        "request_resources": _request_utilization_metrics(sink.request_events),
    }
    task_snapshot = scheduler.task_admission_snapshot()
    request_snapshots = request_controller.pressure.snapshots()
    global_snapshots = request_controller.pressure.global_snapshots()
    request_wait_while_leased = _request_wait_while_task_leased_seconds(sink.scheduler_events, sink.request_events)
    timeline = [{"stream": "scheduler", **_event_payload(event)} for event in sink.scheduler_events] + [
        {"stream": "request", **_event_payload(event)} for event in sink.request_events
    ]
    timeline.sort(key=lambda event: (event["captured_at_monotonic"], event["sequence"]))
    return {
        "measured": measured,
        "wall_time_seconds": wall_time,
        "async_request_wakeup_seconds": async_request_wakeup_seconds,
        "timeline": timeline,
        "traces": traces,
        "request_pressure_metrics": {
            "request_wait_seconds_while_task_leased": request_wait_while_leased,
            "request_pressure_advisory_enabled": inputs.request_pressure_advisory,
            "request_pressure_advisory_skip_count": sum(
                1 for event in sink.scheduler_events if event.event_kind == "request_pressure_advisory_skipped"
            ),
            "first_model_dispatch_column": _first_model_dispatch_column(traces),
        },
        "utilization_metrics": utilization_metrics,
        "capacity_plan": scheduler.capacity_plan(),
        "final_task_snapshot": task_snapshot,
        "final_request_snapshot": {
            "domains": request_snapshots,
            "provider_models": global_snapshots,
            "zero_active_request_leases": all(
                snapshot.active_lease_count == 0 for snapshot in request_snapshots.values()
            ),
            "zero_request_waiters": all(snapshot.waiters == 0 for snapshot in request_snapshots.values()),
        },
        "output_hashes": {
            "completed_task_trace": hashlib.sha256(
                "\n".join(f"{trace['column']}:{trace['row_group']}:{trace['row_index']}" for trace in traces).encode()
            ).hexdigest()
        },
        "per_layer_observed_maxima": {
            "selected_tasks": len(traces),
            "accepted_tasks": len(traces),
            "task_leases_by_resource": _max_task_leases_by_resource(sink.scheduler_events),
            "request_in_flight_by_resource": _max_request_in_flight_by_resource(sink.request_events),
            "active_task_leases_at_end": sum(task_snapshot.leased_resources.values()),
            "active_request_leases_at_end": sum(snapshot.active_lease_count for snapshot in request_snapshots.values()),
        },
        "accepted_task_count": len(traces),
        "selected_task_count": len(traces),
    }


def _schedulable_tasks(record_count: int, groups: tuple[TaskGroupSpec, TaskGroupSpec]) -> tuple[SchedulableTask, ...]:
    tasks: list[SchedulableTask] = []
    for index in range(record_count):
        group = groups[index % len(groups)]
        resource_request = (
            SchedulerResourceRequest({"submission": 1, "llm_wait": 1})
            if group.key.kind == "model"
            else SchedulerResourceRequest({"submission": 1, "local": 1})
        )
        task = Task(
            column=f"col_{index % 4}",
            row_group=index // 16,
            row_index=index % 16,
            task_type="cell",
        )
        digest = hashlib.sha1(f"{task.column}:{task.row_group}:{task.row_index}:{task.task_type}".encode()).hexdigest()[
            :16
        ]
        tasks.append(SchedulableTask(f"task-{digest}", task, group, resource_request))
    return tuple(tasks)


def _row_groups(record_count: int, buffer_size: int) -> list[tuple[int, int]]:
    row_groups: list[tuple[int, int]] = []
    remaining = record_count
    row_group = 0
    while remaining > 0:
        size = min(buffer_size, remaining)
        row_groups.append((row_group, size))
        row_group += 1
        remaining -= size
    return row_groups


def _trace_payload(trace: Any) -> dict[str, Any]:
    return {
        "column": trace.column,
        "row_group": trace.row_group,
        "row_index": trace.row_index,
        "task_type": trace.task_type,
        "dispatched_at": trace.dispatched_at,
        "completed_at": trace.completed_at,
        "status": trace.status,
        "error": trace.error,
    }


def _first_model_dispatch_column(traces: list[dict[str, Any]]) -> str | None:
    model_traces = [trace for trace in traces if trace["column"] in {"a_pressured", "z_open"}]
    if not model_traces:
        return None
    return min(model_traces, key=lambda trace: trace["dispatched_at"])["column"]


def _request_wait_while_task_leased_seconds(
    scheduler_events: list[SchedulerAdmissionEvent],
    request_events: list[Any],
) -> float:
    lease_starts_by_lease: dict[str, float] = {}
    lease_intervals: dict[str, tuple[float, float]] = {}
    for event in _ordered_events(scheduler_events):
        lease_id = getattr(event, "task_lease_id", None)
        if lease_id is None:
            continue
        if event.event_kind == "task_lease_acquired":
            lease_starts_by_lease[lease_id] = float(event.captured_at_monotonic)
        elif event.event_kind == "task_lease_released":
            task_execution_id = getattr(event, "task_execution_id", None)
            started = lease_starts_by_lease.pop(lease_id, None)
            if started is not None and task_execution_id is not None:
                lease_intervals[task_execution_id] = (started, float(event.captured_at_monotonic))

    wait_starts: dict[tuple[str, str | None], float] = {}
    total = 0.0
    for event in _ordered_events(request_events):
        correlation = getattr(event, "captured_correlation", None)
        task_execution_id = getattr(correlation, "task_execution_id", None)
        if task_execution_id is None:
            continue
        attempt_key = (task_execution_id, getattr(event, "request_attempt_id", None))
        if event.event_kind == "request_wait_started":
            wait_starts[attempt_key] = float(event.captured_at_monotonic)
        elif event.event_kind in {
            "request_wait_completed",
            "request_wait_timeout",
            "request_wait_cancelled",
            "request_acquire_denied",
        }:
            wait_started = wait_starts.pop(attempt_key, None)
            lease_interval = lease_intervals.get(task_execution_id)
            if wait_started is None or lease_interval is None:
                continue
            wait_ended = float(event.captured_at_monotonic)
            lease_started, lease_ended = lease_interval
            total += max(0.0, min(wait_ended, lease_ended) - max(wait_started, lease_started))
    return total


def _pipeline_metrics(
    traces: list[dict[str, Any]],
    scheduler_events: list[SchedulerAdmissionEvent],
    *,
    upstream_cols: list[str],
    downstream_cols: list[str],
    submission_capacity: int,
    llm_wait_capacity: int,
    row_group_concurrency: int,
    capacity_plan: AsyncCapacityPlan,
    expected_task_count: int,
) -> dict[str, Any]:
    upstream = [trace for trace in traces if trace["column"] in upstream_cols]
    downstream = [trace for trace in traces if trace["column"] in downstream_cols]
    first_upstream_dispatch = min((trace["dispatched_at"] for trace in upstream), default=None)
    last_upstream_dispatch = max((trace["dispatched_at"] for trace in upstream), default=None)
    last_upstream_complete = max((trace["completed_at"] for trace in upstream), default=None)
    first_downstream_dispatch = min((trace["dispatched_at"] for trace in downstream), default=None)
    ready_gaps = _downstream_ready_gaps(upstream, downstream, upstream_cols, downstream_cols)
    max_task_leases = _max_task_leases_by_resource(scheduler_events)
    observed_row_groups = capacity_plan.observed_maxima.row_groups_in_flight
    observed_queued = dict(capacity_plan.observed_maxima.queued_tasks_by_group)
    has_overlap_points = (
        first_downstream_dispatch is not None
        and last_upstream_dispatch is not None
        and last_upstream_complete is not None
        and first_upstream_dispatch is not None
    )
    overlap_seconds = max(0.0, last_upstream_complete - first_downstream_dispatch) if has_overlap_points else 0.0
    upstream_duration = max(0.0, last_upstream_complete - first_upstream_dispatch) if has_overlap_points else 0.0
    validation = {
        "expected_upstream_task_count": len(upstream) == expected_task_count,
        "expected_downstream_task_count": len(downstream) == expected_task_count,
        "expected_downstream_ready_gap_count": len(ready_gaps) == expected_task_count,
        "submission_cap_respected": max_task_leases.get("submission", 0) <= submission_capacity,
        "llm_wait_cap_respected": max_task_leases.get("llm_wait", 0) <= llm_wait_capacity,
        "row_group_cap_respected": observed_row_groups <= row_group_concurrency,
        "downstream_interleaved_before_all_upstream_dispatched": has_overlap_points
        and first_downstream_dispatch < last_upstream_dispatch,
        "downstream_interleaved_before_all_upstream_completed": has_overlap_points
        and first_downstream_dispatch < last_upstream_complete,
    }
    return {
        "upstream_columns": tuple(upstream_cols),
        "downstream_columns": tuple(downstream_cols),
        "expected_task_count": expected_task_count,
        "upstream_task_count": len(upstream),
        "downstream_task_count": len(downstream),
        "downstream_ready_gap_count": len(ready_gaps),
        "first_downstream_dispatch_delay_seconds": (
            max(0.0, first_downstream_dispatch - first_upstream_dispatch) if has_overlap_points else 0.0
        ),
        "downstream_ready_gap_mean_seconds": statistics.fmean(ready_gaps) if ready_gaps else 0.0,
        "downstream_ready_gap_p95_seconds": _percentile(ready_gaps, 0.95),
        "downstream_ready_gap_max_seconds": max(ready_gaps) if ready_gaps else 0.0,
        "upstream_downstream_overlap_seconds": overlap_seconds,
        "upstream_downstream_overlap_ratio": _safe_ratio(overlap_seconds, upstream_duration),
        "observed_maxima": {
            "row_groups_in_flight": observed_row_groups,
            "task_leases_by_resource": max_task_leases,
            "queued_tasks_by_group": observed_queued,
        },
        "validation": validation,
        "validation_passed": all(validation.values()),
    }


def _downstream_ready_gaps(
    upstream: list[dict[str, Any]],
    downstream: list[dict[str, Any]],
    upstream_cols: list[str],
    downstream_cols: list[str],
) -> list[float]:
    upstream_by_key = {(trace["column"], trace["row_group"], trace["row_index"]): trace for trace in upstream}
    gaps: list[float] = []
    for upstream_col, downstream_col in zip(upstream_cols, downstream_cols):
        for downstream_trace in downstream:
            if downstream_trace["column"] != downstream_col:
                continue
            upstream_trace = upstream_by_key.get(
                (upstream_col, downstream_trace["row_group"], downstream_trace["row_index"])
            )
            if upstream_trace is None:
                continue
            gaps.append(max(0.0, downstream_trace["dispatched_at"] - upstream_trace["completed_at"]))
    return gaps


def _max_task_leases_by_resource(events: list[SchedulerAdmissionEvent]) -> dict[str, int]:
    maxima: dict[str, int] = {}
    for event in events:
        snapshot = event.snapshot
        leased = getattr(snapshot, "leased_resources", {})
        for resource, count in leased.items():
            maxima[str(resource)] = max(maxima.get(str(resource), 0), int(count))
    return maxima


def _max_request_in_flight_by_resource(events: list[Any]) -> dict[str, int]:
    maxima: dict[str, int] = {}
    for event in events:
        snapshot = getattr(event, "pressure_snapshot", None)
        if snapshot is None:
            continue
        resource = getattr(snapshot, "resource", None) or getattr(event, "request_resource_key", None)
        if resource is None:
            continue
        count = int(getattr(snapshot, "in_flight_count", 0))
        maxima[str(resource)] = max(maxima.get(str(resource), 0), count)
    return maxima


def _scheduler_utilization_metrics(events: list[Any]) -> dict[str, dict[str, float | int]]:
    ordered = _ordered_events(events)
    if len(ordered) < 2:
        return {}

    resource_limits: dict[str, int] = {}
    active_by_resource: dict[str, int] = {}
    queued_by_task: dict[str, dict[str, int]] = {}
    queued_at: dict[str, float] = {}
    queue_ages: dict[str, list[float]] = {}
    totals = _empty_utilization_totals()

    for index, event in enumerate(ordered):
        now = float(event.captured_at_monotonic)
        snapshot = getattr(event, "snapshot", None)
        if snapshot is not None:
            for resource, limit in _int_mapping(getattr(snapshot, "resource_limits", {})).items():
                resource_limits[resource] = max(resource_limits.get(resource, 0), limit)
            active_by_resource = _int_mapping(getattr(snapshot, "leased_resources", {}))

        task_id = getattr(event, "task_id", None)
        resource_request = _event_resource_request(event)
        if event.event_kind == "ready_enqueued" and task_id is not None:
            queued_by_task[task_id] = resource_request
            queued_at[task_id] = now
        elif event.event_kind == "task_lease_acquired" and task_id is not None:
            task_resources = queued_by_task.pop(task_id, resource_request)
            started = queued_at.pop(task_id, None)
            if started is not None:
                age = max(0.0, now - started)
                for resource, amount in task_resources.items():
                    if amount > 0:
                        queue_ages.setdefault(resource, []).append(age)

        if index == len(ordered) - 1:
            continue
        next_time = float(ordered[index + 1].captured_at_monotonic)
        interval_seconds = max(0.0, next_time - now)
        queued_demand = _queued_resource_demand(queued_by_task)
        _record_utilization_interval(
            totals,
            resource_limits=resource_limits,
            active_by_resource=active_by_resource,
            queued_or_waiting_by_resource=queued_demand,
            interval_seconds=interval_seconds,
        )

    return _finalize_utilization_metrics(totals, queue_ages)


def _request_utilization_metrics(events: list[Any]) -> dict[str, dict[str, float | int]]:
    ordered = [event for event in _ordered_events(events) if getattr(event, "pressure_snapshot", None) is not None]
    if len(ordered) < 2:
        return {}

    capacities: dict[str, int] = {}
    active_by_resource: dict[str, int] = {}
    waiters_by_resource: dict[str, int] = {}
    wait_started: dict[str, float] = {}
    wait_ages: dict[str, list[float]] = {}
    totals = _empty_utilization_totals()

    for index, event in enumerate(ordered):
        now = float(event.captured_at_monotonic)
        snapshot = event.pressure_snapshot
        resource = str(snapshot.resource)
        capacity = max(1, int(getattr(snapshot, "effective_max", 1)))
        capacities[resource] = max(capacities.get(resource, 0), capacity)
        active_by_resource[resource] = int(getattr(snapshot, "in_flight_count", 0))
        waiters_by_resource[resource] = int(getattr(snapshot, "waiters", 0))

        attempt_id = getattr(event, "request_attempt_id", None)
        if attempt_id is not None and event.event_kind == "request_wait_started":
            wait_started[attempt_id] = now
        elif attempt_id is not None and event.event_kind in {
            "request_wait_completed",
            "request_wait_timeout",
            "request_wait_cancelled",
            "request_acquire_denied",
        }:
            started = wait_started.pop(attempt_id, None)
            if started is not None:
                wait_ages.setdefault(resource, []).append(max(0.0, now - started))

        if index == len(ordered) - 1:
            continue
        next_time = float(ordered[index + 1].captured_at_monotonic)
        interval_seconds = max(0.0, next_time - now)
        _record_utilization_interval(
            totals,
            resource_limits=capacities,
            active_by_resource=active_by_resource,
            queued_or_waiting_by_resource=waiters_by_resource,
            interval_seconds=interval_seconds,
        )

    return _finalize_utilization_metrics(totals, wait_ages)


def _ordered_events(events: list[Any]) -> list[Any]:
    return sorted(events, key=lambda event: (event.captured_at_monotonic, event.sequence))


def _empty_utilization_totals() -> dict[str, dict[str, Any]]:
    return {}


def _record_utilization_interval(
    totals: dict[str, dict[str, Any]],
    *,
    resource_limits: dict[str, int],
    active_by_resource: dict[str, int],
    queued_or_waiting_by_resource: dict[str, int],
    interval_seconds: float,
) -> None:
    if interval_seconds <= 0.0:
        return
    resources = {*resource_limits, *active_by_resource, *queued_or_waiting_by_resource}
    for resource in resources:
        active = max(0, active_by_resource.get(resource, 0))
        capacity = max(1, resource_limits.get(resource, max(active, queued_or_waiting_by_resource.get(resource, 0))))
        idle_slots = max(0, capacity - active)
        queued_or_waiting = queued_or_waiting_by_resource.get(resource, 0)
        resource_totals = totals.setdefault(
            resource,
            {
                "capacity": capacity,
                "active_window_seconds": 0.0,
                "capacity_seconds": 0.0,
                "busy_capacity_seconds": 0.0,
                "idle_capacity_seconds": 0.0,
                "starved_idle_seconds": 0.0,
                "max_in_flight": 0,
                "max_idle_slots": 0,
                "samples": [],
            },
        )
        resource_totals["capacity"] = max(int(resource_totals["capacity"]), capacity)
        resource_totals["active_window_seconds"] += interval_seconds
        resource_totals["capacity_seconds"] += capacity * interval_seconds
        resource_totals["busy_capacity_seconds"] += active * interval_seconds
        resource_totals["idle_capacity_seconds"] += idle_slots * interval_seconds
        if queued_or_waiting > 0 and idle_slots > 0:
            resource_totals["starved_idle_seconds"] += idle_slots * interval_seconds
        resource_totals["max_in_flight"] = max(int(resource_totals["max_in_flight"]), active)
        resource_totals["max_idle_slots"] = max(int(resource_totals["max_idle_slots"]), idle_slots)
        resource_totals["samples"].append((interval_seconds, active))


def _finalize_utilization_metrics(
    totals: dict[str, dict[str, Any]],
    queue_ages: dict[str, list[float]],
) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for resource, resource_totals in totals.items():
        capacity_seconds = float(resource_totals["capacity_seconds"])
        active_window_seconds = float(resource_totals["active_window_seconds"])
        busy_capacity_seconds = float(resource_totals["busy_capacity_seconds"])
        idle_capacity_seconds = float(resource_totals["idle_capacity_seconds"])
        starved_idle_seconds = float(resource_totals["starved_idle_seconds"])
        dependency_horizon_idle_seconds = max(0.0, idle_capacity_seconds - starved_idle_seconds)
        ages = queue_ages.get(resource, [])
        if busy_capacity_seconds == 0.0 and starved_idle_seconds == 0.0 and not ages:
            continue
        active_count_mean = _safe_ratio(busy_capacity_seconds, active_window_seconds)
        active_count_stdev = _weighted_stdev(resource_totals["samples"])
        queue_age_mean_seconds = statistics.fmean(ages) if ages else 0.0
        queue_age_p50_seconds = _percentile(ages, 0.50)
        queue_age_p95_seconds = _percentile(ages, 0.95)
        queue_age_max_seconds = max(ages) if ages else 0.0
        metrics[resource] = {
            "capacity": int(resource_totals["capacity"]),
            "active_window_seconds": active_window_seconds,
            "capacity_seconds": capacity_seconds,
            "busy_capacity_seconds": busy_capacity_seconds,
            "idle_capacity_seconds": idle_capacity_seconds,
            "starved_idle_seconds": starved_idle_seconds,
            "dependency_horizon_idle_seconds": dependency_horizon_idle_seconds,
            "frontier_dependency_horizon_idle_seconds": dependency_horizon_idle_seconds,
            "utilization_ratio": _safe_ratio(busy_capacity_seconds, capacity_seconds),
            "idle_ratio": _safe_ratio(idle_capacity_seconds, capacity_seconds),
            "starved_idle_ratio": _safe_ratio(starved_idle_seconds, capacity_seconds),
            "dependency_horizon_idle_ratio": _safe_ratio(dependency_horizon_idle_seconds, capacity_seconds),
            "frontier_dependency_horizon_idle_ratio": _safe_ratio(
                dependency_horizon_idle_seconds,
                capacity_seconds,
            ),
            "active_count_mean": active_count_mean,
            "active_count_stdev": active_count_stdev,
            "burstiness_coefficient": _safe_ratio(active_count_stdev, active_count_mean),
            "max_in_flight": int(resource_totals["max_in_flight"]),
            "max_idle_slots": int(resource_totals["max_idle_slots"]),
            "scheduler_queue_age_mean_seconds": queue_age_mean_seconds,
            "scheduler_queue_age_p50_seconds": queue_age_p50_seconds,
            "scheduler_queue_age_p95_seconds": queue_age_p95_seconds,
            "scheduler_queue_age_max_seconds": queue_age_max_seconds,
            "scheduler_queue_age_sample_count": len(ages),
            "ready_to_dispatch_gap_mean_seconds": queue_age_mean_seconds,
            "ready_to_dispatch_gap_p50_seconds": queue_age_p50_seconds,
            "ready_to_dispatch_gap_p95_seconds": queue_age_p95_seconds,
            "ready_to_dispatch_gap_max_seconds": queue_age_max_seconds,
            "ready_to_dispatch_sample_count": len(ages),
        }
    return metrics


def _event_resource_request(event: Any) -> dict[str, int]:
    diagnostics = getattr(event, "diagnostics", {}) or {}
    return _int_mapping(diagnostics.get("resource_request", {}))


def _queued_resource_demand(queued_by_task: Mapping[str, Mapping[str, int]]) -> dict[str, int]:
    demand: dict[str, int] = {}
    for resources in queued_by_task.values():
        for resource, amount in resources.items():
            demand[resource] = demand.get(resource, 0) + amount
    return demand


def _int_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {str(resource): int(amount) for resource, amount in value.items()}


def _weighted_stdev(samples: list[tuple[float, int]]) -> float:
    total_weight = sum(weight for weight, _value in samples)
    if total_weight <= 0.0:
        return 0.0
    mean = sum(weight * value for weight, value in samples) / total_weight
    variance = sum(weight * ((value - mean) ** 2) for weight, value in samples) / total_weight
    return variance**0.5


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _pipeline_derived_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        return {}
    ready_gap_p95s = [metric["downstream_ready_gap_p95_seconds"] for metric in metrics]
    ready_gap_maxes = [metric["downstream_ready_gap_max_seconds"] for metric in metrics]
    overlap_ratios = [metric["upstream_downstream_overlap_ratio"] for metric in metrics]
    return {
        "pipeline_validation_passed": all(metric["validation_passed"] for metric in metrics),
        "pipeline_mean_overlap_ratio": statistics.fmean(overlap_ratios) if overlap_ratios else 0.0,
        "pipeline_max_downstream_ready_gap_seconds": max(ready_gap_maxes) if ready_gap_maxes else 0.0,
        "pipeline_p95_downstream_ready_gap_seconds": _percentile(ready_gap_p95s, 0.95),
        "pipeline_downstream_interleaved_before_all_upstream_completed": all(
            metric["validation"]["downstream_interleaved_before_all_upstream_completed"] for metric in metrics
        ),
    }


def _request_pressure_derived_metrics(iterations: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [iteration.get("request_pressure_metrics", {}) for iteration in iterations]
    if not any(metrics):
        return {}
    wait_seconds = [float(metric.get("request_wait_seconds_while_task_leased", 0.0)) for metric in metrics]
    first_dispatches = [metric.get("first_model_dispatch_column") for metric in metrics]
    skip_counts = [int(metric.get("request_pressure_advisory_skip_count", 0) or 0) for metric in metrics]
    return {
        "request_pressure_advisory_enabled": any(
            bool(metric.get("request_pressure_advisory_enabled", False)) for metric in metrics
        ),
        "request_pressure_advisory_skip_count": sum(skip_counts),
        "request_wait_seconds_while_task_leased_mean": statistics.fmean(wait_seconds),
        "request_wait_seconds_while_task_leased_max": max(wait_seconds),
        "first_model_dispatch_column": first_dispatches[0] if first_dispatches else None,
    }


def _aggregate_utilization_metrics(iterations: list[dict[str, Any]]) -> dict[str, Any]:
    scheduler = _aggregate_resource_utilization(iterations, "scheduler_resources")
    request = _aggregate_resource_utilization(iterations, "request_resources")
    all_scheduler = list(scheduler.values())
    all_request = list(request.values())
    return {
        "scheduler_resource_utilization": scheduler,
        "request_resource_utilization": request,
        "scheduler_min_utilization_ratio": min(
            (metric["mean_utilization_ratio"] for metric in all_scheduler),
            default=0.0,
        ),
        "scheduler_max_starved_idle_seconds": max(
            (metric["max_starved_idle_seconds"] for metric in all_scheduler),
            default=0.0,
        ),
        "scheduler_max_ready_to_dispatch_gap_seconds": max(
            (metric["max_ready_to_dispatch_gap_seconds"] for metric in all_scheduler),
            default=0.0,
        ),
        "scheduler_max_scheduler_queue_age_seconds": max(
            (metric["max_scheduler_queue_age_seconds"] for metric in all_scheduler),
            default=0.0,
        ),
        "scheduler_max_burstiness_coefficient": max(
            (metric["max_burstiness_coefficient"] for metric in all_scheduler),
            default=0.0,
        ),
        "request_min_utilization_ratio": min(
            (metric["mean_utilization_ratio"] for metric in all_request),
            default=0.0,
        ),
        "request_max_starved_idle_seconds": max(
            (metric["max_starved_idle_seconds"] for metric in all_request),
            default=0.0,
        ),
        "request_max_ready_to_dispatch_gap_seconds": max(
            (metric["max_ready_to_dispatch_gap_seconds"] for metric in all_request),
            default=0.0,
        ),
        "request_max_burstiness_coefficient": max(
            (metric["max_burstiness_coefficient"] for metric in all_request),
            default=0.0,
        ),
    }


def _aggregate_resource_utilization(
    iterations: list[dict[str, Any]],
    resource_kind: str,
) -> dict[str, dict[str, float | int]]:
    by_resource: dict[str, list[dict[str, float | int]]] = {}
    for iteration in iterations:
        for resource, metrics in iteration.get("utilization_metrics", {}).get(resource_kind, {}).items():
            by_resource.setdefault(resource, []).append(metrics)

    aggregated: dict[str, dict[str, float | int]] = {}
    for resource, metrics in by_resource.items():
        utilization_ratios = [float(metric["utilization_ratio"]) for metric in metrics]
        idle_seconds = [float(metric["idle_capacity_seconds"]) for metric in metrics]
        starved_idle_seconds = [float(metric["starved_idle_seconds"]) for metric in metrics]
        dependency_horizon_idle_seconds = [_dependency_horizon_idle_seconds(metric) for metric in metrics]
        dependency_horizon_idle_ratios = [_dependency_horizon_idle_ratio(metric) for metric in metrics]
        burstiness = [float(metric["burstiness_coefficient"]) for metric in metrics]
        scheduler_queue_ages = [
            float(metric.get("scheduler_queue_age_max_seconds", metric["ready_to_dispatch_gap_max_seconds"]))
            for metric in metrics
        ]
        aggregated[resource] = {
            "iterations_observed": len(metrics),
            "mean_utilization_ratio": statistics.fmean(utilization_ratios),
            "min_utilization_ratio": min(utilization_ratios),
            "max_utilization_ratio": max(utilization_ratios),
            "mean_idle_capacity_seconds": statistics.fmean(idle_seconds),
            "mean_starved_idle_seconds": statistics.fmean(starved_idle_seconds),
            "mean_dependency_horizon_idle_seconds": statistics.fmean(dependency_horizon_idle_seconds),
            "max_dependency_horizon_idle_seconds": max(dependency_horizon_idle_seconds),
            "mean_dependency_horizon_idle_ratio": statistics.fmean(dependency_horizon_idle_ratios),
            "max_dependency_horizon_idle_ratio": max(dependency_horizon_idle_ratios),
            "mean_frontier_dependency_horizon_idle_seconds": statistics.fmean(dependency_horizon_idle_seconds),
            "max_frontier_dependency_horizon_idle_seconds": max(dependency_horizon_idle_seconds),
            "mean_frontier_dependency_horizon_idle_ratio": statistics.fmean(dependency_horizon_idle_ratios),
            "max_frontier_dependency_horizon_idle_ratio": max(dependency_horizon_idle_ratios),
            "max_starved_idle_seconds": max(starved_idle_seconds),
            "max_burstiness_coefficient": max(burstiness),
            "max_scheduler_queue_age_seconds": max(scheduler_queue_ages),
            "max_ready_to_dispatch_gap_seconds": max(scheduler_queue_ages),
        }
    return aggregated


def _dependency_horizon_idle_seconds(metric: Mapping[str, float | int]) -> float:
    if "dependency_horizon_idle_seconds" in metric:
        return float(metric["dependency_horizon_idle_seconds"])
    return max(0.0, float(metric["idle_capacity_seconds"]) - float(metric["starved_idle_seconds"]))


def _dependency_horizon_idle_ratio(metric: Mapping[str, float | int]) -> float:
    if "dependency_horizon_idle_ratio" in metric:
        return float(metric["dependency_horizon_idle_ratio"])
    return _safe_ratio(_dependency_horizon_idle_seconds(metric), float(metric["capacity_seconds"]))


async def _measure_async_request_wakeup_seconds() -> float:
    resource = RequestResourceKey("mock-provider", "mock-model", RequestDomain.CHAT)
    controller = AdaptiveRequestAdmissionController(
        RequestAdmissionConfig(default_queue_wait_timeout_seconds=30.0, increase_after_successes=1)
    )
    controller.register(
        provider_name="mock-provider",
        model_id="mock-model",
        alias="mock-alias",
        max_parallel_requests=1,
    )
    item = RequestAdmissionItem(resource, RequestGroupSpec(resource), queue_wait_timeout_seconds=30.0)
    first_lease = controller.try_acquire(item)
    if not isinstance(first_lease, RequestAdmissionLease):
        raise RuntimeError(f"Expected initial request lease, got {first_lease.reason}")

    queued_task = asyncio.create_task(controller.acquire_async(item))
    for _ in range(100):
        snapshot = controller.pressure.snapshot(resource)
        if snapshot is not None and snapshot.waiters == 1:
            break
        await asyncio.sleep(0)
    else:
        raise RuntimeError("Async request waiter did not enqueue.")

    started = time.perf_counter()
    controller.release(first_lease, RequestReleaseOutcome(kind="success"))
    queued_lease = await asyncio.wait_for(queued_task, timeout=5.0)
    elapsed = time.perf_counter() - started
    controller.release(queued_lease, RequestReleaseOutcome(kind="success"))
    return elapsed


def _artifact(inputs: BenchmarkInputs, iterations: list[dict[str, Any]]) -> dict[str, Any]:
    resource = RequestResourceKey("mock-provider", "mock-model", RequestDomain.CHAT)
    provider_model = ProviderModelKey("mock-provider", "mock-model")
    task_capacity = inputs.task_admission_capacity
    capacity_plan = AsyncCapacityPlan(
        configured=AsyncCapacityConfigured(
            buffer_size=CapacityValue(value=inputs.buffer_size, source="run_config"),
            row_group_admission=RowGroupAdmission(
                row_group_concurrency=CapacityValue(value=inputs.row_group_concurrency, source="benchmark_override"),
                observed_in_flight=0,
            ),
            submission_capacity=CapacityValue(value=task_capacity, source="benchmark_override"),
            task_resource_limits=CapacityValue(
                value={"submission": task_capacity, "llm_wait": max(1, task_capacity // 2), "local": task_capacity},
                source="benchmark_override",
            ),
            request_resources=CapacityValue(value=(resource,), source="benchmark_override"),
            provider_model_static_caps=CapacityValue(
                value={
                    provider_model: ProviderModelStaticCap(
                        cap=max(1, task_capacity // 2),
                        aliases=("mock-alias",),
                        raw_caps={"mock-alias": max(1, task_capacity // 2)},
                    )
                },
                source="model_metadata",
            ),
            request_domain_initial_limits=CapacityValue(
                value={resource: max(1, task_capacity // 2)}, source="benchmark_override"
            ),
            request_admission_config=CapacityValue(
                value=RequestAdmissionConfigSnapshot.from_config(RequestAdmissionConfig()),
                source="engine_internal_config",
            ),
            transport_pool_limits=CapacityValue(value={provider_model: task_capacity}, source="adapter_config"),
        ),
        runtime_snapshot=AsyncCapacityRuntimeSnapshot(
            request_domain_current_limits={resource: max(1, task_capacity // 2)},
            request_domain_effective_max={resource: max(1, task_capacity // 2)},
            request_domain_blocked_until={resource: None},
            provider_model_aggregate_in_flight={provider_model: 0},
        ),
        observed_maxima=AsyncCapacityObservedMaxima(
            row_groups_in_flight=inputs.row_group_concurrency,
            request_waiters_by_resource={resource: 0},
            request_in_flight_by_resource={resource: max(1, task_capacity // 2)},
            provider_model_aggregate_in_flight={provider_model: max(1, task_capacity // 2)},
            request_domain_current_limits={resource: max(1, task_capacity // 2)},
            transport_pool_utilization={provider_model: 0},
        ),
    )
    wall_times = [iteration["wall_time_seconds"] for iteration in iterations]
    async_wakeups = [iteration["async_request_wakeup_seconds"] for iteration in iterations]
    pipeline_metrics = [iteration["pipeline_metrics"] for iteration in iterations if "pipeline_metrics" in iteration]
    utilization_metrics = _aggregate_utilization_metrics(iterations)
    request_pressure_metrics = _request_pressure_derived_metrics(iterations)
    return {
        "scenario_id": inputs.scenario,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "scenario_version": inputs.scenario_version,
        "harness_version": inputs.harness_version,
        "baseline_sha": _git_rev_parse(inputs.baseline_ref),
        "candidate_sha": _git_rev_parse(inputs.candidate_ref),
        "worktree_dirty": _worktree_dirty(),
        "worktree_status_short": _worktree_status_short(),
        "worktree_diff_sha256": _worktree_diff_hash(),
        "command_line": sys.argv,
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
        },
        "inputs": inputs,
        "provider_script": {"provider": "mock-provider", "model": "mock-model", "domains": ["chat"]},
        "clock_script": {"time_source": "time.monotonic", "deterministic_replay": False},
        "capacity_plan": capacity_plan,
        "iterations": iterations,
        "derived_metrics": {
            "mean_wall_time_seconds": statistics.fmean(wall_times) if wall_times else 0.0,
            "p50_wall_time_seconds": statistics.median(wall_times) if wall_times else 0.0,
            "p95_wall_time_seconds": _percentile(wall_times, 0.95),
            "min_wall_time_seconds": min(wall_times) if wall_times else 0.0,
            "max_wall_time_seconds": max(wall_times) if wall_times else 0.0,
            "stdev_wall_time_seconds": statistics.stdev(wall_times) if len(wall_times) > 1 else 0.0,
            "mean_async_request_wakeup_seconds": statistics.fmean(async_wakeups) if async_wakeups else 0.0,
            "p95_async_request_wakeup_seconds": _percentile(async_wakeups, 0.95),
            "max_async_request_wakeup_seconds": max(async_wakeups) if async_wakeups else 0.0,
            **_pipeline_derived_metrics(pipeline_metrics),
            **utilization_metrics,
            **request_pressure_metrics,
            "max_hidden_scheduler_resource_waiters": 0,
            "final_zero_task_leases": all(
                sum(iteration["final_task_snapshot"].leased_resources.values()) == 0 for iteration in iterations
            ),
            "final_zero_request_leases": all(
                iteration["final_request_snapshot"]["zero_active_request_leases"] for iteration in iterations
            ),
            "final_zero_request_waiters": all(
                iteration["final_request_snapshot"]["zero_request_waiters"] for iteration in iterations
            ),
        },
    }


def _event_payload(event: Any) -> dict[str, Any]:
    return {
        "event_kind": event.event_kind,
        "captured_at_monotonic": event.captured_at_monotonic,
        "sequence": event.sequence,
        "captured_correlation": event.captured_correlation,
        "task_id": getattr(event, "task_id", None),
        "task_execution_id": getattr(event, "task_execution_id", None),
        "task_lease_id": getattr(event, "task_lease_id", None),
        "request_attempt_id": getattr(event, "request_attempt_id", None),
        "request_lease_id": getattr(event, "request_lease_id", None),
        "scheduler_resource_key": getattr(event, "scheduler_resource_key", None),
        "request_resource_key": getattr(event, "request_resource_key", None),
        "reason_or_outcome": getattr(event, "reason_or_outcome", None) or getattr(event, "reason_or_result", None),
        "snapshot": getattr(event, "snapshot", None),
        "pressure_snapshot": getattr(event, "pressure_snapshot", None),
        "diagnostics": getattr(event, "diagnostics", {}),
    }


def _iteration_min_utilization_ratio(iteration: Mapping[str, Any], resource_kind: str) -> float:
    resources = iteration.get("utilization_metrics", {}).get(resource_kind, {})
    return min((float(metric["utilization_ratio"]) for metric in resources.values()), default=0.0)


def _iteration_starved_idle_seconds(iteration: Mapping[str, Any], resource_kind: str) -> float:
    resources = iteration.get("utilization_metrics", {}).get(resource_kind, {})
    return sum(float(metric["starved_idle_seconds"]) for metric in resources.values())


def _write_csv(path: Path, artifact: Mapping[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "iteration",
                "wall_time_seconds",
                "async_request_wakeup_seconds",
                "accepted_task_count",
                "scheduler_min_utilization_ratio",
                "scheduler_starved_idle_seconds",
                "output_hash",
            ],
        )
        writer.writeheader()
        for index, iteration in enumerate(artifact["iterations"]):
            writer.writerow(
                {
                    "iteration": index,
                    "wall_time_seconds": iteration["wall_time_seconds"],
                    "async_request_wakeup_seconds": iteration["async_request_wakeup_seconds"],
                    "accepted_task_count": iteration["accepted_task_count"],
                    "scheduler_min_utilization_ratio": _iteration_min_utilization_ratio(
                        iteration,
                        "scheduler_resources",
                    ),
                    "scheduler_starved_idle_seconds": _iteration_starved_idle_seconds(
                        iteration,
                        "scheduler_resources",
                    ),
                    "output_hash": next(iter(iteration["output_hashes"].values())),
                }
            )


def _write_markdown(path: Path, artifact: Mapping[str, Any]) -> None:
    metrics = artifact["derived_metrics"]
    lines = [
        "# Async Scheduling Benchmark Smoke",
        "",
        f"- scenario: `{artifact['scenario_id']}`",
        f"- baseline_sha: `{artifact['baseline_sha']}`",
        f"- candidate_sha: `{artifact['candidate_sha']}`",
        f"- iterations: `{len(artifact['iterations'])}`",
        f"- mean_wall_time_seconds: `{metrics['mean_wall_time_seconds']:.6f}`",
        f"- p95_wall_time_seconds: `{metrics['p95_wall_time_seconds']:.6f}`",
        f"- mean_async_request_wakeup_seconds: `{metrics['mean_async_request_wakeup_seconds']:.6f}`",
        f"- p95_async_request_wakeup_seconds: `{metrics['p95_async_request_wakeup_seconds']:.6f}`",
        f"- max_async_request_wakeup_seconds: `{metrics['max_async_request_wakeup_seconds']:.6f}`",
        f"- scheduler_min_utilization_ratio: `{metrics['scheduler_min_utilization_ratio']:.6f}`",
        f"- scheduler_max_starved_idle_seconds: `{metrics['scheduler_max_starved_idle_seconds']:.6f}`",
        f"- scheduler_max_scheduler_queue_age_seconds: `{metrics['scheduler_max_scheduler_queue_age_seconds']:.6f}`",
        "- scheduler_max_ready_to_dispatch_gap_seconds: "
        f"`{metrics['scheduler_max_ready_to_dispatch_gap_seconds']:.6f}`",
        f"- scheduler_max_burstiness_coefficient: `{metrics['scheduler_max_burstiness_coefficient']:.6f}`",
        f"- max_hidden_scheduler_resource_waiters: `{metrics['max_hidden_scheduler_resource_waiters']}`",
        f"- final_zero_task_leases: `{metrics['final_zero_task_leases']}`",
        f"- final_zero_request_leases: `{metrics['final_zero_request_leases']}`",
        f"- final_zero_request_waiters: `{metrics['final_zero_request_waiters']}`",
    ]
    if "pipeline_validation_passed" in metrics:
        lines.extend(
            [
                f"- pipeline_validation_passed: `{metrics['pipeline_validation_passed']}`",
                f"- pipeline_mean_overlap_ratio: `{metrics['pipeline_mean_overlap_ratio']:.6f}`",
                f"- pipeline_p95_downstream_ready_gap_seconds: `{metrics['pipeline_p95_downstream_ready_gap_seconds']:.6f}`",
                f"- pipeline_max_downstream_ready_gap_seconds: `{metrics['pipeline_max_downstream_ready_gap_seconds']:.6f}`",
                "- pipeline_downstream_interleaved_before_all_upstream_completed: "
                f"`{metrics['pipeline_downstream_interleaved_before_all_upstream_completed']}`",
            ]
        )
    lines.extend(
        _resource_utilization_markdown("Scheduler Resource Utilization", metrics["scheduler_resource_utilization"])
    )
    if metrics["request_resource_utilization"]:
        lines.extend(
            _resource_utilization_markdown("Request Resource Utilization", metrics["request_resource_utilization"])
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _resource_utilization_markdown(title: str, resources: Mapping[str, Mapping[str, Any]]) -> list[str]:
    if not resources:
        return []
    lines = ["", f"## {title}", ""]
    for resource, metrics in sorted(resources.items()):
        lines.append(
            f"- `{resource}`: util=`{float(metrics['mean_utilization_ratio']):.6f}`, "
            f"idle_s=`{float(metrics['mean_idle_capacity_seconds']):.6f}`, "
            f"starved_idle_s=`{float(metrics['mean_starved_idle_seconds']):.6f}`, "
            f"dependency_horizon_idle_s=`{float(metrics['mean_dependency_horizon_idle_seconds']):.6f}`, "
            f"max_scheduler_queue_age_s=`{float(metrics['max_scheduler_queue_age_seconds']):.6f}`, "
            f"burstiness=`{float(metrics['max_burstiness_coefficient']):.6f}`"
        )
    return lines


def _validate_artifact(artifact: Mapping[str, Any]) -> None:
    metrics = artifact["derived_metrics"]
    inputs = artifact["inputs"]
    task_capacity = _input_value(inputs, "task_admission_capacity")
    failures: list[str] = []
    if not metrics["final_zero_task_leases"]:
        failures.append("task leases leaked at end of benchmark")
    if not metrics["final_zero_request_leases"]:
        failures.append("request leases leaked at end of benchmark")
    if not metrics["final_zero_request_waiters"]:
        failures.append("request waiters leaked at end of benchmark")
    if metrics["p95_async_request_wakeup_seconds"] > ASYNC_WAKEUP_GATE_SECONDS:
        failures.append(
            "async request wakeup p95 exceeded "
            f"{ASYNC_WAKEUP_GATE_SECONDS:.3f}s: {metrics['p95_async_request_wakeup_seconds']:.6f}s"
        )
    if "pipeline_validation_passed" in metrics and not metrics["pipeline_validation_passed"]:
        failures.append("real-pipeline-overlap validation failed")
    for index, iteration in enumerate(artifact["iterations"]):
        selected_count = iteration.get(
            "selected_task_count",
            iteration.get("per_layer_observed_maxima", {}).get("selected_tasks", 0),
        )
        accepted_count = iteration["accepted_task_count"]
        if selected_count != accepted_count:
            failures.append(
                f"iteration {index} selected {selected_count} tasks but accepted {accepted_count}; queue drained early"
            )
        observed = iteration.get("per_layer_observed_maxima", {})
        task_leases = observed.get("task_leases_by_resource", {})
        for resource, limit in _task_resource_limits(artifact, iteration, task_capacity).items():
            observed_count = int(task_leases.get(resource, 0))
            if observed_count > limit:
                failures.append(f"iteration {index} exceeded {resource} task cap {limit}: observed {observed_count}")
        request_limits = _request_resource_limits(iteration, default=max(1, task_capacity // 2))
        for resource, observed_count in observed.get("request_in_flight_by_resource", {}).items():
            request_capacity = request_limits.get(str(resource), max(1, task_capacity // 2))
            if int(observed_count) > request_capacity:
                failures.append(
                    f"iteration {index} exceeded request cap {request_capacity} for {resource}: observed {observed_count}"
                )
        utilization = iteration.get("utilization_metrics", {})
        scheduler_utilization = utilization.get("scheduler_resources", {})
        if not scheduler_utilization:
            failures.append(f"iteration {index} did not record scheduler utilization metrics")
        _validate_resource_utilization_metrics(
            scheduler_utilization,
            failures=failures,
            iteration=index,
            resource_kind="scheduler",
        )
        _validate_resource_utilization_metrics(
            utilization.get("request_resources", {}),
            failures=failures,
            iteration=index,
            resource_kind="request",
        )
    if failures:
        joined = "; ".join(failures)
        raise RuntimeError(f"Async scheduling benchmark validation failed: {joined}")


def _task_resource_limits(
    artifact: Mapping[str, Any],
    iteration: Mapping[str, Any],
    task_capacity: int,
) -> dict[str, int]:
    default = {
        "submission": task_capacity,
        "llm_wait": max(1, task_capacity // 2),
        "local": task_capacity,
    }
    plan = iteration.get("capacity_plan") or artifact.get("capacity_plan")
    configured = _field(plan, "configured")
    capacity_value = _field(configured, "task_resource_limits")
    value = _field(capacity_value, "value")
    if not isinstance(value, Mapping):
        return default
    return {str(resource): int(limit) for resource, limit in value.items()}


def _request_resource_limits(iteration: Mapping[str, Any], *, default: int) -> dict[str, int]:
    final_request_snapshot = iteration.get("final_request_snapshot", {})
    domains = _field(final_request_snapshot, "domains")
    if not isinstance(domains, Mapping):
        return {}
    limits: dict[str, int] = {}
    for resource, snapshot in domains.items():
        effective_max = _field(snapshot, "effective_max")
        current_limit = _field(snapshot, "current_limit")
        limit = effective_max if effective_max is not None else current_limit
        limits[str(resource)] = int(limit if limit is not None else default)
    return limits


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _validate_resource_utilization_metrics(
    resource_metrics: Mapping[str, Mapping[str, Any]],
    *,
    failures: list[str],
    iteration: int,
    resource_kind: str,
) -> None:
    for resource, metrics in resource_metrics.items():
        capacity_seconds = float(metrics.get("capacity_seconds", 0.0))
        busy_capacity_seconds = float(metrics.get("busy_capacity_seconds", 0.0))
        idle_capacity_seconds = float(metrics.get("idle_capacity_seconds", 0.0))
        starved_idle_seconds = float(metrics.get("starved_idle_seconds", 0.0))
        dependency_horizon_idle_seconds = float(
            metrics.get(
                "dependency_horizon_idle_seconds",
                max(0.0, idle_capacity_seconds - starved_idle_seconds),
            )
        )
        utilization_ratio = float(metrics.get("utilization_ratio", -1.0))
        dependency_horizon_idle_ratio = float(
            metrics.get(
                "dependency_horizon_idle_ratio",
                _safe_ratio(dependency_horizon_idle_seconds, capacity_seconds),
            )
        )
        if (
            min(
                capacity_seconds,
                busy_capacity_seconds,
                idle_capacity_seconds,
                starved_idle_seconds,
                dependency_horizon_idle_seconds,
            )
            < 0.0
        ):
            failures.append(f"iteration {iteration} has negative {resource_kind} utilization metric for {resource}")
        if busy_capacity_seconds > capacity_seconds + 1e-9:
            failures.append(f"iteration {iteration} has busy {resource_kind} capacity above total for {resource}")
        if starved_idle_seconds > idle_capacity_seconds + 1e-9:
            failures.append(f"iteration {iteration} has starved {resource_kind} idle above total idle for {resource}")
        if abs((starved_idle_seconds + dependency_horizon_idle_seconds) - idle_capacity_seconds) > 1e-9:
            failures.append(f"iteration {iteration} has invalid {resource_kind} idle partition for {resource}")
        if not 0.0 <= utilization_ratio <= 1.0:
            failures.append(f"iteration {iteration} has invalid {resource_kind} utilization ratio for {resource}")
        if not 0.0 <= dependency_horizon_idle_ratio <= 1.0:
            failures.append(
                f"iteration {iteration} has invalid {resource_kind} dependency-horizon idle ratio for {resource}"
            )


def _input_value(inputs: Any, name: str) -> Any:
    if isinstance(inputs, Mapping):
        return inputs[name]
    return getattr(inputs, name)


def _git_rev_parse(ref: str) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", ref], text=True).strip()
    except Exception:
        return f"unresolved:{ref}"


def _worktree_dirty() -> bool:
    try:
        result = subprocess.run(["git", "status", "--short"], check=False, capture_output=True, text=True)
    except Exception:
        return True
    return bool(result.stdout.strip())


def _worktree_status_short() -> str:
    try:
        result = subprocess.run(["git", "status", "--short"], check=False, capture_output=True, text=True)
    except Exception as exc:
        return f"unavailable:{exc}"
    return result.stdout


def _worktree_diff_hash() -> str:
    try:
        diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], text=False)
    except Exception as exc:
        return f"unavailable:{exc}"
    return hashlib.sha256(diff).hexdigest()


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * pct))))
    return ordered[index]


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(_to_jsonable(key)): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list | set):
        return [_to_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
