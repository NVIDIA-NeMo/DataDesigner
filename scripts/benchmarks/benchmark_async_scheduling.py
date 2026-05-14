# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic async scheduling benchmark smoke harness."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from data_designer.engine.capacity import (
    AsyncCapacityConfigured,
    AsyncCapacityObservedMaxima,
    AsyncCapacityPlan,
    AsyncCapacityRuntimeSnapshot,
    CapacityValue,
    RequestAdmissionConfigSnapshot,
    RowGroupAdmission,
)
from data_designer.engine.dataset_builders.utils.fair_task_queue import FairTaskQueue
from data_designer.engine.dataset_builders.utils.task_admission import TaskAdmissionConfig, TaskAdmissionController
from data_designer.engine.dataset_builders.utils.task_model import Task
from data_designer.engine.dataset_builders.utils.task_scheduling import (
    SchedulableTask,
    SchedulerResourceRequest,
    TaskGroupKey,
    TaskGroupSpec,
)
from data_designer.engine.models.clients.request_admission import (
    AdaptiveRequestAdmissionController,
    ProviderModelKey,
    ProviderModelStaticCap,
    RequestAdmissionConfig,
    RequestAdmissionItem,
    RequestDomain,
    RequestGroupSpec,
    RequestReleaseOutcome,
    RequestResourceKey,
)
from data_designer.engine.observability import InMemoryAdmissionEventSink, SchedulerAdmissionEvent

ARTIFACT_SCHEMA_VERSION = "async-scheduling-benchmark-v1"
HARNESS_VERSION = "1.0"


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
    warmups: int
    iterations: int
    seed: int
    scenario_version: str
    harness_version: str


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
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=645)
    parser.add_argument("--scenario-version", default="1")
    parser.add_argument("--harness-version", default=HARNESS_VERSION)
    return parser.parse_args()


def _run_iteration(inputs: BenchmarkInputs, *, measured: bool) -> dict[str, Any]:
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
            )
        )
        request_lease = request_controller.acquire_sync(request_item)
        if inputs.request_latency_seconds:
            time.sleep(inputs.request_latency_seconds)
        request_controller.release(request_lease, RequestReleaseOutcome(kind="success"))
        task_controller.release(lease)
        selected.append(committed.task_id)

    wall_time = time.monotonic() - started
    task_snapshot = task_controller.view()
    request_snapshots = request_controller.pressure.snapshots()
    global_snapshots = request_controller.pressure.global_snapshots()
    output_hash = hashlib.sha256("\n".join(selected).encode()).hexdigest()
    timeline = [{"stream": "scheduler", **_event_payload(event)} for event in sink.scheduler_events] + [
        {"stream": "request", **_event_payload(event)} for event in sink.request_events
    ]
    timeline.sort(key=lambda event: (event["captured_at_monotonic"], event["sequence"]))
    return {
        "measured": measured,
        "wall_time_seconds": wall_time,
        "timeline": timeline,
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
            "active_task_leases_at_end": sum(task_snapshot.leased_resources.values()),
            "active_request_leases_at_end": sum(snapshot.active_lease_count for snapshot in request_snapshots.values()),
        },
        "accepted_task_count": len(accepted),
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
    return {
        "scenario_id": inputs.scenario,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "scenario_version": inputs.scenario_version,
        "harness_version": inputs.harness_version,
        "baseline_sha": _git_rev_parse(inputs.baseline_ref),
        "candidate_sha": _git_rev_parse(inputs.candidate_ref),
        "worktree_dirty": _worktree_dirty(),
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
    }


def _write_csv(path: Path, artifact: Mapping[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["iteration", "wall_time_seconds", "accepted_task_count", "output_hash"]
        )
        writer.writeheader()
        for index, iteration in enumerate(artifact["iterations"]):
            writer.writerow(
                {
                    "iteration": index,
                    "wall_time_seconds": iteration["wall_time_seconds"],
                    "accepted_task_count": iteration["accepted_task_count"],
                    "output_hash": iteration["output_hashes"]["selected_task_ids"],
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
        f"- max_hidden_scheduler_resource_waiters: `{metrics['max_hidden_scheduler_resource_waiters']}`",
        f"- final_zero_task_leases: `{metrics['final_zero_task_leases']}`",
        f"- final_zero_request_leases: `{metrics['final_zero_request_leases']}`",
        f"- final_zero_request_waiters: `{metrics['final_zero_request_waiters']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


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
