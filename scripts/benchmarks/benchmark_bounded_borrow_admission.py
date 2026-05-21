# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import heapq
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from data_designer.engine.dataset_builders.scheduling.queue import FairTaskQueue
from data_designer.engine.dataset_builders.scheduling.resources import (
    SchedulableTask,
    SchedulerResourceRequest,
    TaskGroupKey,
    TaskGroupSpec,
    stable_task_id,
)
from data_designer.engine.dataset_builders.scheduling.task_admission import (
    TaskAdmissionConfig,
    TaskAdmissionController,
    TaskAdmissionLease,
)
from data_designer.engine.dataset_builders.scheduling.task_model import Task
from data_designer.engine.dataset_builders.scheduling.task_policies import BoundedBorrowTaskAdmissionPolicyConfig

PolicyName = Literal["strict", "bounded"]
ScenarioName = Literal["heavy_root_peer_arrival", "neutral_ready_at_start"]


@dataclass(frozen=True)
class BenchmarkTask:
    name: str
    group_name: str
    ready_at: float
    duration: float
    item: SchedulableTask


@dataclass(frozen=True)
class TaskRecord:
    name: str
    group_name: str
    ready_at: float
    dispatch_at: float
    completed_at: float

    @property
    def wait_seconds(self) -> float:
        return self.dispatch_at - self.ready_at


@dataclass(frozen=True)
class ScenarioConfig:
    name: ScenarioName
    capacity: int
    hot_task_count: int
    peer_task_count: int
    hot_ready_at: float
    peer_ready_at: float
    hot_duration: float
    peer_duration: float
    hot_weight: float
    peer_weight: float
    borrow_ceiling: int


@dataclass(frozen=True)
class ScenarioResult:
    policy: PolicyName
    scenario: ScenarioName
    capacity: int
    task_count: int
    wall_time_seconds: float
    utilization_ratio: float
    hot_dispatch_count_before_peer_ready: int
    peer_first_wait_seconds: float
    peer_wait_mean_seconds: float
    peer_wait_p50_seconds: float
    peer_wait_p95_seconds: float
    peer_wait_max_seconds: float
    final_zero_task_leases: bool


@dataclass(frozen=True)
class BenchmarkReport:
    created_at: str
    git_sha: str
    python: str
    platform: str
    scenarios: list[dict[str, object]]
    comparisons: list[dict[str, object]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark strict fair vs bounded-borrow task admission.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".scratch") / "bounded-borrow-admission",
        help="Directory where JSON and Markdown reports are written.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir / time.strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = (
        ScenarioConfig(
            name="heavy_root_peer_arrival",
            capacity=8,
            hot_task_count=512,
            peer_task_count=256,
            hot_ready_at=0.0,
            peer_ready_at=0.05,
            hot_duration=0.5,
            peer_duration=0.05,
            hot_weight=4.0,
            peer_weight=1.0,
            borrow_ceiling=1,
        ),
        ScenarioConfig(
            name="neutral_ready_at_start",
            capacity=8,
            hot_task_count=256,
            peer_task_count=256,
            hot_ready_at=0.0,
            peer_ready_at=0.0,
            hot_duration=0.1,
            peer_duration=0.1,
            hot_weight=1.0,
            peer_weight=1.0,
            borrow_ceiling=1,
        ),
    )

    results = [run_scenario(config, policy) for config in configs for policy in ("strict", "bounded")]
    comparisons = [_compare_results(config.name, results) for config in configs]
    report = BenchmarkReport(
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git_sha=_git_sha(),
        python=platform.python_version(),
        platform=platform.platform(),
        scenarios=[asdict(result) for result in results],
        comparisons=comparisons,
    )

    json_path = output_dir / "bounded_borrow_admission_benchmark.json"
    markdown_path = output_dir / "bounded_borrow_admission_benchmark.md"
    json_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


def run_scenario(config: ScenarioConfig, policy: PolicyName) -> ScenarioResult:
    tasks = _scenario_tasks(config)
    pending = sorted(tasks, key=lambda task: (task.ready_at, task.name))
    queue = FairTaskQueue()
    controller = _controller(config, policy)
    now = 0.0
    records: list[TaskRecord] = []
    running: list[tuple[float, int, BenchmarkTask, TaskAdmissionLease]] = []
    sequence = 0

    while pending or running or queue.has_queued_tasks:
        while pending and pending[0].ready_at <= now:
            ready = []
            while pending and pending[0].ready_at <= now:
                ready.append(pending.pop(0))
            queue.enqueue(task.item for task in ready)

        dispatched = False
        while queue.has_queued_tasks:
            selection = queue.select_next(controller.is_eligible)
            if selection is None:
                break
            decision = controller.try_acquire(selection.item, selection.queue_view)
            if not isinstance(decision, TaskAdmissionLease):
                break
            committed = queue.commit(selection)
            if committed is None:
                controller.release(decision)
                break
            task = next(task for task in tasks if task.item.task_id == committed.task_id)
            sequence += 1
            completed_at = now + task.duration
            heapq.heappush(running, (completed_at, sequence, task, decision))
            records.append(
                TaskRecord(
                    name=task.name,
                    group_name=task.group_name,
                    ready_at=task.ready_at,
                    dispatch_at=now,
                    completed_at=completed_at,
                )
            )
            dispatched = True

        if dispatched:
            continue

        next_ready_at = pending[0].ready_at if pending else None
        next_completion_at = running[0][0] if running else None
        next_times = [value for value in (next_ready_at, next_completion_at) if value is not None]
        if not next_times:
            break
        now = min(next_times)
        while running and running[0][0] <= now:
            _completed_at, _sequence, _task, lease = heapq.heappop(running)
            controller.release(lease)

    total_busy_seconds = sum(task.duration for task in tasks)
    wall_time = max((record.completed_at for record in records), default=0.0)
    peer_records = [record for record in records if record.group_name == "peer"]
    peer_waits = [record.wait_seconds for record in peer_records]
    hot_before_peer = sum(
        1 for record in records if record.group_name == "hot" and record.dispatch_at < config.peer_ready_at
    )
    return ScenarioResult(
        policy=policy,
        scenario=config.name,
        capacity=config.capacity,
        task_count=len(tasks),
        wall_time_seconds=wall_time,
        utilization_ratio=total_busy_seconds / (wall_time * config.capacity) if wall_time else 0.0,
        hot_dispatch_count_before_peer_ready=hot_before_peer,
        peer_first_wait_seconds=peer_waits[0] if peer_waits else 0.0,
        peer_wait_mean_seconds=sum(peer_waits) / len(peer_waits) if peer_waits else 0.0,
        peer_wait_p50_seconds=_percentile(peer_waits, 0.50),
        peer_wait_p95_seconds=_percentile(peer_waits, 0.95),
        peer_wait_max_seconds=max(peer_waits, default=0.0),
        final_zero_task_leases=not controller.view().leased_resources,
    )


def _scenario_tasks(config: ScenarioConfig) -> list[BenchmarkTask]:
    hot_group = TaskGroupSpec(
        TaskGroupKey(kind="model", identity=("benchmark", "hot")),
        weight=config.hot_weight,
        admitted_limit=config.capacity,
    )
    peer_group = TaskGroupSpec(
        TaskGroupKey(kind="model", identity=("benchmark", "peer")),
        weight=config.peer_weight,
        admitted_limit=config.capacity,
    )
    return [
        *_tasks_for_group("hot", hot_group, config.hot_task_count, config.hot_ready_at, config.hot_duration),
        *_tasks_for_group("peer", peer_group, config.peer_task_count, config.peer_ready_at, config.peer_duration),
    ]


def _tasks_for_group(
    group_name: str,
    group: TaskGroupSpec,
    count: int,
    ready_at: float,
    duration: float,
) -> list[BenchmarkTask]:
    tasks = []
    for index in range(count):
        task = Task(column=group_name, row_group=0, row_index=index, task_type="cell")
        item = SchedulableTask(
            task_id=stable_task_id(task),
            payload=task,
            group=group,
            resource_request=SchedulerResourceRequest({"submission": 1, "llm_wait": 1}),
        )
        tasks.append(
            BenchmarkTask(
                name=f"{group_name}-{index}",
                group_name=group_name,
                ready_at=ready_at,
                duration=duration,
                item=item,
            )
        )
    return tasks


def _controller(config: ScenarioConfig, policy: PolicyName) -> TaskAdmissionController:
    bounded_borrow = (
        BoundedBorrowTaskAdmissionPolicyConfig(default_borrow_ceiling=config.borrow_ceiling)
        if policy == "bounded"
        else None
    )
    return TaskAdmissionController(
        TaskAdmissionConfig(
            submission_capacity=config.capacity,
            resource_limits={"llm_wait": config.capacity},
            bounded_borrow=bounded_borrow,
        )
    )


def _compare_results(scenario: ScenarioName, results: list[ScenarioResult]) -> dict[str, object]:
    by_policy = {result.policy: result for result in results if result.scenario == scenario}
    strict = by_policy["strict"]
    bounded = by_policy["bounded"]
    return {
        "scenario": scenario,
        "peer_p95_wait_delta_seconds": bounded.peer_wait_p95_seconds - strict.peer_wait_p95_seconds,
        "peer_p95_wait_reduction_ratio": _reduction_ratio(strict.peer_wait_p95_seconds, bounded.peer_wait_p95_seconds),
        "peer_first_wait_delta_seconds": bounded.peer_first_wait_seconds - strict.peer_first_wait_seconds,
        "wall_time_delta_seconds": bounded.wall_time_seconds - strict.wall_time_seconds,
        "utilization_delta": bounded.utilization_ratio - strict.utilization_ratio,
        "strict_hot_dispatch_before_peer_ready": strict.hot_dispatch_count_before_peer_ready,
        "bounded_hot_dispatch_before_peer_ready": bounded.hot_dispatch_count_before_peer_ready,
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return ordered[index]


def _reduction_ratio(strict_value: float, bounded_value: float) -> float:
    if strict_value == 0.0:
        return 0.0
    return (strict_value - bounded_value) / strict_value


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _markdown_report(report: BenchmarkReport) -> str:
    lines = [
        "# Bounded Borrow Admission Benchmark",
        "",
        f"- Git SHA: `{report.git_sha}`",
        f"- Python: `{report.python}`",
        f"- Platform: `{report.platform}`",
        "",
        "## Scenario Results",
        "",
        "| Scenario | Policy | Tasks | Wall time (s) | Utilization | Hot dispatches before peer ready | Peer wait p95 (s) | Peer first wait (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario in report.scenarios:
        lines.append(
            "| {scenario} | {policy} | {task_count} | {wall_time_seconds:.3f} | {utilization_ratio:.3f} | "
            "{hot_dispatch_count_before_peer_ready} | {peer_wait_p95_seconds:.3f} | "
            "{peer_first_wait_seconds:.3f} |".format(**scenario)
        )
    lines.extend(
        [
            "",
            "## Comparisons",
            "",
            "| Scenario | Peer p95 wait reduction | Peer first wait delta (s) | Wall time delta (s) | Utilization delta |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for comparison in report.comparisons:
        lines.append(
            "| {scenario} | {peer_p95_wait_reduction_ratio:.1%} | {peer_first_wait_delta_seconds:.3f} | "
            "{wall_time_delta_seconds:.3f} | {utilization_delta:.3f} |".format(**comparison)
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
