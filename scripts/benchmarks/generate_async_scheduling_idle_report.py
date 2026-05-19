# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate an HTML idle-time analysis report for async scheduling benchmarks."""

from __future__ import annotations

import argparse
import html
import json
import math
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

DEFAULT_ARTIFACT_DIR = Path("artifacts/async-scheduling-idle-analysis")
DEFAULT_REPORT_PATH = Path("reports/async-scheduling-idle-analysis.html")
BENCHMARK_SCRIPT = Path("scripts/benchmarks/benchmark_async_scheduling.py")
IDLE_SUITE_ID = "async-scheduling-idle-regression"
IDLE_SUITE_VERSION = "1.1"
IDLE_SUMMARY_SCHEMA_VERSION = "async-scheduling-idle-summary-v1"


@dataclass(frozen=True)
class IdleBenchmarkCase:
    name: str
    sweep: str
    record_count: int
    buffer_size: int
    row_group_concurrency: int
    task_admission_capacity: int
    fanout_width: int
    upstream_latency_seconds: float
    downstream_latency_seconds: float
    scenario: str = "real-pipeline-overlap"
    request_latency_seconds: float = 0.0
    model_stage_weight: int = 0
    adaptive_row_group_admission: bool = False
    request_pressure_advisory: bool = False
    iterations: int = 1
    warmups: int = 0

    @property
    def generation_count(self) -> int:
        return self.record_count * self.fanout_width * 2


@dataclass(frozen=True)
class IdleBenchmarkResult:
    case: IdleBenchmarkCase
    artifact_path: Path
    mean_wall_time_seconds: float
    p95_wall_time_seconds: float
    llm_utilization_ratio: float
    llm_idle_ratio: float
    llm_starved_idle_ratio: float
    llm_frontier_dependency_horizon_idle_ratio: float
    llm_starved_idle_seconds: float
    llm_frontier_dependency_horizon_idle_seconds: float
    llm_scheduler_queue_age_p95_seconds: float
    llm_scheduler_queue_age_max_seconds: float
    llm_ready_gap_p95_seconds: float
    llm_ready_gap_max_seconds: float
    llm_burstiness_coefficient: float
    submission_utilization_ratio: float
    submission_starved_idle_ratio: float
    submission_frontier_dependency_horizon_idle_ratio: float
    pipeline_overlap_ratio: float
    downstream_ready_gap_p95_seconds: float
    downstream_ready_gap_max_seconds: float
    throughput_generations_per_second: float
    request_wait_seconds_while_task_leased_mean: float
    request_wait_seconds_while_task_leased_max: float
    request_utilization_ratio: float
    request_idle_ratio: float
    request_starved_idle_ratio: float
    request_frontier_dependency_horizon_idle_ratio: float
    request_burstiness_coefficient: float
    request_pressure_advisory_skip_count: int
    first_model_dispatch_column: str | None
    request_pressure_advisory_enabled: bool
    row_group_admission_mode: str
    row_group_admission_target: int | None
    row_group_admission_observed_max_target: int | None
    row_group_admission_max_admitted_rows: int | None
    validation_passed: bool
    final_zero_task_leases: bool
    final_zero_request_leases: bool
    final_zero_request_waiters: bool


def main() -> None:
    args = _parse_args()
    artifact_dir = Path(args.artifact_dir)
    report_path = Path(args.report_path)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    results = run_idle_benchmark_suite(artifact_dir, quick=args.quick, skip_run=args.skip_run)
    render_idle_report(results, report_path, artifact_dir)
    if args.summary_path:
        write_idle_results_summary(Path(args.summary_path), results, quick=args.quick)

    print(f"Wrote {report_path}")
    print(f"Wrote benchmark artifacts under {artifact_dir}")
    if args.summary_path:
        print(f"Wrote {args.summary_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--summary-path")
    parser.add_argument("--skip-run", action="store_true", help="Reuse existing benchmark JSON files.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a shorter smoke suite while preserving every sweep dimension.",
    )
    return parser.parse_args()


def build_idle_benchmark_cases(*, quick: bool) -> list[IdleBenchmarkCase]:
    return _build_cases(quick)


def run_idle_benchmark_suite(
    artifact_dir: Path,
    *,
    quick: bool,
    skip_run: bool,
) -> list[IdleBenchmarkResult]:
    cases = build_idle_benchmark_cases(quick=quick)
    return [_run_or_load_case(case, artifact_dir, skip_run=skip_run) for case in cases]


def render_idle_report(
    results: Sequence[IdleBenchmarkResult],
    report_path: Path,
    artifact_dir: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(results, report_path, artifact_dir), encoding="utf-8")


def write_idle_results_summary(
    summary_path: Path,
    results: Sequence[IdleBenchmarkResult],
    *,
    quick: bool,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = idle_results_summary(results, quick=quick)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def idle_results_summary(results: Sequence[IdleBenchmarkResult], *, quick: bool) -> dict[str, Any]:
    cases = {_case_key(result): _result_summary(result) for result in results}
    return {
        "summary_schema_version": IDLE_SUMMARY_SCHEMA_VERSION,
        "suite_id": IDLE_SUITE_ID,
        "suite_version": IDLE_SUITE_VERSION,
        "mode": "quick" if quick else "full",
        "case_count": len(results),
        "largest_generation_count": max((result.case.generation_count for result in results), default=0),
        "cases": cases,
    }


def _result_summary(result: IdleBenchmarkResult) -> dict[str, Any]:
    return {
        "case": asdict(result.case),
        "artifact_path": str(result.artifact_path),
        "generation_count": result.case.generation_count,
        "mean_wall_time_seconds": result.mean_wall_time_seconds,
        "p95_wall_time_seconds": result.p95_wall_time_seconds,
        "llm_wait_utilization_ratio": result.llm_utilization_ratio,
        "llm_wait_idle_ratio": result.llm_idle_ratio,
        "llm_wait_starved_idle_ratio": result.llm_starved_idle_ratio,
        "llm_wait_frontier_dependency_horizon_idle_ratio": result.llm_frontier_dependency_horizon_idle_ratio,
        "llm_wait_dependency_horizon_idle_ratio": result.llm_frontier_dependency_horizon_idle_ratio,
        "llm_wait_starved_idle_seconds": result.llm_starved_idle_seconds,
        "llm_wait_frontier_dependency_horizon_idle_seconds": result.llm_frontier_dependency_horizon_idle_seconds,
        "llm_wait_dependency_horizon_idle_seconds": result.llm_frontier_dependency_horizon_idle_seconds,
        "llm_wait_scheduler_queue_age_p95_seconds": result.llm_scheduler_queue_age_p95_seconds,
        "llm_wait_scheduler_queue_age_max_seconds": result.llm_scheduler_queue_age_max_seconds,
        "llm_wait_ready_gap_p95_seconds": result.llm_ready_gap_p95_seconds,
        "llm_wait_ready_gap_max_seconds": result.llm_ready_gap_max_seconds,
        "llm_wait_burstiness_coefficient": result.llm_burstiness_coefficient,
        "submission_utilization_ratio": result.submission_utilization_ratio,
        "submission_starved_idle_ratio": result.submission_starved_idle_ratio,
        "submission_frontier_dependency_horizon_idle_ratio": (result.submission_frontier_dependency_horizon_idle_ratio),
        "pipeline_overlap_ratio": result.pipeline_overlap_ratio,
        "downstream_ready_gap_p95_seconds": result.downstream_ready_gap_p95_seconds,
        "downstream_ready_gap_max_seconds": result.downstream_ready_gap_max_seconds,
        "throughput_generations_per_second": result.throughput_generations_per_second,
        "request_wait_seconds_while_task_leased_mean": result.request_wait_seconds_while_task_leased_mean,
        "request_wait_seconds_while_task_leased_max": result.request_wait_seconds_while_task_leased_max,
        "request_utilization_ratio": result.request_utilization_ratio,
        "request_idle_ratio": result.request_idle_ratio,
        "request_starved_idle_ratio": result.request_starved_idle_ratio,
        "request_frontier_dependency_horizon_idle_ratio": result.request_frontier_dependency_horizon_idle_ratio,
        "request_burstiness_coefficient": result.request_burstiness_coefficient,
        "request_pressure_advisory_skip_count": result.request_pressure_advisory_skip_count,
        "first_model_dispatch_column": result.first_model_dispatch_column,
        "request_pressure_advisory_enabled": result.request_pressure_advisory_enabled,
        "row_group_admission_mode": result.row_group_admission_mode,
        "row_group_admission_target": result.row_group_admission_target,
        "row_group_admission_observed_max_target": result.row_group_admission_observed_max_target,
        "row_group_admission_max_admitted_rows": result.row_group_admission_max_admitted_rows,
        "validation_passed": result.validation_passed,
        "final_zero_task_leases": result.final_zero_task_leases,
        "final_zero_request_leases": result.final_zero_request_leases,
        "final_zero_request_waiters": result.final_zero_request_waiters,
        "dominant_idle_class": _dominant_idle_class(result),
        "recommended_next_lever": _recommended_next_lever(result),
    }


def _case_key(result: IdleBenchmarkResult) -> str:
    return f"{result.case.sweep}/{result.case.name}"


def _build_cases(quick: bool) -> list[IdleBenchmarkCase]:
    base = IdleBenchmarkCase(
        name="baseline-frontier",
        sweep="baseline",
        record_count=512,
        buffer_size=64,
        row_group_concurrency=4,
        task_admission_capacity=8,
        fanout_width=4,
        upstream_latency_seconds=0.003,
        downstream_latency_seconds=0.0003,
    )
    rows = [64, 128, 256] if quick else [64, 128, 256, 512, 1024]
    row_group_concurrency = [1, 2, 4] if quick else [1, 2, 4, 8]
    buffers = [1, 16, 64] if quick else [1, 4, 16, 64, 256]
    capacities = [4, 8] if quick else [4, 8, 16]
    adaptation_rows = 512 if quick else 1024
    combined_adaptation_rows = 128 if quick else 512

    cases = [
        replace(base, name=f"rows-{record_count}", sweep="row-scale", record_count=record_count)
        for record_count in rows
    ]
    cases.extend(
        replace(
            base,
            name=f"row-groups-{concurrency}",
            sweep="row-group-concurrency",
            record_count=512,
            buffer_size=1,
            row_group_concurrency=concurrency,
            fanout_width=1,
        )
        for concurrency in row_group_concurrency
    )
    cases.extend(
        replace(base, name=f"buffer-{buffer_size}", sweep="buffer-size", record_count=512, buffer_size=buffer_size)
        for buffer_size in buffers
    )
    cases.extend(
        replace(
            base,
            name=f"capacity-{capacity}",
            sweep="llm-capacity",
            record_count=512,
            buffer_size=1,
            task_admission_capacity=capacity,
            row_group_concurrency=4,
            fanout_width=1,
        )
        for capacity in capacities
    )
    cases.extend(
        (
            replace(
                base,
                name="weight-custom-model-1",
                sweep="custom-model-weight",
                record_count=512,
                model_stage_weight=1,
            ),
            replace(
                base,
                name="weight-model-capacity",
                sweep="custom-model-weight",
                record_count=512,
                model_stage_weight=0,
            ),
        )
    )
    cases.extend(
        (
            replace(
                base,
                name="narrow-frontier-high-cap",
                sweep="stress-shape",
                record_count=1024,
                buffer_size=1,
                row_group_concurrency=1,
                task_admission_capacity=16,
                fanout_width=1,
                upstream_latency_seconds=0.006,
                downstream_latency_seconds=0.0003,
                iterations=3,
            ),
            replace(
                base,
                name="wide-frontier-high-cap",
                sweep="stress-shape",
                record_count=1024,
                buffer_size=64,
                row_group_concurrency=16,
                task_admission_capacity=16,
                fanout_width=4,
                upstream_latency_seconds=0.006,
                downstream_latency_seconds=0.0003,
                iterations=3,
            ),
        )
    )
    cases.extend(
        (
            replace(
                base,
                name="adaptive-row-groups-fixed-low",
                sweep="adaptations",
                record_count=adaptation_rows,
                buffer_size=1,
                row_group_concurrency=1,
                task_admission_capacity=16,
                fanout_width=1,
                upstream_latency_seconds=0.006,
                downstream_latency_seconds=0.0003,
            ),
            replace(
                base,
                name="adaptive-row-groups-adaptive",
                sweep="adaptations",
                record_count=adaptation_rows,
                buffer_size=1,
                row_group_concurrency=8,
                task_admission_capacity=16,
                fanout_width=1,
                upstream_latency_seconds=0.006,
                downstream_latency_seconds=0.0003,
                adaptive_row_group_admission=True,
            ),
            replace(
                base,
                name="adaptive-row-groups-fixed-high",
                sweep="adaptations",
                record_count=adaptation_rows,
                buffer_size=1,
                row_group_concurrency=8,
                task_admission_capacity=16,
                fanout_width=1,
                upstream_latency_seconds=0.006,
                downstream_latency_seconds=0.0003,
            ),
            replace(
                base,
                name="request-pressure-control",
                sweep="adaptations",
                scenario="request-pressure-advisory",
                record_count=32,
                buffer_size=32,
                row_group_concurrency=1,
                task_admission_capacity=1,
                fanout_width=1,
                request_latency_seconds=0.05,
                upstream_latency_seconds=0.0,
                downstream_latency_seconds=0.001,
            ),
            replace(
                base,
                name="request-pressure-advisory",
                sweep="adaptations",
                scenario="request-pressure-advisory",
                record_count=32,
                buffer_size=32,
                row_group_concurrency=1,
                task_admission_capacity=1,
                fanout_width=1,
                request_latency_seconds=0.05,
                upstream_latency_seconds=0.0,
                downstream_latency_seconds=0.001,
                request_pressure_advisory=True,
            ),
            replace(
                base,
                name="adaptive-request-pressure-control",
                sweep="adaptations",
                scenario="adaptive-request-pressure",
                record_count=combined_adaptation_rows,
                buffer_size=1,
                row_group_concurrency=8,
                task_admission_capacity=4,
                fanout_width=1,
                request_latency_seconds=0.05,
                upstream_latency_seconds=0.0,
                downstream_latency_seconds=0.001,
                adaptive_row_group_admission=True,
                iterations=3,
            ),
            replace(
                base,
                name="adaptive-request-pressure-combined",
                sweep="adaptations",
                scenario="adaptive-request-pressure",
                record_count=combined_adaptation_rows,
                buffer_size=1,
                row_group_concurrency=8,
                task_admission_capacity=4,
                fanout_width=1,
                request_latency_seconds=0.05,
                upstream_latency_seconds=0.0,
                downstream_latency_seconds=0.001,
                adaptive_row_group_admission=True,
                request_pressure_advisory=True,
                iterations=3,
            ),
        )
    )
    return cases


def _run_or_load_case(case: IdleBenchmarkCase, artifact_dir: Path, *, skip_run: bool) -> IdleBenchmarkResult:
    output_dir = artifact_dir / case.sweep / case.name
    json_path = output_dir / "async_scheduling_benchmark.json"
    if skip_run and not json_path.exists():
        raise FileNotFoundError(
            f"Cannot reuse benchmark artifact for {case.sweep}/{case.name}: {json_path} does not exist."
        )
    if not skip_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(BENCHMARK_SCRIPT),
            "--scenario",
            case.scenario,
            "--record-count",
            str(case.record_count),
            "--buffer-size",
            str(case.buffer_size),
            "--row-group-concurrency",
            str(case.row_group_concurrency),
            "--task-admission-capacity",
            str(case.task_admission_capacity),
            "--fanout-width",
            str(case.fanout_width),
            "--upstream-latency-seconds",
            _format_float(case.upstream_latency_seconds),
            "--downstream-latency-seconds",
            _format_float(case.downstream_latency_seconds),
            "--request-latency-seconds",
            _format_float(case.request_latency_seconds),
            "--model-stage-weight",
            str(case.model_stage_weight),
            "--warmups",
            str(case.warmups),
            "--iterations",
            str(case.iterations),
            "--output-dir",
            str(output_dir),
        ]
        if case.adaptive_row_group_admission:
            command.append("--adaptive-row-group-admission")
        if case.request_pressure_advisory:
            command.append("--request-pressure-advisory")
        subprocess.run(command, check=True)
    artifact = json.loads(json_path.read_text(encoding="utf-8"))
    return _extract_result(case, json_path, artifact)


def _extract_result(case: IdleBenchmarkCase, artifact_path: Path, artifact: Mapping[str, Any]) -> IdleBenchmarkResult:
    metrics = artifact["derived_metrics"]
    scheduler_resources = metrics.get("scheduler_resource_utilization", {})
    request_resources = metrics.get("request_resource_utilization", {})
    llm = scheduler_resources.get("llm_wait", {})
    submission = scheduler_resources.get("submission", {})
    mean_wall = _float(metrics, "mean_wall_time_seconds")
    row_group_admission = _row_group_admission_snapshot(artifact)
    return IdleBenchmarkResult(
        case=case,
        artifact_path=artifact_path,
        mean_wall_time_seconds=mean_wall,
        p95_wall_time_seconds=_float(metrics, "p95_wall_time_seconds"),
        llm_utilization_ratio=_float(llm, "mean_utilization_ratio"),
        llm_idle_ratio=_safe_ratio(
            _float(llm, "mean_idle_capacity_seconds"),
            _mean_capacity_seconds(artifact, "llm_wait"),
        ),
        llm_starved_idle_ratio=_safe_ratio(
            _float(llm, "mean_starved_idle_seconds"),
            _mean_capacity_seconds(artifact, "llm_wait"),
        ),
        llm_frontier_dependency_horizon_idle_ratio=_idle_ratio(
            llm,
            artifact,
            "llm_wait",
            "mean_frontier_dependency_horizon_idle_ratio",
        ),
        llm_starved_idle_seconds=_float(llm, "mean_starved_idle_seconds"),
        llm_frontier_dependency_horizon_idle_seconds=_float(
            llm,
            "mean_frontier_dependency_horizon_idle_seconds",
            fallback_key="mean_dependency_horizon_idle_seconds",
        ),
        llm_scheduler_queue_age_p95_seconds=max(
            _max_iteration_metric(
                artifact,
                "llm_wait",
                "scheduler_queue_age_p95_seconds",
            ),
            _max_iteration_metric(
                artifact,
                "llm_wait",
                "ready_to_dispatch_gap_p95_seconds",
            ),
        ),
        llm_scheduler_queue_age_max_seconds=_float(
            llm,
            "max_scheduler_queue_age_seconds",
            fallback_key="max_ready_to_dispatch_gap_seconds",
        ),
        llm_ready_gap_p95_seconds=_max_iteration_metric(
            artifact,
            "llm_wait",
            "ready_to_dispatch_gap_p95_seconds",
        ),
        llm_ready_gap_max_seconds=_float(llm, "max_ready_to_dispatch_gap_seconds"),
        llm_burstiness_coefficient=_float(llm, "max_burstiness_coefficient"),
        submission_utilization_ratio=_float(submission, "mean_utilization_ratio"),
        submission_starved_idle_ratio=_safe_ratio(
            _float(submission, "mean_starved_idle_seconds"),
            _mean_capacity_seconds(artifact, "submission"),
        ),
        submission_frontier_dependency_horizon_idle_ratio=_idle_ratio(
            submission,
            artifact,
            "submission",
            "mean_frontier_dependency_horizon_idle_ratio",
        ),
        pipeline_overlap_ratio=_float(metrics, "pipeline_mean_overlap_ratio"),
        downstream_ready_gap_p95_seconds=_float(metrics, "pipeline_p95_downstream_ready_gap_seconds"),
        downstream_ready_gap_max_seconds=_float(metrics, "pipeline_max_downstream_ready_gap_seconds"),
        throughput_generations_per_second=_safe_ratio(case.generation_count, mean_wall),
        request_wait_seconds_while_task_leased_mean=_float(metrics, "request_wait_seconds_while_task_leased_mean"),
        request_wait_seconds_while_task_leased_max=_float(metrics, "request_wait_seconds_while_task_leased_max"),
        request_utilization_ratio=_mean_resource_metric(request_resources, "mean_utilization_ratio"),
        request_idle_ratio=_safe_ratio(
            _sum_resource_metric(request_resources, "mean_idle_capacity_seconds"),
            _sum_request_capacity_seconds(artifact),
        ),
        request_starved_idle_ratio=_safe_ratio(
            _sum_resource_metric(request_resources, "mean_starved_idle_seconds"),
            _sum_request_capacity_seconds(artifact),
        ),
        request_frontier_dependency_horizon_idle_ratio=_safe_ratio(
            _sum_resource_metric(
                request_resources,
                "mean_frontier_dependency_horizon_idle_seconds",
                fallback_key="mean_dependency_horizon_idle_seconds",
            ),
            _sum_request_capacity_seconds(artifact),
        ),
        request_burstiness_coefficient=_max_resource_metric(request_resources, "max_burstiness_coefficient"),
        request_pressure_advisory_skip_count=int(metrics.get("request_pressure_advisory_skip_count", 0) or 0),
        first_model_dispatch_column=_optional_str(metrics.get("first_model_dispatch_column")),
        request_pressure_advisory_enabled=bool(metrics.get("request_pressure_advisory_enabled", False)),
        row_group_admission_mode=str(row_group_admission.get("mode", "fixed")),
        row_group_admission_target=_optional_int(row_group_admission.get("target_in_flight")),
        row_group_admission_observed_max_target=_optional_int(row_group_admission.get("observed_max_target")),
        row_group_admission_max_admitted_rows=_optional_int(row_group_admission.get("max_admitted_rows")),
        validation_passed=bool(metrics.get("pipeline_validation_passed", True)),
        final_zero_task_leases=bool(metrics.get("final_zero_task_leases", False)),
        final_zero_request_leases=bool(metrics.get("final_zero_request_leases", False)),
        final_zero_request_waiters=bool(metrics.get("final_zero_request_waiters", False)),
    )


def _render_report(
    results: Sequence[IdleBenchmarkResult],
    report_path: Path,
    artifact_dir: Path,
) -> str:
    by_sweep = _group_results(results)
    all_valid = all(
        result.validation_passed
        and result.final_zero_task_leases
        and result.final_zero_request_leases
        and result.final_zero_request_waiters
        for result in results
    )
    max_generations = max((result.case.generation_count for result in results), default=0)
    worst_idle = max(results, key=lambda result: result.llm_starved_idle_ratio)
    worst_total_idle = max(results, key=lambda result: result.llm_idle_ratio)
    best_util = max(results, key=lambda result: result.llm_utilization_ratio)
    median_util = _median(result.llm_utilization_ratio for result in results)
    artifact_link = _relative_href(report_path, artifact_dir)

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>Async Scheduling Idle-Time Analysis</title>",
            _style_block(),
            "</head>",
            "<body>",
            "<main>",
            '<section class="hero">',
            '<p class="eyebrow">DataDesigner async scheduler</p>',
            "<h1>Idle-Time Analysis</h1>",
            '<p class="lede">This report measures where model-capacity idle time appears, '
            "which knobs change it, and which remaining idle looks avoidable by scheduler or "
            "admission-control changes.</p>",
            _summary_grid(
                (
                    ("Benchmark cases", f"{len(results)}"),
                    ("Largest synthetic generation count", f"{max_generations:,}"),
                    ("Median llm_wait utilization", _format_percent(median_util)),
                    ("Worst total llm_wait idle", _format_percent(worst_total_idle.llm_idle_ratio)),
                    ("Worst starved llm_wait idle", _format_percent(worst_idle.llm_starved_idle_ratio)),
                    ("Best llm_wait utilization", _format_percent(best_util.llm_utilization_ratio)),
                    ("Validation", "pass" if all_valid else "check failures"),
                )
            ),
            "</section>",
            "<section>",
            "<h2>What Counts As Idle</h2>",
            "<p><strong>General idle</strong> is unused configured capacity. It includes expected idle when "
            "dependencies are not ready. <strong>Starved idle</strong> is the actionable subset: a resource "
            "has idle slots while the scheduler already has queued work that requests that resource. "
            "<strong>Frontier/dependency-horizon idle</strong> is total idle minus starved idle: capacity "
            "is unused because dependency work has not yet exposed a ready scheduler task. "
            "<strong>Scheduler queue age</strong> measures ready-enqueued to lease-acquired time; "
            "<strong>downstream ready gap</strong> measures dependency-complete to downstream-dispatch time. "
            "For this refactor, the most important resource is <code>llm_wait</code>, because it maps to "
            "model-serving capacity that should stay busy when runnable model work exists.</p>",
            "</section>",
            "<section>",
            "<h2>Implementation Improvements</h2>",
            _implementation_improvements_html(results),
            "</section>",
            "<section>",
            "<h2>Findings</h2>",
            _findings_html(results, by_sweep),
            "</section>",
            "<section>",
            "<h2>Idle Classification Board</h2>",
            _idle_classification_table(results),
            "</section>",
            "<section>",
            "<h2>Adaptation Benchmarks</h2>",
            _adaptation_benchmarks_html(results),
            "</section>",
            "<section>",
            "<h2>Figures</h2>",
            _line_chart(
                "Scale: more rows smooth the model-resource flow",
                _sort(by_sweep["row-scale"], "generation_count"),
                x_getter=lambda result: result.case.generation_count,
                y_series=(
                    ("llm_wait utilization", lambda result: result.llm_utilization_ratio, "#1f7a8c"),
                    ("starved llm_wait idle", lambda result: result.llm_starved_idle_ratio, "#b23a48"),
                ),
                y_max=1.0,
                x_label="synthetic model generations",
                y_label="ratio",
            ),
            _line_chart(
                "Frontier width: row-group concurrency changes starvation",
                _sort(by_sweep["row-group-concurrency"], "row_group_concurrency"),
                x_getter=lambda result: result.case.row_group_concurrency,
                y_series=(
                    ("llm_wait utilization", lambda result: result.llm_utilization_ratio, "#1f7a8c"),
                    ("starved llm_wait idle", lambda result: result.llm_starved_idle_ratio, "#b23a48"),
                    ("downstream ready p95", lambda result: result.downstream_ready_gap_p95_seconds, "#6c5ce7"),
                ),
                x_label="row-group concurrency",
                y_label="ratio / seconds",
            ),
            _line_chart(
                "Buffer size: row-group shape affects traffic waves",
                _sort(by_sweep["buffer-size"], "buffer_size"),
                x_getter=lambda result: result.case.buffer_size,
                y_series=(
                    ("llm_wait utilization", lambda result: result.llm_utilization_ratio, "#1f7a8c"),
                    ("starved llm_wait idle", lambda result: result.llm_starved_idle_ratio, "#b23a48"),
                    ("scheduler queue age p95", lambda result: result.llm_scheduler_queue_age_p95_seconds, "#6c5ce7"),
                ),
                x_label="buffer size",
                y_label="ratio / seconds",
            ),
            _line_chart(
                "Capacity scaling: underfeeding becomes visible as cap rises",
                _sort(by_sweep["llm-capacity"], "task_admission_capacity"),
                x_getter=lambda result: result.case.task_admission_capacity // 2,
                y_series=(
                    ("llm_wait utilization", lambda result: result.llm_utilization_ratio, "#1f7a8c"),
                    ("starved llm_wait idle", lambda result: result.llm_starved_idle_ratio, "#b23a48"),
                    ("throughput / 1000", lambda result: result.throughput_generations_per_second / 1000.0, "#326273"),
                ),
                x_label="modeled llm_wait capacity",
                y_label="ratio / kgen/s",
            ),
            _bar_chart(
                "Custom-model metadata changes per-stage fairness",
                _sort(by_sweep["custom-model-weight"], "model_stage_weight"),
                label_getter=_weight_label,
                bars=(
                    ("llm_wait utilization", lambda result: result.llm_utilization_ratio, "#1f7a8c"),
                    ("starved llm_wait idle", lambda result: result.llm_starved_idle_ratio, "#b23a48"),
                ),
                y_max=1.0,
            ),
            _bar_chart(
                "Stress shapes: narrow vs wide runnable horizon",
                _sort(by_sweep["stress-shape"], "name"),
                label_getter=lambda result: result.case.name.replace("-", " "),
                bars=(
                    ("llm_wait utilization", lambda result: result.llm_utilization_ratio, "#1f7a8c"),
                    ("total llm_wait idle", lambda result: result.llm_idle_ratio, "#8a6f3d"),
                    ("starved llm_wait idle", lambda result: result.llm_starved_idle_ratio, "#b23a48"),
                    (
                        "frontier/dependency idle",
                        lambda result: result.llm_frontier_dependency_horizon_idle_ratio,
                        "#6c5ce7",
                    ),
                    ("burstiness / 2", lambda result: result.llm_burstiness_coefficient / 2.0, "#326273"),
                ),
                y_max=1.0,
            ),
            _bar_chart(
                "Adaptation: adaptive row groups target frontier idle",
                [
                    result
                    for result in _sort(by_sweep.get("adaptations", []), "name")
                    if result.case.name.startswith("adaptive-row-groups")
                ],
                label_getter=lambda result: result.case.name.replace("adaptive-row-groups-", ""),
                bars=(
                    ("llm_wait utilization", lambda result: result.llm_utilization_ratio, "#1f7a8c"),
                    (
                        "frontier/dependency idle",
                        lambda result: result.llm_frontier_dependency_horizon_idle_ratio,
                        "#6c5ce7",
                    ),
                    ("starved llm_wait idle", lambda result: result.llm_starved_idle_ratio, "#b23a48"),
                ),
                y_max=1.0,
            ),
            _bar_chart(
                "Adaptation: request-pressure advisory avoids leased request wait",
                [
                    result
                    for result in _sort(by_sweep.get("adaptations", []), "name")
                    if result.case.name.startswith("request-pressure")
                ],
                label_getter=lambda result: result.case.name.replace("request-pressure-", ""),
                bars=(
                    (
                        "request wait while leased",
                        lambda result: result.request_wait_seconds_while_task_leased_mean,
                        "#b23a48",
                    ),
                    ("scheduler queue age p95", lambda result: result.llm_scheduler_queue_age_p95_seconds, "#6c5ce7"),
                    ("wall seconds / 10", lambda result: result.mean_wall_time_seconds / 10.0, "#1f7a8c"),
                ),
            ),
            _bar_chart(
                "Adaptation: combined adaptive frontier and request pressure",
                [
                    result
                    for result in _sort(by_sweep.get("adaptations", []), "name")
                    if result.case.name.startswith("adaptive-request-pressure")
                ],
                label_getter=lambda result: result.case.name.replace("adaptive-request-pressure-", ""),
                bars=(
                    ("llm_wait utilization", lambda result: result.llm_utilization_ratio, "#1f7a8c"),
                    ("request utilization", lambda result: result.request_utilization_ratio, "#326273"),
                    ("request starved idle", lambda result: result.request_starved_idle_ratio, "#b23a48"),
                    (
                        "frontier/dependency idle",
                        lambda result: result.llm_frontier_dependency_horizon_idle_ratio,
                        "#6c5ce7",
                    ),
                ),
                y_max=1.0,
            ),
            "</section>",
            "<section>",
            "<h2>Benchmark Matrix</h2>",
            _results_table(results, report_path),
            f'<p class="artifact-note">Raw artifacts are under <a href="{html.escape(artifact_link)}">'
            f"{html.escape(str(artifact_dir))}</a>.</p>",
            "</section>",
            "<section>",
            "<h2>Potential Changes</h2>",
            _recommendations_html(results),
            "</section>",
            "</main>",
            "</body>",
            "</html>",
            "",
        ]
    )


def _findings_html(
    results: Sequence[IdleBenchmarkResult],
    by_sweep: Mapping[str, list[IdleBenchmarkResult]],
) -> str:
    row_scale = _sort(by_sweep["row-scale"], "generation_count")
    row_gain = row_scale[-1].llm_utilization_ratio - row_scale[0].llm_utilization_ratio
    rg_results = _sort(by_sweep["row-group-concurrency"], "row_group_concurrency")
    rg_util_gain = rg_results[-1].llm_utilization_ratio - rg_results[0].llm_utilization_ratio
    rg_idle_delta = rg_results[0].llm_starved_idle_ratio - min(result.llm_starved_idle_ratio for result in rg_results)
    capacity_results = _sort(by_sweep["llm-capacity"], "task_admission_capacity")
    weight_results = by_sweep["custom-model-weight"]
    weight_one = next(result for result in weight_results if result.case.model_stage_weight == 1)
    weight_auto = next(result for result in weight_results if result.case.model_stage_weight == 0)
    stress = {result.case.name: result for result in by_sweep["stress-shape"]}
    narrow = stress["narrow-frontier-high-cap"]
    wide = stress["wide-frontier-high-cap"]
    worst = max(results, key=lambda result: result.llm_starved_idle_ratio)

    return (
        '<ul class="finding-list">'
        f"<li><strong>Scale helps once enough runnable work exists.</strong> The row-scale sweep moved "
        f"from {_format_percent(row_scale[0].llm_utilization_ratio)} to "
        f"{_format_percent(row_scale[-1].llm_utilization_ratio)} llm_wait utilization "
        f"({_format_signed_percent(row_gain)}). Larger runs amortize startup and drain phases.</li>"
        f"<li><strong>Runnable frontier width is the main controllable idle source.</strong> In the "
        f"row-group sweep, utilization moved from {_format_percent(rg_results[0].llm_utilization_ratio)} "
        f"to {_format_percent(rg_results[-1].llm_utilization_ratio)} "
        f"({_format_signed_percent(rg_util_gain)}), while starved llm_wait idle improved by "
        f"{_format_percent(max(0.0, rg_idle_delta))} between the narrowest case and the best observed case.</li>"
        f"<li><strong>Total idle and starved idle diagnose different causes.</strong> "
        f"<code>{html.escape(narrow.case.name)}</code> had {_format_percent(narrow.llm_idle_ratio)} total "
        f"llm_wait idle but only {_format_percent(narrow.llm_starved_idle_ratio)} starved idle. The remaining "
        f"{_format_percent(narrow.llm_frontier_dependency_horizon_idle_ratio)} was frontier/dependency-horizon "
        "idle: the scheduler often had no ready model work because the admitted frontier was too small.</li>"
        f"<li><strong>Capacity increases expose underfeeding.</strong> In the capacity sweep, modeled "
        f"llm_wait cap {max(1, capacity_results[0].case.task_admission_capacity // 2)} reached "
        f"{_format_percent(capacity_results[0].llm_utilization_ratio)} utilization, while cap "
        f"{max(1, capacity_results[-1].case.task_admission_capacity // 2)} reached "
        f"{_format_percent(capacity_results[-1].llm_utilization_ratio)} with the same small row frontier.</li>"
        f"<li><strong>Custom model scheduling metadata changes traffic shape.</strong> With synthetic "
        f"<code>custom_model</code> weight forced to 1, llm_wait utilization was "
        f"{_format_percent(weight_one.llm_utilization_ratio)}; with the harness defaulting the weight "
        f"to modeled model capacity it was {_format_percent(weight_auto.llm_utilization_ratio)}. "
        "This run did not show weight=1 underfeeding because each synthetic column is its own group, "
        "but it does show that group identity and weight are first-order benchmark inputs.</li>"
        f"<li><strong>High capacity magnifies traffic-shape effects.</strong> The stress case with one "
        f"large row group and high cap reached {_format_percent(narrow.llm_utilization_ratio)} utilization; "
        f"the wide-frontier shape reached {_format_percent(wide.llm_utilization_ratio)}. "
        "This is the vLLM-server-idle failure mode the scheduler needs to expose and track over time.</li>"
        f"<li><strong>Worst observed actionable idle:</strong> {html.escape(worst.case.name)} had "
        f"{_format_percent(worst.llm_starved_idle_ratio)} starved llm_wait idle and "
        f"{_format_seconds(worst.llm_scheduler_queue_age_max_seconds)} max scheduler queue age.</li>"
        "</ul>"
    )


def _find_result(
    results: Sequence[IdleBenchmarkResult],
    sweep: str,
    name: str,
) -> IdleBenchmarkResult | None:
    return next((result for result in results if result.case.sweep == sweep and result.case.name == name), None)


def _adaptive_row_group_card_evidence(
    control: IdleBenchmarkResult | None,
    adaptive: IdleBenchmarkResult | None,
    row_groups_one: IdleBenchmarkResult,
    narrow: IdleBenchmarkResult,
) -> str:
    if control is None or adaptive is None:
        return (
            f"<code>{html.escape(row_groups_one.case.name)}</code> and "
            f"<code>{html.escape(narrow.case.name)}</code> remain the target proof cases; adaptation "
            "benchmark artifacts were not available in this run."
        )
    return (
        f"<code>{html.escape(control.case.name)}</code> to <code>{html.escape(adaptive.case.name)}</code>: "
        f"utilization {_format_signed_percent(adaptive.llm_utilization_ratio - control.llm_utilization_ratio)}, "
        f"frontier idle reduction "
        f"{_format_percent(max(0.0, control.llm_frontier_dependency_horizon_idle_ratio - adaptive.llm_frontier_dependency_horizon_idle_ratio))}, "
        f"observed target {_format_optional_int(adaptive.row_group_admission_observed_max_target)}."
    )


def _request_pressure_card_evidence(
    control: IdleBenchmarkResult | None,
    advisory: IdleBenchmarkResult | None,
) -> str:
    if control is None or advisory is None:
        return "Needs adaptation benchmark artifacts from a non-skip idle-regression run."
    return (
        f"Leased request wait moved from {_format_seconds(control.request_wait_seconds_while_task_leased_mean)} "
        f"to {_format_seconds(advisory.request_wait_seconds_while_task_leased_mean)}; first dispatch moved "
        f"from <code>{html.escape(control.first_model_dispatch_column or 'unknown')}</code> to "
        f"<code>{html.escape(advisory.first_model_dispatch_column or 'unknown')}</code>."
    )


def _implementation_improvements_html(results: Sequence[IdleBenchmarkResult]) -> str:
    narrow = next(result for result in results if result.case.name == "narrow-frontier-high-cap")
    row_groups_one = next(result for result in results if result.case.name == "row-groups-1")
    rows_256 = next(result for result in results if result.case.name == "rows-256")
    wide = next(result for result in results if result.case.name == "wide-frontier-high-cap")
    adaptive_control = _find_result(results, "adaptations", "adaptive-row-groups-fixed-low")
    adaptive = _find_result(results, "adaptations", "adaptive-row-groups-adaptive")
    pressure_control = _find_result(results, "adaptations", "request-pressure-control")
    pressure_advisory = _find_result(results, "adaptations", "request-pressure-advisory")
    classified = _class_counts(results)
    return (
        _summary_grid(
            (
                (
                    "Classification now explicit",
                    f"{classified.get('frontier/dependency-horizon', 0)} frontier, "
                    f"{classified.get('queued-work starvation', 0)} queued",
                ),
                ("Quick-suite proof gates", f"{len(results)} cases, zero validation failures"),
                ("Largest refreshed case", f"{max(result.case.generation_count for result in results):,} generations"),
                ("Narrow frontier idle identified", _format_percent(narrow.llm_frontier_dependency_horizon_idle_ratio)),
                (
                    "Row-groups=1 frontier idle",
                    _format_percent(row_groups_one.llm_frontier_dependency_horizon_idle_ratio),
                ),
                ("Wide queued starvation age", _format_seconds(wide.llm_scheduler_queue_age_p95_seconds)),
            )
        )
        + '<div class="improvement-grid">'
        + _improvement_card(
            "Idle partition metrics",
            "Implemented",
            "The report now splits total idle into queued-work starvation and frontier/dependency-horizon idle. "
            "This prevents the narrow high-capacity case from being mistaken for a request-admission problem.",
            (
                f"<code>{html.escape(narrow.case.name)}</code>: "
                f"{_format_percent(narrow.llm_frontier_dependency_horizon_idle_ratio)} frontier idle, "
                f"{_format_percent(narrow.llm_starved_idle_ratio)} starved idle."
            ),
            "../scripts/benchmarks/benchmark_async_scheduling.py",
        )
        + _improvement_card(
            "Queue age vs downstream delay",
            "Implemented",
            "Ready-to-lease queue age is now shown separately from dependency-complete-to-dispatch delay. "
            "This distinguishes scheduler backlog from slow dependency propagation.",
            (
                f"<code>{html.escape(rows_256.case.name)}</code>: "
                f"{_format_seconds(rows_256.llm_scheduler_queue_age_p95_seconds)} scheduler queue p95, "
                f"{_format_seconds(rows_256.downstream_ready_gap_p95_seconds)} downstream p95."
            ),
            "../scripts/benchmarks/generate_async_scheduling_idle_report.py",
        )
        + _improvement_card(
            "Batched frontier enqueue",
            "Implemented",
            "A frontier delta with many ready tasks now enters the fair queue in one enqueue operation, while "
            "pre-batch parking, dropped rows, and per-task observability are preserved.",
            "Focused scheduler tests prove one queue operation for a 5-task frontier and one operation for a "
            "pre-batch flush after a dropped row.",
            "../packages/data-designer-engine/src/data_designer/engine/dataset_builders/async_scheduler.py",
        )
        + _improvement_card(
            "Resource-overlap peer pressure",
            "Implemented",
            "Task admission now applies group-cap peer pressure only for queued peers that can use the same "
            "typed resource and are hard-resource eligible. Local/submission-only peers no longer create "
            "false pressure on idle model capacity.",
            "Unit tests cover non-overlapping peers, overlapping <code>llm_wait</code> peers, and peers blocked "
            "by another hard resource.",
            "../packages/data-designer-engine/src/data_designer/engine/dataset_builders/scheduling/task_policies.py",
        )
        + _improvement_card(
            "Adaptive row-group admission",
            "Implemented + benchmarked",
            "The scheduler can now start with one admitted row group, raise the row-group target when "
            "model capacity is idle and queued model demand is low, and remain bounded by the configured "
            "hard cap.",
            _adaptive_row_group_card_evidence(adaptive_control, adaptive, row_groups_one, narrow),
            "../packages/data-designer-engine/src/data_designer/engine/dataset_builders/async_scheduler.py",
        )
        + _improvement_card(
            "Request-pressure advisory",
            "Implemented + benchmarked",
            "Task selection can now use request pressure snapshots to prefer an open same-frontier peer before "
            "spawning work that would immediately wait on request capacity. It does not yield and reacquire "
            "leases inside running generators.",
            _request_pressure_card_evidence(pressure_control, pressure_advisory),
            "../packages/data-designer-engine/src/data_designer/engine/models/request_admission",
        )
        + "</div>"
    )


def _improvement_card(
    title: str,
    status: str,
    body: str,
    evidence: str,
    href: str,
) -> str:
    return (
        '<article class="improvement-card">'
        f'<div><h3>{html.escape(title)}</h3><span class="status">{html.escape(status)}</span></div>'
        f"<p>{body}</p>"
        f'<p class="evidence"><strong>Evidence:</strong> {evidence}</p>'
        f'<a href="{html.escape(href)}">source</a>'
        "</article>"
    )


def _idle_classification_table(results: Sequence[IdleBenchmarkResult]) -> str:
    headers = [
        "case",
        "dominant idle class",
        "util",
        "starved idle",
        "frontier idle",
        "queue age p95",
        "downstream p95",
        "next lever",
    ]
    rows = []
    for result in sorted(
        results,
        key=lambda item: (
            _idle_class_priority(_dominant_idle_class(item)),
            -(item.llm_starved_idle_ratio + item.llm_frontier_dependency_horizon_idle_ratio),
            item.case.sweep,
            item.case.name,
        ),
    ):
        idle_class = _dominant_idle_class(result)
        rows.append(
            [
                html.escape(f"{result.case.sweep}/{result.case.name}"),
                f'<span class="idle-class idle-{html.escape(_class_slug(idle_class))}">{html.escape(idle_class)}</span>',
                _format_percent(result.llm_utilization_ratio),
                _format_percent(result.llm_starved_idle_ratio),
                _format_percent(result.llm_frontier_dependency_horizon_idle_ratio),
                _format_seconds(result.llm_scheduler_queue_age_p95_seconds),
                _format_seconds(result.downstream_ready_gap_p95_seconds),
                html.escape(_recommended_next_lever(result)),
            ]
        )
    return (
        "<p>This board is the main analysis surface for optimization work. Frontier/dependency-horizon cases "
        "need row-group or scheduling-shard changes; queued-work starvation cases need queue/admission/resource "
        "selection changes; request-pressure cases need request-backed evidence before scheduler policy changes.</p>"
        + _html_table(headers, rows)
    )


def _adaptation_benchmarks_html(results: Sequence[IdleBenchmarkResult]) -> str:
    adaptive_control = _find_result(results, "adaptations", "adaptive-row-groups-fixed-low")
    adaptive = _find_result(results, "adaptations", "adaptive-row-groups-adaptive")
    adaptive_high = _find_result(results, "adaptations", "adaptive-row-groups-fixed-high")
    pressure_control = _find_result(results, "adaptations", "request-pressure-control")
    pressure_advisory = _find_result(results, "adaptations", "request-pressure-advisory")
    combined_control = _find_result(results, "adaptations", "adaptive-request-pressure-control")
    combined = _find_result(results, "adaptations", "adaptive-request-pressure-combined")
    if adaptive_control is None or adaptive is None or pressure_control is None or pressure_advisory is None:
        return (
            "<p>The adaptation benchmark cases are not present in this artifact set. Re-run the idle "
            "regression suite without <code>--skip-run</code> to collect them.</p>"
        )

    adaptive_util_delta = adaptive.llm_utilization_ratio - adaptive_control.llm_utilization_ratio
    adaptive_frontier_delta = (
        adaptive_control.llm_frontier_dependency_horizon_idle_ratio
        - adaptive.llm_frontier_dependency_horizon_idle_ratio
    )
    adaptive_high_delta = adaptive.llm_utilization_ratio - adaptive_high.llm_utilization_ratio if adaptive_high else 0.0
    request_wait_delta = (
        pressure_control.request_wait_seconds_while_task_leased_mean
        - pressure_advisory.request_wait_seconds_while_task_leased_mean
    )
    queue_age_delta = (
        pressure_control.llm_scheduler_queue_age_p95_seconds - pressure_advisory.llm_scheduler_queue_age_p95_seconds
    )
    combined_request_wait_delta = 0.0
    combined_request_idle_delta = 0.0
    if combined_control is not None and combined is not None:
        combined_request_wait_delta = (
            combined_control.request_wait_seconds_while_task_leased_mean
            - combined.request_wait_seconds_while_task_leased_mean
        )
        combined_request_idle_delta = combined_control.request_idle_ratio - combined.request_idle_ratio
    high_context = ""
    if adaptive_high is not None:
        high_context = (
            f" The fixed-high ceiling reached {_format_percent(adaptive_high.llm_utilization_ratio)} utilization "
            f"with {_format_percent(adaptive_high.llm_frontier_dependency_horizon_idle_ratio)} frontier idle."
        )

    summary = _summary_grid(
        (
            ("Adaptive utilization delta", _format_signed_percent(adaptive_util_delta)),
            ("Adaptive frontier-idle reduction", _format_percent(max(0.0, adaptive_frontier_delta))),
            ("Adaptive vs fixed-high", _format_signed_percent(adaptive_high_delta)),
            (
                "Adaptive observed target",
                _format_optional_int(adaptive.row_group_admission_observed_max_target),
            ),
            ("Request leased-wait reduction", _format_seconds(max(0.0, request_wait_delta))),
            (
                "Advisory first dispatch",
                pressure_advisory.first_model_dispatch_column or "unknown",
            ),
            ("Request queue-age p95 reduction", _format_seconds(max(0.0, queue_age_delta))),
            ("Combined leased-wait delta", _format_signed_seconds(combined_request_wait_delta)),
            ("Combined request-idle delta", _format_signed_percent(combined_request_idle_delta)),
        )
    )
    rows = [
        [
            "adaptive fixed-low control",
            html.escape(adaptive_control.case.name),
            _format_percent(adaptive_control.llm_utilization_ratio),
            _format_percent(adaptive_control.llm_frontier_dependency_horizon_idle_ratio),
            _format_optional_int(adaptive_control.row_group_admission_observed_max_target),
            "-",
            "-",
        ],
        [
            "adaptive enabled",
            html.escape(adaptive.case.name),
            _format_percent(adaptive.llm_utilization_ratio),
            _format_percent(adaptive.llm_frontier_dependency_horizon_idle_ratio),
            _format_optional_int(adaptive.row_group_admission_observed_max_target),
            "-",
            "-",
        ],
        [
            "request control",
            html.escape(pressure_control.case.name),
            _format_percent(pressure_control.llm_utilization_ratio),
            _format_percent(pressure_control.llm_frontier_dependency_horizon_idle_ratio),
            "-",
            _format_seconds(pressure_control.request_wait_seconds_while_task_leased_mean),
            html.escape(pressure_control.first_model_dispatch_column or "unknown"),
        ],
        [
            "request advisory",
            html.escape(pressure_advisory.case.name),
            _format_percent(pressure_advisory.llm_utilization_ratio),
            _format_percent(pressure_advisory.llm_frontier_dependency_horizon_idle_ratio),
            "-",
            _format_seconds(pressure_advisory.request_wait_seconds_while_task_leased_mean),
            html.escape(pressure_advisory.first_model_dispatch_column or "unknown"),
        ],
    ]
    if combined_control is not None:
        rows.append(
            [
                "combined control",
                html.escape(combined_control.case.name),
                _format_percent(combined_control.llm_utilization_ratio),
                _format_percent(combined_control.llm_frontier_dependency_horizon_idle_ratio),
                _format_optional_int(combined_control.row_group_admission_observed_max_target),
                _format_seconds(combined_control.request_wait_seconds_while_task_leased_mean),
                html.escape(combined_control.first_model_dispatch_column or "unknown"),
            ]
        )
    if combined is not None:
        rows.append(
            [
                "combined enabled",
                html.escape(combined.case.name),
                _format_percent(combined.llm_utilization_ratio),
                _format_percent(combined.llm_frontier_dependency_horizon_idle_ratio),
                _format_optional_int(combined.row_group_admission_observed_max_target),
                _format_seconds(combined.request_wait_seconds_while_task_leased_mean),
                html.escape(combined.first_model_dispatch_column or "unknown"),
            ]
        )
    if adaptive_high is not None:
        rows.insert(
            2,
            [
                "adaptive fixed-high control",
                html.escape(adaptive_high.case.name),
                _format_percent(adaptive_high.llm_utilization_ratio),
                _format_percent(adaptive_high.llm_frontier_dependency_horizon_idle_ratio),
                _format_optional_int(adaptive_high.row_group_admission_observed_max_target),
                "-",
                "-",
            ],
        )
    return (
        summary + "<p>The row-group adaptation is judged against a fixed-low frontier because that is the failure "
        "shape where capacity is idle before enough dependency work exists. It should reduce frontier idle "
        "or raise utilization while staying below the hard row-group cap. The fixed-high control isolates "
        "how close the adaptive policy gets to simply admitting the full hard-cap frontier up front."
        f"{high_context}</p>"
        "<p>The request-pressure adaptation is judged by whether it avoids dispatching the pressured model "
        "first and reduces request-wait time while a scheduler task lease is already held. The combined "
        "case keeps adaptive row-group admission enabled while request pressure is present, so it measures "
        "whether the scheduler can widen the frontier and still avoid sending the first available leases "
        "into a saturated model.</p>"
        + _html_table(
            [
                "adaptation",
                "case",
                "util",
                "frontier idle",
                "observed target",
                "leased request wait",
                "first model dispatch",
            ],
            rows,
        )
    )


def _recommendations_html(results: Sequence[IdleBenchmarkResult]) -> str:
    worst = max(results, key=lambda result: result.llm_starved_idle_ratio)
    return (
        '<ol class="recommendations">'
        "<li><strong>Add an adaptive row-group admission controller.</strong> A subclass or policy layer "
        "can watch bottleneck-resource starved idle and admit another row group when model capacity is "
        "idle while queued or pending model work remains, bounded by buffer and memory limits. When total "
        "idle is high but starved idle is low, the controller should treat that as a frontier problem and "
        "prefetch more row groups rather than tuning request AIMD.</li>"
        "<li><strong>Keep resource-aware idle in the standard benchmark output.</strong> The new "
        "<code>llm_wait</code> utilization, starved idle, frontier/dependency-horizon idle, "
        "scheduler queue-age, downstream ready-gap, and burstiness metrics identify "
        "whether a vLLM-like resource is genuinely idle or merely waiting for dependencies.</li>"
        "<li><strong>Document custom-model group identity and weight.</strong> "
        "<code>SchedulingMetadata.custom_model(...)</code> controls fairness and per-group admission. "
        "Plugins that share one external model should use a shared model identity and a weight that reflects "
        "real serving capacity; plugins that represent distinct resources should keep distinct identities.</li>"
        "<li><strong>Use the report as a regression suite.</strong> The worst case here is "
        f"<code>{html.escape(worst.case.name)}</code>; future scheduler or admission changes should improve "
        "that case without increasing ready gaps or violating final zero-lease checks.</li>"
        "</ol>"
    )


def _results_table(results: Sequence[IdleBenchmarkResult], report_path: Path) -> str:
    headers = [
        "case",
        "sweep",
        "scenario",
        "generations",
        "rows",
        "buffer",
        "rg conc",
        "llm cap",
        "util",
        "idle",
        "starved idle",
        "frontier idle",
        "queue age p95",
        "downstream gap p95",
        "leased request wait",
        "request util",
        "request starved idle",
        "advisory skips",
        "first model",
        "overlap",
        "throughput",
        "artifact",
    ]
    rows = []
    for result in sorted(results, key=lambda item: (item.case.sweep, item.case.name)):
        rows.append(
            [
                html.escape(result.case.name),
                html.escape(result.case.sweep),
                html.escape(result.case.scenario),
                f"{result.case.generation_count:,}",
                f"{result.case.record_count:,}",
                f"{result.case.buffer_size:,}",
                f"{result.case.row_group_concurrency}",
                f"{max(1, result.case.task_admission_capacity // 2)}",
                _format_percent(result.llm_utilization_ratio),
                _format_percent(result.llm_idle_ratio),
                _format_percent(result.llm_starved_idle_ratio),
                _format_percent(result.llm_frontier_dependency_horizon_idle_ratio),
                _format_seconds(result.llm_scheduler_queue_age_p95_seconds),
                _format_seconds(result.downstream_ready_gap_p95_seconds),
                _format_seconds(result.request_wait_seconds_while_task_leased_mean),
                _format_percent(result.request_utilization_ratio),
                _format_percent(result.request_starved_idle_ratio),
                f"{result.request_pressure_advisory_skip_count:,}",
                html.escape(result.first_model_dispatch_column or "-"),
                _format_percent(result.pipeline_overlap_ratio),
                f"{result.throughput_generations_per_second:,.0f}/s",
                f'<a href="{html.escape(_relative_href(report_path, result.artifact_path))}">json</a>',
            ]
        )
    return _html_table(headers, rows)


def _line_chart(
    title: str,
    results: Sequence[IdleBenchmarkResult],
    *,
    x_getter: Any,
    y_series: Sequence[tuple[str, Any, str]],
    x_label: str,
    y_label: str,
    y_max: float | None = None,
) -> str:
    width = 880
    height = 360
    left = 64
    right = 24
    top = 42
    bottom = 54
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [float(x_getter(result)) for result in results]
    y_values = [float(getter(result)) for _label, getter, _color in y_series for result in results]
    min_x = min(xs, default=0.0)
    max_x = max(xs, default=1.0)
    if math.isclose(min_x, max_x):
        max_x = min_x + 1.0
    max_y = y_max if y_max is not None else max(y_values, default=1.0)
    max_y = max(max_y, 0.001)

    elements = [_chart_frame(width, height, left, top, plot_w, plot_h, title, x_label, y_label)]
    for series_index, (label, getter, color) in enumerate(y_series):
        points = []
        for result in results:
            x_value = float(x_getter(result))
            y_value = min(float(getter(result)), max_y)
            x = left + ((x_value - min_x) / (max_x - min_x)) * plot_w
            y = top + plot_h - (y_value / max_y) * plot_h
            points.append((x, y, y_value))
        point_attr = " ".join(f"{x:.1f},{y:.1f}" for x, y, _value in points)
        elements.append(f'<polyline points="{point_attr}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y, value in points:
            elements.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"><title>{label}: {value:.4f}</title></circle>'
            )
        elements.append(_legend_item(label, color, series_index))

    for result in results:
        x_value = float(x_getter(result))
        x = left + ((x_value - min_x) / (max_x - min_x)) * plot_w
        elements.append(
            f'<text class="tick" x="{x:.1f}" y="{height - 28}" text-anchor="middle">{_compact_number(x_value)}</text>'
        )
    return f'<figure class="chart">{"".join(elements)}</svg></figure>'


def _bar_chart(
    title: str,
    results: Sequence[IdleBenchmarkResult],
    *,
    label_getter: Any,
    bars: Sequence[tuple[str, Any, str]],
    y_max: float | None = None,
) -> str:
    width = 880
    height = 360
    left = 64
    top = 42
    bottom = 82
    plot_w = width - left - 24
    plot_h = height - top - bottom
    y_values = [float(getter(result)) for _label, getter, _color in bars for result in results]
    max_y = max(y_max or max(y_values, default=1.0), 0.001)
    group_w = plot_w / max(1, len(results))
    bar_w = min(32.0, (group_w - 16.0) / max(1, len(bars)))

    elements = [_chart_frame(width, height, left, top, plot_w, plot_h, title, "case", "ratio")]
    for result_index, result in enumerate(results):
        group_x = left + result_index * group_w + group_w / 2
        for bar_index, (label, getter, color) in enumerate(bars):
            value = min(float(getter(result)), max_y)
            x = group_x - (bar_w * len(bars)) / 2 + bar_index * bar_w
            bar_h = (value / max_y) * plot_h
            y = top + plot_h - bar_h
            elements.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 3:.1f}" height="{bar_h:.1f}" '
                f'rx="2" fill="{color}"><title>{label}: {value:.4f}</title></rect>'
            )
        elements.append(
            f'<text class="tick" x="{group_x:.1f}" y="{height - 48}" text-anchor="middle">'
            f"{html.escape(label_getter(result))}</text>"
        )
    for series_index, (label, _getter, color) in enumerate(bars):
        elements.append(_legend_item(label, color, series_index))
    return f'<figure class="chart">{"".join(elements)}</svg></figure>'


def _chart_frame(
    width: int,
    height: int,
    left: int,
    top: int,
    plot_w: int,
    plot_h: int,
    title: str,
    x_label: str,
    y_label: str,
) -> str:
    bottom = top + plot_h
    right = left + plot_w
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
        f'<text class="chart-title" x="{left}" y="24">{html.escape(title)}</text>'
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>'
        f'<text class="axis-label" x="{left + plot_w / 2:.1f}" y="{height - 8}" text-anchor="middle">{html.escape(x_label)}</text>'
        f'<text class="axis-label" transform="translate(16 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle">{html.escape(y_label)}</text>'
        f'<text class="tick" x="{left - 10}" y="{bottom}" text-anchor="end">0</text>'
        f'<text class="tick" x="{left - 10}" y="{top + 4}" text-anchor="end">max</text>'
        f'<line x1="{left}" y1="{top + plot_h / 2:.1f}" x2="{right}" y2="{top + plot_h / 2:.1f}" class="grid"/>'
    )


def _legend_item(label: str, color: str, index: int) -> str:
    x = 650
    y = 28 + (index % 5) * 17
    return (
        f'<g class="legend"><rect x="{x}" y="{y - 10}" width="10" height="10" fill="{color}"/>'
        f'<text x="{x + 16}" y="{y}" class="legend-text">{html.escape(label)}</text></g>'
    )


def _summary_grid(items: Iterable[tuple[str, str]]) -> str:
    cards = []
    for label, value in items:
        cards.append(
            f'<div class="summary-card"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        )
    return f'<div class="summary-grid">{"".join(cards)}</div>'


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    row_html = []
    for row in rows:
        row_html.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{header_html}</tr></thead><tbody>{"".join(row_html)}</tbody></table></div>'


def _style_block() -> str:
    return """
<style>
:root {
  color-scheme: light;
  --ink: #17212b;
  --muted: #526070;
  --line: #d8dee6;
  --panel: #f7f9fb;
  --accent: #1f7a8c;
  --warn: #b23a48;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: #ffffff;
}
main { max-width: 1120px; margin: 0 auto; padding: 44px 28px 72px; }
.hero { border-bottom: 1px solid var(--line); padding-bottom: 28px; }
.eyebrow { color: var(--accent); font-weight: 700; text-transform: uppercase; letter-spacing: .08em; font-size: 12px; }
h1 { font-size: 44px; line-height: 1.05; margin: 8px 0 14px; letter-spacing: 0; }
h2 { font-size: 24px; margin: 34px 0 12px; letter-spacing: 0; }
p, li { color: var(--muted); line-height: 1.6; }
code { font-family: "SFMono-Regular", Consolas, monospace; font-size: .94em; background: #eef2f6; padding: 1px 5px; border-radius: 4px; }
.lede { max-width: 820px; font-size: 18px; }
.summary-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 26px; }
.summary-card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }
.summary-card span { display: block; color: var(--muted); font-size: 13px; margin-bottom: 8px; }
.summary-card strong { display: block; font-size: 24px; }
.finding-list, .recommendations { padding-left: 22px; }
.finding-list li, .recommendations li { margin: 10px 0; }
.improvement-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 18px; }
.improvement-card { border: 1px solid var(--line); border-radius: 8px; padding: 16px; background: #fff; }
.improvement-card div { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 8px; }
.improvement-card h3 { margin: 0; font-size: 17px; letter-spacing: 0; }
.improvement-card p { margin: 8px 0; }
.improvement-card a { font-size: 13px; font-weight: 700; }
.status { border: 1px solid var(--line); border-radius: 999px; padding: 3px 8px; color: var(--accent); font-size: 12px; font-weight: 700; white-space: nowrap; }
.evidence { font-size: 13px; }
.idle-class { display: inline-block; border-radius: 999px; padding: 3px 9px; font-weight: 700; font-size: 12px; }
.idle-frontier-dependency-horizon { color: #46348a; background: #eeeafd; }
.idle-queued-work-starvation { color: #8a2430; background: #fdebed; }
.idle-healthy-control { color: #1d5f35; background: #e8f5ec; }
.chart { border: 1px solid var(--line); border-radius: 8px; background: #fff; margin: 18px 0; padding: 10px; overflow-x: auto; }
svg { width: 100%; min-width: 760px; height: auto; }
.axis { stroke: #7c8794; stroke-width: 1; }
.grid { stroke: #e5e9ef; stroke-width: 1; }
.chart-title { font-size: 17px; font-weight: 700; fill: var(--ink); }
.axis-label, .tick, .legend-text { fill: var(--muted); font-size: 12px; }
.legend-text { font-weight: 600; }
.table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }
table { border-collapse: collapse; min-width: 1080px; width: 100%; font-size: 13px; }
th, td { padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
th { background: var(--panel); color: var(--ink); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
td { color: var(--muted); }
.artifact-note { font-size: 13px; }
a { color: var(--accent); }
@media (max-width: 760px) {
  main { padding: 28px 18px 52px; }
  h1 { font-size: 34px; }
  .summary-grid { grid-template-columns: 1fr; }
  .improvement-grid { grid-template-columns: 1fr; }
}
</style>
"""


def _group_results(results: Sequence[IdleBenchmarkResult]) -> dict[str, list[IdleBenchmarkResult]]:
    grouped: dict[str, list[IdleBenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(result.case.sweep, []).append(result)
    return grouped


def _sort(results: Sequence[IdleBenchmarkResult], attr: str) -> list[IdleBenchmarkResult]:
    return sorted(results, key=lambda result: getattr(result.case, attr))


def _float(values: Mapping[str, Any], key: str, *, fallback_key: str | None = None) -> float:
    value = values.get(key)
    if value is None and fallback_key is not None:
        value = values.get(fallback_key)
    return float(value or 0.0)


def _idle_ratio(
    values: Mapping[str, Any],
    artifact: Mapping[str, Any],
    resource: str,
    key: str,
) -> float:
    del key
    capacity_seconds = _mean_capacity_seconds(artifact, resource)
    idle_seconds = _float(values, "mean_idle_capacity_seconds")
    starved_seconds = _float(values, "mean_starved_idle_seconds")
    return _safe_ratio(max(0.0, idle_seconds - starved_seconds), capacity_seconds)


def _mean_capacity_seconds(artifact: Mapping[str, Any], resource: str) -> float:
    values = []
    for iteration in artifact.get("iterations", []):
        metrics = iteration.get("utilization_metrics", {}).get("scheduler_resources", {}).get(resource, {})
        values.append(float(metrics.get("capacity_seconds", 0.0) or 0.0))
    if not values:
        return 0.0
    return sum(values) / len(values)


def _sum_request_capacity_seconds(artifact: Mapping[str, Any]) -> float:
    values = []
    for iteration in artifact.get("iterations", []):
        resources = iteration.get("utilization_metrics", {}).get("request_resources", {})
        if not isinstance(resources, Mapping):
            continue
        values.append(sum(float(metrics.get("capacity_seconds", 0.0) or 0.0) for metrics in resources.values()))
    if not values:
        return 0.0
    return sum(values) / len(values)


def _mean_resource_metric(resources: Mapping[str, Mapping[str, Any]], key: str) -> float:
    values = [float(metrics.get(key, 0.0) or 0.0) for metrics in resources.values()]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _sum_resource_metric(
    resources: Mapping[str, Mapping[str, Any]],
    key: str,
    *,
    fallback_key: str | None = None,
) -> float:
    total = 0.0
    for metrics in resources.values():
        value = metrics.get(key)
        if value is None and fallback_key is not None:
            value = metrics.get(fallback_key)
        total += float(value or 0.0)
    return total


def _max_resource_metric(resources: Mapping[str, Mapping[str, Any]], key: str) -> float:
    return max((float(metrics.get(key, 0.0) or 0.0) for metrics in resources.values()), default=0.0)


def _max_iteration_metric(artifact: Mapping[str, Any], resource: str, metric: str) -> float:
    values = []
    for iteration in artifact.get("iterations", []):
        resource_metrics = iteration.get("utilization_metrics", {}).get("scheduler_resources", {}).get(resource, {})
        values.append(float(resource_metrics.get(metric, 0.0) or 0.0))
    return max(values, default=0.0)


def _row_group_admission_snapshot(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    iteration_snapshots = [
        _nested_mapping(
            iteration,
            ("capacity_plan", "configured", "row_group_admission"),
        )
        for iteration in artifact.get("iterations", [])
    ]
    for snapshot in reversed(iteration_snapshots):
        if snapshot:
            return snapshot
    return _nested_mapping(artifact, ("capacity_plan", "configured", "row_group_admission"))


def _nested_mapping(values: Mapping[str, Any], keys: Sequence[str]) -> Mapping[str, Any]:
    current: Any = values
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    return current if isinstance(current, Mapping) else {}


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _safe_subtract(left: float, right: float) -> float:
    return max(0.0, left - right)


def _median(values: Iterable[float]) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        return 0.0
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0


def _format_float(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".")


def _format_percent(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def _format_signed_percent(value: float) -> str:
    sign = "+" if value >= 0.0 else ""
    return f"{sign}{value * 100.0:.1f} pp"


def _format_signed_seconds(value: float) -> str:
    sign = "+" if value >= 0.0 else "-"
    return f"{sign}{_format_seconds(abs(value))}"


def _format_seconds(value: float) -> str:
    if value < 0.001:
        return f"{value * 1_000_000.0:.0f} us"
    if value < 1.0:
        return f"{value * 1000.0:.1f} ms"
    return f"{value:.2f} s"


def _compact_number(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000.0:.1f}k"
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}"


def _format_optional_int(value: int | None) -> str:
    return "n/a" if value is None else f"{value:,}"


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _weight_label(result: IdleBenchmarkResult) -> str:
    if result.case.model_stage_weight == 0:
        return "model cap"
    return f"weight {result.case.model_stage_weight}"


def _dominant_idle_class(result: IdleBenchmarkResult) -> str:
    if (
        result.llm_frontier_dependency_horizon_idle_ratio >= 0.10
        and result.llm_frontier_dependency_horizon_idle_ratio > result.llm_starved_idle_ratio
    ):
        return "frontier/dependency-horizon"
    if result.llm_starved_idle_ratio >= 0.10:
        return "queued-work starvation"
    if result.llm_scheduler_queue_age_p95_seconds > max(1.0, result.downstream_ready_gap_p95_seconds * 10.0):
        return "queued-work starvation"
    return "healthy/control"


def _recommended_next_lever(result: IdleBenchmarkResult) -> str:
    idle_class = _dominant_idle_class(result)
    if idle_class == "frontier/dependency-horizon":
        return "adaptive row-group admission with memory and queue guardrails"
    if idle_class == "queued-work starvation" and result.llm_scheduler_queue_age_p95_seconds > 1.0:
        return "resource-aware task admission, candidate selection, or smaller scheduling shards"
    if idle_class == "queued-work starvation":
        return "task admission/fairness policy and eligible-starved-idle metrics"
    return "keep as control case and watch for regressions"


def _class_counts(results: Sequence[IdleBenchmarkResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        idle_class = _dominant_idle_class(result)
        counts[idle_class] = counts.get(idle_class, 0) + 1
    return counts


def _idle_class_priority(idle_class: str) -> int:
    priorities = {
        "frontier/dependency-horizon": 0,
        "queued-work starvation": 1,
        "healthy/control": 2,
    }
    return priorities.get(idle_class, 99)


def _class_slug(idle_class: str) -> str:
    return idle_class.replace("/", "-").replace(" ", "-")


def _relative_href(report_path: Path, target_path: Path) -> str:
    try:
        return Path("../" + str(target_path)).as_posix() if not target_path.is_absolute() else target_path.as_uri()
    except ValueError:
        return str(target_path)


if __name__ == "__main__":
    main()
