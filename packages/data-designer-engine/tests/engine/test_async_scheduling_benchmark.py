# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


def _load_benchmark_module() -> ModuleType:
    benchmark_path = Path(__file__).resolve().parents[4] / "scripts" / "benchmarks" / "benchmark_async_scheduling.py"
    spec = importlib.util.spec_from_file_location("benchmark_async_scheduling", benchmark_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_idle_regression_module() -> ModuleType:
    benchmark_dir = Path(__file__).resolve().parents[4] / "scripts" / "benchmarks"
    module_name = "run_async_scheduling_idle_regression"
    spec = importlib.util.spec_from_file_location(module_name, benchmark_dir / f"{module_name}.py")
    assert spec is not None
    assert spec.loader is not None
    sys.path.insert(0, str(benchmark_dir))
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(benchmark_dir))
    return module


def _capacity_plan() -> SimpleNamespace:
    return SimpleNamespace(
        observed_maxima=SimpleNamespace(
            row_groups_in_flight=1,
            queued_tasks_by_group={},
        )
    )


def _valid_artifact(iteration: dict[str, Any]) -> dict[str, Any]:
    iteration.setdefault(
        "utilization_metrics",
        {
            "scheduler_resources": {
                "submission": {
                    "capacity_seconds": 1.0,
                    "busy_capacity_seconds": 0.0,
                    "idle_capacity_seconds": 1.0,
                    "starved_idle_seconds": 0.0,
                    "utilization_ratio": 0.0,
                }
            },
            "request_resources": {},
        },
    )
    return {
        "inputs": SimpleNamespace(task_admission_capacity=2),
        "derived_metrics": {
            "final_zero_task_leases": True,
            "final_zero_request_leases": True,
            "final_zero_request_waiters": True,
            "p95_async_request_wakeup_seconds": 0.0,
        },
        "iterations": [iteration],
    }


def test_pipeline_metrics_fails_when_downstream_traces_are_missing() -> None:
    benchmark = _load_benchmark_module()
    traces = [
        {
            "column": "heavy_0",
            "row_group": 0,
            "row_index": 0,
            "dispatched_at": 1.0,
            "completed_at": 2.0,
        }
    ]

    metrics = benchmark._pipeline_metrics(
        traces,
        [],
        upstream_cols=["heavy_0"],
        downstream_cols=["fast_0"],
        submission_capacity=1,
        llm_wait_capacity=1,
        row_group_concurrency=1,
        capacity_plan=_capacity_plan(),
        expected_task_count=1,
    )

    assert metrics["upstream_task_count"] == 1
    assert metrics["downstream_task_count"] == 0
    assert metrics["downstream_ready_gap_count"] == 0
    assert not metrics["validation"]["expected_downstream_task_count"]
    assert not metrics["validation"]["expected_downstream_ready_gap_count"]
    assert not metrics["validation_passed"]


def test_benchmark_validation_rejects_early_queue_exit() -> None:
    benchmark = _load_benchmark_module()
    artifact = _valid_artifact(
        {
            "accepted_task_count": 2,
            "selected_task_count": 1,
            "per_layer_observed_maxima": {
                "task_leases_by_resource": {},
                "request_in_flight_by_resource": {},
            },
        }
    )

    with pytest.raises(RuntimeError, match="queue drained early"):
        benchmark._validate_artifact(artifact)


def test_benchmark_validation_rejects_cap_violations() -> None:
    benchmark = _load_benchmark_module()
    artifact = _valid_artifact(
        {
            "accepted_task_count": 2,
            "selected_task_count": 2,
            "per_layer_observed_maxima": {
                "task_leases_by_resource": {"submission": 3},
                "request_in_flight_by_resource": {"mock": 2},
            },
        }
    )

    with pytest.raises(RuntimeError, match="exceeded submission task cap"):
        benchmark._validate_artifact(artifact)


def test_scheduler_utilization_metrics_record_idle_and_starved_capacity() -> None:
    benchmark = _load_benchmark_module()
    events = [
        _scheduler_event(
            "ready_enqueued",
            timestamp=0.0,
            sequence=1,
            task_id="task-a",
            leased_resources={},
            resource_request={"submission": 1},
        ),
        _scheduler_event(
            "task_lease_acquired",
            timestamp=1.0,
            sequence=2,
            task_id="task-a",
            leased_resources={"submission": 1},
            resource_request={"submission": 1},
        ),
        _scheduler_event(
            "task_lease_released",
            timestamp=3.0,
            sequence=3,
            task_id="task-a",
            leased_resources={},
            resource_request={"submission": 1},
        ),
    ]

    metrics = benchmark._scheduler_utilization_metrics(events)["submission"]

    assert metrics["capacity"] == 2
    assert metrics["busy_capacity_seconds"] == pytest.approx(2.0)
    assert metrics["idle_capacity_seconds"] == pytest.approx(4.0)
    assert metrics["starved_idle_seconds"] == pytest.approx(2.0)
    assert metrics["dependency_horizon_idle_seconds"] == pytest.approx(2.0)
    assert metrics["frontier_dependency_horizon_idle_ratio"] == pytest.approx(1 / 3)
    assert metrics["utilization_ratio"] == pytest.approx(1 / 3)
    assert metrics["scheduler_queue_age_max_seconds"] == pytest.approx(1.0)
    assert metrics["ready_to_dispatch_gap_max_seconds"] == pytest.approx(1.0)
    assert metrics["ready_to_dispatch_sample_count"] == 1


def test_request_utilization_metrics_record_waiter_starved_idle() -> None:
    benchmark = _load_benchmark_module()
    events = [
        _request_event("request_wait_started", timestamp=0.0, sequence=1, in_flight=0, waiters=1),
        _request_event("request_wait_completed", timestamp=1.0, sequence=2, in_flight=1, waiters=0),
        _request_event("request_lease_released", timestamp=3.0, sequence=3, in_flight=0, waiters=0),
    ]

    metrics = benchmark._request_utilization_metrics(events)["mock-resource"]

    assert metrics["capacity"] == 2
    assert metrics["busy_capacity_seconds"] == pytest.approx(2.0)
    assert metrics["idle_capacity_seconds"] == pytest.approx(4.0)
    assert metrics["starved_idle_seconds"] == pytest.approx(2.0)
    assert metrics["dependency_horizon_idle_seconds"] == pytest.approx(2.0)
    assert metrics["scheduler_queue_age_max_seconds"] == pytest.approx(1.0)
    assert metrics["ready_to_dispatch_gap_max_seconds"] == pytest.approx(1.0)


def test_benchmark_validation_requires_scheduler_utilization_metrics() -> None:
    benchmark = _load_benchmark_module()
    artifact = _valid_artifact(
        {
            "accepted_task_count": 1,
            "selected_task_count": 1,
            "per_layer_observed_maxima": {
                "task_leases_by_resource": {},
                "request_in_flight_by_resource": {},
            },
            "utilization_metrics": {"scheduler_resources": {}, "request_resources": {}},
        }
    )

    with pytest.raises(RuntimeError, match="did not record scheduler utilization metrics"):
        benchmark._validate_artifact(artifact)


def test_idle_regression_guardrails_pass_for_expected_summary() -> None:
    regression = _load_idle_regression_module()
    summary = _idle_regression_summary()

    checks = regression.evaluate_idle_regression_summary(summary)

    assert all(check.passed for check in checks if check.severity == "error")


def test_idle_regression_detects_validation_failure() -> None:
    regression = _load_idle_regression_module()
    summary = _idle_regression_summary()
    summary["cases"]["row-scale/rows-64"]["validation_passed"] = False

    checks = regression.evaluate_idle_regression_summary(summary)

    assert any(not check.passed and check.name == "row-scale/rows-64 validation" for check in checks)


def test_idle_regression_detects_baseline_utilization_drop() -> None:
    regression = _load_idle_regression_module()
    baseline = _idle_regression_summary(utilization=0.80)
    current = _idle_regression_summary(utilization=0.70)

    checks = regression.evaluate_idle_regression_summary(current, baseline_summary=baseline)

    assert any(
        not check.passed and check.category == "baseline" and check.name.endswith("utilization regression")
        for check in checks
    )


def test_idle_regression_detects_bad_idle_partition() -> None:
    regression = _load_idle_regression_module()
    summary = _idle_regression_summary()
    summary["cases"]["row-scale/rows-64"]["llm_wait_frontier_dependency_horizon_idle_ratio"] = 0.99

    checks = regression.evaluate_idle_regression_summary(summary)

    assert any(not check.passed and check.name == "row-scale/rows-64 idle partition" for check in checks)


def test_idle_regression_requires_adaptation_controls() -> None:
    regression = _load_idle_regression_module()
    summary = _idle_regression_summary()
    del summary["cases"]["adaptations/adaptive-row-groups-fixed-high"]

    checks = regression.evaluate_idle_regression_summary(summary)

    assert any(
        not check.passed and check.name == "required case adaptations/adaptive-row-groups-fixed-high"
        for check in checks
    )


def test_idle_regression_detects_bad_adaptive_row_group_response() -> None:
    regression = _load_idle_regression_module()
    summary = _idle_regression_summary()
    adaptive = summary["cases"]["adaptations/adaptive-row-groups-adaptive"]
    adaptive["llm_wait_utilization_ratio"] = 0.45
    adaptive["llm_wait_idle_ratio"] = 0.55
    adaptive["llm_wait_starved_idle_ratio"] = 0.10
    adaptive["llm_wait_frontier_dependency_horizon_idle_ratio"] = 0.45

    checks = regression.evaluate_idle_regression_summary(summary)

    assert any(not check.passed and check.name == "adaptive row-group utilization response" for check in checks)
    assert any(not check.passed and check.name == "adaptive row-group fixed-high isolation" for check in checks)


def test_idle_regression_detects_request_pressure_control_not_pressured() -> None:
    regression = _load_idle_regression_module()
    summary = _idle_regression_summary()
    control = summary["cases"]["adaptations/request-pressure-control"]
    control["first_model_dispatch_column"] = "z_open"
    control["request_wait_seconds_while_task_leased_mean"] = 0.0

    checks = regression.evaluate_idle_regression_summary(summary)

    assert any(not check.passed and check.name == "request-pressure control dispatch choice" for check in checks)
    assert any(not check.passed and check.name == "request-pressure control leased wait present" for check in checks)


def test_idle_regression_detects_request_pressure_advisory_regression() -> None:
    regression = _load_idle_regression_module()
    summary = _idle_regression_summary()
    summary["cases"]["adaptations/request-pressure-advisory"]["request_wait_seconds_while_task_leased_mean"] = 0.05

    checks = regression.evaluate_idle_regression_summary(summary)

    assert any(not check.passed and check.name == "request-pressure advisory leased-wait response" for check in checks)


def test_idle_regression_detects_combined_adaptive_request_pressure_regression() -> None:
    regression = _load_idle_regression_module()
    summary = _idle_regression_summary()
    combined = summary["cases"]["adaptations/adaptive-request-pressure-combined"]
    combined["row_group_admission_observed_max_target"] = 1
    combined["request_pressure_advisory_skip_count"] = 0
    combined["request_idle_ratio"] = 0.70

    checks = regression.evaluate_idle_regression_summary(summary)

    assert any(not check.passed and check.name == "combined adaptive/request target grew" for check in checks)
    assert any(
        not check.passed and check.name == "combined adaptive/request advisory skipped pressured work"
        for check in checks
    )
    assert any(not check.passed and check.name == "combined adaptive/request request-idle response" for check in checks)


def test_idle_regression_detects_request_cleanup_failure() -> None:
    regression = _load_idle_regression_module()
    summary = _idle_regression_summary()
    summary["cases"]["adaptations/request-pressure-advisory"]["final_zero_request_waiters"] = False

    checks = regression.evaluate_idle_regression_summary(summary)

    assert any(
        not check.passed and check.name == "adaptations/request-pressure-advisory final_zero_request_waiters"
        for check in checks
    )


def test_benchmark_validation_uses_scenario_task_resource_limits() -> None:
    benchmark = _load_benchmark_module()
    artifact = _valid_artifact(
        {
            "accepted_task_count": 2,
            "selected_task_count": 2,
            "capacity_plan": SimpleNamespace(
                configured=SimpleNamespace(
                    task_resource_limits=SimpleNamespace(
                        value={"submission": 2, "llm_wait": 2, "local": 2},
                    )
                )
            ),
            "final_request_snapshot": {
                "zero_active_request_leases": True,
                "zero_request_waiters": True,
                "domains": {
                    "pressured": SimpleNamespace(effective_max=1, current_limit=1),
                },
            },
            "per_layer_observed_maxima": {
                "task_leases_by_resource": {"submission": 2, "llm_wait": 2},
                "request_in_flight_by_resource": {"pressured": 1},
            },
        }
    )

    benchmark._validate_artifact(artifact)


def _scheduler_event(
    event_kind: str,
    *,
    timestamp: float,
    sequence: int,
    task_id: str,
    leased_resources: dict[str, int],
    resource_request: dict[str, int],
) -> SimpleNamespace:
    return SimpleNamespace(
        event_kind=event_kind,
        captured_at_monotonic=timestamp,
        sequence=sequence,
        task_id=task_id,
        diagnostics={"resource_request": resource_request},
        snapshot=SimpleNamespace(
            resource_limits={"submission": 2},
            leased_resources=leased_resources,
        ),
    )


def _idle_regression_summary(*, utilization: float = 0.80) -> dict[str, Any]:
    cases = {
        "row-scale/rows-64": _idle_regression_case(utilization=utilization, record_count=64, generation_count=512),
        "row-scale/rows-256": _idle_regression_case(
            utilization=utilization,
            record_count=256,
            generation_count=2048,
        ),
        "row-group-concurrency/row-groups-1": _idle_regression_case(
            utilization=0.50,
            idle=0.50,
            starved=0.20,
            row_group_concurrency=1,
            generation_count=1024,
        ),
        "row-group-concurrency/row-groups-4": _idle_regression_case(
            utilization=utilization,
            idle=1.0 - utilization,
            starved=0.10,
            row_group_concurrency=4,
            generation_count=1024,
        ),
        "buffer-size/buffer-1": _idle_regression_case(
            utilization=utilization,
            idle=1.0 - utilization,
            starved=0.10,
            buffer_size=1,
            generation_count=4096,
        ),
        "stress-shape/narrow-frontier-high-cap": _idle_regression_case(
            utilization=0.30,
            idle=0.70,
            starved=0.20,
            record_count=1024,
            generation_count=2048,
        ),
        "stress-shape/wide-frontier-high-cap": _idle_regression_case(
            utilization=utilization,
            idle=1.0 - utilization,
            starved=0.10,
            record_count=1024,
            generation_count=8192,
        ),
        "custom-model-weight/weight-model-capacity": _idle_regression_case(
            utilization=utilization,
            idle=1.0 - utilization,
            starved=0.10,
            generation_count=4096,
        ),
        "adaptations/adaptive-row-groups-fixed-low": _idle_regression_case(
            utilization=0.40,
            idle=0.60,
            starved=0.10,
            record_count=512,
            row_group_concurrency=1,
            generation_count=1024,
        ),
        "adaptations/adaptive-row-groups-adaptive": _idle_regression_case(
            utilization=0.75,
            idle=0.25,
            starved=0.08,
            record_count=512,
            row_group_concurrency=8,
            generation_count=1024,
            row_group_admission_observed_max_target=4,
        ),
        "adaptations/adaptive-row-groups-fixed-high": _idle_regression_case(
            utilization=0.78,
            idle=0.22,
            starved=0.08,
            record_count=512,
            row_group_concurrency=8,
            generation_count=1024,
        ),
        "adaptations/request-pressure-control": _idle_regression_case(
            utilization=0.70,
            idle=0.30,
            starved=0.10,
            generation_count=64,
            request_utilization=0.50,
            request_idle=0.50,
            request_starved=0.30,
            request_wait_seconds_while_task_leased_mean=0.05,
            first_model_dispatch_column="a_pressured",
        ),
        "adaptations/request-pressure-advisory": _idle_regression_case(
            utilization=0.70,
            idle=0.30,
            starved=0.10,
            generation_count=64,
            request_utilization=0.60,
            request_idle=0.40,
            request_starved=0.20,
            request_wait_seconds_while_task_leased_mean=0.0,
            request_pressure_advisory_enabled=True,
            request_pressure_advisory_skip_count=4,
            first_model_dispatch_column="z_open",
        ),
        "adaptations/adaptive-request-pressure-control": _idle_regression_case(
            utilization=0.65,
            idle=0.35,
            starved=0.12,
            record_count=512,
            row_group_concurrency=8,
            generation_count=1024,
            request_utilization=0.50,
            request_idle=0.50,
            request_starved=0.30,
            request_wait_seconds_while_task_leased_mean=0.20,
            first_model_dispatch_column="a_pressured",
            row_group_admission_observed_max_target=4,
        ),
        "adaptations/adaptive-request-pressure-combined": _idle_regression_case(
            utilization=0.70,
            idle=0.30,
            starved=0.10,
            record_count=512,
            row_group_concurrency=8,
            generation_count=1024,
            request_utilization=0.65,
            request_idle=0.35,
            request_starved=0.20,
            request_wait_seconds_while_task_leased_mean=0.05,
            request_pressure_advisory_enabled=True,
            request_pressure_advisory_skip_count=8,
            first_model_dispatch_column="z_open",
            row_group_admission_observed_max_target=4,
        ),
    }
    for index in range(3):
        cases[f"filler/filler-{index}"] = _idle_regression_case(
            utilization=utilization,
            idle=1.0 - utilization,
            starved=0.10,
        )
    return {
        "suite_id": "async-scheduling-idle-regression",
        "suite_version": "1.1",
        "mode": "quick",
        "largest_generation_count": 8192,
        "cases": cases,
    }


def _idle_regression_case(
    *,
    utilization: float,
    idle: float | None = None,
    starved: float = 0.10,
    record_count: int = 512,
    generation_count: int = 1024,
    row_group_concurrency: int = 4,
    buffer_size: int = 64,
    request_utilization: float = 0.0,
    request_idle: float = 0.0,
    request_starved: float = 0.0,
    request_wait_seconds_while_task_leased_mean: float = 0.0,
    request_pressure_advisory_enabled: bool = False,
    request_pressure_advisory_skip_count: int = 0,
    first_model_dispatch_column: str | None = None,
    row_group_admission_observed_max_target: int | None = None,
) -> dict[str, Any]:
    resolved_idle = 1.0 - utilization if idle is None else idle
    return {
        "case": {
            "name": "case",
            "sweep": "sweep",
            "record_count": record_count,
            "buffer_size": buffer_size,
            "row_group_concurrency": row_group_concurrency,
            "task_admission_capacity": 8,
            "fanout_width": 1,
            "upstream_latency_seconds": 0.003,
            "downstream_latency_seconds": 0.0003,
            "scenario": "real-pipeline-overlap",
            "request_latency_seconds": 0.0,
            "model_stage_weight": 0,
            "adaptive_row_group_admission": False,
            "request_pressure_advisory": False,
            "iterations": 1,
            "warmups": 0,
        },
        "generation_count": generation_count,
        "llm_wait_utilization_ratio": utilization,
        "llm_wait_idle_ratio": resolved_idle,
        "llm_wait_starved_idle_ratio": starved,
        "llm_wait_frontier_dependency_horizon_idle_ratio": max(0.0, resolved_idle - starved),
        "llm_wait_scheduler_queue_age_p95_seconds": 0.001,
        "request_utilization_ratio": request_utilization,
        "request_idle_ratio": request_idle,
        "request_starved_idle_ratio": request_starved,
        "request_frontier_dependency_horizon_idle_ratio": max(0.0, request_idle - request_starved),
        "request_wait_seconds_while_task_leased_mean": request_wait_seconds_while_task_leased_mean,
        "request_wait_seconds_while_task_leased_max": request_wait_seconds_while_task_leased_mean,
        "request_pressure_advisory_enabled": request_pressure_advisory_enabled,
        "request_pressure_advisory_skip_count": request_pressure_advisory_skip_count,
        "first_model_dispatch_column": first_model_dispatch_column,
        "row_group_admission_observed_max_target": row_group_admission_observed_max_target,
        "throughput_generations_per_second": 100.0,
        "validation_passed": True,
        "final_zero_task_leases": True,
        "final_zero_request_leases": True,
        "final_zero_request_waiters": True,
    }


def _request_event(
    event_kind: str, *, timestamp: float, sequence: int, in_flight: int, waiters: int
) -> SimpleNamespace:
    return SimpleNamespace(
        event_kind=event_kind,
        captured_at_monotonic=timestamp,
        sequence=sequence,
        request_attempt_id="request-a",
        pressure_snapshot=SimpleNamespace(
            resource="mock-resource",
            effective_max=2,
            in_flight_count=in_flight,
            waiters=waiters,
        ),
    )
