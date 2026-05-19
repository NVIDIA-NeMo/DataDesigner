# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the canonical async scheduling idle-time regression suite."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from generate_async_scheduling_idle_report import (
    IDLE_SUITE_ID,
    IDLE_SUITE_VERSION,
    idle_results_summary,
    render_idle_report,
    run_idle_benchmark_suite,
    write_idle_results_summary,
)

DEFAULT_ARTIFACT_DIR = Path("artifacts/async-scheduling-idle-regression")
DEFAULT_REPORT_PATH = Path("reports/async-scheduling-idle-regression.html")
DEFAULT_SUMMARY_PATH = DEFAULT_ARTIFACT_DIR / "idle_regression_summary.json"
DEFAULT_CHECKS_PATH = DEFAULT_ARTIFACT_DIR / "idle_regression_checks.json"
CHECKS_SCHEMA_VERSION = "async-scheduling-idle-checks-v1"


@dataclass(frozen=True)
class RegressionCheck:
    name: str
    category: str
    severity: str
    passed: bool
    observed: float | int | str | bool
    expected: str
    detail: str = ""


def main() -> None:
    args = _parse_args()
    artifact_dir = Path(args.artifact_dir)
    report_path = Path(args.report_path)
    summary_path = Path(args.summary_path)
    checks_path = Path(args.checks_path)

    results = run_idle_benchmark_suite(artifact_dir, quick=args.quick, skip_run=args.skip_run)
    render_idle_report(results, report_path, artifact_dir)
    write_idle_results_summary(summary_path, results, quick=args.quick)

    summary = idle_results_summary(results, quick=args.quick)
    baseline_summary = _load_json(Path(args.baseline_summary)) if args.baseline_summary else None
    checks = evaluate_idle_regression_summary(
        summary,
        baseline_summary=baseline_summary,
        utilization_tolerance=args.utilization_tolerance,
        idle_tolerance=args.idle_tolerance,
    )
    checks_payload = _checks_payload(checks)
    checks_path.parent.mkdir(parents=True, exist_ok=True)
    checks_path.write_text(json.dumps(checks_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _print_check_summary(checks, report_path, summary_path, checks_path)
    if _has_error(checks) and not args.allow_failures:
        sys.exit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--checks-path", default=str(DEFAULT_CHECKS_PATH))
    parser.add_argument("--baseline-summary")
    parser.add_argument("--skip-run", action="store_true", help="Reuse existing benchmark JSON files.")
    parser.add_argument("--quick", action="store_true", help="Run the shorter suite while preserving coverage shape.")
    parser.add_argument(
        "--utilization-tolerance",
        type=float,
        default=0.05,
        help="Allowed absolute llm_wait utilization drop when comparing to --baseline-summary.",
    )
    parser.add_argument(
        "--idle-tolerance",
        type=float,
        default=0.05,
        help="Allowed absolute llm_wait idle increase when comparing to --baseline-summary.",
    )
    parser.add_argument("--allow-failures", action="store_true", help="Write artifacts but exit zero on failed checks.")
    return parser.parse_args()


def evaluate_idle_regression_summary(
    summary: Mapping[str, Any],
    *,
    baseline_summary: Mapping[str, Any] | None = None,
    utilization_tolerance: float = 0.05,
    idle_tolerance: float = 0.05,
) -> list[RegressionCheck]:
    cases = _cases(summary)
    mode = str(summary.get("mode", "full"))
    checks: list[RegressionCheck] = [
        _check(
            "suite id",
            category="shape",
            passed=summary.get("suite_id") == IDLE_SUITE_ID,
            observed=str(summary.get("suite_id")),
            expected=IDLE_SUITE_ID,
        ),
        _check(
            "suite version",
            category="shape",
            passed=summary.get("suite_version") == IDLE_SUITE_VERSION,
            observed=str(summary.get("suite_version")),
            expected=IDLE_SUITE_VERSION,
        ),
        _check(
            "case count",
            category="shape",
            passed=len(cases) >= _minimum_case_count(mode),
            observed=len(cases),
            expected=f">= {_minimum_case_count(mode)}",
        ),
        _check(
            "large generation case present",
            category="shape",
            passed=int(summary.get("largest_generation_count", 0)) >= 8192,
            observed=int(summary.get("largest_generation_count", 0)),
            expected=">= 8192 generations",
        ),
    ]

    checks.extend(_required_case_checks(cases, mode=mode))
    checks.extend(_per_case_metric_checks(cases))
    checks.extend(_suite_behavior_checks(cases, mode=mode))
    if baseline_summary is not None:
        checks.extend(
            _baseline_comparison_checks(
                cases,
                _cases(baseline_summary),
                utilization_tolerance=utilization_tolerance,
                idle_tolerance=idle_tolerance,
            )
        )
    return checks


def _required_case_checks(cases: Mapping[str, Mapping[str, Any]], *, mode: str) -> list[RegressionCheck]:
    required = {
        "adaptations/adaptive-row-groups-adaptive",
        "adaptations/adaptive-row-groups-fixed-high",
        "adaptations/adaptive-row-groups-fixed-low",
        "adaptations/adaptive-request-pressure-combined",
        "adaptations/adaptive-request-pressure-control",
        "adaptations/request-pressure-advisory",
        "adaptations/request-pressure-control",
        "row-scale/rows-64",
        "row-group-concurrency/row-groups-1",
        "buffer-size/buffer-1",
        "stress-shape/narrow-frontier-high-cap",
        "stress-shape/wide-frontier-high-cap",
        "custom-model-weight/weight-model-capacity",
    }
    if mode != "quick":
        required.update(
            {
                "row-scale/rows-1024",
                "row-group-concurrency/row-groups-8",
                "buffer-size/buffer-256",
                "llm-capacity/capacity-16",
            }
        )
    else:
        required.add("row-scale/rows-256")
    return [
        _check(
            f"required case {case_key}",
            category="shape",
            passed=case_key in cases,
            observed=case_key in cases,
            expected="present",
        )
        for case_key in sorted(required)
    ]


def _per_case_metric_checks(cases: Mapping[str, Mapping[str, Any]]) -> list[RegressionCheck]:
    checks: list[RegressionCheck] = []
    for case_key, case in sorted(cases.items()):
        checks.append(
            _check(
                f"{case_key} validation",
                category="correctness",
                passed=bool(case.get("validation_passed")),
                observed=bool(case.get("validation_passed")),
                expected="true",
            )
        )
        for cleanup_metric in (
            "final_zero_task_leases",
            "final_zero_request_leases",
            "final_zero_request_waiters",
        ):
            checks.append(
                _check(
                    f"{case_key} {cleanup_metric}",
                    category="correctness",
                    passed=bool(case.get(cleanup_metric)),
                    observed=bool(case.get(cleanup_metric)),
                    expected="true",
                )
            )
        for metric in (
            "llm_wait_utilization_ratio",
            "llm_wait_idle_ratio",
            "llm_wait_starved_idle_ratio",
            "llm_wait_frontier_dependency_horizon_idle_ratio",
            "request_utilization_ratio",
            "request_idle_ratio",
            "request_starved_idle_ratio",
            "request_frontier_dependency_horizon_idle_ratio",
        ):
            value = _metric(case, metric)
            checks.append(
                _check(
                    f"{case_key} {metric}",
                    category="metrics",
                    passed=math.isfinite(value) and 0.0 <= value <= 1.0,
                    observed=value,
                    expected="finite ratio in [0, 1]",
                )
            )
        queue_age_p95 = _metric(case, "llm_wait_scheduler_queue_age_p95_seconds")
        checks.append(
            _check(
                f"{case_key} llm_wait_scheduler_queue_age_p95_seconds",
                category="metrics",
                passed=math.isfinite(queue_age_p95) and queue_age_p95 >= 0.0,
                observed=queue_age_p95,
                expected="finite seconds >= 0",
            )
        )
        checks.append(
            _check(
                f"{case_key} idle dominates starved idle",
                category="metrics",
                passed=_metric(case, "llm_wait_idle_ratio") + 1e-9 >= _metric(case, "llm_wait_starved_idle_ratio"),
                observed=_metric(case, "llm_wait_idle_ratio") - _metric(case, "llm_wait_starved_idle_ratio"),
                expected="idle ratio >= starved idle ratio",
            )
        )
        checks.append(
            _check(
                f"{case_key} idle partition",
                category="metrics",
                passed=abs(
                    _metric(case, "llm_wait_idle_ratio")
                    - _metric(case, "llm_wait_starved_idle_ratio")
                    - _metric(case, "llm_wait_frontier_dependency_horizon_idle_ratio")
                )
                <= 1e-6,
                observed=(
                    _metric(case, "llm_wait_starved_idle_ratio")
                    + _metric(case, "llm_wait_frontier_dependency_horizon_idle_ratio")
                    - _metric(case, "llm_wait_idle_ratio")
                ),
                expected="starved + frontier/dependency-horizon idle == total idle",
            )
        )
    return checks


def _suite_behavior_checks(cases: Mapping[str, Mapping[str, Any]], *, mode: str) -> list[RegressionCheck]:
    checks: list[RegressionCheck] = []
    row_group_cases = [case for key, case in cases.items() if key.startswith("row-group-concurrency/")]
    if row_group_cases:
        low = min(row_group_cases, key=lambda case: int(case["case"]["row_group_concurrency"]))
        high = max(row_group_cases, key=lambda case: int(case["case"]["row_group_concurrency"]))
        low_util = _metric(low, "llm_wait_utilization_ratio")
        high_util = _metric(high, "llm_wait_utilization_ratio")
        checks.append(
            _check(
                "row-group frontier response",
                category="optimization",
                passed=high_util >= low_util + 0.20 or low_util >= 0.60,
                observed=high_util - low_util,
                expected="highest row-group concurrency improves utilization by >= 20pp unless low case is already >= 60%",
            )
        )
        checks.append(
            _check(
                "wide row-group frontier utilization",
                category="optimization",
                passed=high_util >= _wide_row_group_utilization_floor(mode),
                observed=high_util,
                expected=f">= {_wide_row_group_utilization_floor(mode):.2f} llm_wait utilization",
            )
        )

    if "buffer-size/buffer-1" in cases:
        checks.append(
            _check(
                "small-buffer model utilization",
                category="optimization",
                passed=_metric(cases["buffer-size/buffer-1"], "llm_wait_utilization_ratio") >= 0.65,
                observed=_metric(cases["buffer-size/buffer-1"], "llm_wait_utilization_ratio"),
                expected=">= 0.65 llm_wait utilization",
            )
        )
    if "llm-capacity/capacity-16" in cases:
        checks.append(
            _check(
                "high-capacity underfeeding floor",
                category="optimization",
                passed=_metric(cases["llm-capacity/capacity-16"], "llm_wait_utilization_ratio") >= 0.25,
                observed=_metric(cases["llm-capacity/capacity-16"], "llm_wait_utilization_ratio"),
                expected=">= 0.25 llm_wait utilization",
            )
        )
    if "stress-shape/wide-frontier-high-cap" in cases:
        checks.append(
            _check(
                "wide high-capacity stress utilization",
                category="optimization",
                passed=_metric(cases["stress-shape/wide-frontier-high-cap"], "llm_wait_utilization_ratio") >= 0.55,
                observed=_metric(cases["stress-shape/wide-frontier-high-cap"], "llm_wait_utilization_ratio"),
                expected=">= 0.55 llm_wait utilization",
            )
        )
    if {
        "stress-shape/narrow-frontier-high-cap",
        "stress-shape/wide-frontier-high-cap",
    }.issubset(cases):
        narrow_dependency_idle = _metric(
            cases["stress-shape/narrow-frontier-high-cap"],
            "llm_wait_frontier_dependency_horizon_idle_ratio",
        )
        wide_dependency_idle = _metric(
            cases["stress-shape/wide-frontier-high-cap"],
            "llm_wait_frontier_dependency_horizon_idle_ratio",
        )
        checks.append(
            _check(
                "wide frontier dependency-horizon idle response",
                category="optimization",
                passed=wide_dependency_idle <= narrow_dependency_idle + 0.10,
                observed=wide_dependency_idle - narrow_dependency_idle,
                expected="wide frontier dependency-horizon idle does not exceed narrow by > 10pp",
            )
        )
    if {
        "adaptations/adaptive-row-groups-fixed-low",
        "adaptations/adaptive-row-groups-fixed-high",
        "adaptations/adaptive-row-groups-adaptive",
    }.issubset(cases):
        control = cases["adaptations/adaptive-row-groups-fixed-low"]
        fixed_high = cases["adaptations/adaptive-row-groups-fixed-high"]
        adaptive = cases["adaptations/adaptive-row-groups-adaptive"]
        control_util = _metric(control, "llm_wait_utilization_ratio")
        fixed_high_util = _metric(fixed_high, "llm_wait_utilization_ratio")
        adaptive_util = _metric(adaptive, "llm_wait_utilization_ratio")
        control_frontier = _metric(control, "llm_wait_frontier_dependency_horizon_idle_ratio")
        fixed_high_frontier = _metric(fixed_high, "llm_wait_frontier_dependency_horizon_idle_ratio")
        adaptive_frontier = _metric(adaptive, "llm_wait_frontier_dependency_horizon_idle_ratio")
        checks.append(
            _check(
                "adaptive row-group utilization response",
                category="optimization",
                passed=adaptive_util >= control_util + 0.10 or control_util >= 0.70,
                observed=adaptive_util - control_util,
                expected="adaptive row groups improve low-frontier utilization by >= 10pp unless control is >= 70%",
            )
        )
        checks.append(
            _check(
                "adaptive row-group fixed-high isolation",
                category="optimization",
                passed=adaptive_util + 0.10 >= fixed_high_util,
                observed=adaptive_util - fixed_high_util,
                expected="adaptive row groups stay within 10pp utilization of fixed-high hard-cap control",
            )
        )
        checks.append(
            _check(
                "adaptive row-group frontier-idle response",
                category="optimization",
                passed=adaptive_frontier <= min(control_frontier, fixed_high_frontier) + 0.05,
                observed=adaptive_frontier - min(control_frontier, fixed_high_frontier),
                expected="adaptive row groups do not exceed the better fixed-control frontier idle by > 5pp",
            )
        )
        checks.append(
            _check(
                "adaptive row-group target grew",
                category="optimization",
                passed=int(adaptive.get("row_group_admission_observed_max_target", 0) or 0) > 1,
                observed=int(adaptive.get("row_group_admission_observed_max_target", 0) or 0),
                expected="observed adaptive target > 1",
            )
        )
    if {
        "adaptations/request-pressure-control",
        "adaptations/request-pressure-advisory",
    }.issubset(cases):
        control = cases["adaptations/request-pressure-control"]
        advisory = cases["adaptations/request-pressure-advisory"]
        control_wait = _metric(control, "request_wait_seconds_while_task_leased_mean")
        advisory_wait = _metric(advisory, "request_wait_seconds_while_task_leased_mean")
        checks.append(
            _check(
                "request-pressure control dispatch choice",
                category="optimization",
                passed=control.get("first_model_dispatch_column") == "a_pressured",
                observed=str(control.get("first_model_dispatch_column")),
                expected="a_pressured",
            )
        )
        checks.append(
            _check(
                "request-pressure control leased wait present",
                category="optimization",
                passed=control_wait >= 0.01,
                observed=control_wait,
                expected=">= 0.01 seconds",
            )
        )
        checks.append(
            _check(
                "request-pressure advisory leased-wait response",
                category="optimization",
                passed=advisory_wait <= control_wait - 0.005,
                observed=advisory_wait - control_wait,
                expected="advisory leased request wait at least 5ms lower than control",
            )
        )
        checks.append(
            _check(
                "request-pressure advisory dispatch choice",
                category="optimization",
                passed=advisory.get("first_model_dispatch_column") == "z_open",
                observed=str(advisory.get("first_model_dispatch_column")),
                expected="z_open",
            )
        )
    if {
        "adaptations/adaptive-request-pressure-control",
        "adaptations/adaptive-request-pressure-combined",
    }.issubset(cases):
        control = cases["adaptations/adaptive-request-pressure-control"]
        combined = cases["adaptations/adaptive-request-pressure-combined"]
        control_wait = _metric(control, "request_wait_seconds_while_task_leased_mean")
        combined_wait = _metric(combined, "request_wait_seconds_while_task_leased_mean")
        control_llm_util = _metric(control, "llm_wait_utilization_ratio")
        combined_llm_util = _metric(combined, "llm_wait_utilization_ratio")
        control_request_idle = _metric(control, "request_idle_ratio")
        combined_request_idle = _metric(combined, "request_idle_ratio")
        checks.extend(
            [
                _check(
                    "combined adaptive/request target grew",
                    category="optimization",
                    passed=int(combined.get("row_group_admission_observed_max_target", 0) or 0) > 1,
                    observed=int(combined.get("row_group_admission_observed_max_target", 0) or 0),
                    expected="observed adaptive target > 1",
                ),
                _check(
                    "combined adaptive/request advisory enabled",
                    category="optimization",
                    passed=bool(combined.get("request_pressure_advisory_enabled")),
                    observed=bool(combined.get("request_pressure_advisory_enabled")),
                    expected="true",
                ),
                _check(
                    "combined adaptive/request advisory skipped pressured work",
                    category="optimization",
                    passed=int(combined.get("request_pressure_advisory_skip_count", 0) or 0) > 0,
                    observed=int(combined.get("request_pressure_advisory_skip_count", 0) or 0),
                    expected="> 0 skip events",
                ),
                _check(
                    "combined adaptive/request leased wait captured",
                    category="optimization",
                    passed=math.isfinite(combined_wait) and combined_wait >= 0.0 and control_wait >= 0.0,
                    observed=combined_wait - control_wait,
                    expected="finite leased request wait delta",
                ),
                _check(
                    "combined adaptive/request llm utilization response",
                    category="optimization",
                    passed=combined_llm_util + 0.05 >= control_llm_util,
                    observed=combined_llm_util - control_llm_util,
                    expected="combined llm utilization stays within 5pp of control or improves",
                ),
                _check(
                    "combined adaptive/request request-idle response",
                    category="optimization",
                    passed=combined_request_idle <= control_request_idle + 0.05,
                    observed=combined_request_idle - control_request_idle,
                    expected="combined request idle does not exceed control by > 5pp",
                ),
            ]
        )
    return checks


def _baseline_comparison_checks(
    cases: Mapping[str, Mapping[str, Any]],
    baseline_cases: Mapping[str, Mapping[str, Any]],
    *,
    utilization_tolerance: float,
    idle_tolerance: float,
) -> list[RegressionCheck]:
    checks: list[RegressionCheck] = []
    for case_key, baseline in sorted(baseline_cases.items()):
        current = cases.get(case_key)
        if current is None:
            checks.append(
                _check(
                    f"{case_key} baseline case present",
                    category="baseline",
                    passed=False,
                    observed="missing",
                    expected="present",
                )
            )
            continue
        baseline_util = _metric(baseline, "llm_wait_utilization_ratio")
        current_util = _metric(current, "llm_wait_utilization_ratio")
        checks.append(
            _check(
                f"{case_key} utilization regression",
                category="baseline",
                passed=current_util + utilization_tolerance >= baseline_util,
                observed=current_util - baseline_util,
                expected=f">= -{utilization_tolerance:.3f}",
            )
        )
        for metric in (
            "llm_wait_idle_ratio",
            "llm_wait_starved_idle_ratio",
            "llm_wait_frontier_dependency_horizon_idle_ratio",
        ):
            baseline_idle = _metric(baseline, metric)
            current_idle = _metric(current, metric)
            checks.append(
                _check(
                    f"{case_key} {metric} regression",
                    category="baseline",
                    passed=current_idle <= baseline_idle + idle_tolerance,
                    observed=current_idle - baseline_idle,
                    expected=f"<= +{idle_tolerance:.3f}",
                )
            )
        baseline_throughput = _metric(baseline, "throughput_generations_per_second")
        current_throughput = _metric(current, "throughput_generations_per_second")
        checks.append(
            _check(
                f"{case_key} throughput smoke",
                category="baseline",
                severity="warning",
                passed=current_throughput >= baseline_throughput * 0.75,
                observed=current_throughput - baseline_throughput,
                expected=">= 75% of baseline throughput",
            )
        )
    return checks


def _wide_row_group_utilization_floor(mode: str) -> float:
    return 0.55 if mode == "quick" else 0.70


def _checks_payload(checks: Sequence[RegressionCheck]) -> dict[str, Any]:
    return {
        "checks_schema_version": CHECKS_SCHEMA_VERSION,
        "suite_id": IDLE_SUITE_ID,
        "suite_version": IDLE_SUITE_VERSION,
        "passed": not _has_error(checks),
        "error_count": sum(1 for check in checks if check.severity == "error" and not check.passed),
        "warning_count": sum(1 for check in checks if check.severity == "warning" and not check.passed),
        "checks": [asdict(check) for check in checks],
    }


def _print_check_summary(
    checks: Sequence[RegressionCheck],
    report_path: Path,
    summary_path: Path,
    checks_path: Path,
) -> None:
    errors = [check for check in checks if check.severity == "error" and not check.passed]
    warnings = [check for check in checks if check.severity == "warning" and not check.passed]
    status = "PASS" if not errors else "FAIL"
    print(f"Idle regression suite: {status} ({len(errors)} errors, {len(warnings)} warnings)")
    print(f"Wrote {report_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {checks_path}")
    for check in errors[:10]:
        print(f"ERROR {check.name}: observed {check.observed}; expected {check.expected}")
    for check in warnings[:5]:
        print(f"WARN {check.name}: observed {check.observed}; expected {check.expected}")


def _has_error(checks: Sequence[RegressionCheck]) -> bool:
    return any(check.severity == "error" and not check.passed for check in checks)


def _check(
    name: str,
    *,
    category: str,
    passed: bool,
    observed: float | int | str | bool,
    expected: str,
    severity: str = "error",
    detail: str = "",
) -> RegressionCheck:
    return RegressionCheck(
        name=name,
        category=category,
        severity=severity,
        passed=passed,
        observed=observed,
        expected=expected,
        detail=detail,
    )


def _cases(summary: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    cases = summary.get("cases", {})
    if not isinstance(cases, Mapping):
        return {}
    return cases


def _minimum_case_count(mode: str) -> int:
    return 15 if mode == "quick" else 21


def _metric(case: Mapping[str, Any], name: str) -> float:
    return float(case.get(name, 0.0) or 0.0)


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
