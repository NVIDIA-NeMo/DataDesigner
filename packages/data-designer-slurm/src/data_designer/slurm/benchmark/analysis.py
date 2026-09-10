# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Point-in-time analysis of ordinary benchmark child runs."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from data_designer.slurm.benchmark.compiler import BenchmarkCompiler
from data_designer.slurm.benchmark.records import (
    BenchmarkCaseResult,
    BenchmarkChildRun,
    BenchmarkManifest,
    BenchmarkOutcome,
    BenchmarkRecommendation,
    BenchmarkRecommendationKind,
    BenchmarkReport,
)
from data_designer.slurm.config import DataDesignerSlurmBenchmarkConfig, DataDesignerSlurmConfig
from data_designer.slurm.config.utils import convert_duration_to_seconds
from data_designer.slurm.contracts import ArtifactReference, compute_canonical_json_sha256
from data_designer.slurm.planning import ResolvedSlurmRunPlan


@dataclass(frozen=True, slots=True)
class BenchmarkMeasurements:
    """Complete timing and output facts for one successful ordinary run."""

    actual_records: int
    boot_seconds: float
    generation_seconds: float
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class BenchmarkRunObservation:
    """Verified ordinary run inputs and one status snapshot."""

    authored_config: DataDesignerSlurmConfig
    resolved_plan: ResolvedSlurmRunPlan
    outcome: BenchmarkOutcome
    measurements: BenchmarkMeasurements | None = None


class BenchmarkRunObserver(Protocol):
    """Observe one ordinary child run from durable state."""

    def observe(self, run_id: str, *, refresh_state: bool) -> BenchmarkRunObservation:
        """Return one verified child snapshot or raise an observation failure."""


class BenchmarkObservationFailure(Exception):
    """Preserve a non-success child outcome without aborting the report."""

    def __init__(self, outcome: BenchmarkOutcome) -> None:
        if outcome not in {
            BenchmarkOutcome.MISSING,
            BenchmarkOutcome.STALE,
            BenchmarkOutcome.SCHEDULER_INCONSISTENT,
        }:
            raise ValueError("invalid benchmark observation failure outcome")
        self.outcome = outcome
        super().__init__(outcome.value)


class BenchmarkManifestMismatchError(ValueError):
    """Raised when persisted benchmark children do not match compilation."""


class BenchmarkAnalyzer:
    """Analyze one immutable benchmark manifest in manifest order."""

    def __init__(self, observer: BenchmarkRunObserver, clock: Callable[[], datetime]) -> None:
        self._observer = observer
        self._clock = clock

    def analyze(
        self,
        config: DataDesignerSlurmBenchmarkConfig,
        base_run: DataDesignerSlurmConfig,
        manifest: BenchmarkManifest,
        manifest_reference: ArtifactReference,
        *,
        refresh_state: bool,
    ) -> BenchmarkReport:
        requested_records = _validate_and_get_requested_records(config, base_run, manifest)
        results = tuple(
            self._analyze_case(
                config,
                child,
                requested_records[index],
                refresh_state=refresh_state,
            )
            for index, child in enumerate(manifest.children)
        )
        recommendations = _recommend(results)
        created_at = self._clock()
        analysis_digest = compute_canonical_json_sha256(
            {
                "benchmark_id": manifest.benchmark_id,
                "created_at": created_at.isoformat(),
                "cases": [case.model_dump(mode="json") for case in results],
                "recommendations": [item.model_dump(mode="json") for item in recommendations],
            }
        )
        return BenchmarkReport(
            schema_version=1,
            benchmark_id=manifest.benchmark_id,
            analysis_id=f"analysis-{analysis_digest[:32]}",
            benchmark_manifest=manifest_reference,
            created_at=created_at,
            cases=results,
            recommendations=recommendations,
        )

    def _analyze_case(
        self,
        config: DataDesignerSlurmBenchmarkConfig,
        child: BenchmarkChildRun,
        requested_records: int,
        *,
        refresh_state: bool,
    ) -> BenchmarkCaseResult:
        fallback = _fallback_facts(child, requested_records)
        try:
            observed = self._observer.observe(child.child_run_id, refresh_state=refresh_state)
        except BenchmarkObservationFailure as error:
            return _update_result(fallback, outcome=error.outcome)
        if (
            observed.authored_config.compute_sha256() != child.child_authored_config.sha256
            or observed.resolved_plan.run_id != child.child_run_id
        ):
            return _update_result(fallback, outcome=BenchmarkOutcome.STALE)

        result = _resolved_facts(child, observed.authored_config, observed.resolved_plan)
        if observed.outcome is not BenchmarkOutcome.SUCCEEDED:
            return _update_result(result, outcome=observed.outcome)
        if observed.measurements is None:
            return _update_result(result, outcome=BenchmarkOutcome.INCOMPLETE)
        measurements = observed.measurements
        actual_records = measurements.actual_records
        boot_seconds = measurements.boot_seconds
        generation_seconds = measurements.generation_seconds
        wall_seconds = measurements.wall_seconds
        requested_records = observed.authored_config.invocation.num_records
        if actual_records != requested_records or generation_seconds <= 0 or wall_seconds <= 0:
            return _update_result(
                result,
                outcome=BenchmarkOutcome.INCOMPLETE,
                actual_records=actual_records,
            )
        gpus_per_job = result.gpus_per_job
        if gpus_per_job is None:
            raise ValueError("resolved benchmark case has no GPU count")
        rows_per_second = actual_records / generation_seconds
        budget_seconds = convert_duration_to_seconds(config.analysis.target_runtime)
        effective_generation_seconds = max(0.0, budget_seconds - boot_seconds)
        feasible = boot_seconds < budget_seconds
        gpu_hours_per_job = gpus_per_job * budget_seconds / 3600
        if not feasible:
            return _update_result(
                result,
                outcome=BenchmarkOutcome.SUCCEEDED,
                actual_records=actual_records,
                boot_seconds=boot_seconds,
                generation_seconds=effective_generation_seconds,
                wall_seconds=wall_seconds,
                rows_per_second=rows_per_second,
                gpu_hours_per_job=gpu_hours_per_job,
                feasible=False,
            )
        target_jobs = math.ceil(config.analysis.target_total_records / (rows_per_second * effective_generation_seconds))
        return _update_result(
            result,
            outcome=BenchmarkOutcome.SUCCEEDED,
            actual_records=actual_records,
            boot_seconds=boot_seconds,
            generation_seconds=effective_generation_seconds,
            wall_seconds=wall_seconds,
            rows_per_second=rows_per_second,
            gpu_hours_per_job=gpu_hours_per_job,
            total_gpu_hours=gpu_hours_per_job * target_jobs,
            target_jobs=target_jobs,
            feasible=True,
        )


def _fallback_facts(
    child: BenchmarkChildRun,
    requested_records: int,
) -> BenchmarkCaseResult:
    return BenchmarkCaseResult(
        case_id=child.case_id,
        child_run_id=child.child_run_id,
        outcome=BenchmarkOutcome.MISSING,
        requested_records=requested_records,
    )


def _resolved_facts(
    child: BenchmarkChildRun,
    authored: DataDesignerSlurmConfig,
    plan: ResolvedSlurmRunPlan,
) -> BenchmarkCaseResult:
    nodes = sum(len(deployment.node_indices) for deployment in plan.deployments)
    gpus = sum(len(deployment.node_indices) * deployment.gpus_per_node for deployment in plan.deployments)
    topology_digest = compute_canonical_json_sha256(
        [
            {
                "deployment_id": deployment.deployment_id,
                "model_alias": deployment.authored.model_alias,
                "node_indices": deployment.node_indices,
                "gpus_per_node": deployment.gpus_per_node,
                "topology": deployment.topology.model_dump(mode="json"),
            }
            for deployment in plan.deployments
        ]
    )
    return BenchmarkCaseResult(
        case_id=child.case_id,
        child_run_id=child.child_run_id,
        outcome=BenchmarkOutcome.INCOMPLETE,
        topology_digest=topology_digest,
        requested_records=authored.invocation.num_records,
        gpus_per_job=gpus,
        nodes_per_job=nodes,
    )


def _validate_and_get_requested_records(
    config: DataDesignerSlurmBenchmarkConfig,
    base_run: DataDesignerSlurmConfig,
    manifest: BenchmarkManifest,
) -> tuple[int, ...]:
    try:
        compiled = BenchmarkCompiler.compile(config, base_run)
    except ValueError as error:
        raise BenchmarkManifestMismatchError("benchmark configuration cannot reproduce its children") from error
    expected = tuple(
        (case.case_id, case.child_run_id, case.child_run_config.compute_sha256()) for case in compiled.cases
    )
    actual = tuple(
        (child.case_id, child.child_run_id, child.child_authored_config.sha256) for child in manifest.children
    )
    if manifest.benchmark_id != compiled.benchmark_id or actual != expected:
        raise BenchmarkManifestMismatchError("benchmark manifest does not match deterministic child expansion")
    return tuple(case.child_run_config.invocation.num_records for case in compiled.cases)


def _update_result(result: BenchmarkCaseResult, **updates: object) -> BenchmarkCaseResult:
    return BenchmarkCaseResult.model_validate(result.model_dump(mode="python") | updates)


def _recommend(cases: tuple[BenchmarkCaseResult, ...]) -> tuple[BenchmarkRecommendation, ...]:
    eligible = tuple(
        case
        for case in cases
        if case.outcome is BenchmarkOutcome.SUCCEEDED
        and case.feasible is True
        and case.target_jobs is not None
        and case.total_gpu_hours is not None
    )
    if not eligible:
        return ()
    pareto = tuple(
        case
        for case in eligible
        if not any(
            other.case_id != case.case_id
            and other.target_jobs <= case.target_jobs
            and other.total_gpu_hours <= case.total_gpu_hours
            and (other.target_jobs < case.target_jobs or other.total_gpu_hours < case.total_gpu_hours)
            for other in eligible
        )
    )
    minimum_jobs = min(eligible, key=lambda case: (case.target_jobs, case.total_gpu_hours, case.case_id))
    minimum_gpu_hours = min(eligible, key=lambda case: (case.total_gpu_hours, case.target_jobs, case.case_id))
    return tuple(
        BenchmarkRecommendation(kind=BenchmarkRecommendationKind.PARETO, case_id=case.case_id) for case in pareto
    ) + (
        BenchmarkRecommendation(kind=BenchmarkRecommendationKind.MINIMUM_JOBS, case_id=minimum_jobs.case_id),
        BenchmarkRecommendation(
            kind=BenchmarkRecommendationKind.MINIMUM_GPU_HOURS,
            case_id=minimum_gpu_hours.case_id,
        ),
    )


__all__ = [
    "BenchmarkAnalyzer",
    "BenchmarkMeasurements",
    "BenchmarkManifestMismatchError",
    "BenchmarkObservationFailure",
    "BenchmarkRunObservation",
    "BenchmarkRunObserver",
]
