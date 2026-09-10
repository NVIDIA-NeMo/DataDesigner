# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from data_designer.slurm.benchmark.analysis import (
    BenchmarkAnalyzer,
    BenchmarkMeasurements,
    BenchmarkObservationFailure,
    BenchmarkRunObservation,
)
from data_designer.slurm.benchmark.compiler import BenchmarkCompiler
from data_designer.slurm.benchmark.execution import SystemBenchmarkBackend
from data_designer.slurm.benchmark.records import (
    BenchmarkChildRun,
    BenchmarkOutcome,
    BenchmarkRecommendationKind,
)
from data_designer.slurm.benchmark.store import BenchmarkConflictError, BenchmarkStore
from data_designer.slurm.config import (
    BenchmarkBaseRun,
    DataDesignerSlurmBenchmarkConfig,
    DataDesignerSlurmConfig,
)
from data_designer.slurm.contracts import ArtifactReference
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.services import (
    SlurmRunExecution,
    SlurmServiceError,
    SlurmServiceErrorCode,
    SlurmServiceOperation,
)

NOW = datetime(2026, 8, 19, 12, tzinfo=timezone.utc)


class _Observer:
    def __init__(self, responses=None) -> None:
        self.responses = {} if responses is None else responses
        self.calls = []

    def observe(self, run_id: str, *, refresh_state: bool) -> BenchmarkRunObservation:
        self.calls.append((run_id, refresh_state))
        response = self.responses[run_id]
        if isinstance(response, Exception):
            raise response
        return response


class _RunService:
    def __init__(self, run_id, configs, submissions, calls, benchmark_root, failures) -> None:
        self.run_id = run_id
        self.configs = configs
        self.submissions = submissions
        self.calls = calls
        self.benchmark_root = benchmark_root
        self.failures = failures

    def execute(self, config, *, source_root, dry_run, force):
        assert (self.benchmark_root / "benchmark.json").is_file()
        self.calls.append((self.run_id, config, source_root, dry_run, force))
        self.configs[self.run_id] = config
        if self.run_id in self.failures:
            raise SlurmServiceError(
                SlurmServiceErrorCode.UNAVAILABLE,
                SlurmServiceOperation.EXECUTE_RUN,
                "submission unavailable",
            )
        self.submissions.add(self.run_id)
        return SlurmRunExecution(
            run_id=self.run_id,
            state="submitted",
            plan_sha256="1" * 64,
            shard_count=1,
            job_id=4101,
        )


def _inline_config(
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    authored_run: DataDesignerSlurmConfig,
) -> DataDesignerSlurmBenchmarkConfig:
    return benchmark_config.model_copy(
        update={
            "base_run": BenchmarkBaseRun(inline=authored_run),
            "concurrency_values": [32],
        }
    )


def _manifest(workspace: Path, config, compiled):
    store = BenchmarkStore(workspace, compiled.benchmark_id)
    children = tuple(
        BenchmarkChildRun(
            case_id=case.case_id,
            child_run_id=case.child_run_id,
            child_authored_config=ArtifactReference(
                path=(workspace / "runs" / case.child_run_id / "authored-config.json").as_posix(),
                sha256=case.child_run_config.compute_sha256(),
            ),
        )
        for case in compiled.cases
    )
    return store, store.build_manifest(config, children)


def test_store_is_convergent_and_rejects_tampered_metadata(
    tmp_path: Path,
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    authored_run: DataDesignerSlurmConfig,
) -> None:
    config = _inline_config(benchmark_config, authored_run)
    compiled = BenchmarkCompiler.compile(config, authored_run)
    store, manifest = _manifest(tmp_path, config, compiled)

    assert store.publish(config, authored_run, manifest) == manifest
    assert store.publish(config, authored_run, manifest) == manifest
    assert store.load() == (config, authored_run, manifest)

    store.config_path.write_text("{}\n")
    with pytest.raises(BenchmarkConflictError):
        store.load()


def test_run_persists_manifest_before_submission_and_is_idempotent(
    tmp_path: Path,
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    authored_run: DataDesignerSlurmConfig,
) -> None:
    config = _inline_config(benchmark_config, authored_run)
    compiled = BenchmarkCompiler.compile(config, authored_run)
    configs = {}
    submissions = set()
    calls = []
    failures = set()
    benchmark_root = tmp_path / "benchmarks" / compiled.benchmark_id
    backend = SystemBenchmarkBackend(
        tmp_path,
        lambda run_id: _RunService(run_id, configs, submissions, calls, benchmark_root, failures),
        _Observer(),
        lambda: NOW,
        child_config_loader=configs.get,
        child_submission_loader=submissions.__contains__,
    )

    first = backend.run(config, source_root=tmp_path, force=False)
    second = backend.run(config, source_root=tmp_path, force=False)

    assert first == second
    assert len(calls) == len(compiled.cases)
    assert tuple(configs) == tuple(case.child_run_id for case in compiled.cases)
    assert all(force is False for *_, force in calls)


def test_initialized_child_without_submission_requires_force_to_resume(
    tmp_path: Path,
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    authored_run: DataDesignerSlurmConfig,
) -> None:
    config = _inline_config(benchmark_config, authored_run)
    compiled = BenchmarkCompiler.compile(config, authored_run)
    missing_id = compiled.cases[0].child_run_id
    configs = {}
    submissions = set()
    calls = []
    failures = {missing_id}
    benchmark_root = tmp_path / "benchmarks" / compiled.benchmark_id
    backend = SystemBenchmarkBackend(
        tmp_path,
        lambda run_id: _RunService(run_id, configs, submissions, calls, benchmark_root, failures),
        _Observer(),
        lambda: NOW,
        child_config_loader=configs.get,
        child_submission_loader=submissions.__contains__,
    )

    with pytest.raises(SlurmServiceError, match="1 of 2"):
        backend.run(config, source_root=tmp_path, force=False)
    failures.clear()
    with pytest.raises(SlurmServiceError, match="1 of 2") as conflict:
        backend.run(config, source_root=tmp_path, force=False)
    assert conflict.value.code is SlurmServiceErrorCode.CONFLICT

    backend.run(config, source_root=tmp_path, force=True)

    assert [run_id for run_id, *_ in calls].count(missing_id) == 2


def test_partial_submission_attempts_all_children_and_resumes_only_missing(
    tmp_path: Path,
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    authored_run: DataDesignerSlurmConfig,
) -> None:
    config = _inline_config(benchmark_config, authored_run)
    compiled = BenchmarkCompiler.compile(config, authored_run)
    missing_id = compiled.cases[0].child_run_id
    configs = {}
    submissions = set()
    calls = []
    failures = {missing_id}
    benchmark_root = tmp_path / "benchmarks" / compiled.benchmark_id
    backend = SystemBenchmarkBackend(
        tmp_path,
        lambda run_id: _RunService(run_id, configs, submissions, calls, benchmark_root, failures),
        _Observer(),
        lambda: NOW,
        child_config_loader=configs.get,
        child_submission_loader=submissions.__contains__,
    )

    with pytest.raises(SlurmServiceError, match="1 of 2"):
        backend.run(config, source_root=tmp_path, force=True)
    assert len(calls) == 2

    failures.clear()
    backend.run(config, source_root=tmp_path, force=True)

    assert [run_id for run_id, *_ in calls].count(missing_id) == 2
    assert len(calls) == 3
    assert all(force is False for *_, force in calls)


def test_analysis_preserves_mixed_outcomes_and_stable_order(
    tmp_path: Path,
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    config = _inline_config(benchmark_config, authored_run)
    compiled = BenchmarkCompiler.compile(config, authored_run)
    store, manifest = _manifest(tmp_path, config, compiled)
    successful = compiled.cases[0]
    plan = multi_node_plan.model_copy(update={"run_id": successful.child_run_id})
    observer = _Observer(
        {
            successful.child_run_id: BenchmarkRunObservation(
                authored_config=successful.child_run_config,
                resolved_plan=plan,
                outcome=BenchmarkOutcome.SUCCEEDED,
                measurements=BenchmarkMeasurements(
                    actual_records=1000,
                    boot_seconds=60,
                    generation_seconds=120,
                    wall_seconds=180,
                ),
            ),
            compiled.cases[1].child_run_id: BenchmarkObservationFailure(BenchmarkOutcome.MISSING),
        }
    )

    report = BenchmarkAnalyzer(observer, lambda: NOW).analyze(
        config,
        authored_run,
        manifest,
        store.manifest_reference(manifest),
        refresh_state=True,
    )
    repeated = BenchmarkAnalyzer(observer, lambda: NOW).analyze(
        config,
        authored_run,
        manifest,
        store.manifest_reference(manifest),
        refresh_state=True,
    )

    assert tuple(case.outcome for case in report.cases) == (
        BenchmarkOutcome.SUCCEEDED,
        BenchmarkOutcome.MISSING,
    )
    assert report.cases[0].rows_per_second == 1000 / 120
    assert report.cases[0].generation_seconds == 14340
    assert report.cases[0].target_jobs == 9
    assert report.serialize_json() == repeated.serialize_json()
    assert observer.calls == [(case.child_run_id, True) for case in compiled.cases] * 2


def test_analysis_keeps_measured_infeasible_case_successful(
    tmp_path: Path,
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    config = _inline_config(benchmark_config, authored_run).model_copy(
        update={"deployment_cases": [benchmark_config.deployment_cases[0]]}
    )
    compiled = BenchmarkCompiler.compile(config, authored_run)
    store, manifest = _manifest(tmp_path, config, compiled)
    case = compiled.cases[0]
    observation = BenchmarkRunObservation(
        authored_config=case.child_run_config,
        resolved_plan=multi_node_plan.model_copy(update={"run_id": case.child_run_id}),
        outcome=BenchmarkOutcome.SUCCEEDED,
        measurements=BenchmarkMeasurements(
            actual_records=case.child_run_config.invocation.num_records,
            boot_seconds=15000,
            generation_seconds=120,
            wall_seconds=15120,
        ),
    )

    report = BenchmarkAnalyzer(_Observer({case.child_run_id: observation}), lambda: NOW).analyze(
        config,
        authored_run,
        manifest,
        store.manifest_reference(manifest),
        refresh_state=False,
    )

    assert report.cases[0].outcome is BenchmarkOutcome.SUCCEEDED
    assert report.cases[0].feasible is False
    assert report.cases[0].generation_seconds == 0
    assert report.cases[0].target_jobs is None
    assert report.cases[0].total_gpu_hours is None


def test_analysis_rejects_manifest_rewritten_away_from_compiled_children(
    tmp_path: Path,
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    authored_run: DataDesignerSlurmConfig,
) -> None:
    config = _inline_config(benchmark_config, authored_run)
    compiled = BenchmarkCompiler.compile(config, authored_run)
    store, manifest = _manifest(tmp_path, config, compiled)
    first = manifest.children[0]
    rewritten_id = f"{first.child_run_id}-other"
    rewritten = BenchmarkChildRun(
        case_id=first.case_id,
        child_run_id=rewritten_id,
        child_authored_config=ArtifactReference(
            path=(tmp_path / "runs" / rewritten_id / "authored-config.json").as_posix(),
            sha256="f" * 64,
        ),
    )
    tampered = manifest.model_copy(update={"children": (rewritten, *manifest.children[1:])})

    with pytest.raises(ValueError, match="deterministic child expansion"):
        BenchmarkAnalyzer(_Observer(), lambda: NOW).analyze(
            config,
            authored_run,
            tampered,
            store.manifest_reference(tampered),
            refresh_state=False,
        )


def test_analysis_recommendations_are_stable_and_select_dominating_case(
    tmp_path: Path,
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    config = _inline_config(benchmark_config, authored_run)
    compiled = BenchmarkCompiler.compile(config, authored_run)
    store, manifest = _manifest(tmp_path, config, compiled)
    responses = {}
    for index, case in enumerate(compiled.cases, start=1):
        responses[case.child_run_id] = BenchmarkRunObservation(
            authored_config=case.child_run_config,
            resolved_plan=multi_node_plan.model_copy(update={"run_id": case.child_run_id}),
            outcome=BenchmarkOutcome.SUCCEEDED,
            measurements=BenchmarkMeasurements(
                actual_records=case.child_run_config.invocation.num_records,
                boot_seconds=60,
                generation_seconds=120 * index,
                wall_seconds=180 * index,
            ),
        )

    report = BenchmarkAnalyzer(_Observer(responses), lambda: NOW).analyze(
        config,
        authored_run,
        manifest,
        store.manifest_reference(manifest),
        refresh_state=False,
    )

    assert tuple(item.kind for item in report.recommendations) == (
        BenchmarkRecommendationKind.PARETO,
        BenchmarkRecommendationKind.MINIMUM_JOBS,
        BenchmarkRecommendationKind.MINIMUM_GPU_HOURS,
    )
    assert {item.case_id for item in report.recommendations} == {compiled.cases[0].case_id}


@pytest.mark.parametrize(
    "outcome",
    [
        BenchmarkOutcome.PENDING,
        BenchmarkOutcome.ACCOUNTING_LAG,
        BenchmarkOutcome.FAILED,
        BenchmarkOutcome.INCOMPLETE,
        BenchmarkOutcome.MISSING,
        BenchmarkOutcome.STALE,
        BenchmarkOutcome.SCHEDULER_INCONSISTENT,
    ],
)
def test_analysis_keeps_each_non_success_outcome_explicit(
    tmp_path: Path,
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
    outcome: BenchmarkOutcome,
) -> None:
    config = _inline_config(benchmark_config, authored_run).model_copy(
        update={"deployment_cases": [benchmark_config.deployment_cases[0]]}
    )
    compiled = BenchmarkCompiler.compile(config, authored_run)
    store, manifest = _manifest(tmp_path, config, compiled)
    case = compiled.cases[0]
    response = (
        BenchmarkObservationFailure(outcome)
        if outcome
        in {
            BenchmarkOutcome.MISSING,
            BenchmarkOutcome.STALE,
            BenchmarkOutcome.SCHEDULER_INCONSISTENT,
        }
        else BenchmarkRunObservation(
            authored_config=case.child_run_config,
            resolved_plan=multi_node_plan.model_copy(update={"run_id": case.child_run_id}),
            outcome=outcome,
        )
    )

    report = BenchmarkAnalyzer(_Observer({case.child_run_id: response}), lambda: NOW).analyze(
        config,
        authored_run,
        manifest,
        store.manifest_reference(manifest),
        refresh_state=False,
    )

    assert report.cases[0].outcome is outcome


def test_fail_if_incomplete_persists_the_point_in_time_report_first(
    tmp_path: Path,
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    authored_run: DataDesignerSlurmConfig,
) -> None:
    config = _inline_config(benchmark_config, authored_run)
    compiled = BenchmarkCompiler.compile(config, authored_run)
    store, manifest = _manifest(tmp_path, config, compiled)
    store.publish(config, authored_run, manifest)
    observer = _Observer(
        {case.child_run_id: BenchmarkObservationFailure(BenchmarkOutcome.MISSING) for case in compiled.cases}
    )
    backend = SystemBenchmarkBackend(
        tmp_path,
        lambda _: pytest.fail("analysis must not create a run service"),
        observer,
        lambda: NOW,
    )

    with pytest.raises(SlurmServiceError, match="2 incomplete cases"):
        backend.analyze(
            manifest.benchmark_id,
            refresh_state=False,
            fail_if_incomplete=True,
        )

    assert len(tuple(store.reports_root.iterdir())) == 1
