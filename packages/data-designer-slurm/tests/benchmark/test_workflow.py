# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Lock

import pytest

from data_designer.slurm.benchmark.analysis import (
    BenchmarkAnalyzer,
    BenchmarkMeasurements,
    BenchmarkObservationFailure,
    BenchmarkRunObservation,
    BenchmarkShardMeasurements,
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
    FixedRecordPolicy,
)
from data_designer.slurm.contracts import ArtifactReference
from data_designer.slurm.planning import ResolvedDependencyLock, ResolvedSlurmRunPlan
from data_designer.slurm.planning.compiler import SlurmRunCompiler
from data_designer.slurm.planning.resolution import resolve_slurm_config
from data_designer.slurm.services import (
    SlurmRunExecution,
    SlurmServiceError,
    SlurmServiceErrorCode,
    SlurmServiceOperation,
)

NOW = datetime(2026, 8, 19, 12, tzinfo=UTC)


@pytest.mark.parametrize(
    ("gpu_start_index", "topology_digest"),
    [
        (None, "19380747e9269b80f4d64da44d67a1e010eb61923c34365792df18aab01bc9b7"),
        (0, "59b41934fbfddc75c4e8d268533737d38c11580e6a041ca27e00ecd457e83dad"),
        (1, "205f985652cf46653c02b5958d95e3b2fb2775dcec56a014ecdde9bd86093080"),
    ],
)
def test_shared_benchmark_counts_physical_nodes_and_allocated_gpu_budget(
    tmp_path: Path,
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    shared_node_config: DataDesignerSlurmConfig,
    shared_node_plan: ResolvedSlurmRunPlan,
    dependency_lock_single: ResolvedDependencyLock,
    gpu_start_index: int | None,
    topology_digest: str,
) -> None:
    payload = shared_node_config.model_dump(mode="python")
    payload["deployments"][0]["resources"]["gpus_per_node"] = 2
    payload["deployments"][1]["resources"]["gpu_start_index"] = gpu_start_index
    base_run = DataDesignerSlurmConfig.model_validate(payload)
    benchmark_payload = benchmark_config.model_dump(mode="python")
    benchmark_payload.update(
        base_run={"inline": base_run.model_dump(mode="python")},
        model_aliases=["writer", "judge"],
        concurrency_values=[1],
        record_policy={"type": "fixed", "records": 4},
        deployment_cases=[{"name": "shared", "deployments": {"writer": {"nodes": 1, "nodes_per_replica": 1}}}],
    )
    config = DataDesignerSlurmBenchmarkConfig.model_validate(benchmark_payload)
    compiled = BenchmarkCompiler.compile(config, base_run)
    store, manifest = _manifest(tmp_path, config, compiled)
    case = compiled.cases[0]
    expected = shared_node_plan
    plan = SlurmRunCompiler.compile(
        resolve_slurm_config(
            case.child_run_config,
            selected_profile=expected.selected_profile,
            client_runtime=expected.client.runtime,
            deployment_images=tuple(deployment.image for deployment in expected.deployments),
            dependency_lock=dependency_lock_single,
            runtime_bundle=expected.runtime_bundle,
            run_id=case.child_run_id,
            package_version=expected.package_version,
        )
    )
    observer = _Observer(
        {
            case.child_run_id: BenchmarkRunObservation(
                authored_config=case.child_run_config,
                resolved_plan=plan,
                outcome=BenchmarkOutcome.SUCCEEDED,
                measurements=_measurements(4, 1, 2, 3, 1),
            )
        }
    )
    report = BenchmarkAnalyzer(observer, lambda: NOW).analyze(
        config, base_run, manifest, store.manifest_reference(manifest), refresh_state=False
    )
    assert sum(deployment.gpus_per_node for deployment in plan.deployments) == 6
    assert report.cases[0].nodes_per_job == 1
    assert report.cases[0].gpus_per_job == 8
    assert report.cases[0].gpu_hours_per_job == 32
    assert report.cases[0].topology_digest == topology_digest


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

    def execute(self, config, *, source_root, dry_run):
        assert (self.benchmark_root / "benchmark.json").is_file()
        self.calls.append((self.run_id, config, source_root, dry_run))
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


class _ConcurrentRunService(_RunService):
    def __init__(self, *args, second_check: Event) -> None:
        super().__init__(*args)
        self.second_check = second_check

    def execute(self, config, *, source_root, dry_run):
        self.second_check.wait(timeout=1)
        return super().execute(config, source_root=source_root, dry_run=dry_run)


class _ConcurrentSubmissionLoader:
    def __init__(self, submissions: set[str], second_check: Event) -> None:
        self._submissions = submissions
        self._second_check = second_check
        self._lock = Lock()
        self._checks = 0

    def __call__(self, run_id: str) -> bool:
        with self._lock:
            self._checks += 1
            if self._checks == 2:
                self._second_check.set()
        return run_id in self._submissions


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


def _measurements(
    actual_records: int,
    boot_seconds: float,
    generation_seconds: float,
    wall_seconds: float,
    shard_count: int,
) -> BenchmarkMeasurements:
    records_per_shard = actual_records // shard_count
    return BenchmarkMeasurements(
        shards=tuple(
            BenchmarkShardMeasurements(
                actual_records=(
                    actual_records - records_per_shard * (shard_count - 1)
                    if index == shard_count - 1
                    else records_per_shard
                ),
                boot_seconds=boot_seconds,
                generation_seconds=generation_seconds,
                wall_seconds=wall_seconds,
            )
            for index in range(shard_count)
        )
    )


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
    progress: list[str] = []
    benchmark_root = tmp_path / "benchmarks" / compiled.benchmark_id
    backend = SystemBenchmarkBackend(
        tmp_path,
        lambda run_id: _RunService(run_id, configs, submissions, calls, benchmark_root, failures),
        _Observer(),
        lambda: NOW,
        child_config_loader=configs.get,
        child_submission_loader=submissions.__contains__,
        progress=progress.append,
    )

    first = backend.run(config, source_root=tmp_path, force=False)
    second = backend.run(config, source_root=tmp_path, force=False)

    assert first == second
    assert len(calls) == len(compiled.cases)
    assert tuple(configs) == tuple(case.child_run_id for case in compiled.cases)
    assert progress == [
        "Benchmark manifest saved; checking 2 child runs.",
        "Benchmark case 1/2: submitted.",
        "Benchmark case 2/2: submitted.",
        "Benchmark launch: 2 submitted, 0 already submitted, 0 failed.",
        "Benchmark manifest saved; checking 2 child runs.",
        "Benchmark case 1/2: already submitted.",
        "Benchmark case 2/2: already submitted.",
        "Benchmark launch: 0 submitted, 2 already submitted, 0 failed.",
    ]


def test_progress_output_failure_cannot_change_benchmark_submissions(
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

    def broken_progress(_: str) -> None:
        raise RuntimeError("progress stream failed")

    backend = SystemBenchmarkBackend(
        tmp_path,
        lambda run_id: _RunService(run_id, configs, submissions, calls, benchmark_root, failures),
        _Observer(),
        lambda: NOW,
        child_config_loader=configs.get,
        child_submission_loader=submissions.__contains__,
        progress=broken_progress,
    )

    backend.run(config, source_root=tmp_path, force=False)

    assert len(calls) == len(compiled.cases)
    assert submissions == {case.child_run_id for case in compiled.cases}


def test_concurrent_runs_submit_each_deterministic_child_once(
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
    second_check = Event()
    submission_loader = _ConcurrentSubmissionLoader(submissions, second_check)
    store, manifest = _manifest(tmp_path, config, compiled)
    store.publish(config, authored_run, manifest)

    backend = SystemBenchmarkBackend(
        tmp_path,
        lambda run_id: _ConcurrentRunService(
            run_id,
            configs,
            submissions,
            calls,
            store.benchmark_root,
            failures,
            second_check=second_check,
        ),
        _Observer(),
        lambda: NOW,
        child_config_loader=configs.get,
        child_submission_loader=submission_loader,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(backend.run, config, source_root=tmp_path, force=False) for _ in range(2))
        assert tuple(future.result() for future in futures) == (manifest, manifest)

    assert len(calls) == len(compiled.cases)


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
    progress: list[str] = []
    benchmark_root = tmp_path / "benchmarks" / compiled.benchmark_id
    backend = SystemBenchmarkBackend(
        tmp_path,
        lambda run_id: _RunService(run_id, configs, submissions, calls, benchmark_root, failures),
        _Observer(),
        lambda: NOW,
        child_config_loader=configs.get,
        child_submission_loader=submissions.__contains__,
        progress=progress.append,
    )

    with pytest.raises(SlurmServiceError, match="1 of 2") as failure:
        backend.run(config, source_root=tmp_path, force=True)
    assert len(calls) == 2
    assert compiled.benchmark_id in str(failure.value)
    assert missing_id in str(failure.value)
    assert "submission unavailable" in str(failure.value)
    assert (
        f"Benchmark case 1/2 submission failed: {compiled.cases[0].case_id} "
        f"(run {missing_id}): submission unavailable; continuing."
    ) in progress
    assert "Benchmark launch: 1 submitted, 0 already submitted, 1 failed." in progress

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
                measurements=_measurements(1000, 60, 120, 180, len(plan.shards)),
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
    assert report.cases[0].target_jobs == 17
    assert report.cases[0].total_gpu_hours == report.cases[0].gpu_hours_per_job * 17
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
        measurements=_measurements(
            case.child_run_config.invocation.num_records,
            15000,
            120,
            15120,
            len(multi_node_plan.shards),
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


def test_analysis_uses_actual_rows_from_nonempty_partial_case(
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
    requested_records = case.child_run_config.invocation.num_records
    actual_records = requested_records - 9
    observation = BenchmarkRunObservation(
        authored_config=case.child_run_config,
        resolved_plan=multi_node_plan.model_copy(update={"run_id": case.child_run_id}),
        outcome=BenchmarkOutcome.SUCCEEDED,
        measurements=_measurements(
            actual_records,
            60,
            120,
            180,
            len(multi_node_plan.shards),
        ),
    )

    report = BenchmarkAnalyzer(_Observer({case.child_run_id: observation}), lambda: NOW).analyze(
        config,
        authored_run,
        manifest,
        store.manifest_reference(manifest),
        refresh_state=False,
    )

    result = report.cases[0]
    assert result.outcome is BenchmarkOutcome.SUCCEEDED
    assert result.requested_records == requested_records
    assert result.actual_records == actual_records
    assert result.rows_per_second == actual_records / 120
    assert result.feasible is True
    assert result.target_jobs is not None


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


def test_analysis_accepts_saved_benchmark_from_previous_compiler(
    tmp_path: Path,
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    authored_run: DataDesignerSlurmConfig,
) -> None:
    config = _inline_config(benchmark_config, authored_run)
    legacy = BenchmarkCompiler._compile(config, authored_run, legacy=True)
    assert legacy.benchmark_id == "benchmark-996c47990258ceb4a3ca7b245ddee971"
    assert legacy.cases[0].child_run_id == "run-996c47990258ceb4-0000-89d3e0eccd48"
    assert all(case.child_run_config.array_tasks.count == 2 for case in legacy.cases)
    store, manifest = _manifest(tmp_path, config, legacy)
    store.publish(config, authored_run, manifest)
    saved_config, saved_base_run, saved_manifest = store.load()
    observer = _Observer(
        {case.child_run_id: BenchmarkObservationFailure(BenchmarkOutcome.MISSING) for case in legacy.cases}
    )

    report = BenchmarkAnalyzer(observer, lambda: NOW).analyze(
        saved_config,
        saved_base_run,
        saved_manifest,
        store.manifest_reference(saved_manifest),
        refresh_state=False,
    )

    assert report.benchmark_id == legacy.benchmark_id
    assert tuple(case.outcome for case in report.cases) == (BenchmarkOutcome.MISSING,) * len(legacy.cases)
    assert tuple(case.requested_records for case in report.cases) == (1000,) * len(legacy.cases)


def test_analysis_accepts_single_task_case_below_base_array_count(
    tmp_path: Path,
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    authored_run: DataDesignerSlurmConfig,
) -> None:
    config = _inline_config(benchmark_config, authored_run).model_copy(
        update={"record_policy": FixedRecordPolicy(type="fixed", records=1)}
    )
    compiled = BenchmarkCompiler.compile(config, authored_run)
    assert authored_run.array_tasks.count == 2
    assert all(case.child_run_config.array_tasks.count == 1 for case in compiled.cases)
    store, manifest = _manifest(tmp_path, config, compiled)
    observer = _Observer(
        {case.child_run_id: BenchmarkObservationFailure(BenchmarkOutcome.MISSING) for case in compiled.cases}
    )

    report = BenchmarkAnalyzer(observer, lambda: NOW).analyze(
        config,
        authored_run,
        manifest,
        store.manifest_reference(manifest),
        refresh_state=False,
    )

    assert tuple(case.requested_records for case in report.cases) == (1,) * len(compiled.cases)


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
            measurements=_measurements(
                case.child_run_config.invocation.num_records,
                60,
                120 * index,
                180 * index,
                len(multi_node_plan.shards),
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
