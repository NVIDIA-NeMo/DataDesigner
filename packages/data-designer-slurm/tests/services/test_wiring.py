# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from data_designer.slurm.benchmark.compiler import BenchmarkCompiler
from data_designer.slurm.client.dependencies import ResolvedClientDependencies
from data_designer.slurm.config import (
    BenchmarkBaseRun,
    BuilderInput,
    DataDesignerSlurmBenchmarkConfig,
    DataDesignerSlurmConfig,
    ImageBuildRequest,
    SecretRef,
    SlurmProfile,
    SlurmProfileCatalog,
    select_profile,
)
from data_designer.slurm.contracts import ArtifactReference, canonical_json
from data_designer.slurm.images.records import RegisteredImage
from data_designer.slurm.images.registry import ImageRegistryStore
from data_designer.slurm.launcher.errors import SlurmLauncherError
from data_designer.slurm.launcher.models import (
    SlurmAccountingEntry,
    SlurmJobSubmissionReceipt,
    SlurmProcessExitCode,
    SlurmQueueEntry,
)
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.services import (
    SlurmServiceError,
    SlurmServiceErrorCode,
    create_slurm_benchmark_service,
    create_slurm_image_service,
    create_slurm_run_service,
)
from data_designer.slurm.state import (
    AttemptLifecycleState,
    AttemptManifest,
    AttemptTerminalClassification,
    RunManifest,
    SchedulerIdentity,
    SchedulerState,
    ShardManifest,
    SlurmStateWriter,
    StateConflictError,
    StateNotFoundError,
)


class _Launcher:
    def __init__(
        self,
        gpu_counts: tuple[int, ...] = (),
        *,
        cancel_error: Exception | None = None,
        job_id: int = 42,
        release_error: Exception | None = None,
        submission_error: Exception | None = None,
    ) -> None:
        self.submissions: list[str] = []
        self.cancellations: list[int] = []
        self.releases: list[int] = []
        self.held_submissions: list[bool] = []
        self.exported_environments: list[dict[str, str]] = []
        self.queue_entries: tuple[SlurmQueueEntry, ...] = ()
        self.accounting_entries: tuple[SlurmAccountingEntry, ...] = ()
        self.gpu_counts = gpu_counts
        self.job_id = job_id
        self.cancel_error = cancel_error
        self.release_error = release_error
        self.submission_error = submission_error

    def submit_script(
        self,
        script: str,
        *,
        hold: bool = False,
        export_environment: Mapping[str, str] | None = None,
    ) -> SlurmJobSubmissionReceipt:
        self.submissions.append(script)
        self.held_submissions.append(hold)
        self.exported_environments.append(dict(export_environment or {}))
        if self.submission_error is not None:
            raise self.submission_error
        return SlurmJobSubmissionReceipt(job_id=self.job_id)

    def cancel(self, job_id: int) -> None:
        self.cancellations.append(job_id)
        if self.cancel_error is not None:
            raise self.cancel_error

    def release(self, job_id: int) -> None:
        self.releases.append(job_id)
        if self.release_error is not None:
            raise self.release_error

    def query_gpu_counts(self, *, partition: str | None = None) -> tuple[int, ...]:
        assert partition is not None
        return self.gpu_counts

    def query_queue(self, selectors: object) -> tuple[SlurmQueueEntry, ...]:
        del selectors
        return self.queue_entries

    def query_accounting(self, selectors: object) -> tuple[SlurmAccountingEntry, ...]:
        del selectors
        return self.accounting_entries


class _Publisher:
    def __init__(
        self,
        *,
        initialization_error: BaseException | None = None,
        submission_error: BaseException | None = None,
    ) -> None:
        self.initializations: list[tuple[str, bool]] = []
        self.plans: list[ResolvedSlurmRunPlan] = []
        self.submissions: list[tuple[str, int, datetime]] = []
        self.submission_failures: list[tuple[str, int, datetime]] = []
        self.initialization_error = initialization_error
        self.submission_error = submission_error

    def initialize(
        self,
        authored: DataDesignerSlurmConfig,
        plan: ResolvedSlurmRunPlan,
        dependencies: ResolvedClientDependencies,
        builder_payload: dict[str, object] | None,
        *,
        force: bool,
    ) -> None:
        if self.initialization_error is not None:
            raise self.initialization_error
        assert plan.authored_config.sha256 == authored.compute_sha256()
        assert builder_payload is None
        assert all(path.is_file() for path in dependencies.wheel_sources)
        self.plans.append(plan)
        self.initializations.append((plan.run_id, force))

    def record_submission(self, plan: ResolvedSlurmRunPlan, job_id: int, *, submitted_at: datetime) -> None:
        if self.submission_error is not None:
            raise self.submission_error
        self.submissions.append((plan.run_id, job_id, submitted_at))

    def record_submission_failure(
        self,
        plan: ResolvedSlurmRunPlan,
        job_id: int,
        *,
        failed_at: datetime,
    ) -> None:
        self.submission_failures.append((plan.run_id, job_id, failed_at))


def _profile(tmp_path: Path, profile_catalog: SlurmProfileCatalog) -> SlurmProfile:
    return profile_catalog.clusters["primary"].model_copy(update={"workspace_root": tmp_path.as_posix()})


def _register_images(
    tmp_path: Path,
    authored: DataDesignerSlurmConfig,
    plan: ResolvedSlurmRunPlan,
) -> None:
    store = ImageRegistryStore(tmp_path)
    pairs = ((authored.client.image, plan.client.image),) + tuple(
        (deployment.server.image, resolved.image)
        for deployment, resolved in zip(authored.deployments, plan.deployments, strict=True)
    )
    for index, (reference, resolved) in enumerate(pairs):
        path = tmp_path / f"image-{index}.sqsh"
        path.write_bytes(f"image-{index}".encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        inspection = resolved.inspection.model_copy(update={"sqsh_sha256": digest})
        assert reference.name is not None
        image = RegisteredImage(
            schema_version=1,
            name=reference.name,
            path=path.as_posix(),
            sqsh_sha256=digest,
            inspection=inspection,
        )
        store.register(image, verify_before_publish=lambda _: None)


def test_production_wiring_dry_run_resolves_and_renders_without_submission(
    tmp_path: Path,
    profile_catalog: SlurmProfileCatalog,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    _register_images(tmp_path, authored_run_single, single_node_plan)
    launcher = _Launcher()
    service = create_slurm_run_service(
        profile=_profile(tmp_path, profile_catalog),
        launcher=launcher,  # type: ignore[arg-type]
        run_id_factory=lambda: "run-wired",
        package_version="0.9.2",
    )

    result = service.execute(authored_run_single, source_root=tmp_path, dry_run=True)

    assert result.run_id == "run-wired"
    assert result.state == "dry_run"
    assert result.batch_script is not None
    assert "#SBATCH --array=0" in result.batch_script
    assert launcher.submissions == []


def test_production_benchmark_wiring_submits_each_case_as_an_ordinary_run(
    tmp_path: Path,
    profile_catalog: SlurmProfileCatalog,
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    config = benchmark_config.model_copy(
        update={
            "base_run": BenchmarkBaseRun(inline=authored_run_single),
            "concurrency_values": [32],
            "deployment_cases": [benchmark_config.deployment_cases[0]],
        }
    )
    _register_images(tmp_path, authored_run_single, single_node_plan)
    launcher = _Launcher()
    publisher = _Publisher()
    service = create_slurm_benchmark_service(
        profile=_profile(tmp_path, profile_catalog),
        launcher=launcher,  # type: ignore[arg-type]
        artifact_publisher=publisher,
        package_version="0.9.2",
    )

    manifest = service.run(config, source_root=tmp_path, force=True)

    assert len(manifest.children) == 1
    assert publisher.initializations == [(manifest.children[0].child_run_id, False)]
    assert len(launcher.submissions) == 1
    assert (tmp_path / "benchmarks" / manifest.benchmark_id / "benchmark.json").is_file()


def test_production_benchmark_children_preserve_catalog_profile_provenance(
    tmp_path: Path,
    profile_catalog: SlurmProfileCatalog,
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    config = benchmark_config.model_copy(
        update={
            "base_run": BenchmarkBaseRun(inline=authored_run_single),
            "concurrency_values": [32],
            "deployment_cases": [benchmark_config.deployment_cases[0]],
        }
    )
    catalog = profile_catalog.model_copy(
        update={
            "clusters": {
                name: profile.model_copy(update={"workspace_root": tmp_path.as_posix()})
                for name, profile in profile_catalog.clusters.items()
            }
        }
    )
    _register_images(tmp_path, authored_run_single, single_node_plan)
    publisher = _Publisher()
    service = create_slurm_benchmark_service(
        catalog=catalog,
        cluster="primary",
        launcher=_Launcher(),  # type: ignore[arg-type]
        artifact_publisher=publisher,
        package_version="0.9.2",
    )

    service.run(config, source_root=tmp_path)

    assert publisher.plans[0].selected_profile == select_profile(catalog, cluster="primary")


@pytest.mark.parametrize("drop_run_manifest", [False, True])
def test_production_benchmark_force_resumes_initialized_child_without_submission(
    tmp_path: Path,
    profile_catalog: SlurmProfileCatalog,
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    drop_run_manifest: bool,
) -> None:
    config = benchmark_config.model_copy(
        update={
            "base_run": BenchmarkBaseRun(inline=authored_run_single),
            "concurrency_values": [32],
            "deployment_cases": [benchmark_config.deployment_cases[0]],
        }
    )
    child_run_id = BenchmarkCompiler.compile(config, authored_run_single).cases[0].child_run_id
    _register_images(tmp_path, authored_run_single, single_node_plan)
    launcher = _Launcher(submission_error=SlurmLauncherError("unavailable"))
    service = create_slurm_benchmark_service(
        profile=_profile(tmp_path, profile_catalog),
        launcher=launcher,  # type: ignore[arg-type]
        package_version="0.9.2",
    )

    with pytest.raises(SlurmServiceError, match="1 of 1") as unavailable:
        service.run(config, source_root=tmp_path)
    assert unavailable.value.code is SlurmServiceErrorCode.UNAVAILABLE
    run_root = tmp_path / "runs" / child_run_id
    if drop_run_manifest:
        (run_root / "run.json").unlink()
    launcher.submission_error = None

    manifest = service.run(config, source_root=tmp_path, force=True)

    writer = SlurmStateWriter(tmp_path, manifest.children[0].child_run_id)
    shard = writer.load_shards()[0]
    assert len(launcher.submissions) == 2
    assert writer.load_attempts(shard.shard_id)[0].scheduler is not None


def test_public_plan_resolves_builder_relative_to_explicit_source_root(
    tmp_path: Path,
    profile_catalog: SlurmProfileCatalog,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    payload = authored_run_single.builder.inline
    assert payload is not None
    (tmp_path / "builder.json").write_bytes(canonical_json(payload))
    authored = authored_run_single.model_copy(update={"builder": BuilderInput(source="builder.json")})
    _register_images(tmp_path, authored, single_node_plan)
    service = create_slurm_run_service(
        profile=_profile(tmp_path, profile_catalog),
        launcher=_Launcher(),  # type: ignore[arg-type]
        run_id_factory=lambda: "run-wired",
        package_version="0.9.2",
    )

    plan = service.plan(authored, source_root=tmp_path)

    assert plan.builder.authored_source == "builder.json"
    assert plan.builder.source is not None
    assert plan.builder.source.path == (tmp_path / "runs/run-wired/builder-config.json").as_posix()


def test_production_wiring_submits_after_publisher_initialization(
    tmp_path: Path,
    profile_catalog: SlurmProfileCatalog,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    _register_images(tmp_path, authored_run_single, single_node_plan)
    launcher = _Launcher()
    publisher = _Publisher()
    submitted_at = datetime(2026, 9, 8, tzinfo=timezone.utc)
    service = create_slurm_run_service(
        profile=_profile(tmp_path, profile_catalog),
        artifact_publisher=publisher,  # type: ignore[arg-type]
        launcher=launcher,  # type: ignore[arg-type]
        run_id_factory=lambda: "run-wired",
        clock=lambda: submitted_at,
        package_version="0.9.2",
    )

    result = service.execute(authored_run_single, source_root=tmp_path, force=True)

    assert result.state == "submitted"
    assert result.job_id == 42
    assert publisher.initializations == [("run-wired", True)]
    assert publisher.submissions == [("run-wired", 42, submitted_at)]
    assert len(launcher.submissions) == 1
    assert launcher.held_submissions == [True]
    assert launcher.releases == [42]
    assert launcher.exported_environments == [{"SLURM_EXPORT_ENV": "ALL"}]
    assert (tmp_path / "managed-assets").is_dir()


def test_production_publisher_rejects_force_before_submission(
    tmp_path: Path,
    profile_catalog: SlurmProfileCatalog,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    _register_images(tmp_path, authored_run_single, single_node_plan)
    launcher = _Launcher()
    service = create_slurm_run_service(
        profile=_profile(tmp_path, profile_catalog),
        launcher=launcher,  # type: ignore[arg-type]
        run_id_factory=lambda: "run-wired",
        package_version="0.9.2",
    )

    with pytest.raises(SlurmServiceError, match="different inputs") as caught:
        service.execute(authored_run_single, source_root=tmp_path, force=True)

    assert caught.value.code is SlurmServiceErrorCode.CONFLICT
    assert launcher.submissions == []


def test_production_wiring_exports_referenced_secrets_to_the_allocation(
    tmp_path: Path,
    profile_catalog: SlurmProfileCatalog,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    deployment = authored_run_single.deployments[0]
    server = deployment.server.model_copy(
        update={"environment": {"HF_TOKEN": SecretRef(type="secret", environment="SOURCE_TOKEN")}}
    )
    authored = authored_run_single.model_copy(
        update={"deployments": [deployment.model_copy(update={"server": server})]}
    )
    _register_images(tmp_path, authored, single_node_plan)
    launcher = _Launcher()
    service = create_slurm_run_service(
        profile=_profile(tmp_path, profile_catalog),
        artifact_publisher=_Publisher(),  # type: ignore[arg-type]
        launcher=launcher,  # type: ignore[arg-type]
        run_id_factory=lambda: "run-wired",
        package_version="0.9.2",
        source_environment={"SOURCE_TOKEN": "secret-value"},
    )

    service.execute(authored, source_root=tmp_path)

    assert launcher.exported_environments == [{"SLURM_EXPORT_ENV": "ALL", "SOURCE_TOKEN": "secret-value"}]


def test_production_wiring_rejects_missing_secret_before_publishing(
    tmp_path: Path,
    profile_catalog: SlurmProfileCatalog,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    deployment = authored_run_single.deployments[0]
    server = deployment.server.model_copy(
        update={"environment": {"HF_TOKEN": SecretRef(type="secret", environment="MISSING_TOKEN")}}
    )
    authored = authored_run_single.model_copy(
        update={"deployments": [deployment.model_copy(update={"server": server})]}
    )
    _register_images(tmp_path, authored, single_node_plan)
    launcher = _Launcher()
    publisher = _Publisher()
    service = create_slurm_run_service(
        profile=_profile(tmp_path, profile_catalog),
        artifact_publisher=publisher,  # type: ignore[arg-type]
        launcher=launcher,  # type: ignore[arg-type]
        run_id_factory=lambda: "run-wired",
        package_version="0.9.2",
        source_environment={},
    )

    with pytest.raises(SlurmServiceError, match="MISSING_TOKEN") as caught:
        service.execute(authored, source_root=tmp_path)

    assert caught.value.code is SlurmServiceErrorCode.INVALID_REQUEST
    assert publisher.initializations == []
    assert launcher.submissions == []


def test_production_wiring_publishes_initial_state_before_releasing_submission(
    tmp_path: Path,
    profile_catalog: SlurmProfileCatalog,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    _register_images(tmp_path, authored_run_single, single_node_plan)
    launcher = _Launcher()
    service = create_slurm_run_service(
        profile=_profile(tmp_path, profile_catalog),
        launcher=launcher,  # type: ignore[arg-type]
        run_id_factory=lambda: "run-wired",
        package_version="0.9.2",
    )

    result = service.execute(authored_run_single, source_root=tmp_path)

    writer = SlurmStateWriter(tmp_path, result.run_id)
    attempts = writer.load_attempts("shard-00000")
    assert result.state == "submitted"
    assert writer.load_run().run_id == result.run_id
    dependency_bytes = (tmp_path / "runs/run-wired/dependency-lock.json").read_bytes()
    assert hashlib.sha256(dependency_bytes).hexdigest() == writer.load_resolved_plan().client.dependency_lock.sha256
    assert attempts[0].state is AttemptLifecycleState.SUBMITTED
    assert attempts[0].scheduler == SchedulerIdentity(array_job_id=42, array_task_id=0)
    assert launcher.held_submissions == [True]
    assert launcher.releases == [42]


def test_status_reconciles_cancelled_scheduler_attempt(
    tmp_path: Path,
    profile_catalog: SlurmProfileCatalog,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    _register_images(tmp_path, authored_run_single, single_node_plan)
    launcher = _Launcher()
    service = create_slurm_run_service(
        profile=_profile(tmp_path, profile_catalog),
        launcher=launcher,  # type: ignore[arg-type]
        run_id_factory=lambda: "run-wired",
        clock=lambda: datetime(2026, 9, 8, tzinfo=timezone.utc),
        package_version="0.9.2",
    )
    result = service.execute(authored_run_single, source_root=tmp_path)
    service.cancel(result.run_id)
    scheduler = SchedulerIdentity(array_job_id=42, array_task_id=0)
    launcher.accounting_entries = (
        SlurmAccountingEntry(
            job_identity=scheduler,
            state=SchedulerState.CANCELLED,
            process_exit_code=SlurmProcessExitCode(exit_status=0, termination_signal=15),
        ),
    )

    status = service.status(result.run_id)

    attempt = status.shards[0].attempts[0].attempt
    assert attempt.state is AttemptLifecycleState.FAILED
    assert attempt.terminal_classification is AttemptTerminalClassification.CANCELLED


def test_status_expires_unrequeued_preemption_and_cancel_skips_terminal_job(
    tmp_path: Path,
    profile_catalog: SlurmProfileCatalog,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    _register_images(tmp_path, authored_run_single, single_node_plan)
    launcher = _Launcher()
    current_time = [datetime(2026, 9, 8, tzinfo=timezone.utc)]
    service = create_slurm_run_service(
        profile=_profile(tmp_path, profile_catalog),
        launcher=launcher,  # type: ignore[arg-type]
        run_id_factory=lambda: "run-wired",
        clock=lambda: current_time[0],
        package_version="0.9.2",
    )
    result = service.execute(authored_run_single, source_root=tmp_path)
    scheduler = SchedulerIdentity(array_job_id=42, array_task_id=0)
    launcher.accounting_entries = (
        SlurmAccountingEntry(
            job_identity=scheduler,
            state=SchedulerState.PREEMPTED,
            process_exit_code=SlurmProcessExitCode(exit_status=0, termination_signal=0),
        ),
    )

    pending = service.status(result.run_id)
    current_time[0] += timedelta(minutes=5, seconds=1)
    failed = service.status(result.run_id)
    cancellation = service.cancel(result.run_id)

    assert pending.shards[0].attempts[0].attempt.state is AttemptLifecycleState.PENDING
    assert failed.shards[0].attempts[0].attempt.state is AttemptLifecycleState.FAILED
    assert cancellation.job_ids == ()
    assert launcher.cancellations == []


def test_status_does_not_expire_requeue_window_from_another_attempt_clock(
    tmp_path: Path,
    profile_catalog: SlurmProfileCatalog,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    authored = authored_run_single.model_copy(
        update={"array_tasks": authored_run_single.array_tasks.model_copy(update={"count": 2})}
    )
    _register_images(tmp_path, authored, single_node_plan)
    launcher = _Launcher()
    current_time = [datetime(2026, 9, 8, tzinfo=timezone.utc)]
    service = create_slurm_run_service(
        profile=_profile(tmp_path, profile_catalog),
        launcher=launcher,  # type: ignore[arg-type]
        run_id_factory=lambda: "run-wired",
        clock=lambda: current_time[0],
        package_version="0.9.2",
    )
    result = service.execute(authored, source_root=tmp_path)
    preempted = SchedulerIdentity(array_job_id=42, array_task_id=0)
    running = SchedulerIdentity(array_job_id=42, array_task_id=1)
    launcher.queue_entries = (SlurmQueueEntry(job_identity=running, state=SchedulerState.RUNNING),)
    launcher.accounting_entries = (
        SlurmAccountingEntry(
            job_identity=preempted,
            state=SchedulerState.PREEMPTED,
            process_exit_code=SlurmProcessExitCode(exit_status=0, termination_signal=0),
        ),
    )
    service.status(result.run_id)
    writer = SlurmStateWriter(tmp_path, result.run_id)
    other_attempt = writer.load_attempt("shard-00001", "attempt-0001")
    writer.update_attempt(other_attempt.model_copy(update={"updated_at": current_time[0] + timedelta(minutes=10)}))

    current_time[0] += timedelta(minutes=1)
    still_requeueing = service.status(result.run_id)
    current_time[0] += timedelta(minutes=1)
    launcher.queue_entries = (
        SlurmQueueEntry(job_identity=preempted, state=SchedulerState.PENDING),
        SlurmQueueEntry(job_identity=running, state=SchedulerState.RUNNING),
    )
    requeued = service.status(result.run_id)

    assert still_requeueing.shards[0].attempts[0].attempt.state is AttemptLifecycleState.PENDING
    assert requeued.shards[0].attempts[0].attempt.state is AttemptLifecycleState.PENDING


def test_auto_gpu_resolution_rejects_mixed_node_shapes(
    tmp_path: Path,
    profile_catalog: SlurmProfileCatalog,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    _register_images(tmp_path, authored_run_single, single_node_plan)
    profile = profile_catalog.clusters["lab"].model_copy(update={"workspace_root": tmp_path.as_posix()})
    service = create_slurm_run_service(
        profile=profile,
        launcher=_Launcher((4, 8)),  # type: ignore[arg-type]
        run_id_factory=lambda: "run-wired",
        package_version="0.9.2",
    )

    with pytest.raises(SlurmServiceError) as caught:
        service.execute(authored_run_single, source_root=tmp_path, dry_run=True)

    assert caught.value.code is SlurmServiceErrorCode.UNAVAILABLE


def test_publisher_value_error_is_not_misattributed_to_run_preparation(
    tmp_path: Path,
    profile_catalog: SlurmProfileCatalog,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    _register_images(tmp_path, authored_run_single, single_node_plan)
    launcher = _Launcher()
    service = create_slurm_run_service(
        profile=_profile(tmp_path, profile_catalog),
        artifact_publisher=_Publisher(initialization_error=ValueError("backend bug")),  # type: ignore[arg-type]
        launcher=launcher,  # type: ignore[arg-type]
        run_id_factory=lambda: "run-wired",
        package_version="0.9.2",
    )

    with pytest.raises(SlurmServiceError) as caught:
        service.execute(authored_run_single, source_root=tmp_path)

    assert caught.value.code is SlurmServiceErrorCode.INTERNAL
    assert str(caught.value) == "execute run failed"
    assert launcher.submissions == []


def test_recording_conflict_cancels_the_accepted_job(
    tmp_path: Path,
    profile_catalog: SlurmProfileCatalog,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    _register_images(tmp_path, authored_run_single, single_node_plan)
    launcher = _Launcher()
    publisher = _Publisher(submission_error=StateConflictError("conflict"))
    service = create_slurm_run_service(
        profile=_profile(tmp_path, profile_catalog),
        artifact_publisher=publisher,  # type: ignore[arg-type]
        launcher=launcher,  # type: ignore[arg-type]
        run_id_factory=lambda: "run-wired",
        package_version="0.9.2",
    )

    with pytest.raises(SlurmServiceError) as caught:
        service.execute(authored_run_single, source_root=tmp_path)

    assert caught.value.code is SlurmServiceErrorCode.CONFLICT
    assert launcher.cancellations == [42]
    assert len(publisher.submission_failures) == 1


def test_recording_conflict_does_not_fail_another_submission_attempt(
    tmp_path: Path,
    profile_catalog: SlurmProfileCatalog,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    _register_images(tmp_path, authored_run_single, single_node_plan)
    profile = _profile(tmp_path, profile_catalog)
    first = create_slurm_run_service(
        profile=profile,
        launcher=_Launcher(job_id=41),  # type: ignore[arg-type]
        run_id_factory=lambda: "run-wired",
        package_version="0.9.2",
    )
    first.execute(authored_run_single, source_root=tmp_path)
    launcher = _Launcher(job_id=42)
    second = create_slurm_run_service(
        profile=profile,
        launcher=launcher,  # type: ignore[arg-type]
        run_id_factory=lambda: "run-wired",
        package_version="0.9.2",
    )

    with pytest.raises(SlurmServiceError) as caught:
        second.execute(authored_run_single, source_root=tmp_path)

    assert caught.value.code is SlurmServiceErrorCode.CONFLICT
    assert launcher.cancellations == [42]
    attempt = SlurmStateWriter(tmp_path, "run-wired").load_attempt("shard-00000", "attempt-0001")
    assert attempt.state is AttemptLifecycleState.SUBMITTED
    assert attempt.scheduler == SchedulerIdentity(array_job_id=41, array_task_id=0)


def test_partial_submission_recording_failure_cancels_job_and_fails_created_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile_catalog: SlurmProfileCatalog,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    authored = authored_run_single.model_copy(
        update={"array_tasks": authored_run_single.array_tasks.model_copy(update={"count": 2})}
    )
    _register_images(tmp_path, authored, single_node_plan)
    original_create = SlurmStateWriter.create_attempt
    created = 0

    def fail_second_attempt(writer: SlurmStateWriter, attempt: AttemptManifest) -> AttemptManifest:
        nonlocal created
        created += 1
        if created == 2:
            raise StateConflictError("injected partial submission failure")
        return original_create(writer, attempt)

    monkeypatch.setattr(SlurmStateWriter, "create_attempt", fail_second_attempt)
    launcher = _Launcher()
    failed_at = datetime(2026, 9, 8, tzinfo=timezone.utc)
    service = create_slurm_run_service(
        profile=_profile(tmp_path, profile_catalog),
        launcher=launcher,  # type: ignore[arg-type]
        run_id_factory=lambda: "run-wired",
        clock=lambda: failed_at,
        package_version="0.9.2",
    )

    with pytest.raises(SlurmServiceError) as caught:
        service.execute(authored, source_root=tmp_path)

    assert caught.value.code is SlurmServiceErrorCode.CONFLICT
    assert launcher.cancellations == [42]
    writer = SlurmStateWriter(tmp_path, "run-wired")
    attempt = writer.load_attempt("shard-00000", "attempt-0001")
    assert attempt.state is AttemptLifecycleState.FAILED
    assert attempt.terminal_classification is AttemptTerminalClassification.CANCELLED
    with pytest.raises(StateNotFoundError):
        writer.load_attempt("shard-00001", "attempt-0001")


def test_release_failure_cancels_the_held_job(
    tmp_path: Path,
    profile_catalog: SlurmProfileCatalog,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    _register_images(tmp_path, authored_run_single, single_node_plan)
    launcher = _Launcher(release_error=SlurmLauncherError("release failed"))
    failed_at = datetime(2026, 9, 8, tzinfo=timezone.utc)
    service = create_slurm_run_service(
        profile=_profile(tmp_path, profile_catalog),
        launcher=launcher,  # type: ignore[arg-type]
        run_id_factory=lambda: "run-wired",
        clock=lambda: failed_at,
        package_version="0.9.2",
    )

    with pytest.raises(SlurmServiceError, match="could not be released and was cancelled") as caught:
        service.execute(authored_run_single, source_root=tmp_path)

    assert caught.value.code is SlurmServiceErrorCode.UNAVAILABLE
    assert launcher.releases == [42]
    assert launcher.cancellations == [42]
    attempt = SlurmStateWriter(tmp_path, "run-wired").load_attempt("shard-00000", "attempt-0001")
    assert attempt.state is AttemptLifecycleState.FAILED
    assert attempt.terminal_classification is AttemptTerminalClassification.CANCELLED
    assert attempt.updated_at == failed_at


def test_failed_compensating_cancel_reports_accepted_job_id(
    tmp_path: Path,
    profile_catalog: SlurmProfileCatalog,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    _register_images(tmp_path, authored_run_single, single_node_plan)
    launcher = _Launcher(cancel_error=RuntimeError("cancel failed"))
    service = create_slurm_run_service(
        profile=_profile(tmp_path, profile_catalog),
        artifact_publisher=_Publisher(submission_error=StateConflictError("conflict")),  # type: ignore[arg-type]
        launcher=launcher,  # type: ignore[arg-type]
        run_id_factory=lambda: "run-wired",
        package_version="0.9.2",
    )

    with pytest.raises(SlurmServiceError, match="Slurm job 42") as caught:
        service.execute(authored_run_single, source_root=tmp_path)

    assert caught.value.code is SlurmServiceErrorCode.UNAVAILABLE


def test_recording_interrupt_is_preserved_when_compensating_cancel_fails(
    tmp_path: Path,
    profile_catalog: SlurmProfileCatalog,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    _register_images(tmp_path, authored_run_single, single_node_plan)
    launcher = _Launcher(cancel_error=RuntimeError("cancel failed"))
    service = create_slurm_run_service(
        profile=_profile(tmp_path, profile_catalog),
        artifact_publisher=_Publisher(submission_error=KeyboardInterrupt()),  # type: ignore[arg-type]
        launcher=launcher,  # type: ignore[arg-type]
        run_id_factory=lambda: "run-wired",
        package_version="0.9.2",
    )

    with pytest.raises(KeyboardInterrupt):
        service.execute(authored_run_single, source_root=tmp_path)

    assert launcher.cancellations == [42]


def test_production_status_and_cancel_use_only_persisted_m2_records(
    tmp_path: Path,
    profile_catalog: SlurmProfileCatalog,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    _register_images(tmp_path, authored_run_single, single_node_plan)
    launcher = _Launcher()
    service = create_slurm_run_service(
        profile=_profile(tmp_path, profile_catalog),
        launcher=launcher,  # type: ignore[arg-type]
        run_id_factory=lambda: "run-wired",
        package_version="0.9.2",
    )
    plan = service.plan(authored_run_single)
    now = datetime(2026, 9, 8, tzinfo=timezone.utc)
    plan_reference = ArtifactReference(
        path=(tmp_path / "runs/run-wired/resolved-plan.json").as_posix(),
        sha256=plan.compute_sha256(),
    )
    run = RunManifest(
        schema_version=1,
        run_id=plan.run_id,
        created_at=now,
        authored_config=plan.authored_config,
        resolved_plan=plan_reference,
        shard_count=len(plan.shards),
    )
    shards = tuple(
        ShardManifest(
            schema_version=1,
            run_id=plan.run_id,
            shard_id=shard.shard_id,
            shard_index=shard.shard_index,
            record_range=shard.record_range,
            input_partition=shard.input_partition,
            resume_workspace=shard.resume_workspace,
            created_at=now,
        )
        for shard in plan.shards
    )
    writer = SlurmStateWriter(tmp_path, plan.run_id)
    writer.initialize_run(authored_run_single, plan, run, shards)
    created = AttemptManifest(
        schema_version=1,
        run_id=plan.run_id,
        shard_id=shards[0].shard_id,
        attempt_id="attempt-0001",
        attempt_ordinal=1,
        resolved_plan=plan_reference,
        state=AttemptLifecycleState.CREATED,
        created_at=now,
        updated_at=now,
    )
    writer.create_attempt(created)
    assert service.cancel(plan.run_id).job_ids == ()
    assert launcher.cancellations == []
    writer.update_attempt(
        created.model_copy(
            update={
                "state": AttemptLifecycleState.SUBMITTED,
                "scheduler": SchedulerIdentity(array_job_id=42, array_task_id=0),
            }
        )
    )

    status = service.status(plan.run_id)
    cancellation = service.cancel(plan.run_id)
    persisted_after_cancel = service.status(plan.run_id)

    assert status.run == run
    assert status.shards[0].attempts[0].attempt.state is AttemptLifecycleState.SUBMITTED
    assert cancellation.job_ids == (42,)
    assert persisted_after_cancel.shards[0].attempts[0].attempt.state is AttemptLifecycleState.SUBMITTED
    assert launcher.cancellations == [42]


def test_production_status_maps_unknown_run_to_not_found(
    tmp_path: Path,
    profile_catalog: SlurmProfileCatalog,
) -> None:
    service = create_slurm_run_service(
        profile=_profile(tmp_path, profile_catalog),
        launcher=_Launcher(),  # type: ignore[arg-type]
        package_version="0.9.2",
    )

    with pytest.raises(SlurmServiceError) as caught:
        service.status("run-missing")

    assert caught.value.code is SlurmServiceErrorCode.NOT_FOUND


def test_production_image_registry_operations_and_lifecycle_gap(
    tmp_path: Path,
    profile_catalog: SlurmProfileCatalog,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    _register_images(tmp_path, authored_run_single, single_node_plan)
    service = create_slurm_image_service(profile=_profile(tmp_path, profile_catalog))

    images = service.list()
    selected = service.get(images[0].name)
    removed = service.remove(images[0].name)

    assert selected == removed == images[0]
    assert len(service.list()) == 1
    with pytest.raises(SlurmServiceError) as caught:
        service.add(
            ImageBuildRequest(
                name="new-image",
                kind="client",
                source=f"registry.example/client@sha256:{'1' * 64}",
            )
        )
    assert caught.value.code is SlurmServiceErrorCode.UNAVAILABLE
    assert str(caught.value) == "image registration is not available; use a pre-registered image"
