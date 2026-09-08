# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from data_designer.slurm.client.dependencies import ResolvedClientDependencies
from data_designer.slurm.config import (
    BuilderInput,
    DataDesignerSlurmConfig,
    ImageBuildRequest,
    SlurmProfile,
    SlurmProfileCatalog,
)
from data_designer.slurm.contracts import ArtifactReference, canonical_json
from data_designer.slurm.images.records import RegisteredImage
from data_designer.slurm.images.registry import ImageRegistryStore
from data_designer.slurm.launcher.models import SlurmJobSubmissionReceipt
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.services import (
    SlurmServiceError,
    SlurmServiceErrorCode,
    create_slurm_image_service,
    create_slurm_run_service,
)
from data_designer.slurm.state import (
    AttemptLifecycleState,
    AttemptManifest,
    RunManifest,
    SchedulerIdentity,
    ShardManifest,
    SlurmStateWriter,
    StateConflictError,
)


class _Launcher:
    def __init__(self, gpu_counts: tuple[int, ...] = (), *, cancel_error: Exception | None = None) -> None:
        self.submissions: list[str] = []
        self.cancellations: list[int] = []
        self.gpu_counts = gpu_counts
        self.cancel_error = cancel_error

    def submit_script(self, script: str) -> SlurmJobSubmissionReceipt:
        self.submissions.append(script)
        return SlurmJobSubmissionReceipt(job_id=42)

    def cancel(self, job_id: int) -> None:
        self.cancellations.append(job_id)
        if self.cancel_error is not None:
            raise self.cancel_error

    def query_gpu_counts(self, *, partition: str | None = None) -> tuple[int, ...]:
        assert partition is not None
        return self.gpu_counts


class _Publisher:
    def __init__(
        self,
        *,
        initialization_error: BaseException | None = None,
        submission_error: BaseException | None = None,
    ) -> None:
        self.initializations: list[tuple[str, bool]] = []
        self.submissions: list[tuple[str, int, datetime]] = []
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
        self.initializations.append((plan.run_id, force))

    def record_submission(self, plan: ResolvedSlurmRunPlan, job_id: int, *, submitted_at: datetime) -> None:
        if self.submission_error is not None:
            raise self.submission_error
        self.submissions.append((plan.run_id, job_id, submitted_at))


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
    submitted_at = datetime(2026, 9, 8, tzinfo=UTC)
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


def test_production_wiring_stops_before_submission_without_state_publisher(
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

    with pytest.raises(SlurmServiceError) as caught:
        service.execute(authored_run_single, source_root=tmp_path)

    assert caught.value.code is SlurmServiceErrorCode.UNAVAILABLE
    assert str(caught.value) == "run submission is not available; use --dry-run"
    assert launcher.submissions == []


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
    now = datetime(2026, 9, 8, tzinfo=UTC)
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

    assert status.run == run
    assert status.shards[0].attempts[0].attempt.state is AttemptLifecycleState.SUBMITTED
    assert cancellation.job_ids == (42,)
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
