# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package-owned production wiring for public Slurm services."""

from __future__ import annotations

import importlib.metadata
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol, TypeVar
from uuid import uuid4

from pydantic import JsonValue

from data_designer.slurm.client.dependencies import (
    ClientDependencyResolutionError,
    ClientDependencyResolver,
    ResolvedClientDependencies,
)
from data_designer.slurm.client.errors import ClientWorkerError
from data_designer.slurm.client.filesystem import ensure_private_directory
from data_designer.slurm.config import (
    DataDesignerSlurmConfig,
    ImageBuildRequest,
    ImageKind,
    ImageRef,
    SelectedSlurmProfile,
    SlurmConfigLoadError,
    SlurmProfile,
    SlurmProfileCatalog,
    collect_secret_environment_names,
    load_builder_payload,
    resolve_profile,
)
from data_designer.slurm.contracts import Identifier, ShardId
from data_designer.slurm.images.errors import ImageConflictError, ImageNotFoundError, SlurmImageError
from data_designer.slurm.images.records import RegisteredImage
from data_designer.slurm.images.registry import ImageRegistryStore
from data_designer.slurm.images.service import VerifiedImageRegistry
from data_designer.slurm.launcher.client import SlurmCommandClient
from data_designer.slurm.launcher.errors import SlurmLauncherError
from data_designer.slurm.launcher.renderer import render_generation_attempt_script
from data_designer.slurm.planning import ResolvedImage, ResolvedSlurmRunPlan
from data_designer.slurm.planning.compiler import SlurmRunCompiler
from data_designer.slurm.planning.resolution import resolve_slurm_config
from data_designer.slurm.runtime.bundle import stage_runtime_bundle
from data_designer.slurm.runtime.errors import SlurmRuntimeError
from data_designer.slurm.services.artifacts import StateRunArtifactPublisher
from data_designer.slurm.services.errors import SlurmServiceError, SlurmServiceErrorCode, SlurmServiceOperation
from data_designer.slurm.services.images import SlurmImageService
from data_designer.slurm.services.results import (
    SlurmCollectionExecution,
    SlurmPersistedAttemptStatus,
    SlurmPersistedRunStatus,
    SlurmPersistedShardStatus,
    SlurmRetryExecution,
    SlurmRunCancellation,
    SlurmRunExecution,
)
from data_designer.slurm.services.retry_collection import RunRetryCollectionBackend
from data_designer.slurm.services.run import SlurmRunService
from data_designer.slurm.serving.resolver import resolve_vllm_server
from data_designer.slurm.state import (
    AttemptLifecycleState,
    AttemptManifest,
    AttemptStatus,
    AttemptTerminalClassification,
    EffectiveAttemptState,
    SchedulerState,
    SlurmStateError,
    SlurmStateReconciler,
    SlurmStateWriter,
    StateConflictError,
    StateNotFoundError,
)

RunIdFactory = Callable[[], str]
Clock = Callable[[], datetime]
_ResultT = TypeVar("_ResultT")
_MAX_VISIBLE_JOB_IDS = 16
_ACTIVE_ATTEMPT_STATES = frozenset(
    {AttemptLifecycleState.SUBMITTED, AttemptLifecycleState.PENDING, AttemptLifecycleState.RUNNING}
)
_FAILURE_CLASSIFICATIONS = {
    SchedulerState.FAILED: AttemptTerminalClassification.FAILED,
    SchedulerState.CANCELLED: AttemptTerminalClassification.CANCELLED,
    SchedulerState.TIMED_OUT: AttemptTerminalClassification.TIMED_OUT,
    SchedulerState.NODE_FAILED: AttemptTerminalClassification.NODE_FAILED,
    SchedulerState.PREEMPTED: AttemptTerminalClassification.PREEMPTED,
    SchedulerState.REQUEUED: AttemptTerminalClassification.REQUEUED,
    SchedulerState.OUT_OF_MEMORY: AttemptTerminalClassification.OUT_OF_MEMORY,
}


class SlurmRunArtifactPublisher(Protocol):
    """Persist submission-critical run artifacts through the STATE boundary."""

    def initialize(
        self,
        authored: DataDesignerSlurmConfig,
        plan: ResolvedSlurmRunPlan,
        dependencies: ResolvedClientDependencies,
        builder_payload: dict[str, JsonValue] | None,
        *,
        force: bool,
    ) -> None:
        """Atomically publish all immutable run inputs before submission."""

    def record_submission(self, plan: ResolvedSlurmRunPlan, job_id: int, *, submitted_at: datetime) -> None:
        """Persist the submitted scheduler identity for every initial attempt."""

    def record_submission_failure(self, plan: ResolvedSlurmRunPlan, *, failed_at: datetime) -> None:
        """Mark initial attempts failed after a held submission is cancelled."""


@dataclass(frozen=True, slots=True)
class _PreparedRun:
    plan: ResolvedSlurmRunPlan
    dependencies: ResolvedClientDependencies
    builder_payload: dict[str, JsonValue] | None
    batch_script: str


class _RunPreparer:
    def __init__(
        self,
        selected_profile: SelectedSlurmProfile,
        image_registry: VerifiedImageRegistry,
        dependency_resolver: ClientDependencyResolver,
        launcher: SlurmCommandClient,
        run_id_factory: RunIdFactory,
        package_version: str,
    ) -> None:
        self._profile = selected_profile
        self._images = image_registry
        self._dependencies = dependency_resolver
        self._launcher = launcher
        self._run_id_factory = run_id_factory
        self._package_version = package_version

    @contextmanager
    def prepare(
        self,
        authored: DataDesignerSlurmConfig,
        *,
        source_root: Path,
        operation: SlurmServiceOperation,
    ) -> Iterator[_PreparedRun]:
        stack = ExitStack()
        try:
            prepared = self._prepare(authored, source_root=source_root, operation=operation, stack=stack)
        except BaseException:
            stack.close()
            raise
        try:
            yield prepared
        finally:
            stack.close()

    def _prepare(
        self,
        authored: DataDesignerSlurmConfig,
        *,
        source_root: Path,
        operation: SlurmServiceOperation,
        stack: ExitStack,
    ) -> _PreparedRun:
        try:
            run_id = self._run_id_factory()
            workspace_root = self._profile.profile.workspace_root
            run_root = Path(workspace_root) / "runs" / run_id
            client_image = self._images.resolve_for_planning(authored.client.image, expected_kind=ImageKind.CLIENT)
            deployment_images = tuple(
                self._images.resolve_for_planning(deployment.server.image, expected_kind=ImageKind.SERVING)
                for deployment in authored.deployments
            )
            builder_payload = _resolve_builder_payload(authored, source_root)
            dependencies = stack.enter_context(
                self._dependencies.resolve(
                    authored.client.dependencies,
                    client_image,
                    run_root=run_root,
                    source_root=source_root,
                )
            )
            runtime_bundle = stage_runtime_bundle(workspace_root)
            effective = resolve_slurm_config(
                authored,
                selected_profile=self._profile,
                client_image=client_image,
                deployment_images=deployment_images,
                dependency_lock=dependencies.lock,
                runtime_bundle=runtime_bundle,
                run_id=run_id,
                package_version=self._package_version,
                resolved_gpus_per_node=self._resolve_gpu_count(authored, operation),
                builder_payload=builder_payload,
            )
            plan = SlurmRunCompiler.compile(effective)
            for deployment in plan.deployments:
                resolve_vllm_server(plan, deployment.deployment_id)
            return _PreparedRun(
                plan=plan,
                dependencies=dependencies,
                builder_payload=builder_payload,
                batch_script=render_generation_attempt_script(plan, attempt_ordinal=1),
            )
        except SlurmServiceError:
            raise
        except ImageNotFoundError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.NOT_FOUND, operation, "required image is not registered"
            ) from None
        except SlurmImageError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.CONFLICT, operation, "required image cannot be verified"
            ) from None
        except (ClientDependencyResolutionError, SlurmConfigLoadError, ValueError):
            raise SlurmServiceError(
                SlurmServiceErrorCode.INVALID_REQUEST, operation, "run configuration cannot be prepared"
            ) from None
        except SlurmLauncherError:
            raise SlurmServiceError(SlurmServiceErrorCode.UNAVAILABLE, operation, "Slurm is unavailable") from None
        except (SlurmRuntimeError, OSError):
            raise SlurmServiceError(
                SlurmServiceErrorCode.UNAVAILABLE, operation, "run artifacts cannot be staged"
            ) from None

    def _resolve_gpu_count(
        self,
        authored: DataDesignerSlurmConfig,
        operation: SlurmServiceOperation,
    ) -> int | None:
        if self._profile.profile.gpus_per_node != "auto":
            return None
        partition = authored.submission.partition or self._profile.profile.scheduler.partition
        counts = tuple(sorted(set(self._launcher.query_gpu_counts(partition=partition))))
        if len(counts) != 1:
            raise SlurmServiceError(
                SlurmServiceErrorCode.UNAVAILABLE,
                operation,
                "eligible Slurm nodes do not report one GPU count",
            )
        return counts[0]


class _SystemRunPlanner:
    def __init__(self, preparer: _RunPreparer) -> None:
        self._preparer = preparer

    def plan(self, config: DataDesignerSlurmConfig, *, source_root: Path) -> ResolvedSlurmRunPlan:
        with self._preparer.prepare(
            config, source_root=source_root, operation=SlurmServiceOperation.PLAN_RUN
        ) as prepared:
            return prepared.plan


class _SystemRunBackend:
    def __init__(
        self,
        preparer: _RunPreparer,
        selected_profile: SelectedSlurmProfile,
        launcher: SlurmCommandClient,
        publisher: SlurmRunArtifactPublisher,
        clock: Clock,
        source_environment: Mapping[str, str],
    ) -> None:
        self._preparer = preparer
        self._profile = selected_profile
        self._launcher = launcher
        self._publisher = publisher
        self._clock = clock
        self._source_environment = source_environment
        self._retry_collection = RunRetryCollectionBackend(
            selected_profile.profile.workspace_root,
            launcher,
            clock,
        )

    def execute(
        self,
        config: DataDesignerSlurmConfig,
        *,
        source_root: Path,
        dry_run: bool,
        force: bool,
    ) -> SlurmRunExecution:
        with self._preparer.prepare(
            config,
            source_root=source_root,
            operation=SlurmServiceOperation.EXECUTE_RUN,
        ) as prepared:
            plan = prepared.plan
            if dry_run:
                return SlurmRunExecution(
                    run_id=plan.run_id,
                    state="dry_run",
                    plan_sha256=plan.compute_sha256(),
                    shard_count=len(plan.shards),
                    batch_script=prepared.batch_script,
                )
            self._materialize_default_managed_assets(config, plan)
            export_environment = self._build_export_environment(config)
            publisher = self._publisher
            self._initialize_run(
                publisher,
                config,
                prepared,
                force=force,
            )
            try:
                receipt = self._launcher.submit_script(
                    prepared.batch_script,
                    hold=True,
                    export_environment=export_environment,
                )
            except SlurmLauncherError:
                raise SlurmServiceError(
                    SlurmServiceErrorCode.UNAVAILABLE,
                    SlurmServiceOperation.EXECUTE_RUN,
                    "Slurm submission is unavailable",
                ) from None
            try:
                self._record_submission(publisher, plan, receipt.job_id)
            except Exception as error:
                try:
                    self._launcher.cancel(receipt.job_id)
                except Exception:
                    raise SlurmServiceError(
                        SlurmServiceErrorCode.UNAVAILABLE,
                        SlurmServiceOperation.EXECUTE_RUN,
                        f"Slurm job {receipt.job_id} was submitted but could not be recorded or cancelled",
                    ) from error
                try:
                    self._record_submission_failure(publisher, plan)
                except SlurmServiceError as state_error:
                    raise SlurmServiceError(
                        SlurmServiceErrorCode.INTERNAL,
                        SlurmServiceOperation.EXECUTE_RUN,
                        f"Slurm job {receipt.job_id} was cancelled but its partial submission state could not be updated",
                    ) from state_error
                raise
            except BaseException:
                try:
                    self._launcher.cancel(receipt.job_id)
                except Exception:
                    pass
                else:
                    try:
                        self._record_submission_failure(publisher, plan)
                    except BaseException:
                        pass
                raise
            try:
                self._launcher.release(receipt.job_id)
            except SlurmLauncherError as error:
                try:
                    self._launcher.cancel(receipt.job_id)
                except Exception:
                    raise SlurmServiceError(
                        SlurmServiceErrorCode.UNAVAILABLE,
                        SlurmServiceOperation.EXECUTE_RUN,
                        f"held Slurm job {receipt.job_id} could not be released or cancelled",
                    ) from error
                try:
                    self._record_submission_failure(publisher, plan)
                except SlurmServiceError as state_error:
                    raise SlurmServiceError(
                        SlurmServiceErrorCode.INTERNAL,
                        SlurmServiceOperation.EXECUTE_RUN,
                        f"held Slurm job {receipt.job_id} was cancelled but run {plan.run_id!r} could not be updated",
                    ) from state_error
                raise SlurmServiceError(
                    SlurmServiceErrorCode.UNAVAILABLE,
                    SlurmServiceOperation.EXECUTE_RUN,
                    f"held Slurm job {receipt.job_id} could not be released and was cancelled",
                ) from None
            return SlurmRunExecution(
                run_id=plan.run_id,
                state="submitted",
                plan_sha256=plan.compute_sha256(),
                shard_count=len(plan.shards),
                job_id=receipt.job_id,
            )

    @staticmethod
    def _materialize_default_managed_assets(
        config: DataDesignerSlurmConfig,
        plan: ResolvedSlurmRunPlan,
    ) -> None:
        if config.invocation.input_bindings.managed_assets_path is not None:
            return
        path = plan.invocation.effective_input_bindings.managed_assets_path
        assert path is not None
        try:
            ensure_private_directory(Path(path))
        except (ClientWorkerError, OSError):
            raise SlurmServiceError(
                SlurmServiceErrorCode.UNAVAILABLE,
                SlurmServiceOperation.EXECUTE_RUN,
                "managed assets workspace cannot be prepared",
            ) from None

    def _build_export_environment(self, config: DataDesignerSlurmConfig) -> dict[str, str]:
        environment = {"SLURM_EXPORT_ENV": "ALL"}
        if "SLURM_CONF" in self._source_environment:
            slurm_conf = self._source_environment["SLURM_CONF"]
            if type(slurm_conf) is not str or "\0" in slurm_conf:
                raise SlurmServiceError(
                    SlurmServiceErrorCode.INVALID_REQUEST,
                    SlurmServiceOperation.EXECUTE_RUN,
                    "SLURM_CONF is invalid",
                )
            environment["SLURM_CONF"] = slurm_conf
        for name in collect_secret_environment_names(config):
            try:
                value = self._source_environment[name]
            except KeyError:
                raise SlurmServiceError(
                    SlurmServiceErrorCode.INVALID_REQUEST,
                    SlurmServiceOperation.EXECUTE_RUN,
                    f"required secret environment {name!r} is unavailable",
                ) from None
            if type(value) is not str or "\0" in value:
                raise SlurmServiceError(
                    SlurmServiceErrorCode.INVALID_REQUEST,
                    SlurmServiceOperation.EXECUTE_RUN,
                    f"required secret environment {name!r} is invalid",
                )
            environment[name] = value
        return environment

    def _initialize_run(
        self,
        publisher: SlurmRunArtifactPublisher,
        config: DataDesignerSlurmConfig,
        prepared: _PreparedRun,
        *,
        force: bool,
    ) -> None:
        try:
            publisher.initialize(
                config,
                prepared.plan,
                prepared.dependencies,
                prepared.builder_payload,
                force=force,
            )
        except StateNotFoundError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.NOT_FOUND,
                SlurmServiceOperation.EXECUTE_RUN,
                "required run state was not found",
            ) from None
        except StateConflictError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.CONFLICT,
                SlurmServiceOperation.EXECUTE_RUN,
                "run state already contains different inputs",
            ) from None
        except SlurmStateError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.INTERNAL,
                SlurmServiceOperation.EXECUTE_RUN,
                "run state cannot be initialized",
            ) from None

    def _record_submission(
        self,
        publisher: SlurmRunArtifactPublisher,
        plan: ResolvedSlurmRunPlan,
        job_id: int,
    ) -> None:
        try:
            publisher.record_submission(plan, job_id, submitted_at=self._clock())
        except StateNotFoundError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.NOT_FOUND,
                SlurmServiceOperation.EXECUTE_RUN,
                "submitted run state was not found",
            ) from None
        except StateConflictError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.CONFLICT,
                SlurmServiceOperation.EXECUTE_RUN,
                "submitted run state conflicts with persisted state",
            ) from None
        except SlurmStateError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.INTERNAL,
                SlurmServiceOperation.EXECUTE_RUN,
                "submission state cannot be recorded",
            ) from None

    def _record_submission_failure(self, publisher: SlurmRunArtifactPublisher, plan: ResolvedSlurmRunPlan) -> None:
        try:
            publisher.record_submission_failure(plan, failed_at=self._clock())
        except (StateNotFoundError, StateConflictError, SlurmStateError):
            raise SlurmServiceError(
                SlurmServiceErrorCode.INTERNAL,
                SlurmServiceOperation.EXECUTE_RUN,
                "cancelled submission state cannot be recorded",
            ) from None

    def status(self, run_id: Identifier) -> SlurmPersistedRunStatus:
        operation = SlurmServiceOperation.STATUS_RUN
        try:
            writer = SlurmStateWriter(self._profile.profile.workspace_root, run_id)
            run = writer.load_run()
            persisted_shards = writer.load_shards()
            self._reconcile_attempts(
                writer,
                tuple(attempt for shard in persisted_shards for attempt in writer.load_attempts(shard.shard_id)),
            )
            shards = []
            for shard in persisted_shards:
                writer.resume_incomplete_finalization(shard.shard_id, published_at=self._clock())
                attempts = tuple(
                    SlurmPersistedAttemptStatus(
                        attempt=attempt,
                        readiness=_load_optional(
                            lambda shard_id=shard.shard_id, attempt_id=attempt.attempt_id: writer.load_readiness(
                                shard_id, attempt_id
                            )
                        ),
                    )
                    for attempt in writer.load_attempts(shard.shard_id)
                )
                shards.append(
                    SlurmPersistedShardStatus(
                        shard=shard,
                        attempts=attempts,
                        winner=_load_optional(lambda shard_id=shard.shard_id: writer.load_winner(shard_id)),
                    )
                )
            return SlurmPersistedRunStatus(run=run, shards=tuple(shards))
        except StateNotFoundError:
            raise SlurmServiceError(SlurmServiceErrorCode.NOT_FOUND, operation, "run state was not found") from None
        except StateConflictError:
            raise SlurmServiceError(SlurmServiceErrorCode.CONFLICT, operation, "run state is inconsistent") from None
        except SlurmStateError:
            raise SlurmServiceError(SlurmServiceErrorCode.INTERNAL, operation, "run state cannot be read") from None

    def _reconcile_attempts(
        self,
        writer: SlurmStateWriter,
        attempts: tuple[AttemptManifest, ...],
    ) -> None:
        active = tuple(
            attempt for attempt in attempts if attempt.state in _ACTIVE_ATTEMPT_STATES and attempt.scheduler is not None
        )
        if not active:
            return
        try:
            reconciled = SlurmStateReconciler(
                self._profile.profile.workspace_root,
                writer.load_run().run_id,
                self._launcher,
            ).refresh(observed_at=self._clock())
        except SlurmStateError as error:
            if isinstance(error.__cause__, SlurmLauncherError):
                return
            raise
        active_identities = {(attempt.shard_id, attempt.attempt_id) for attempt in active}
        for shard in reconciled.shards:
            for status in shard.attempts:
                if (status.attempt.shard_id, status.attempt.attempt_id) in active_identities:
                    self._update_reconciled_attempt(writer, status)

    @staticmethod
    def _update_reconciled_attempt(
        writer: SlurmStateWriter,
        status: AttemptStatus,
    ) -> None:
        attempt = status.attempt
        scheduler = status.scheduler
        if scheduler is None:  # pragma: no cover - active attempts always have scheduler evidence
            raise AssertionError("active attempt has no reconciled scheduler evidence")
        update: dict[str, object] = {"updated_at": scheduler.observed_at}
        if status.effective_state is EffectiveAttemptState.PENDING and attempt.state is AttemptLifecycleState.SUBMITTED:
            update["state"] = AttemptLifecycleState.PENDING
        elif status.effective_state is EffectiveAttemptState.RUNNING and attempt.state in {
            AttemptLifecycleState.SUBMITTED,
            AttemptLifecycleState.PENDING,
        }:
            update["state"] = AttemptLifecycleState.RUNNING
        elif status.effective_state is EffectiveAttemptState.FAILED:
            update.update(
                state=AttemptLifecycleState.FAILED,
                terminal_classification=_FAILURE_CLASSIFICATIONS.get(
                    scheduler.state,
                    AttemptTerminalClassification.UNKNOWN,
                ),
            )
        else:
            return
        try:
            writer.update_attempt(attempt.model_copy(update=update))
        except StateConflictError:
            return

    def cancel(self, run_id: Identifier) -> SlurmRunCancellation:
        status = self.status(run_id)
        job_ids = tuple(
            sorted(
                {
                    attempt.attempt.scheduler.array_job_id
                    for shard in status.shards
                    for attempt in shard.attempts
                    if attempt.attempt.state in _ACTIVE_ATTEMPT_STATES and attempt.attempt.scheduler is not None
                }
            )
        )
        failures = []
        for job_id in job_ids:
            try:
                self._launcher.cancel(job_id)
            except SlurmLauncherError:
                failures.append(job_id)
        if failures:
            raise SlurmServiceError(
                SlurmServiceErrorCode.UNAVAILABLE,
                SlurmServiceOperation.CANCEL_RUN,
                f"failed to cancel managed Slurm jobs: {_format_job_ids(failures)}",
            )
        return SlurmRunCancellation(run_id=run_id, job_ids=job_ids)

    def retry(
        self,
        run_or_job_id: Identifier,
        *,
        shard_ids: tuple[ShardId, ...] | None,
        resume: Literal["never", "always", "if_possible"],
        dry_run: bool,
        force: bool,
    ) -> SlurmRetryExecution:
        return self._retry_collection.retry(
            run_or_job_id,
            shard_ids=shard_ids,
            resume=resume,
            dry_run=dry_run,
            force=force,
        )

    def collect(
        self,
        input_path: Path,
        *,
        destination: Path,
        num_partitions: int,
    ) -> SlurmCollectionExecution:
        return self._retry_collection.collect(
            input_path,
            destination=destination,
            num_partitions=num_partitions,
        )


class _RegistryImageBackend:
    def __init__(self, workspace_root: str) -> None:
        self._verified = VerifiedImageRegistry(workspace_root)
        self._store = ImageRegistryStore(workspace_root)

    def resolve(self, reference: ImageRef, *, expected_kind: ImageKind) -> ResolvedImage:
        try:
            return self._verified.resolve_for_planning(reference, expected_kind=expected_kind)
        except ImageNotFoundError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.NOT_FOUND,
                SlurmServiceOperation.RESOLVE_IMAGE,
                "image is not registered",
            ) from None
        except SlurmImageError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.CONFLICT,
                SlurmServiceOperation.RESOLVE_IMAGE,
                "registered image cannot be verified",
            ) from None

    def add(self, request: ImageBuildRequest, *, replace: bool) -> RegisteredImage:
        del request, replace
        raise SlurmServiceError(
            SlurmServiceErrorCode.UNAVAILABLE,
            SlurmServiceOperation.ADD_IMAGE,
            "image registration is not available; use a pre-registered image",
        )

    def list(self) -> tuple[RegisteredImage, ...]:
        return self._invoke_registry(SlurmServiceOperation.LIST_IMAGES, self._store.list_images)

    def get(self, name: Identifier) -> RegisteredImage:
        return self._invoke_registry(SlurmServiceOperation.GET_IMAGE, lambda: self._store.get_by_name(name))

    def remove(self, name: Identifier) -> RegisteredImage:
        return self._invoke_registry(SlurmServiceOperation.REMOVE_IMAGE, lambda: self._verified.unregister(name))

    @staticmethod
    def _invoke_registry(operation: SlurmServiceOperation, call: Callable[[], _ResultT]) -> _ResultT:
        try:
            return call()
        except ImageNotFoundError:
            raise SlurmServiceError(SlurmServiceErrorCode.NOT_FOUND, operation, "image is not registered") from None
        except ImageConflictError:
            raise SlurmServiceError(SlurmServiceErrorCode.CONFLICT, operation, "image registry conflict") from None
        except SlurmImageError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.INTERNAL, operation, "image registry operation failed"
            ) from None


def create_slurm_run_service(
    *,
    profile: SlurmProfile | None = None,
    catalog: SlurmProfileCatalog | None = None,
    profile_file: str | Path | None = None,
    cluster: str | None = None,
    artifact_publisher: SlurmRunArtifactPublisher | None = None,
    dependency_resolver: ClientDependencyResolver | None = None,
    launcher: SlurmCommandClient | None = None,
    run_id_factory: RunIdFactory | None = None,
    clock: Clock | None = None,
    package_version: str | None = None,
    source_environment: Mapping[str, str] | None = None,
) -> SlurmRunService:
    """Create the production run service for one selected cluster profile."""
    selected = resolve_profile(profile=profile, catalog=catalog, profile_file=profile_file, cluster=cluster)
    command_client = launcher or SlurmCommandClient()
    selected_clock = clock or _utc_now
    preparer = _RunPreparer(
        selected,
        VerifiedImageRegistry(selected.profile.workspace_root),
        dependency_resolver or ClientDependencyResolver(),
        command_client,
        run_id_factory or _new_run_id,
        package_version or importlib.metadata.version("data-designer-slurm"),
    )
    backend = _SystemRunBackend(
        preparer,
        selected,
        command_client,
        artifact_publisher or StateRunArtifactPublisher(selected.profile.workspace_root, selected_clock),
        selected_clock,
        dict(os.environ if source_environment is None else source_environment),
    )
    return SlurmRunService(_SystemRunPlanner(preparer), render_generation_attempt_script, backend)


def create_slurm_image_service(
    *,
    profile: SlurmProfile | None = None,
    catalog: SlurmProfileCatalog | None = None,
    profile_file: str | Path | None = None,
    cluster: str | None = None,
) -> SlurmImageService:
    """Create the production image service for one selected cluster profile."""
    selected = resolve_profile(profile=profile, catalog=catalog, profile_file=profile_file, cluster=cluster)
    backend = _RegistryImageBackend(selected.profile.workspace_root)
    return SlurmImageService(backend, backend)


def _resolve_builder_payload(
    authored: DataDesignerSlurmConfig,
    source_root: Path,
) -> dict[str, JsonValue] | None:
    if authored.builder.source is None:
        return None
    return load_builder_payload(source_root / authored.builder.source)


def _load_optional(call: Callable[[], _ResultT]) -> _ResultT | None:
    try:
        return call()
    except StateNotFoundError:
        return None


def _new_run_id() -> str:
    return f"run-{uuid4().hex}"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_job_ids(job_ids: list[int]) -> str:
    visible = ", ".join(str(job_id) for job_id in job_ids[:_MAX_VISIBLE_JOB_IDS])
    remaining = len(job_ids) - _MAX_VISIBLE_JOB_IDS
    return visible if remaining <= 0 else f"{visible}, and {remaining} more"


__all__ = [
    "SlurmRunArtifactPublisher",
    "create_slurm_image_service",
    "create_slurm_run_service",
]
