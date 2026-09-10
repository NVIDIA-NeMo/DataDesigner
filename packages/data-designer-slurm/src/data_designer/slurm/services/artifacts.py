# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish immutable inputs and initial state for one submitted run."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from pydantic import JsonValue

from data_designer.slurm.client.dependencies import ResolvedClientDependencies
from data_designer.slurm.client.filesystem import ensure_private_directory
from data_designer.slurm.config import DataDesignerSlurmConfig
from data_designer.slurm.contracts import ArtifactReference, pretty_json
from data_designer.slurm.filesystem import create_restrictive_temporary_file, get_file_facts
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.state import (
    AttemptLifecycleState,
    AttemptManifest,
    AttemptTerminalClassification,
    RunManifest,
    SchedulerIdentity,
    ShardManifest,
    SlurmStateError,
    SlurmStateWriter,
    StateConflictError,
    StateNotFoundError,
)
from data_designer.slurm.state.filesystem import (
    open_verified_directory,
    open_verified_regular_file,
    publish_immutable_text,
    sync_directory,
)
from data_designer.slurm.state.storage import StateStorage

_MAXIMUM_RECORD_SIZE = 16 * 1024 * 1024
_TEMPORARY_PREFIX = ".artifact."
_TEMPORARY_SUFFIX = ".tmp"


class StateRunArtifactPublisher:
    """Persist submission inputs through the package-owned state workspace."""

    def __init__(self, workspace_root: str | Path, clock: Callable[[], datetime]) -> None:
        self._workspace_root = Path(workspace_root)
        self._clock = clock

    def initialize(
        self,
        authored: DataDesignerSlurmConfig,
        plan: ResolvedSlurmRunPlan,
        dependencies: ResolvedClientDependencies,
        builder_payload: dict[str, JsonValue] | None,
        *,
        force: bool,
    ) -> None:
        if force:
            raise StateConflictError("force cannot replace durable run state")
        writer = SlurmStateWriter(self._workspace_root, plan.run_id)
        try:
            created_at = writer.load_run().created_at
        except StateNotFoundError:
            try:
                created_at = (
                    StateStorage(self._workspace_root, plan.run_id).read_shard(plan.shards[0].shard_id).created_at
                )
            except FileNotFoundError:
                created_at = self._clock()
        plan_reference = ArtifactReference(
            path=(writer.run_root / "resolved-plan.json").as_posix(),
            sha256=plan.compute_sha256(),
        )
        run = RunManifest(
            schema_version=1,
            run_id=plan.run_id,
            created_at=created_at,
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
                created_at=created_at,
            )
            for shard in plan.shards
        )
        writer.initialize_run(authored, plan, run, shards)
        try:
            self._publish_inputs(writer.run_root, plan, dependencies, builder_payload)
        except SlurmStateError:
            raise
        except Exception as error:
            raise SlurmStateError(f"cannot publish immutable inputs for run {plan.run_id!r}") from error

    def record_submission(self, plan: ResolvedSlurmRunPlan, job_id: int, *, submitted_at: datetime) -> None:
        writer = SlurmStateWriter(self._workspace_root, plan.run_id)
        plan_reference = ArtifactReference(
            path=(writer.run_root / "resolved-plan.json").as_posix(),
            sha256=plan.compute_sha256(),
        )
        for shard in plan.shards:
            writer.create_attempt(
                AttemptManifest(
                    schema_version=1,
                    run_id=plan.run_id,
                    shard_id=shard.shard_id,
                    attempt_id="attempt-0001",
                    attempt_ordinal=1,
                    resolved_plan=plan_reference,
                    state=AttemptLifecycleState.SUBMITTED,
                    scheduler=SchedulerIdentity(array_job_id=job_id, array_task_id=shard.array_task_index),
                    created_at=submitted_at,
                    updated_at=submitted_at,
                )
            )

    def record_submission_failure(self, plan: ResolvedSlurmRunPlan, *, failed_at: datetime) -> None:
        """Mark every initial attempt failed after its held job is cancelled."""
        writer = SlurmStateWriter(self._workspace_root, plan.run_id)
        for shard in plan.shards:
            try:
                attempt = writer.load_attempt(shard.shard_id, "attempt-0001")
            except StateNotFoundError:
                continue
            writer.update_attempt(
                attempt.model_copy(
                    update={
                        "state": AttemptLifecycleState.FAILED,
                        "terminal_classification": AttemptTerminalClassification.CANCELLED,
                        "updated_at": failed_at,
                    }
                )
            )

    @staticmethod
    def _publish_inputs(
        run_root: Path,
        plan: ResolvedSlurmRunPlan,
        dependencies: ResolvedClientDependencies,
        builder_payload: dict[str, JsonValue] | None,
    ) -> None:
        _publish_text(run_root, plan.client.dependency_lock, dependencies.lock.serialize_json())
        if (plan.builder.source is None) != (builder_payload is None):
            raise SlurmStateError("resolved builder source does not match its staged payload")
        if plan.builder.source is not None:
            assert builder_payload is not None
            _publish_text(run_root, plan.builder.source, pretty_json(builder_payload))
        if (dependencies.lock.source is None) != (dependencies.lock_source is None):
            raise SlurmStateError("dependency lock source does not match its staged artifact")
        if dependencies.lock.source is not None:
            assert dependencies.lock_source is not None
            _publish_file(run_root, dependencies.lock.source, dependencies.lock_source)
        packages = dependencies.lock.overlay_packages
        if len(packages) != len(dependencies.wheel_sources):
            raise SlurmStateError("dependency wheel sources do not match the resolved lock")
        for package, source in zip(packages, dependencies.wheel_sources, strict=True):
            _publish_file(run_root, package.artifact, source)
        seed_path = plan.invocation.effective_input_bindings.seed_path
        for shard in plan.shards:
            if shard.input_partition is None:
                continue
            _publish_text(
                run_root,
                shard.input_partition,
                pretty_json(
                    {
                        "record_range": shard.record_range.model_dump(mode="json"),
                        "seed_path": seed_path,
                    }
                ),
            )


def _publish_text(run_root: Path, reference: ArtifactReference, content: str) -> None:
    target = _validate_target(run_root, reference)
    if hashlib.sha256(content.encode()).hexdigest() != reference.sha256:
        raise SlurmStateError(f"staged artifact {target.name!r} does not match its resolved digest")
    ensure_private_directory(target.parent)
    with open_verified_directory(target.parent, require_private=True) as descriptor:
        publish_immutable_text(
            descriptor,
            target.name,
            content,
            target,
            maximum_size=_MAXIMUM_RECORD_SIZE,
        )


def _publish_file(run_root: Path, reference: ArtifactReference, source: Path) -> None:
    target = _validate_target(run_root, reference)
    ensure_private_directory(target.parent)
    source_descriptor: int | None = None
    try:
        source_before = source.lstat()
        source_descriptor = os.open(
            source,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        source_opened = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_opened.st_mode) or get_file_facts(source_before) != get_file_facts(source_opened):
            raise OSError(f"source artifact {source} is not a stable regular file")
        with open_verified_directory(target.parent, require_private=True) as target_descriptor:
            output_descriptor, temporary_name = create_restrictive_temporary_file(
                target_descriptor,
                prefix=_TEMPORARY_PREFIX,
                suffix=_TEMPORARY_SUFFIX,
            )
            try:
                digest = hashlib.sha256()
                try:
                    while chunk := os.read(source_descriptor, 1024 * 1024):
                        digest.update(chunk)
                        remaining = memoryview(chunk)
                        while remaining:
                            written = os.write(output_descriptor, remaining)
                            if written == 0:
                                raise OSError("artifact copy made no progress")
                            remaining = remaining[written:]
                    os.fsync(output_descriptor)
                finally:
                    os.close(output_descriptor)
                source_after = os.fstat(source_descriptor)
                source_current = source.lstat()
                if (
                    get_file_facts(source_opened) != get_file_facts(source_after)
                    or get_file_facts(source_after) != get_file_facts(source_current)
                    or digest.hexdigest() != reference.sha256
                ):
                    raise OSError(f"source artifact {source} changed or has an unexpected digest")
                try:
                    os.link(
                        temporary_name,
                        target.name,
                        src_dir_fd=target_descriptor,
                        dst_dir_fd=target_descriptor,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    with open_verified_regular_file(
                        target_descriptor,
                        target.name,
                        target,
                        expected_size=source_after.st_size,
                        expected_sha256=reference.sha256,
                    ):
                        pass
                os.unlink(temporary_name, dir_fd=target_descriptor)
                temporary_name = None
                sync_directory(target_descriptor)
            finally:
                if temporary_name is not None:
                    try:
                        os.unlink(temporary_name, dir_fd=target_descriptor)
                    except OSError:
                        pass
    finally:
        if source_descriptor is not None:
            os.close(source_descriptor)


def _validate_target(run_root: Path, reference: ArtifactReference) -> Path:
    target = Path(reference.path)
    try:
        target.relative_to(run_root)
    except ValueError:
        raise SlurmStateError("resolved artifact path is outside the run workspace") from None
    return target


__all__ = ["StateRunArtifactPublisher"]
