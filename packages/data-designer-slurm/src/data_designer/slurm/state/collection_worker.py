# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allocation-only worker for one persisted collection plan."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

import data_designer.lazy_heavy_imports as lazy
from data_designer.slurm.contracts import Identifier, validate_absolute_path
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.state.collection_filesystem import derive_collection_staging_directory, stage_collection
from data_designer.slurm.state.collection_inputs import CollectionInputResolver
from data_designer.slurm.state.collection_merge import CollectionMerger
from data_designer.slurm.state.collection_records import CollectionResult, CollectionState, CollectionStatus
from data_designer.slurm.state.collection_storage import CollectionStorage
from data_designer.slurm.state.collection_validation import (
    validate_collection_result,
    validate_collection_status_transition,
)
from data_designer.slurm.state.destinations import CollectionDestinationResolver
from data_designer.slurm.state.errors import SlurmStateError, StateConflictError, StateCorruptionError
from data_designer.slurm.state.outputs import CandidateOutputManifest, CollectionPlan
from data_designer.slurm.state.reader import StateReader
from data_designer.slurm.state.storage import StateStorage

_IDENTIFIER_ADAPTER = TypeAdapter(Identifier)


class SlurmCollectionWorker:
    """Execute bulk collection only inside its recorded CPU Slurm job."""

    def __init__(
        self,
        workspace_root: str | Path,
        run_id: Identifier,
        collection_id: Identifier,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        root, normalized_run_id, normalized_collection_id = _validate_location(workspace_root, run_id, collection_id)
        self._state = StateStorage(root, normalized_run_id)
        self._collections = CollectionStorage(self._state)
        self._reader = StateReader(self._state, normalized_run_id)
        self._inputs = CollectionInputResolver(self._state, self._reader)
        self._destinations = CollectionDestinationResolver()
        self._environment = dict(os.environ if environment is None else environment)
        self._run_id = normalized_run_id
        self._collection_id = normalized_collection_id

    def run(self, *, completed_at: datetime | None = None) -> CollectionResult:
        """Validate, merge, and atomically publish one collection output."""
        started_at = _utc_now() if completed_at is None else completed_at
        try:
            return self._run_locked(started_at, completed_at)
        except (StateConflictError, SlurmStateError):
            raise
        except (OSError, ValueError, lazy.pa.ArrowException) as error:
            raise SlurmStateError(f"collection {self._collection_id!r} failed") from error

    def _run_locked(self, started_at: datetime, completed_at: datetime | None) -> CollectionResult:
        with self._collections.acquire_lock():
            status = self._collections.read_status(self._collection_id)
            plan = self._load_bound_plan(status)
            self._validate_identity(plan.run_id, status.run_id)
            self._require_scheduler_job(status)
            resolved_plan, candidates = self._inputs.resolve(plan)
            self._destinations.validate_persisted(resolved_plan, plan)
            if status.state is CollectionState.SUCCEEDED:
                return self._load_valid_result(plan, status)
            recovered = self._load_optional_result(plan)
            if recovered is not None:
                return self._publish_success_status(plan, status, recovered, recovered.completed_at)
            running = self._advance_status(status, CollectionState.RUNNING, started_at)
            return self._execute_collection(plan, running, resolved_plan, candidates, completed_at)

    def _execute_collection(
        self,
        plan: CollectionPlan,
        running: CollectionStatus,
        resolved_plan: ResolvedSlurmRunPlan,
        candidates: tuple[CandidateOutputManifest, ...],
        completed_at: datetime | None,
    ) -> CollectionResult:
        try:
            destination = self._destinations.validate_persisted(resolved_plan, plan)
            with stage_collection(
                Path(plan.container_destination),
                running.staging_directory,
                Path(destination.mount.target),
            ) as staged:
                merger = CollectionMerger(resolved_plan.output.format, completed_at=completed_at)
                result = merger.merge(
                    plan,
                    candidates,
                    staged,
                )
        except (SlurmStateError, OSError, ValueError, lazy.pa.ArrowException) as error:
            return self._recover_or_fail(plan, running, completed_at, error)
        return self._publish_success_status(plan, running, result, result.completed_at)

    def _recover_or_fail(
        self,
        plan: CollectionPlan,
        running: CollectionStatus,
        completed_at: datetime | None,
        error: Exception,
    ) -> CollectionResult:
        try:
            recovered = self._load_optional_result(plan)
        except (SlurmStateError, OSError, ValueError) as recovery_error:
            self._advance_status(running, CollectionState.FAILED, self._completion_time(completed_at))
            raise recovery_error from error
        if recovered is not None:
            return self._publish_success_status(plan, running, recovered, recovered.completed_at)
        self._advance_status(running, CollectionState.FAILED, self._completion_time(completed_at))
        raise error

    def _load_optional_result(self, plan: CollectionPlan) -> CollectionResult | None:
        try:
            return self._load_valid_result(plan)
        except FileNotFoundError:
            return None

    def _load_valid_result(
        self,
        plan: CollectionPlan,
        status: CollectionStatus | None = None,
    ) -> CollectionResult:
        result = self._collections.read_result_from(plan, Path(plan.container_destination))
        resolved_plan = self._reader.load_resolved_plan()
        validated = validate_collection_result(
            plan,
            result,
            expected_records=resolved_plan.invocation.authored.num_records,
            output_format=resolved_plan.output.format,
        )
        if status is not None and status.result != self._collections.get_result_reference(plan, validated):
            raise StateCorruptionError("collection status does not bind its published result")
        self._collections.verify_result_files(plan, validated, Path(plan.container_destination))
        return validated

    def _publish_success_status(
        self,
        plan: CollectionPlan,
        previous: CollectionStatus,
        result: CollectionResult,
        updated_at: datetime,
    ) -> CollectionResult:
        current = CollectionStatus(
            schema_version=1,
            collection_id=previous.collection_id,
            run_id=previous.run_id,
            collection_plan=previous.collection_plan,
            staging_directory=previous.staging_directory,
            revision=previous.revision + 1,
            updated_at=updated_at,
            state=CollectionState.SUCCEEDED,
            scheduler=previous.scheduler,
            scheduler_observation=previous.scheduler_observation,
            result=self._collections.get_result_reference(plan, result),
        )
        validate_collection_status_transition(previous, current)
        self._collections.replace_status(current)
        return result

    def _advance_status(
        self,
        previous: CollectionStatus,
        state: CollectionState,
        updated_at: datetime,
    ) -> CollectionStatus:
        current = _updated_status(
            previous,
            revision=previous.revision + 1,
            updated_at=updated_at,
            state=state,
        )
        validate_collection_status_transition(previous, current)
        self._collections.replace_status(current)
        return current

    def _require_scheduler_job(self, status: CollectionStatus) -> None:
        scheduler = status.scheduler
        if type(scheduler) is not int:
            raise StateConflictError("collection worker requires an ordinary Slurm job identity")
        observed_job_id = self._environment.get("SLURM_JOB_ID")
        if observed_job_id != str(scheduler):
            raise StateConflictError("collection worker must run inside its recorded Slurm job")

    def _validate_identity(self, plan_run_id: Identifier, status_run_id: Identifier) -> None:
        if plan_run_id != self._run_id or status_run_id != self._run_id:
            raise StateConflictError("collection records do not match the requested run")

    def _load_bound_plan(self, status: CollectionStatus) -> CollectionPlan:
        plan = self._collections.read_plan(status.collection_id)
        if status.collection_plan != self._collections.get_plan_reference(plan):
            raise StateCorruptionError("collection status does not bind its persisted collection plan")
        if status.staging_directory != derive_collection_staging_directory(plan):
            raise StateCorruptionError("collection status does not bind its exact staging directory")
        return plan

    def _completion_time(self, completed_at: datetime | None) -> datetime:
        return _utc_now() if completed_at is None else completed_at


def _updated_status(previous: CollectionStatus, **updates: object) -> CollectionStatus:
    payload = previous.model_dump(mode="python")
    payload.update(updates)
    return CollectionStatus.model_validate(payload)


def _validate_location(
    workspace_root: str | Path,
    run_id: Identifier,
    collection_id: Identifier,
) -> tuple[Path, Identifier, Identifier]:
    try:
        root = validate_absolute_path(Path(workspace_root).as_posix())
        normalized_run_id = _IDENTIFIER_ADAPTER.validate_python(run_id, strict=True)
        normalized_collection_id = _IDENTIFIER_ADAPTER.validate_python(collection_id, strict=True)
    except (ValidationError, ValueError) as error:
        raise SlurmStateError("invalid persisted collection worker location") from error
    return Path(root), normalized_run_id, normalized_collection_id


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one allocation-local collection from explicit persisted identities."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--collection-id", required=True)
    arguments = parser.parse_args(argv)
    SlurmCollectionWorker(
        arguments.workspace_root,
        arguments.run_id,
        arguments.collection_id,
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SlurmCollectionWorker", "main"]
