# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process preparation and reconciliation of CPU collection jobs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from pydantic import TypeAdapter, ValidationError

from data_designer.slurm.contracts import Identifier, validate_absolute_path
from data_designer.slurm.launcher.client import SlurmCommandClient
from data_designer.slurm.launcher.collection import render_collection_script
from data_designer.slurm.launcher.errors import SlurmLauncherError, SlurmSubmissionError
from data_designer.slurm.launcher.models import SlurmJobSubmissionReceipt
from data_designer.slurm.state.collection_filesystem import (
    derive_collection_staging_directory,
    prepare_collection_destination,
    remove_collection_stage,
)
from data_designer.slurm.state.collection_inputs import CollectionInputResolver
from data_designer.slurm.state.collection_records import CollectionResult, CollectionState, CollectionStatus
from data_designer.slurm.state.collection_storage import CollectionStorage
from data_designer.slurm.state.collection_validation import (
    derive_collection_state,
    validate_collection_result,
    validate_collection_status_transition,
)
from data_designer.slurm.state.destinations import CollectionDestinationResolver
from data_designer.slurm.state.errors import SlurmStateError, StateConflictError, StateCorruptionError
from data_designer.slurm.state.observation import SchedulerObservationClient, SchedulerObservationCollector
from data_designer.slurm.state.outputs import CollectionPlan
from data_designer.slurm.state.reader import StateReader
from data_designer.slurm.state.scheduler import SchedulerState
from data_designer.slurm.state.storage import StateStorage

_IDENTIFIER_ADAPTER = TypeAdapter(Identifier)


class CollectionScheduler(SchedulerObservationClient, Protocol):
    """Scheduler operations required by collection submission and refresh."""

    def submit_script(self, script: str) -> SlurmJobSubmissionReceipt:
        """Submit one rendered CPU collection job."""
        ...


class SlurmCollectionCoordinator:
    """Persist, submit, and refresh winner-driven collection jobs."""

    def __init__(
        self,
        workspace_root: str | Path,
        run_id: Identifier,
        scheduler: CollectionScheduler | None = None,
    ) -> None:
        root, normalized_run_id = _validate_location(workspace_root, run_id)
        self._scheduler = scheduler if scheduler is not None else SlurmCommandClient()
        self._state = StateStorage(root, normalized_run_id)
        self._reader = StateReader(self._state, normalized_run_id)
        self._collections = CollectionStorage(self._state)
        self._inputs = CollectionInputResolver(self._state, self._reader)
        self._destinations = CollectionDestinationResolver()
        self._collector = SchedulerObservationCollector(self._scheduler)
        self._run_id = normalized_run_id

    def submit(
        self,
        *,
        destination: str | Path | None = None,
        submitted_at: datetime | None = None,
    ) -> CollectionStatus:
        """Validate winners and submit a CPU-only collection job."""
        timestamp = datetime.now(timezone.utc) if submitted_at is None else submitted_at
        try:
            with self._collections.acquire_lock():
                self._collections.discard_incomplete_tail()
                current = self._get_current_status()
                if current is not None:
                    current_plan = self._load_bound_plan(current)
                    if current.state is CollectionState.PREPARED:
                        raise StateConflictError("previous collection submission has an ambiguous scheduler outcome")
                    if current.state is not CollectionState.FAILED:
                        self._validate_existing_destination(current, destination)
                        return current
                    resolved_destination = self._destinations.validate_persisted(
                        self._reader.load_resolved_plan(),
                        current_plan,
                    )
                    remove_collection_stage(
                        Path(current_plan.host_destination),
                        current.staging_directory,
                        Path(resolved_destination.mount.source),
                    )
                run, resolved_plan, _ = self._reader.load_context()
                resolved_destination = self._destinations.resolve(resolved_plan, destination)
                collection_plan = CollectionPlan(
                    schema_version=1,
                    collection_id=self._collections.get_next_collection_id(),
                    run_id=run.run_id,
                    created_at=timestamp,
                    resolved_plan=run.resolved_plan,
                    planned_shards=self._inputs.get_winner_shards(),
                    host_destination=resolved_destination.host_path,
                    container_destination=resolved_destination.container_path,
                    num_partitions=resolved_plan.output.partitions,
                )
                self._inputs.resolve(collection_plan)
                prepare_collection_destination(
                    Path(collection_plan.host_destination),
                    Path(resolved_destination.mount.source),
                )
                self._collections.ensure_collection(collection_plan.collection_id)
                self._collections.publish_plan(collection_plan)
                prepared = CollectionStatus(
                    schema_version=1,
                    collection_id=collection_plan.collection_id,
                    run_id=run.run_id,
                    collection_plan=self._collections.get_plan_reference(collection_plan),
                    staging_directory=derive_collection_staging_directory(collection_plan),
                    revision=1,
                    updated_at=timestamp,
                    state=CollectionState.PREPARED,
                )
                self._collections.publish_status(prepared)
                script = render_collection_script(resolved_plan, collection_plan, resolved_destination)
                return self._submit_prepared(collection_plan, prepared, script, timestamp)
        except (StateConflictError, StateCorruptionError, SlurmStateError):
            raise
        except (OSError, ValidationError, ValueError) as error:
            raise SlurmStateError(f"cannot submit collection for run {self._run_id!r}") from error

    def refresh(
        self,
        *,
        collection_id: Identifier | None = None,
        observed_at: datetime | None = None,
    ) -> CollectionStatus:
        """Reconcile one persisted collection from scheduler and publication evidence."""
        timestamp = datetime.now(timezone.utc) if observed_at is None else observed_at
        try:
            with self._collections.acquire_lock():
                selected_id = self._get_selected_collection_id(collection_id)
                previous = self._collections.read_status(selected_id)
                plan = self._load_bound_plan(previous)
                resolved_plan, _ = self._inputs.resolve(plan)
                destination = self._destinations.validate_persisted(resolved_plan, plan)
                if previous.state is CollectionState.SUCCEEDED:
                    self._load_valid_result(plan, previous)
                    return previous
                recovered = self._load_optional_result(plan)
                if recovered is not None:
                    return self._publish_succeeded(plan, previous, recovered, timestamp)
                if previous.state is CollectionState.FAILED:
                    remove_collection_stage(
                        Path(plan.host_destination),
                        previous.staging_directory,
                        Path(destination.mount.source),
                    )
                    return previous
                if previous.scheduler is None:
                    raise StateConflictError("collection submission has an ambiguous scheduler outcome")
                observations = self._collector.collect(
                    (previous.scheduler,),
                    observed_at=timestamp,
                    previous={previous.scheduler: previous.scheduler_observation},
                )
                observation = observations[0]
                state = derive_collection_state(observation.state)
                if observation.state is SchedulerState.COMPLETED:
                    state = CollectionState.FAILED
                current = _updated_status(
                    previous,
                    revision=previous.revision + 1,
                    updated_at=timestamp,
                    state=state,
                    scheduler_observation=observation,
                )
                validate_collection_status_transition(previous, current)
                self._collections.replace_status(current)
                if current.state is CollectionState.FAILED:
                    remove_collection_stage(
                        Path(plan.host_destination),
                        current.staging_directory,
                        Path(destination.mount.source),
                    )
                return current
        except (StateConflictError, StateCorruptionError, SlurmStateError):
            raise
        except (OSError, ValidationError, ValueError) as error:
            raise SlurmStateError(f"cannot refresh collection for run {self._run_id!r}") from error

    def _submit_prepared(
        self,
        plan: CollectionPlan,
        prepared: CollectionStatus,
        script: str,
        submitted_at: datetime,
    ) -> CollectionStatus:
        try:
            receipt = self._scheduler.submit_script(script)
        except SlurmSubmissionError as error:
            if not error.may_have_succeeded:
                failed = _updated_status(
                    prepared,
                    revision=2,
                    updated_at=submitted_at,
                    state=CollectionState.FAILED,
                )
                validate_collection_status_transition(prepared, failed)
                self._collections.replace_status(failed)
            raise SlurmStateError(f"cannot submit collection {plan.collection_id!r}") from error
        except SlurmLauncherError as error:
            raise SlurmStateError(f"cannot submit collection {plan.collection_id!r}") from error
        submitted = _updated_status(
            prepared,
            revision=2,
            updated_at=submitted_at,
            state=CollectionState.SUBMITTED,
            scheduler=receipt.job_id,
        )
        validate_collection_status_transition(prepared, submitted)
        self._collections.replace_status(submitted)
        return submitted

    def _get_current_status(self) -> CollectionStatus | None:
        collection_ids = self._collections.list_collection_ids()
        return None if not collection_ids else self._collections.read_status(collection_ids[-1])

    def _validate_existing_destination(
        self,
        status: CollectionStatus,
        requested_destination: str | Path | None,
    ) -> None:
        plan = self._load_bound_plan(status)
        resolved_plan = self._reader.load_resolved_plan()
        resolved = self._destinations.validate_persisted(resolved_plan, plan)
        requested = self._destinations.resolve(resolved_plan, requested_destination)
        if requested != resolved:
            raise StateCorruptionError("persisted collection destination does not match the requested destination")
        if plan.host_destination != resolved.host_path or plan.container_destination != resolved.container_path:
            raise StateCorruptionError("persisted collection destination does not match the resolved plan")
        if status.state is CollectionState.SUCCEEDED:
            self._load_valid_result(plan, status)

    def _get_selected_collection_id(self, collection_id: Identifier | None) -> Identifier:
        collection_ids = self._collections.list_collection_ids()
        if not collection_ids:
            raise StateConflictError("run has no persisted collection")
        selected = collection_ids[-1] if collection_id is None else collection_id
        if selected not in collection_ids:
            raise StateConflictError("requested collection is not persisted for this run")
        return selected

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
        self._inputs.resolve(plan)
        result = self._collections.read_result(plan)
        resolved_plan = self._reader.load_resolved_plan()
        validated = validate_collection_result(
            plan,
            result,
            expected_records=resolved_plan.invocation.authored.num_records,
            output_format=resolved_plan.output.format,
        )
        if status is not None and status.result != self._collections.get_result_reference(plan, validated):
            raise StateCorruptionError("collection status does not bind its published result")
        self._collections.verify_result_files(
            plan,
            validated,
            Path(plan.host_destination),
            verify_digests=False,
        )
        return validated

    def _publish_succeeded(
        self,
        plan: CollectionPlan,
        previous: CollectionStatus,
        result: CollectionResult,
        timestamp: datetime,
    ) -> CollectionStatus:
        current = _updated_status(
            previous,
            revision=previous.revision + 1,
            updated_at=timestamp,
            state=CollectionState.SUCCEEDED,
            result=self._collections.get_result_reference(plan, result),
        )
        validate_collection_status_transition(previous, current)
        self._collections.replace_status(current)
        return current

    def _load_bound_plan(self, status: CollectionStatus) -> CollectionPlan:
        plan = self._collections.read_plan(status.collection_id)
        if status.collection_plan != self._collections.get_plan_reference(plan):
            raise StateCorruptionError("collection status does not bind its persisted collection plan")
        if status.staging_directory != derive_collection_staging_directory(plan):
            raise StateCorruptionError("collection status does not bind its exact staging directory")
        return plan


def _validate_location(workspace_root: str | Path, run_id: Identifier) -> tuple[Path, Identifier]:
    try:
        root = validate_absolute_path(Path(workspace_root).as_posix())
        normalized_run_id = _IDENTIFIER_ADAPTER.validate_python(run_id, strict=True)
    except (ValidationError, ValueError) as error:
        raise SlurmStateError("invalid persisted collection location") from error
    return Path(root), normalized_run_id


def _updated_status(previous: CollectionStatus, **updates: object) -> CollectionStatus:
    payload = previous.model_dump(mode="python")
    payload.update(updates)
    return CollectionStatus.model_validate(payload)


__all__ = ["CollectionScheduler", "SlurmCollectionCoordinator"]
