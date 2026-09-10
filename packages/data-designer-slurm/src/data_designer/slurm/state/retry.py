# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process failed-shard retry orchestration."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from pydantic import TypeAdapter, ValidationError

from data_designer.slurm.contracts import Identifier, ShardId, validate_absolute_path
from data_designer.slurm.launcher.client import SlurmCommandClient
from data_designer.slurm.launcher.errors import SlurmLauncherError, SlurmSubmissionError
from data_designer.slurm.launcher.models import SlurmJobSubmissionReceipt, SlurmSubmissionMatch
from data_designer.slurm.launcher.renderer import render_generation_retry_script
from data_designer.slurm.state.base import SchedulerIdentity
from data_designer.slurm.state.errors import SlurmStateError, StateConflictError, StateCorruptionError
from data_designer.slurm.state.execution import AttemptLifecycleState, AttemptManifest
from data_designer.slurm.state.finalization import WinnerFinalizer
from data_designer.slurm.state.observation import SchedulerObservationClient
from data_designer.slurm.state.observer import SlurmStateReconciler
from data_designer.slurm.state.outputs import RetryPlan, RetryShard
from data_designer.slurm.state.reader import StateReader
from data_designer.slurm.state.retry_records import RetryState, RetryStatus, validate_retry_status_transition
from data_designer.slurm.state.retry_storage import RetryStorage
from data_designer.slurm.state.scheduler import EffectiveAttemptState
from data_designer.slurm.state.status import RunStatus, ShardStatus
from data_designer.slurm.state.storage import StateStorage
from data_designer.slurm.state.submission_recovery import (
    SUBMISSION_VISIBILITY_WINDOW,
    PreparedSubmission,
    resolve_prepared_submission,
)
from data_designer.slurm.state.validation import (
    StateContractError,
    validate_attempt_transition,
    validate_shard_attempt_set,
)

_IDENTIFIER_ADAPTER = TypeAdapter(Identifier)


class RetryScheduler(SchedulerObservationClient, Protocol):
    """Scheduler operations required by fresh-process retry."""

    def submit_script(self, script: str) -> SlurmJobSubmissionReceipt:
        """Submit one rendered retry array."""
        ...

    def query_submissions_by_name(
        self,
        job_name: Identifier,
        *,
        submitted_after: datetime,
    ) -> tuple[SlurmSubmissionMatch, ...]:
        """Return allocations matching one exact retry submission name."""
        ...


class SlurmRetryCoordinator:
    """Select retryable shards and durably submit their next attempts."""

    def __init__(
        self,
        workspace_root: str | Path,
        run_id: Identifier,
        scheduler: RetryScheduler | None = None,
    ) -> None:
        root, normalized_run_id = _validate_location(workspace_root, run_id)
        self._scheduler = scheduler if scheduler is not None else SlurmCommandClient()
        self._state = StateStorage(root, normalized_run_id)
        self._reader = StateReader(self._state, normalized_run_id)
        self._retries = RetryStorage(self._state)
        self._finalizer = WinnerFinalizer(self._state, self._reader)
        self._reconciler = SlurmStateReconciler(root, normalized_run_id, self._scheduler)
        self._run_id = normalized_run_id

    def retry(
        self,
        *,
        shard_ids: Sequence[ShardId] | None = None,
        effective_resume_mode: Literal["never", "always"],
        observed_at: datetime | None = None,
    ) -> tuple[AttemptManifest, ...]:
        """Refresh state, submit exactly the retryable selection, and publish attempts."""
        timestamp = datetime.now(timezone.utc) if observed_at is None else observed_at
        try:
            if effective_resume_mode not in {"never", "always"}:
                raise StateConflictError("effective resume mode must be 'never' or 'always'")
            with self._retries.acquire_lock():
                self._retries.discard_incomplete_tail()
                self._settle_pending_retry(timestamp)
                status = self._reconciler.refresh(observed_at=timestamp)
                active = self._load_active_retry(status, shard_ids, effective_resume_mode)
                if active is not None:
                    return active
                selected = _select_retryable_shards(status, shard_ids)
                plan = self._build_retry_plan(status, selected, effective_resume_mode, timestamp)
                script = render_generation_retry_script(self._reader.load_resolved_plan(status.run), plan)
                with self._acquire_selection_locks(selected):
                    self._require_fresh_selection(status, selected)
                    self._persist_prepared_retry(plan, timestamp)
                    receipt = self._submit(plan, script, timestamp)
                    attempts, superseded = self._publish_attempts(plan, receipt.job_id, timestamp)
                    self._settle_retry(plan, receipt.job_id, timestamp, superseded=superseded)
                    return attempts
        except (StateConflictError, StateCorruptionError, SlurmStateError):
            raise
        except (OSError, ValidationError, ValueError) as error:
            raise SlurmStateError(f"cannot retry persisted run {self._run_id!r}") from error

    def _settle_pending_retry(self, updated_at: datetime) -> None:
        retry_ids = self._retries.list_retry_ids()
        if not retry_ids:
            return
        latest_id = retry_ids[-1]
        status = self._retries.read_status(latest_id)
        plan = self._load_bound_plan(status)
        if status.state is RetryState.PREPARED:
            status = self._reconcile_prepared_retry(plan, status, updated_at)
        if status.state is not RetryState.SUBMITTED:
            return
        assert status.array_job_id is not None
        shard_ids = tuple(shard.shard_id for shard in plan.planned_shards)
        run_status = self._reconciler.refresh(observed_at=updated_at)
        selected = tuple(_get_shard_status(run_status, shard_id) for shard_id in shard_ids)
        with self._acquire_selection_locks(selected):
            _, superseded = self._publish_attempts(plan, status.array_job_id, plan.created_at)
            self._settle_retry(plan, status.array_job_id, updated_at, superseded=superseded)

    def _reconcile_prepared_retry(
        self,
        plan: RetryPlan,
        status: RetryStatus,
        updated_at: datetime,
    ) -> RetryStatus:
        assert status.reconciliation_deadline is not None
        job_id = resolve_prepared_submission(
            self._scheduler,
            PreparedSubmission(
                job_name=plan.submission_job_name,
                submitted_after=plan.created_at,
                reconciliation_deadline=status.reconciliation_deadline,
                expected_array_task_ids=tuple(shard.array_task_index for shard in plan.planned_shards),
            ),
            observed_at=updated_at,
        )
        if job_id is not None:
            return self._publish_submitted_status(plan, job_id, updated_at)
        self._fail_retry(plan, updated_at)
        return self._retries.read_status(plan.retry_id)

    def _build_retry_plan(
        self,
        status: RunStatus,
        selected: tuple[ShardStatus, ...],
        effective_resume_mode: Literal["never", "always"],
        created_at: datetime,
    ) -> RetryPlan:
        plan = self._reader.load_resolved_plan(status.run)
        requested_resume = plan.invocation.authored.resume
        if requested_resume != "if_possible" and effective_resume_mode != requested_resume:
            raise StateConflictError("effective resume mode does not match the pinned resolved plan")
        return RetryPlan(
            schema_version=1,
            retry_id=self._retries.get_next_retry_id(),
            run_id=status.run.run_id,
            created_at=created_at,
            resolved_plan=status.run.resolved_plan,
            planned_shards=tuple(
                RetryShard(
                    shard_id=shard_status.shard.shard_id,
                    attempt_id=f"attempt-{len(shard_status.attempts) + 1:04d}",
                    attempt_ordinal=len(shard_status.attempts) + 1,
                    array_task_index=plan.shards[shard_status.shard.shard_index].array_task_index,
                )
                for shard_status in selected
            ),
            effective_resume_mode=effective_resume_mode,
        )

    def _load_active_retry(
        self,
        status: RunStatus,
        requested_shard_ids: Sequence[ShardId] | None,
        effective_resume_mode: Literal["never", "always"],
    ) -> tuple[AttemptManifest, ...] | None:
        retry_ids = self._retries.list_retry_ids()
        if not retry_ids:
            return None
        retry_status = self._retries.read_status(retry_ids[-1])
        retry_plan = self._load_bound_plan(retry_status)
        if retry_status.state is not RetryState.COMPLETED:
            return None
        planned_ids = tuple(shard.shard_id for shard in retry_plan.planned_shards)
        if requested_shard_ids is None and any(
            _is_retryable(shard) and shard.shard.shard_id not in planned_ids for shard in status.shards
        ):
            return None
        if retry_plan.effective_resume_mode != effective_resume_mode or not self._matches_requested_shards(
            requested_shard_ids, planned_ids
        ):
            return None
        return self._get_active_attempts(status, retry_plan)

    @staticmethod
    def _matches_requested_shards(
        requested_shard_ids: Sequence[ShardId] | None,
        planned_ids: tuple[ShardId, ...],
    ) -> bool:
        if requested_shard_ids is None:
            return True
        requested = tuple(requested_shard_ids)
        return len(requested) == len(set(requested)) and set(requested) == set(planned_ids)

    @staticmethod
    def _get_active_attempts(status: RunStatus, retry_plan: RetryPlan) -> tuple[AttemptManifest, ...] | None:
        statuses_by_shard = {shard.shard.shard_id: shard for shard in status.shards}
        attempts: list[AttemptManifest] = []
        for planned in retry_plan.planned_shards:
            shard_status = statuses_by_shard.get(planned.shard_id)
            if shard_status is None or not shard_status.attempts:
                return None
            matching = next(
                (item for item in shard_status.attempts if item.attempt.attempt_id == planned.attempt_id),
                None,
            )
            if shard_status.winner is not None:
                if (
                    shard_status.winner.attempt_id != planned.attempt_id
                    or matching is None
                    or matching.effective_state is not EffectiveAttemptState.SUCCEEDED
                ):
                    return None
                attempts.append(matching.attempt)
                continue
            latest = shard_status.attempts[-1]
            if latest.attempt.attempt_id != planned.attempt_id or latest.effective_state not in {
                EffectiveAttemptState.PENDING,
                EffectiveAttemptState.RUNNING,
                EffectiveAttemptState.ACCOUNTING_LAG,
            }:
                return None
            attempts.append(latest.attempt)
        return tuple(attempts)

    @contextmanager
    def _acquire_selection_locks(self, selected: tuple[ShardStatus, ...]) -> Iterator[None]:
        with ExitStack() as resources:
            for shard_status in selected:
                resources.enter_context(self._state.acquire_resume_and_shard_locks(shard_status.shard.shard_id))
            yield

    def _require_fresh_selection(self, status: RunStatus, selected: tuple[ShardStatus, ...]) -> None:
        for shard_status in selected:
            run, plan, shard = self._reader.load_shard_context(shard_status.shard.shard_id)
            attempts = self._reader.load_validated_shard_attempts(run, plan, shard)
            winner = self._finalizer.load_optional_winner(run, plan, shard, attempts)
            expected_attempts = tuple(attempt_status.attempt for attempt_status in shard_status.attempts)
            expected_observations = tuple(attempt_status.scheduler for attempt_status in shard_status.attempts)
            current_observations = tuple(
                self._reader.load_optional_scheduler_observation(attempt) for attempt in attempts
            )
            if run != status.run or shard != shard_status.shard or attempts != expected_attempts:
                raise StateConflictError("persisted shard changed after retry reconciliation; retry again")
            if current_observations != expected_observations:
                raise StateConflictError("scheduler evidence changed after retry reconciliation; retry again")
            if winner is not None:
                raise StateConflictError(f"shard {shard.shard_id!r} already has an immutable winner")

    def _persist_prepared_retry(self, plan: RetryPlan, timestamp: datetime) -> None:
        self._retries.ensure_retry(plan.retry_id)
        self._retries.publish_plan(plan)
        self._retries.publish_status(
            RetryStatus(
                schema_version=1,
                retry_id=plan.retry_id,
                run_id=plan.run_id,
                retry_plan=self._retries.get_plan_reference(plan),
                revision=1,
                updated_at=timestamp,
                state=RetryState.PREPARED,
                reconciliation_deadline=timestamp + SUBMISSION_VISIBILITY_WINDOW,
            )
        )

    def _submit(self, plan: RetryPlan, script: str, timestamp: datetime) -> SlurmJobSubmissionReceipt:
        try:
            receipt = self._scheduler.submit_script(script)
        except SlurmSubmissionError as error:
            if not error.may_have_succeeded:
                self._fail_retry(plan, timestamp)
            raise SlurmStateError(f"cannot submit retry {plan.retry_id!r}") from error
        except SlurmLauncherError as error:
            raise SlurmStateError(f"cannot submit retry {plan.retry_id!r}") from error
        self._publish_submitted_status(plan, receipt.job_id, timestamp)
        return receipt

    def _publish_submitted_status(self, plan: RetryPlan, array_job_id: int, timestamp: datetime) -> RetryStatus:
        plan_reference = self._retries.get_plan_reference(plan)
        submitted = RetryStatus(
            schema_version=1,
            retry_id=plan.retry_id,
            run_id=plan.run_id,
            retry_plan=plan_reference,
            revision=2,
            updated_at=timestamp,
            state=RetryState.SUBMITTED,
            array_job_id=array_job_id,
        )
        validate_retry_status_transition(self._retries.read_status(plan.retry_id), submitted)
        self._retries.replace_status(submitted)
        return submitted

    def _publish_attempts(
        self,
        retry_plan: RetryPlan,
        array_job_id: int,
        timestamp: datetime,
    ) -> tuple[tuple[AttemptManifest, ...], bool]:
        run = self._reader.load_run()
        plan = self._reader.load_resolved_plan(run)
        if retry_plan.run_id != run.run_id or retry_plan.resolved_plan != run.resolved_plan:
            raise StateConflictError("retry plan does not bind the current persisted run")
        prepared: list[tuple[AttemptManifest, bool]] = []
        superseded = False
        for selected in retry_plan.planned_shards:
            shard = self._reader.load_shard_context(selected.shard_id)[2]
            attempts = self._reader.load_validated_shard_attempts(run, plan, shard)
            winner = self._finalizer.load_optional_winner(run, plan, shard, attempts)
            attempt = AttemptManifest(
                schema_version=1,
                run_id=run.run_id,
                shard_id=selected.shard_id,
                attempt_id=selected.attempt_id,
                attempt_ordinal=selected.attempt_ordinal,
                resolved_plan=run.resolved_plan,
                state=AttemptLifecycleState.SUBMITTED,
                scheduler=SchedulerIdentity(
                    array_job_id=array_job_id,
                    array_task_id=selected.array_task_index,
                ),
                created_at=timestamp,
                updated_at=timestamp,
            )
            existing = next((item for item in attempts if item.attempt_id == selected.attempt_id), None)
            if existing is not None:
                try:
                    validate_attempt_transition(attempt, existing)
                except StateContractError as error:
                    raise StateConflictError(
                        f"retry attempt {selected.attempt_id!r} contains incompatible state"
                    ) from error
                if winner is not None and winner.attempt_id != selected.attempt_id:
                    superseded = True
                    continue
                prepared.append((existing, False))
                continue
            if winner is not None:
                superseded = True
                continue
            self._finalizer.require_no_winner(run, plan, shard, attempts)
            if selected.attempt_ordinal != len(attempts) + 1:
                raise StateConflictError("retry attempt ordinal is no longer next for its shard")
            self._reader.validate_attempt_against_plan(run, plan, shard, attempt)
            validate_shard_attempt_set(run, shard, attempts + (attempt,))
            prepared.append((attempt, True))
        published: list[AttemptManifest] = []
        for attempt, requires_publication in prepared:
            if requires_publication:
                self._state.publish_attempt(attempt)
            published.append(attempt)
        return tuple(published), superseded

    def _settle_retry(
        self,
        plan: RetryPlan,
        array_job_id: int,
        timestamp: datetime,
        *,
        superseded: bool,
    ) -> None:
        if superseded:
            self._fail_retry(plan, timestamp, array_job_id=array_job_id)
        else:
            self._complete_retry(plan, array_job_id, timestamp)

    def _fail_retry(self, plan: RetryPlan, timestamp: datetime, *, array_job_id: int | None = None) -> None:
        previous = self._retries.read_status(plan.retry_id)
        failed = RetryStatus(
            schema_version=1,
            retry_id=plan.retry_id,
            run_id=plan.run_id,
            retry_plan=self._retries.get_plan_reference(plan),
            revision=previous.revision + 1,
            updated_at=timestamp,
            state=RetryState.FAILED,
            array_job_id=array_job_id,
        )
        validate_retry_status_transition(previous, failed)
        self._retries.replace_status(failed)

    def _complete_retry(self, plan: RetryPlan, array_job_id: int, timestamp: datetime) -> None:
        previous = self._retries.read_status(plan.retry_id)
        if previous.state is RetryState.COMPLETED:
            return
        completed = RetryStatus(
            schema_version=1,
            retry_id=plan.retry_id,
            run_id=plan.run_id,
            retry_plan=self._retries.get_plan_reference(plan),
            revision=previous.revision + 1,
            updated_at=timestamp,
            state=RetryState.COMPLETED,
            array_job_id=array_job_id,
        )
        validate_retry_status_transition(previous, completed)
        self._retries.replace_status(completed)

    def _load_bound_plan(self, status: RetryStatus) -> RetryPlan:
        plan = self._retries.read_plan(status.retry_id)
        if status.retry_plan != self._retries.get_plan_reference(plan):
            raise StateCorruptionError("retry status does not bind its persisted retry plan")
        return plan


def _select_retryable_shards(
    status: RunStatus,
    shard_ids: Sequence[ShardId] | None,
) -> tuple[ShardStatus, ...]:
    requested = None if shard_ids is None else tuple(shard_ids)
    if requested is not None and len(requested) != len(set(requested)):
        raise StateConflictError("explicit retry shard IDs must be unique")
    known = {shard.shard.shard_id for shard in status.shards}
    if requested is not None and not set(requested).issubset(known):
        raise StateConflictError("explicit retry selection contains an unknown shard")
    selected = tuple(
        shard
        for shard in status.shards
        if (requested is None or shard.shard.shard_id in requested) and _is_retryable(shard)
    )
    expected_count = len(
        tuple(shard for shard in status.shards if requested is None or shard.shard.shard_id in requested)
    )
    if requested is not None and len(selected) != expected_count:
        raise StateConflictError("explicit retry selection includes a sealed or nonterminal shard")
    if not selected:
        raise StateConflictError("run has no retryable shards")
    return selected


def _is_retryable(shard: ShardStatus) -> bool:
    return (
        shard.winner is None
        and bool(shard.attempts)
        and shard.attempts[-1].effective_state in {EffectiveAttemptState.FAILED, EffectiveAttemptState.UNKNOWN}
    )


def _get_shard_status(status: RunStatus, shard_id: ShardId) -> ShardStatus:
    shard_status = next((shard for shard in status.shards if shard.shard.shard_id == shard_id), None)
    if shard_status is None:
        raise StateCorruptionError(f"retry plan references unknown shard {shard_id!r}")
    return shard_status


def _validate_location(workspace_root: str | Path, run_id: Identifier) -> tuple[Path, Identifier]:
    try:
        root = validate_absolute_path(Path(workspace_root).as_posix())
        normalized_run_id = _IDENTIFIER_ADAPTER.validate_python(run_id, strict=True)
    except (ValidationError, ValueError) as error:
        raise SlurmStateError("invalid persisted retry location") from error
    return Path(root), normalized_run_id


__all__ = ["RetryScheduler", "SlurmRetryCoordinator"]
