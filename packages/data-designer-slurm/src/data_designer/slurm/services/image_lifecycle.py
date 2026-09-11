# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Production composition for synchronous Slurm image lifecycle jobs."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from data_designer.slurm.config import ImageBuildRequest, SelectedSlurmProfile
from data_designer.slurm.contracts import Identifier
from data_designer.slurm.images.errors import (
    ImageConflictError,
    ImageLifecycleError,
    ImageRegistryError,
    ImageVerificationError,
)
from data_designer.slurm.images.lifecycle import (
    PreparedImageLifecycleJob,
    cleanup_prepared_image_lifecycle,
    prepare_image_lifecycle_job,
    publish_completed_image_lifecycle,
    submit_prepared_image_lifecycle,
)
from data_designer.slurm.images.records import RegisteredImage
from data_designer.slurm.launcher.client import SlurmCommandClient
from data_designer.slurm.launcher.errors import SlurmLauncherError, SlurmSubmissionError
from data_designer.slurm.launcher.models import SlurmAccountingEntry, SlurmQueueEntry
from data_designer.slurm.services.errors import SlurmServiceError, SlurmServiceErrorCode, SlurmServiceOperation
from data_designer.slurm.state import (
    SchedulerJobIdentity,
    SchedulerObservation,
    SchedulerObservationCollector,
    SchedulerState,
    SlurmStateError,
)
from data_designer.slurm.state.scheduler import is_scheduler_failure_state, is_scheduler_terminal_state

LifecycleIdFactory = Callable[[], str]
Clock = Callable[[], datetime]
Sleeper = Callable[[float], None]

_POLL_INTERVAL_SECONDS = 300.0
_ACCOUNTING_EXIT_LAG = timedelta(minutes=5)


class _TerminalLifecycleError(RuntimeError):
    pass


class _RecordingObservationClient:
    def __init__(self, launcher: SlurmCommandClient) -> None:
        self._launcher = launcher
        self.accounting: tuple[SlurmAccountingEntry, ...] = ()

    def query_queue(self, selectors: Sequence[SchedulerJobIdentity]) -> tuple[SlurmQueueEntry, ...]:
        return self._launcher.query_queue(selectors)

    def query_accounting(self, selectors: Sequence[SchedulerJobIdentity]) -> tuple[SlurmAccountingEntry, ...]:
        self.accounting = self._launcher.query_accounting(selectors)
        return self.accounting


class SlurmImageLifecycleManager:
    """Prepare, submit, reconcile, publish, and register one image."""

    def __init__(
        self,
        selected_profile: SelectedSlurmProfile,
        launcher: SlurmCommandClient,
        *,
        lifecycle_id_factory: LifecycleIdFactory | None = None,
        clock: Clock | None = None,
        sleep: Sleeper | None = None,
    ) -> None:
        self._profile = selected_profile
        self._launcher = launcher
        self._lifecycle_id_factory = lifecycle_id_factory or _new_lifecycle_id
        self._clock = clock or _utc_now
        self._sleep = sleep or time.sleep

    def add(self, request: ImageBuildRequest, *, replace: bool) -> RegisteredImage:
        """Run one lifecycle job and publish only a successful verified result."""
        operation = SlurmServiceOperation.ADD_IMAGE
        try:
            prepared = prepare_image_lifecycle_job(
                request,
                self._profile,
                lifecycle_id=self._lifecycle_id_factory(),
            )
        except (ImageLifecycleError, OSError, ValueError):
            raise _unavailable("image lifecycle job cannot be prepared") from None

        try:
            receipt = submit_prepared_image_lifecycle(prepared, self._launcher)
        except SlurmSubmissionError as error:
            if not error.may_have_succeeded:
                _cleanup_failed_lifecycle(prepared)
                raise _unavailable("image lifecycle job cannot be submitted") from None
            raise _unavailable(
                f"image lifecycle submission outcome is unknown; lifecycle {prepared.plan.lifecycle_id} was retained"
            ) from None
        except (ImageLifecycleError, SlurmLauncherError, OSError, ValueError):
            _cleanup_failed_lifecycle(prepared)
            raise _unavailable("image lifecycle job cannot be submitted") from None

        try:
            self._wait_for_success(receipt.job_id)
            return publish_completed_image_lifecycle(prepared, replace=replace)
        except (ImageConflictError, ImageVerificationError):
            raise SlurmServiceError(
                SlurmServiceErrorCode.CONFLICT,
                operation,
                "image lifecycle result cannot be verified or registered",
            ) from None
        except ImageRegistryError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.INTERNAL,
                operation,
                "image registry operation failed",
            ) from None
        except ImageLifecycleError:
            raise _unavailable("image lifecycle result cannot be published") from None
        except _TerminalLifecycleError:
            _cleanup_failed_lifecycle(prepared)
            raise _unavailable(f"image lifecycle job {receipt.job_id} did not complete successfully") from None
        except (SlurmLauncherError, SlurmStateError, OSError, ValueError):
            self._cancel_and_cleanup(receipt.job_id, prepared)
            raise _unavailable(f"image lifecycle job {receipt.job_id} did not complete successfully") from None
        except BaseException:
            self._cancel_and_cleanup(receipt.job_id, prepared)
            raise

    def _wait_for_success(self, job_id: int) -> None:
        client = _RecordingObservationClient(self._launcher)
        observations = SchedulerObservationCollector(client)
        previous: SchedulerObservation | None = None
        completed_without_accounting_deadline: datetime | None = None
        while True:
            observed_at = self._clock()
            observation = observations.collect(
                (job_id,),
                observed_at=observed_at,
                previous={job_id: previous},
            )[0]
            if observation.state is SchedulerState.COMPLETED:
                accounting = next((entry for entry in client.accounting if entry.job_identity == job_id), None)
                if accounting is None or accounting.state is not SchedulerState.COMPLETED:
                    if completed_without_accounting_deadline is None:
                        completed_without_accounting_deadline = observed_at + _ACCOUNTING_EXIT_LAG
                    elif observed_at > completed_without_accounting_deadline:
                        raise SlurmStateError("completed image lifecycle job has no exit evidence")
                    previous = observation
                    self._sleep(_POLL_INTERVAL_SECONDS)
                    continue
                if (
                    accounting.process_exit_code.exit_status != 0
                    or accounting.process_exit_code.termination_signal != 0
                ):
                    raise _TerminalLifecycleError("completed image lifecycle job has no successful exit evidence")
                return
            if is_scheduler_failure_state(observation.state):
                raise _TerminalLifecycleError("image lifecycle job did not complete successfully")
            if observation.state is SchedulerState.UNKNOWN:
                raise SlurmStateError("image lifecycle job has unknown scheduler state")
            previous = observation
            self._sleep(_POLL_INTERVAL_SECONDS)

    def _cancel_and_cleanup(self, job_id: int, prepared: PreparedImageLifecycleJob) -> None:
        try:
            self._launcher.cancel(job_id)
            if not self._wait_for_termination(job_id):
                return
        except (SlurmLauncherError, SlurmStateError, OSError, ValueError):
            return
        _cleanup_failed_lifecycle(prepared)

    def _wait_for_termination(self, job_id: int) -> bool:
        observations = SchedulerObservationCollector(self._launcher)
        previous: SchedulerObservation | None = None
        deadline = self._clock() + _ACCOUNTING_EXIT_LAG
        while True:
            observed_at = self._clock()
            observation = observations.collect(
                (job_id,),
                observed_at=observed_at,
                previous={job_id: previous},
            )[0]
            if is_scheduler_terminal_state(observation.state):
                return True
            if observation.state is SchedulerState.UNKNOWN or observed_at >= deadline:
                return False
            previous = observation
            self._sleep(_POLL_INTERVAL_SECONDS)


def _cleanup_failed_lifecycle(prepared: PreparedImageLifecycleJob) -> None:
    try:
        cleanup_prepared_image_lifecycle(prepared)
    except ImageLifecycleError:
        pass


def _unavailable(message: str) -> SlurmServiceError:
    return SlurmServiceError(SlurmServiceErrorCode.UNAVAILABLE, SlurmServiceOperation.ADD_IMAGE, message)


def _new_lifecycle_id() -> Identifier:
    return f"image-{uuid4().hex}"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = ["Clock", "LifecycleIdFactory", "Sleeper", "SlurmImageLifecycleManager"]
