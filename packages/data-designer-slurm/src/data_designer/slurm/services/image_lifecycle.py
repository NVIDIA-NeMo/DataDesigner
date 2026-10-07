# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Production composition for synchronous Slurm image lifecycle jobs."""

from __future__ import annotations

import codecs
import re
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
    read_image_lifecycle_log,
    submit_prepared_image_lifecycle,
)
from data_designer.slurm.images.records import RegisteredImage
from data_designer.slurm.launcher.client import SlurmCommandClient
from data_designer.slurm.launcher.errors import SlurmLauncherError, SlurmSubmissionError
from data_designer.slurm.launcher.models import SlurmAccountingEntry, SlurmQueueEntry
from data_designer.slurm.security import redact_sensitive_diagnostic
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
ProgressReporter = Callable[[str], None]
LogReporter = Callable[[str], None]

_POLL_INTERVAL_SECONDS = 30.0
_ACCOUNTING_EXIT_LAG = timedelta(minutes=5)
_LOG_READ_BYTES = 64 * 1024
_LOG_TOTAL_BYTES = 1024 * 1024
_LOG_LIMIT_NOTICE = "Image job log display reached its 1 MiB limit; further output is hidden."
_UNSAFE_TERMINAL_CHARACTERS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


class _TerminalLifecycleError(RuntimeError):
    pass


class _JobLogFollower:
    """Best-effort, bounded local output for the exact submitted lifecycle job."""

    def __init__(self, prepared: PreparedImageLifecycleJob, job_id: int, report: LogReporter) -> None:
        self._prepared = prepared
        self._job_id = job_id
        self._report = report
        self._offsets = {"stdout": 0, "stderr": 0}
        self._decoders = {stream: codecs.getincrementaldecoder("utf-8")(errors="replace") for stream in self._offsets}
        self._pending_lines = {"stdout": "", "stderr": ""}
        self._skip_next_lf = {"stdout": False, "stderr": False}
        self._source_bytes = 0
        self._displayed_bytes = 0
        self._unavailable_reported = False
        self._limit_reported = False

    def drain(self, *, final: bool = False) -> None:
        """Print newly appended bytes; after termination, drain the bounded remainder."""
        if self._limit_reported:
            return
        while self._source_bytes < _LOG_TOTAL_BYTES:
            received = False
            for stream in ("stdout", "stderr"):
                allowance = min(_LOG_READ_BYTES, _LOG_TOTAL_BYTES - self._source_bytes)
                if allowance == 0:
                    break
                try:
                    content, offset = read_image_lifecycle_log(
                        self._prepared,
                        self._job_id,
                        stream,
                        offset=self._offsets[stream],
                        maximum_bytes=allowance,
                    )
                except (ImageLifecycleError, OSError, ValueError):
                    if not self._unavailable_reported:
                        self._emit("Image job logs are unavailable.")
                        self._unavailable_reported = True
                    return
                self._offsets[stream] = offset
                self._source_bytes += len(content)
                if content:
                    received = True
                    self._append(stream, content)
                    if self._limit_reported:
                        return
            if not final or not received:
                break
        if final or self._source_bytes >= _LOG_TOTAL_BYTES:
            for stream in ("stdout", "stderr"):
                self._append(stream, b"", final=True)
                if self._pending_lines[stream]:
                    self._emit_line(stream, self._pending_lines[stream])
                    self._pending_lines[stream] = ""
            if self._source_bytes >= _LOG_TOTAL_BYTES:
                self._emit_limit()

    def _append(self, stream: str, content: bytes, *, final: bool = False) -> None:
        text = self._decoders[stream].decode(content, final=final)
        if self._skip_next_lf[stream]:
            if text.startswith("\n"):
                text = text[1:]
            self._skip_next_lf[stream] = False
        parts = re.split(r"(\r\n|\r|\n)", text)
        pending = self._pending_lines[stream]
        for index in range(0, len(parts) - 1, 2):
            self._emit_line(stream, pending + parts[index])
            pending = ""
            if self._limit_reported:
                return
        self._pending_lines[stream] = pending + parts[-1]
        self._skip_next_lf[stream] = len(parts) > 1 and parts[-2] == "\r" and not parts[-1]

    def _emit_line(self, stream: str, line: str) -> None:
        sanitized = redact_sensitive_diagnostic(line)
        self._emit(f"[image {stream}] {_UNSAFE_TERMINAL_CHARACTERS.sub('?', sanitized)}")

    def _emit(self, message: str) -> None:
        if self._limit_reported:
            return
        available = _LOG_TOTAL_BYTES - len(_LOG_LIMIT_NOTICE.encode()) - 1 - self._displayed_bytes
        encoded = message.encode("utf-8")
        if len(encoded) + 1 > available:
            if available > 1:
                self._send(encoded[: available - 1].decode("utf-8", errors="ignore"))
            self._emit_limit()
            return
        self._send(message)

    def _send(self, message: str) -> None:
        self._displayed_bytes += len(message.encode("utf-8")) + 1
        try:
            self._report(message)
        except Exception:
            # Log display must never change a submitted job's outcome.
            pass

    def _emit_limit(self) -> None:
        if not self._limit_reported:
            self._limit_reported = True
            self._send(_LOG_LIMIT_NOTICE)


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
        progress: ProgressReporter | None = None,
        logs: LogReporter | None = None,
    ) -> None:
        self._profile = selected_profile
        self._launcher = launcher
        self._lifecycle_id_factory = lifecycle_id_factory or _new_lifecycle_id
        self._clock = clock or _utc_now
        self._sleep = sleep or time.sleep
        self._progress = progress
        self._logs = logs

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
            self._report_progress("Submitting image inspection job...")
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

        follower = _JobLogFollower(prepared, receipt.job_id, self._logs) if self._logs is not None else None
        try:
            self._report_progress("Image inspection job submitted; checking every 30 seconds.")
            self._wait_for_success(receipt.job_id, follower)
            self._report_progress("Inspection complete; verifying and registering image...")
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
            self._cancel_and_cleanup(receipt.job_id, prepared, follower)
            raise _unavailable(f"image lifecycle job {receipt.job_id} did not complete successfully") from None
        except BaseException:
            self._cancel_and_cleanup(receipt.job_id, prepared, follower)
            raise

    def _wait_for_success(self, job_id: int, follower: _JobLogFollower | None) -> None:
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
            self._report_progress(f"Image inspection job: {observation.state.value.replace('_', ' ')}.")
            if follower is not None:
                follower.drain()
            if observation.state is SchedulerState.COMPLETED:
                accounting = next((entry for entry in client.accounting if entry.job_identity == job_id), None)
                if accounting is None or accounting.state is not SchedulerState.COMPLETED:
                    self._report_progress("Waiting for successful job exit evidence...")
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
                    if follower is not None:
                        follower.drain(final=True)
                    raise _TerminalLifecycleError("completed image lifecycle job has no successful exit evidence")
                if follower is not None:
                    follower.drain(final=True)
                return
            if is_scheduler_failure_state(observation.state):
                if follower is not None:
                    follower.drain(final=True)
                raise _TerminalLifecycleError("image lifecycle job did not complete successfully")
            if observation.state is SchedulerState.UNKNOWN:
                raise SlurmStateError("image lifecycle job has unknown scheduler state")
            previous = observation
            self._sleep(_POLL_INTERVAL_SECONDS)

    def _report_progress(self, message: str) -> None:
        if self._progress is not None:
            try:
                self._progress(message)
            except Exception:
                # A closed progress stream must not cancel an already submitted job.
                pass

    def _cancel_and_cleanup(
        self,
        job_id: int,
        prepared: PreparedImageLifecycleJob,
        follower: _JobLogFollower | None = None,
    ) -> None:
        terminated = False
        try:
            self._report_progress("Cancelling image inspection job...")
            self._launcher.cancel(job_id)
            terminated = self._wait_for_termination(job_id)
        except (SlurmLauncherError, SlurmStateError, OSError, ValueError):
            pass
        if follower is not None:
            follower.drain(final=True)
        if terminated:
            _cleanup_failed_lifecycle(prepared)

    def _wait_for_termination(self, job_id: int) -> bool:
        client = _RecordingObservationClient(self._launcher)
        observations = SchedulerObservationCollector(client)
        previous: SchedulerObservation | None = None
        deadline = self._clock() + _ACCOUNTING_EXIT_LAG
        while True:
            observed_at = self._clock()
            observation = observations.collect(
                (job_id,),
                observed_at=observed_at,
                previous={job_id: previous},
            )[0]
            self._report_progress("Waiting for image inspection job to terminate...")
            accounting = next((entry for entry in client.accounting if entry.job_identity == job_id), None)
            if accounting is not None and is_scheduler_terminal_state(accounting.state):
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
