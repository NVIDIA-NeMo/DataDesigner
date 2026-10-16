# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One owner for allocation-local process launch, observation, and cleanup."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from data_designer.slurm.runtime.errors import SlurmRuntimeError, SlurmRuntimeErrorCode
from data_designer.slurm.runtime.logs import open_step_logs
from data_designer.slurm.runtime.models import RuntimeStep
from data_designer.slurm.runtime.signals import TerminationSignalCoordinator


class RuntimeClock(Protocol):
    """Monotonic time and sleep boundary used by runtime polling."""

    def monotonic(self) -> float:
        """Return monotonic seconds."""
        ...

    def now(self) -> datetime:
        """Return timezone-aware UTC wall-clock time."""
        ...

    def sleep(self, seconds: float) -> None:
        """Advance or wait for positive seconds."""
        ...


class SystemRuntimeClock:
    """Production runtime clock."""

    def monotonic(self) -> float:
        """Return system monotonic seconds."""
        return time.monotonic()

    def now(self) -> datetime:
        """Return current UTC wall-clock time."""
        return datetime.now(timezone.utc)

    def sleep(self, seconds: float) -> None:
        """Sleep for the requested duration."""
        time.sleep(seconds)


class StepProcess(Protocol):
    """Minimal process handle required by the allocation supervisor."""

    @property
    def pid(self) -> int:
        """Return the child process identity."""
        ...

    def poll(self) -> int | None:
        """Return the exit status when complete."""
        ...

    def terminate(self) -> None:
        """Request graceful termination."""
        ...

    def kill(self) -> None:
        """Force termination."""
        ...

    def wait(self, timeout: float | None = None) -> int:
        """Wait for completion and return the exit status."""
        ...


class RuntimeStepRunner(Protocol):
    """Start structured runtime steps without a shell."""

    def start(self, step: RuntimeStep) -> StepProcess:
        """Start one process and return its handle."""
        ...


class SubprocessStepRunner:
    """Production process boundary using ``Popen`` with exact argv and environment."""

    def start(self, step: RuntimeStep) -> StepProcess:
        """Start one step with restrictive package-owned log files."""
        with open_step_logs(step) as (stdout_descriptor, stderr_descriptor):
            process = subprocess.Popen(
                step.command,
                stdin=subprocess.DEVNULL,
                stdout=stdout_descriptor,
                stderr=stderr_descriptor,
                env=dict(step.environment),
                shell=False,
                start_new_session=True,
                close_fds=True,
            )
        return _SessionStepProcess(process)


@dataclass(frozen=True, slots=True)
class _SessionStepProcess:
    """Process handle that signals the complete session-owned process group."""

    process: subprocess.Popen[bytes]

    @property
    def pid(self) -> int:
        return self.process.pid

    def poll(self) -> int | None:
        return self.process.poll()

    def terminate(self) -> None:
        self._signal_group(signal.SIGTERM)

    def kill(self) -> None:
        self._signal_group(signal.SIGKILL)

    def wait(self, timeout: float | None = None) -> int:
        return self.process.wait(timeout=timeout)

    def _signal_group(self, selected: signal.Signals) -> None:
        try:
            os.killpg(self.pid, selected)
        except ProcessLookupError:
            if self.poll() is None:
                raise


@dataclass(frozen=True, slots=True)
class ManagedStep:
    """A started structured step and its process identity."""

    step: RuntimeStep
    process: StepProcess


class StepSupervisor:
    """Own every child process and perform idempotent reverse-order cleanup."""

    def __init__(
        self,
        runner: RuntimeStepRunner,
        *,
        signals: TerminationSignalCoordinator,
        clock: RuntimeClock | None = None,
        poll_interval_seconds: float = 0.1,
        termination_grace_seconds: float = 10.0,
    ) -> None:
        if poll_interval_seconds <= 0 or termination_grace_seconds <= 0:
            raise ValueError("runtime polling and termination intervals must be positive")
        self._runner = runner
        self._signals = signals
        self._clock = clock or SystemRuntimeClock()
        self._poll_interval_seconds = poll_interval_seconds
        self._termination_grace_seconds = termination_grace_seconds
        self._managed: list[ManagedStep] = []
        self._cleanup_started = False
        self._cleanup_in_progress = False
        self._cleanup_complete = False

    @property
    def managed_steps(self) -> tuple[ManagedStep, ...]:
        """Return the started process identities in launch order."""
        return tuple(self._managed)

    @property
    def cleanup_complete(self) -> bool:
        """Return whether every registered process is confirmed exited."""
        return self._cleanup_complete

    def start(self, step: RuntimeStep) -> ManagedStep:
        """Start and register one structured step."""
        if self._cleanup_started:
            raise SlurmRuntimeError(SlurmRuntimeErrorCode.CLEANUP_FAILED, "cannot start a step after cleanup")
        try:
            with self._signals.defer_termination():
                managed = ManagedStep(step=step, process=self._runner.start(step))
                self._managed.append(managed)
        except (OSError, subprocess.SubprocessError) as error:
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.STEP_FAILED,
                f"cannot start runtime step {step.step_id!r}",
            ) from error
        return managed

    def wait(self, target: ManagedStep, *, required: tuple[ManagedStep, ...] = ()) -> None:
        """Wait for one finite step while requiring long-running peers to stay alive."""
        while True:
            self.require_running(required)
            returncode = target.process.poll()
            if returncode is not None:
                if returncode != 0:
                    raise SlurmRuntimeError(
                        SlurmRuntimeErrorCode.STEP_FAILED,
                        f"runtime step {target.step.step_id!r} exited with status {returncode}",
                    )
                return
            self._clock.sleep(self._poll_interval_seconds)

    @staticmethod
    def require_running(required: tuple[ManagedStep, ...]) -> None:
        """Fail when any required long-running step has exited."""
        for managed in required:
            returncode = managed.process.poll()
            if returncode is not None:
                raise SlurmRuntimeError(
                    SlurmRuntimeErrorCode.STEP_FAILED,
                    f"required runtime step {managed.step.step_id!r} exited with status {returncode}",
                )

    def cleanup(self) -> None:
        """Terminate every live child exactly once, escalating after a bounded grace period."""
        if self._cleanup_complete or self._cleanup_in_progress:
            return
        self._cleanup_started = True
        self._cleanup_in_progress = True
        try:
            with self._signals.block_termination():
                failures = self._stop_managed_steps()
                self._cleanup_complete = self._are_all_steps_stopped()
        finally:
            self._cleanup_in_progress = False
        if not self._cleanup_complete:
            failures.append(RuntimeError("one or more runtime processes are still running"))
        if failures:
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.CLEANUP_FAILED,
                f"cleanup failed for {len(failures)} runtime process operation(s)",
            ) from failures[0]

    def _stop_managed_steps(self) -> list[BaseException]:
        failures: list[BaseException] = []
        live = tuple(managed for managed in reversed(self._managed) if self._is_running(managed, failures))
        for managed in live:
            try:
                managed.process.terminate()
            except BaseException as error:
                failures.append(error)
        deadline = self._clock.monotonic() + self._termination_grace_seconds
        for managed in live:
            self._stop_step(managed, deadline, failures)
        return failures

    def _stop_step(self, managed: ManagedStep, deadline: float, failures: list[BaseException]) -> None:
        while self._is_running(managed, failures) and self._clock.monotonic() < deadline:
            self._clock.sleep(self._poll_interval_seconds)
        if self._is_running(managed, failures):
            try:
                managed.process.kill()
            except BaseException as error:
                failures.append(error)
        try:
            managed.process.wait(timeout=self._termination_grace_seconds)
        except BaseException as error:
            failures.append(error)

    @staticmethod
    def _is_running(managed: ManagedStep, failures: list[BaseException]) -> bool:
        try:
            return managed.process.poll() is None
        except BaseException as error:
            failures.append(error)
            return True

    def _are_all_steps_stopped(self) -> bool:
        try:
            return all(managed.process.poll() is not None for managed in self._managed)
        except BaseException:
            return False
