# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal typed argument-vector client for Slurm command-line tools."""

from __future__ import annotations

import os
import re
import subprocess
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from data_designer.slurm.contracts import Identifier
from data_designer.slurm.launcher.errors import SlurmCommandError, SlurmCommandOutputError, SlurmSubmissionError
from data_designer.slurm.launcher.models import (
    SlurmAccountingEntry,
    SlurmJobSubmissionReceipt,
    SlurmNamedJobEntry,
    SlurmQueueEntry,
    SlurmSubmissionMatch,
)
from data_designer.slurm.launcher.parsing import (
    parse_accounting,
    parse_gpu_counts,
    parse_named_jobs,
    parse_queue,
    parse_submission,
)
from data_designer.slurm.launcher.runner import CommandRunner, SubprocessRunner
from data_designer.slurm.state import SchedulerIdentity, SchedulerJobIdentity

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_SLURM_INTEGER = (1 << 32) - 1


@dataclass(frozen=True)
class SlurmExecutables:
    """Executable paths used for bounded Slurm operations."""

    sbatch: str = "sbatch"
    squeue: str = "squeue"
    sacct: str = "sacct"
    scancel: str = "scancel"
    sinfo: str = "sinfo"

    def __post_init__(self) -> None:
        for executable in (self.sbatch, self.squeue, self.sacct, self.scancel, self.sinfo):
            _validate_argument(executable, field_name="Slurm executable")
            if any(character.isspace() for character in executable):
                raise ValueError("Slurm executable must be one argument-vector token")


class SlurmCommandClient:
    """Submit, observe, and cancel Slurm jobs through structured commands."""

    _executables: SlurmExecutables
    _runner: CommandRunner

    def __init__(
        self,
        runner: CommandRunner | None = None,
        *,
        executables: SlurmExecutables | None = None,
    ) -> None:
        self._runner = runner if runner is not None else SubprocessRunner()
        self._executables = executables if executables is not None else SlurmExecutables()

    def submit(self, script_path: str | Path) -> SlurmJobSubmissionReceipt:
        """Submit one rendered batch script and return its assigned job ID."""
        path = str(script_path)
        _validate_argument(path, field_name="batch script path")
        if path.startswith("-"):
            raise ValueError("batch script path must not begin with '-'; prefix relative paths with './'")
        output = self._run((self._executables.sbatch, "--parsable", "--export=NIL", path))
        return parse_submission(output)

    def submit_script(self, script: str) -> SlurmJobSubmissionReceipt:
        """Submit verified batch-script text through standard input."""
        if type(script) is not str or not script or "\0" in script:
            raise ValueError("batch script text must be non-empty UTF-8 text without NUL")
        try:
            output = self._run(
                (self._executables.sbatch, "--parsable", "--export=NIL"),
                input_text=script,
            )
        except SlurmCommandError as error:
            raise SlurmSubmissionError(
                str(error),
                may_have_succeeded=error.command_may_have_completed,
            ) from error
        try:
            return parse_submission(output)
        except SlurmCommandOutputError as error:
            raise SlurmSubmissionError(str(error), may_have_succeeded=True) from error

    def query_queue(self, selectors: Sequence[SchedulerJobIdentity]) -> tuple[SlurmQueueEntry, ...]:
        """Return normalized active-queue rows for explicit managed jobs."""
        requested = tuple(selectors)
        jobs = _format_selectors(requested)
        output = self._run(
            (
                self._executables.squeue,
                "--noheader",
                "--array",
                "--format=%i|%T",
                f"--jobs={jobs}",
            )
        )
        entries = parse_queue(output)
        ignored = _validate_observed_job_identities(
            tuple(entry.job_identity for entry in entries),
            requested,
            command="squeue",
        )
        return tuple(entry for entry in entries if entry.job_identity not in ignored)

    def query_accounting(self, selectors: Sequence[SchedulerJobIdentity]) -> tuple[SlurmAccountingEntry, ...]:
        """Return normalized accounting rows for explicit managed jobs."""
        requested = tuple(selectors)
        jobs = _format_selectors(requested)
        output = self._run(
            (
                self._executables.sacct,
                "--noheader",
                "--array",
                "--allocations",
                "--parsable2",
                "--format=JobID,State,ExitCode",
                f"--jobs={jobs}",
            )
        )
        entries = parse_accounting(output)
        ignored = _validate_observed_job_identities(
            tuple(entry.job_identity for entry in entries),
            requested,
            command="sacct",
        )
        return tuple(entry for entry in entries if entry.job_identity not in ignored)

    def query_submissions_by_name(
        self,
        job_name: Identifier,
        *,
        submitted_after: datetime,
    ) -> tuple[SlurmSubmissionMatch, ...]:
        """Return current-user allocations matching one exact recovery name."""
        if type(job_name) is not str or _IDENTIFIER_PATTERN.fullmatch(job_name) is None:
            raise ValueError("Slurm job name must be a valid identifier")
        accounting_start = _format_accounting_start(submitted_after)
        queue_output = self._run(
            (
                self._executables.squeue,
                "--noheader",
                "--array",
                "--format=%i|%.128j",
                "--me",
                f"--name={job_name}",
            )
        )
        accounting_output = self._run(
            (
                self._executables.sacct,
                "--noheader",
                "--array",
                "--allocations",
                "--parsable2",
                "--format=JobIDRaw,JobName%128",
                f"--uid={os.getuid()}",
                f"--starttime={accounting_start}",
                f"--name={job_name}",
            )
        )
        entries = (
            *parse_named_jobs(queue_output, command="squeue"),
            *parse_named_jobs(accounting_output, command="sacct"),
        )
        if any(entry.job_name != job_name for entry in entries):
            raise SlurmCommandOutputError("scheduler returned a job outside the requested exact name")
        return _merge_submission_matches(entries)

    def cancel(self, selector: SchedulerJobIdentity) -> None:
        """Cancel one managed Slurm job, array, or array task."""
        self._run((self._executables.scancel, _format_selector(selector)))

    def query_gpu_counts(self, *, partition: Identifier | None = None) -> tuple[int, ...]:
        """Return configured GPU counts reported for eligible node groups."""
        command = [self._executables.sinfo, "--noheader", "--format=%G"]
        if partition is not None:
            if type(partition) is not str or _IDENTIFIER_PATTERN.fullmatch(partition) is None:
                raise ValueError("Slurm partition must be a valid identifier")
            command.append(f"--partition={partition}")
        return parse_gpu_counts(self._run(command))

    def _run(self, command: Sequence[str], *, input_text: str | None = None) -> str:
        command_name = Path(command[0]).name
        try:
            completed = (
                self._runner.run(command) if input_text is None else self._runner.run(command, input_text=input_text)
            )
        except subprocess.TimeoutExpired as error:
            raise SlurmCommandError(
                f"{command_name} could not be executed: {_format_error_detail(error)}",
                command_may_have_completed=True,
            ) from error
        except OSError as error:
            raise SlurmCommandError(f"{command_name} could not be executed: {_format_error_detail(error)}") from error
        except subprocess.SubprocessError as error:
            raise SlurmCommandError(
                f"{command_name} could not be executed: {_format_error_detail(error)}",
                command_may_have_completed=True,
            ) from error
        returncode = getattr(completed, "returncode", None)
        stdout = getattr(completed, "stdout", None)
        stderr = getattr(completed, "stderr", None)
        if type(returncode) is not int or not isinstance(stdout, str) or not isinstance(stderr, str):
            raise SlurmCommandError(
                f"{command_name} returned a malformed process result",
                command_may_have_completed=True,
            )
        if returncode:
            detail = _normalize_bounded_text(stderr) or "no diagnostic output"
            raise SlurmCommandError(f"{command_name} failed with exit code {returncode}: {detail}")
        return stdout


def _format_selectors(selectors: Sequence[SchedulerJobIdentity]) -> str:
    if not selectors:
        raise ValueError("at least one managed Slurm job selector is required")
    return ",".join(dict.fromkeys(_format_selector(selector) for selector in selectors))


def _format_selector(selector: SchedulerJobIdentity) -> str:
    if isinstance(selector, SchedulerIdentity):
        job_id = _format_job_id(selector.array_job_id)
        if selector.array_task_id > _MAX_SLURM_INTEGER:
            raise ValueError("Slurm array-task IDs must be non-negative 32-bit integers")
        return f"{job_id}_{selector.array_task_id}"
    return _format_job_id(selector)


def _format_job_id(value: object) -> str:
    if type(value) is not int or not 0 < value <= _MAX_SLURM_INTEGER:
        raise ValueError("Slurm job IDs must be positive 32-bit integers")
    return str(value)


def _validate_observed_job_identities(
    job_identities: Sequence[SchedulerJobIdentity],
    selectors: Sequence[SchedulerJobIdentity],
    *,
    command: str,
) -> frozenset[SchedulerJobIdentity]:
    """Validate result correlation and return unselected aggregate rows."""
    selected_job_ids = {selector for selector in selectors if type(selector) is int}
    selected_array_tasks = {selector for selector in selectors if isinstance(selector, SchedulerIdentity)}
    selected_array_job_ids = {selector.array_job_id for selector in selected_array_tasks}
    ignored: set[SchedulerJobIdentity] = set()
    for job_identity in job_identities:
        if type(job_identity) is int and job_identity in selected_job_ids:
            continue
        if type(job_identity) is int and job_identity in selected_array_job_ids:
            ignored.add(job_identity)
            continue
        if isinstance(job_identity, SchedulerIdentity) and (
            job_identity in selected_array_tasks or job_identity.array_job_id in selected_job_ids
        ):
            continue
        raise SlurmCommandOutputError(f"{command} returned an unrequested job or array-task ID")
    return frozenset(ignored)


def _validate_argument(value: str, *, field_name: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{field_name} must not be empty")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field_name} must not contain control characters")


def _normalize_bounded_text(value: str, *, limit: int = 512) -> str:
    sanitized = "".join(" " if unicodedata.category(character).startswith("C") else character for character in value)
    normalized = " ".join(sanitized.split())
    return normalized if len(normalized) <= limit else f"{normalized[: limit - 3]}..."


def _format_error_detail(error: BaseException) -> str:
    if isinstance(error, subprocess.TimeoutExpired):
        return "command timed out"
    return _normalize_bounded_text(str(error)) or error.__class__.__name__


def _format_accounting_start(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("submission lookup timestamp must be timezone-aware")
    return (value - timedelta(minutes=1)).astimezone().strftime("%Y-%m-%dT%H:%M:%S")


def _merge_submission_matches(entries: Sequence[SlurmNamedJobEntry]) -> tuple[SlurmSubmissionMatch, ...]:
    grouped: dict[tuple[int, str], set[int]] = {}
    ordinary: set[tuple[int, str]] = set()
    for entry in entries:
        key = (entry.job_id, entry.job_name)
        if entry.array_task_id is None:
            ordinary.add(key)
        else:
            grouped.setdefault(key, set()).add(entry.array_task_id)
    matches: list[SlurmSubmissionMatch] = []
    for job_id, job_name in sorted(ordinary | set(grouped)):
        key = (job_id, job_name)
        task_ids = tuple(sorted(grouped[key])) if key in grouped else None
        matches.append(SlurmSubmissionMatch(job_id=job_id, job_name=job_name, array_task_ids=task_ids))
    return tuple(matches)
