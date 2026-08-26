# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed argument-vector client for Slurm command-line tools."""

from __future__ import annotations

import re
import subprocess
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from data_designer.slurm.contracts import Identifier
from data_designer.slurm.launcher.errors import SlurmCommandError, SlurmParseError
from data_designer.slurm.launcher.models import AccountingRecord, QueueRecord, SlurmJobIdentity, SlurmSubmission
from data_designer.slurm.launcher.parsing import (
    parse_accounting,
    parse_gpu_counts,
    parse_queue,
    parse_submission,
)
from data_designer.slurm.launcher.runner import CommandRunner, SubprocessRunner
from data_designer.slurm.state import SchedulerIdentity

JobSelector: TypeAlias = SlurmJobIdentity
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_SLURM_INTEGER = (1 << 32) - 1


@dataclass(frozen=True, slots=True)
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

    def submit(self, script_path: str | Path) -> SlurmSubmission:
        """Submit one rendered batch script and return its assigned job ID."""
        path = str(script_path)
        _validate_argument(path, field_name="batch script path")
        if path.startswith("-"):
            raise ValueError("batch script path must not begin with '-'; prefix relative paths with './'")
        output = self._run((self._executables.sbatch, "--parsable", "--export=NIL", path))
        return parse_submission(output)

    def query_queue(self, selectors: Sequence[JobSelector]) -> tuple[QueueRecord, ...]:
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
        records = parse_queue(output)
        ignored = _validate_selected_schedulers(
            tuple(record.scheduler for record in records),
            requested,
            command="squeue",
        )
        return tuple(record for record in records if record.scheduler not in ignored)

    def query_accounting(self, selectors: Sequence[JobSelector]) -> tuple[AccountingRecord, ...]:
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
        records = parse_accounting(output)
        ignored = _validate_selected_schedulers(
            tuple(record.scheduler for record in records),
            requested,
            command="sacct",
        )
        return tuple(record for record in records if record.scheduler not in ignored)

    def cancel(self, selector: JobSelector) -> None:
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

    def _run(self, command: Sequence[str]) -> str:
        command_name = Path(command[0]).name
        try:
            completed = self._runner.run(command)
        except (OSError, subprocess.SubprocessError) as error:
            raise SlurmCommandError(f"{command_name} could not be executed: {_format_error_detail(error)}") from error
        if completed.returncode:
            detail = _normalize_bounded_text(completed.stderr) or "no diagnostic output"
            raise SlurmCommandError(f"{command_name} failed with exit code {completed.returncode}: {detail}")
        if not isinstance(completed.stdout, str):
            raise SlurmCommandError(f"{command_name} did not return text output")
        return completed.stdout


def _format_selectors(selectors: Sequence[JobSelector]) -> str:
    if not selectors:
        raise ValueError("at least one managed Slurm job selector is required")
    return ",".join(dict.fromkeys(_format_selector(selector) for selector in selectors))


def _format_selector(selector: JobSelector) -> str:
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


def _validate_selected_schedulers(
    schedulers: Sequence[SlurmJobIdentity],
    selectors: Sequence[JobSelector],
    *,
    command: str,
) -> frozenset[SlurmJobIdentity]:
    """Validate result correlation and identify aggregate rows to omit."""
    ignored: set[SlurmJobIdentity] = set()
    for scheduler in schedulers:
        explicitly_selected = any(type(selector) is int and selector == scheduler for selector in selectors)
        is_array_parent = type(scheduler) is int and any(
            isinstance(selector, SchedulerIdentity) and selector.array_job_id == scheduler for selector in selectors
        )
        if is_array_parent and not explicitly_selected:
            ignored.add(scheduler)
            continue
        if any(_selector_matches(scheduler, selector) for selector in selectors):
            continue
        raise SlurmParseError(f"{command} returned an unrequested job or array-task ID")
    return frozenset(ignored)


def _selector_matches(scheduler: SlurmJobIdentity, selector: JobSelector) -> bool:
    if isinstance(selector, SchedulerIdentity):
        return scheduler == selector
    if type(scheduler) is int:
        return scheduler == selector
    return scheduler.array_job_id == selector


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
