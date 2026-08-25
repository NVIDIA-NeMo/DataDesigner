# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict parsers for bounded, machine-readable Slurm output."""

from __future__ import annotations

import re

from data_designer.slurm.launcher.errors import SlurmParseError
from data_designer.slurm.launcher.models import (
    AccountingRecord,
    QueueRecord,
    SlurmExitCode,
    SlurmSubmission,
)
from data_designer.slurm.state import SchedulerIdentity, SchedulerState

_ARRAY_ID_PATTERN = re.compile(r"^(?P<job>[1-9][0-9]*)_(?P<task>[0-9]+)$")
_ARRAY_STEP_ID_PATTERN = re.compile(r"^[1-9][0-9]*_[0-9]+\.[^\s|]+$")
_JOB_ID_PATTERN = re.compile(r"^[1-9][0-9]*$")
_CLUSTER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_EXIT_CODE_PATTERN = re.compile(r"^(?P<status>[0-9]+):(?P<signal>[0-9]+)$")
_GRES_GPU_PATTERN = re.compile(r"^gpu:(?:(?:[^:,()]+):)*(?P<count>[1-9][0-9]*)(?:\([^\r\n]*\))?$")

_STATE_MAP = {
    "BOOT_FAIL": SchedulerState.FAILED,
    "CANCELLED": SchedulerState.CANCELLED,
    "COMPLETED": SchedulerState.COMPLETED,
    "COMPLETING": SchedulerState.RUNNING,
    "CONFIGURING": SchedulerState.PENDING,
    "DEADLINE": SchedulerState.FAILED,
    "FAILED": SchedulerState.FAILED,
    "NODE_FAIL": SchedulerState.NODE_FAILED,
    "OUT_OF_MEMORY": SchedulerState.OUT_OF_MEMORY,
    "PENDING": SchedulerState.PENDING,
    "PREEMPTED": SchedulerState.PREEMPTED,
    "REQUEUED": SchedulerState.REQUEUED,
    "REQUEUE_FED": SchedulerState.PENDING,
    "REQUEUE_HOLD": SchedulerState.PENDING,
    "RESIZING": SchedulerState.RUNNING,
    "REVOKED": SchedulerState.FAILED,
    "RUNNING": SchedulerState.RUNNING,
    "SIGNALING": SchedulerState.RUNNING,
    "SPECIAL_EXIT": SchedulerState.FAILED,
    "STAGE_OUT": SchedulerState.RUNNING,
    "STOPPED": SchedulerState.RUNNING,
    "SUSPENDED": SchedulerState.RUNNING,
    "TIMEOUT": SchedulerState.TIMED_OUT,
}


def parse_submission(output: str) -> SlurmSubmission:
    """Parse ``sbatch --parsable`` output."""
    value = output.strip()
    job_id, separator, cluster_name = value.partition(";")
    if not job_id.isascii() or not job_id.isdecimal() or int(job_id) <= 0:
        raise SlurmParseError("sbatch returned an invalid job ID")
    if separator and _CLUSTER_NAME_PATTERN.fullmatch(cluster_name) is None:
        raise SlurmParseError("sbatch returned an invalid cluster name")
    return SlurmSubmission(array_job_id=int(job_id), cluster_name=cluster_name or None)


def parse_queue(output: str) -> tuple[QueueRecord, ...]:
    """Parse ``squeue --format=%i|%T`` rows."""
    records: list[QueueRecord] = []
    identities: set[SchedulerIdentity] = set()
    for line_number, line in _collect_nonempty_lines(output):
        fields = line.split("|")
        if len(fields) != 2:
            raise SlurmParseError(f"squeue line {line_number} must contain two fields")
        scheduler = _parse_array_identity(fields[0], command="squeue", line_number=line_number)
        _reject_duplicate(scheduler, identities, command="squeue", line_number=line_number)
        records.append(QueueRecord(scheduler=scheduler, state=parse_state(fields[1])))
    return tuple(records)


def parse_accounting(output: str) -> tuple[AccountingRecord, ...]:
    """Parse array-task rows from ``sacct --format=JobIDRaw,State,ExitCode``."""
    records: list[AccountingRecord] = []
    identities: set[SchedulerIdentity] = set()
    for line_number, line in _collect_nonempty_lines(output):
        fields = line.split("|")
        if len(fields) != 3:
            raise SlurmParseError(f"sacct line {line_number} must contain three fields")
        if _JOB_ID_PATTERN.fullmatch(fields[0]) is not None or _ARRAY_STEP_ID_PATTERN.fullmatch(fields[0]) is not None:
            continue
        scheduler = _parse_array_identity(fields[0], command="sacct", line_number=line_number)
        _reject_duplicate(scheduler, identities, command="sacct", line_number=line_number)
        records.append(
            AccountingRecord(
                scheduler=scheduler,
                state=parse_state(fields[1]),
                exit_code=_parse_exit_code(fields[2], line_number=line_number),
            )
        )
    return tuple(records)


def parse_gpu_counts(output: str) -> tuple[int, ...]:
    """Parse configured per-node GPU counts from ``sinfo --format=%G`` rows."""
    counts: list[int] = []
    for line_number, line in _collect_nonempty_lines(output):
        if line in {"(null)", "N/A"}:
            continue
        line_counts: list[int] = []
        for gres in _split_gres_fields(line, line_number=line_number):
            if not gres.startswith("gpu:"):
                continue
            match = _GRES_GPU_PATTERN.fullmatch(gres)
            if match is None:
                raise SlurmParseError(f"sinfo line {line_number} contains an invalid GPU resource")
            line_counts.append(int(match.group("count")))
        if line_counts:
            counts.append(sum(line_counts))
    return tuple(counts)


def _split_gres_fields(value: str, *, line_number: int) -> tuple[str, ...]:
    fields: list[str] = []
    start = 0
    annotation_depth = 0
    for index, character in enumerate(value):
        if character == "(":
            annotation_depth += 1
            if annotation_depth > 1:
                raise SlurmParseError(f"sinfo line {line_number} contains an invalid GPU resource")
        elif character == ")":
            annotation_depth -= 1
            if annotation_depth < 0:
                raise SlurmParseError(f"sinfo line {line_number} contains an invalid GPU resource")
        elif character == "," and annotation_depth == 0:
            fields.append(value[start:index])
            start = index + 1
    if annotation_depth:
        raise SlurmParseError(f"sinfo line {line_number} contains an invalid GPU resource")
    fields.append(value[start:])
    if any(not field for field in fields):
        raise SlurmParseError(f"sinfo line {line_number} contains an invalid GPU resource")
    return tuple(fields)


def parse_state(value: str) -> SchedulerState:
    """Normalize one Slurm long state spelling without guessing unknown states."""
    normalized = value.strip().upper().removesuffix("+")
    if not normalized:
        raise SlurmParseError("scheduler state must not be empty")
    if normalized.startswith("CANCELLED BY "):
        canceller = normalized.removeprefix("CANCELLED BY ")
        if not canceller.isascii() or not canceller.isdecimal():
            raise SlurmParseError("cancelled scheduler state has an invalid owner")
        normalized = "CANCELLED"
    elif any(character.isspace() for character in normalized):
        raise SlurmParseError("scheduler state contains unexpected whitespace")
    return _STATE_MAP.get(normalized, SchedulerState.UNKNOWN)


def _collect_nonempty_lines(output: str) -> tuple[tuple[int, str], ...]:
    return tuple(
        (line_number, line)
        for line_number, raw_line in enumerate(output.splitlines(), start=1)
        if (line := raw_line.strip())
    )


def _parse_array_identity(value: str, *, command: str, line_number: int) -> SchedulerIdentity:
    match = _ARRAY_ID_PATTERN.fullmatch(value)
    if match is None:
        raise SlurmParseError(f"{command} line {line_number} contains an invalid array-task ID")
    return SchedulerIdentity(
        array_job_id=int(match.group("job")),
        array_task_id=int(match.group("task")),
    )


def _parse_exit_code(value: str, *, line_number: int) -> SlurmExitCode:
    match = _EXIT_CODE_PATTERN.fullmatch(value)
    if match is None:
        raise SlurmParseError(f"sacct line {line_number} contains an invalid exit code")
    return SlurmExitCode(status=int(match.group("status")), signal=int(match.group("signal")))


def _reject_duplicate(
    scheduler: SchedulerIdentity,
    identities: set[SchedulerIdentity],
    *,
    command: str,
    line_number: int,
) -> None:
    if scheduler in identities:
        raise SlurmParseError(f"{command} line {line_number} duplicates an array-task ID")
    identities.add(scheduler)
