# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal strict parsers for bounded, machine-readable Slurm output."""

from __future__ import annotations

import re

from data_designer.slurm.launcher.errors import SlurmCommandOutputError
from data_designer.slurm.launcher.models import (
    SlurmAccountingEntry,
    SlurmJobSubmissionReceipt,
    SlurmProcessExitCode,
    SlurmQueueEntry,
)
from data_designer.slurm.state import SchedulerIdentity, SchedulerJobIdentity, SchedulerState

_ARRAY_ID_PATTERN = re.compile(r"^(?P<job>[1-9][0-9]*)_(?P<task>[0-9]+)$")
_JOB_ID_PATTERN = re.compile(r"^[1-9][0-9]*$")
_CLUSTER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_EXIT_CODE_PATTERN = re.compile(r"^(?P<status>[0-9]+):(?P<signal>[0-9]+)$")
_GRES_GPU_PATTERN = re.compile(r"^gpu:(?:(?:[^:,()]+):)*(?P<count>[1-9][0-9]*)(?:\([^\r\n]*\))?$")
_MAX_SLURM_INTEGER = (1 << 32) - 1

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
    "RESV_DEL_HOLD": SchedulerState.PENDING,
    "RESIZING": SchedulerState.RUNNING,
    "REVOKED": SchedulerState.FAILED,
    "RUNNING": SchedulerState.RUNNING,
    "SIGNALING": SchedulerState.RUNNING,
    "SPECIAL_EXIT": SchedulerState.PENDING,
    "STAGE_OUT": SchedulerState.RUNNING,
    "STOPPED": SchedulerState.RUNNING,
    "SUSPENDED": SchedulerState.RUNNING,
    "TIMEOUT": SchedulerState.TIMED_OUT,
}


def parse_submission(output: str) -> SlurmJobSubmissionReceipt:
    """Parse non-federated ``sbatch --parsable`` output."""
    value = output.strip()
    job_id, separator, cluster_name = value.partition(";")
    if not job_id.isascii() or not job_id.isdecimal():
        raise SlurmCommandOutputError("sbatch returned an invalid job ID")
    parsed_job_id = _parse_decimal(job_id, message="sbatch returned an invalid job ID")
    if parsed_job_id <= 0:
        raise SlurmCommandOutputError("sbatch returned an invalid job ID")
    if separator:
        if _CLUSTER_NAME_PATTERN.fullmatch(cluster_name) is None:
            raise SlurmCommandOutputError("sbatch returned an invalid cluster name")
        raise SlurmCommandOutputError("federated Slurm submissions are not supported")
    return SlurmJobSubmissionReceipt(job_id=parsed_job_id)


def parse_queue(output: str) -> tuple[SlurmQueueEntry, ...]:
    """Parse ``squeue --format=%i|%T`` rows."""
    entries: list[SlurmQueueEntry] = []
    identities: set[SchedulerJobIdentity] = set()
    for line_number, line in _collect_nonempty_lines(output):
        fields = line.split("|")
        if len(fields) != 2:
            raise SlurmCommandOutputError(f"squeue line {line_number} must contain two fields")
        job_identity = _parse_job_identity(fields[0], command="squeue", line_number=line_number)
        _reject_duplicate(job_identity, identities, command="squeue", line_number=line_number)
        entries.append(SlurmQueueEntry(job_identity=job_identity, state=parse_state(fields[1])))
    return tuple(entries)


def parse_accounting(output: str) -> tuple[SlurmAccountingEntry, ...]:
    """Parse job and array-task rows from ``sacct --format=JobID,State,ExitCode``."""
    entries: list[SlurmAccountingEntry] = []
    identities: set[SchedulerJobIdentity] = set()
    for line_number, line in _collect_nonempty_lines(output):
        fields = line.split("|")
        if len(fields) != 3:
            raise SlurmCommandOutputError(f"sacct line {line_number} must contain three fields")
        job_identity = _parse_job_identity(fields[0], command="sacct", line_number=line_number)
        _reject_duplicate(job_identity, identities, command="sacct", line_number=line_number)
        entries.append(
            SlurmAccountingEntry(
                job_identity=job_identity,
                state=parse_state(fields[1]),
                process_exit_code=_parse_exit_code(fields[2], line_number=line_number),
            )
        )
    return tuple(entries)


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
                raise SlurmCommandOutputError(f"sinfo line {line_number} contains an invalid GPU resource")
            line_counts.append(
                _parse_decimal(
                    match.group("count"),
                    message=f"sinfo line {line_number} contains an invalid GPU resource",
                )
            )
        if line_counts:
            counts.append(sum(line_counts))
    return tuple(counts)


def parse_state(value: str) -> SchedulerState:
    """Normalize one Slurm long state spelling without guessing unknown states."""
    normalized = value.strip().upper().removesuffix("+")
    if not normalized:
        raise SlurmCommandOutputError("scheduler state must not be empty")
    if normalized.startswith("CANCELLED BY "):
        canceller = normalized.removeprefix("CANCELLED BY ")
        if not canceller.isascii() or not canceller.isdecimal():
            raise SlurmCommandOutputError("cancelled scheduler state has an invalid owner")
        normalized = "CANCELLED"
    elif any(character.isspace() for character in normalized):
        raise SlurmCommandOutputError("scheduler state contains unexpected whitespace")
    return _STATE_MAP.get(normalized, SchedulerState.UNKNOWN)


def _split_gres_fields(value: str, *, line_number: int) -> tuple[str, ...]:
    fields: list[str] = []
    start = 0
    annotation_depth = 0
    for index, character in enumerate(value):
        if character == "(":
            annotation_depth += 1
            if annotation_depth > 1:
                raise SlurmCommandOutputError(f"sinfo line {line_number} contains an invalid GPU resource")
        elif character == ")":
            annotation_depth -= 1
            if annotation_depth < 0:
                raise SlurmCommandOutputError(f"sinfo line {line_number} contains an invalid GPU resource")
        elif character == "," and annotation_depth == 0:
            fields.append(value[start:index])
            start = index + 1
    if annotation_depth:
        raise SlurmCommandOutputError(f"sinfo line {line_number} contains an invalid GPU resource")
    fields.append(value[start:])
    if any(not field for field in fields):
        raise SlurmCommandOutputError(f"sinfo line {line_number} contains an invalid GPU resource")
    return tuple(fields)


def _collect_nonempty_lines(output: str) -> tuple[tuple[int, str], ...]:
    return tuple(
        (line_number, line)
        for line_number, raw_line in enumerate(output.splitlines(), start=1)
        if (line := raw_line.strip())
    )


def _parse_array_identity(value: str, *, command: str, line_number: int) -> SchedulerIdentity:
    match = _ARRAY_ID_PATTERN.fullmatch(value)
    if match is None:
        raise SlurmCommandOutputError(f"{command} line {line_number} contains an invalid array-task ID")
    message = f"{command} line {line_number} contains an invalid array-task ID"
    return SchedulerIdentity(
        array_job_id=_parse_decimal(match.group("job"), message=message),
        array_task_id=_parse_decimal(match.group("task"), message=message),
    )


def _parse_job_identity(value: str, *, command: str, line_number: int) -> SchedulerJobIdentity:
    message = f"{command} line {line_number} contains an invalid job or array-task ID"
    if _JOB_ID_PATTERN.fullmatch(value) is not None:
        return _parse_decimal(value, message=message)
    try:
        return _parse_array_identity(value, command=command, line_number=line_number)
    except SlurmCommandOutputError as error:
        raise SlurmCommandOutputError(message) from error


def _parse_exit_code(value: str, *, line_number: int) -> SlurmProcessExitCode:
    match = _EXIT_CODE_PATTERN.fullmatch(value)
    if match is None:
        raise SlurmCommandOutputError(f"sacct line {line_number} contains an invalid exit code")
    message = f"sacct line {line_number} contains an invalid exit code"
    return SlurmProcessExitCode(
        exit_status=_parse_decimal(match.group("status"), message=message),
        termination_signal=_parse_decimal(match.group("signal"), message=message),
    )


def _parse_decimal(value: str, *, message: str) -> int:
    if len(value) > 10:
        raise SlurmCommandOutputError(message)
    try:
        parsed = int(value)
    except ValueError as error:
        raise SlurmCommandOutputError(message) from error
    if parsed > _MAX_SLURM_INTEGER:
        raise SlurmCommandOutputError(message)
    return parsed


def _reject_duplicate(
    job_identity: SchedulerJobIdentity,
    identities: set[SchedulerJobIdentity],
    *,
    command: str,
    line_number: int,
) -> None:
    if job_identity in identities:
        raise SlurmCommandOutputError(f"{command} line {line_number} duplicates a job or array-task ID")
    identities.add(job_identity)
