# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transient typed values returned by Slurm commands."""

from __future__ import annotations

from dataclasses import dataclass

from data_designer.slurm.state import SchedulerJobIdentity, SchedulerState


@dataclass(frozen=True)
class SlurmJobSubmissionReceipt:
    """Job identity returned for one accepted non-federated submission."""

    job_id: int


@dataclass(frozen=True)
class SlurmProcessExitCode:
    """Slurm's process status and terminating signal pair."""

    exit_status: int
    termination_signal: int


@dataclass(frozen=True)
class SlurmQueueEntry:
    """One transient normalized active-queue entry."""

    job_identity: SchedulerJobIdentity
    state: SchedulerState


@dataclass(frozen=True)
class SlurmAccountingEntry:
    """One transient normalized accounting entry."""

    job_identity: SchedulerJobIdentity
    state: SchedulerState
    process_exit_code: SlurmProcessExitCode
