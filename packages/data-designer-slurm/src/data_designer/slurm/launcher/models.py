# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transient typed values returned by Slurm commands."""

from __future__ import annotations

from dataclasses import dataclass

from data_designer.slurm.contracts import Identifier
from data_designer.slurm.state import SchedulerIdentity, SchedulerState


@dataclass(frozen=True, slots=True)
class SlurmSubmission:
    """Identity assigned by Slurm to one accepted array submission."""

    array_job_id: int
    cluster_name: Identifier | None = None


@dataclass(frozen=True, slots=True)
class SlurmExitCode:
    """Slurm's process status and terminating signal pair."""

    status: int
    signal: int


@dataclass(frozen=True, slots=True)
class QueueRecord:
    """One normalized active-queue row."""

    scheduler: SchedulerIdentity
    state: SchedulerState


@dataclass(frozen=True, slots=True)
class AccountingRecord:
    """One normalized accounting row."""

    scheduler: SchedulerIdentity
    state: SchedulerState
    exit_code: SlurmExitCode
