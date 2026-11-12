# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared recovery policy for ambiguous Slurm submission receipts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from data_designer.slurm.contracts import Identifier
from data_designer.slurm.launcher.errors import SlurmLauncherError
from data_designer.slurm.launcher.models import SlurmSubmissionMatch
from data_designer.slurm.state.errors import SlurmStateError, StateConflictError

SUBMISSION_VISIBILITY_WINDOW = timedelta(minutes=5)


class SubmissionLookup(Protocol):
    """Scheduler lookup needed to recover one immutable submission plan."""

    def query_submissions_by_name(
        self,
        job_name: Identifier,
        *,
        submitted_after: datetime,
    ) -> tuple[SlurmSubmissionMatch, ...]:
        """Return allocations matching one exact submission name."""
        ...


@dataclass(frozen=True)
class PreparedSubmission:
    """Immutable scheduler correlation facts for one prepared operation."""

    job_name: Identifier
    submitted_after: datetime
    reconciliation_deadline: datetime
    expected_array_task_ids: tuple[int, ...] | None


def resolve_prepared_submission(
    scheduler: SubmissionLookup,
    prepared: PreparedSubmission,
    *,
    observed_at: datetime,
) -> int | None:
    """Return one recovered job ID, or ``None`` after definitive bounded absence."""
    try:
        matches = scheduler.query_submissions_by_name(
            prepared.job_name,
            submitted_after=prepared.submitted_after,
        )
    except SlurmLauncherError as error:
        raise SlurmStateError("cannot reconcile ambiguous Slurm submission") from error
    if len(matches) > 1:
        raise StateConflictError("multiple scheduler jobs match the prepared submission")
    if matches:
        match = matches[0]
        if match.array_task_ids == prepared.expected_array_task_ids:
            return match.job_id
        if _is_partial_array_view(match.array_task_ids, prepared.expected_array_task_ids):
            if observed_at <= prepared.reconciliation_deadline:
                raise StateConflictError("prepared submission is still being reconciled")
            return None
        raise StateConflictError("scheduler job shape does not match the prepared submission")
    if observed_at <= prepared.reconciliation_deadline:
        raise StateConflictError("prepared submission is still being reconciled")
    return None


def _is_partial_array_view(
    observed: tuple[int, ...] | None,
    expected: tuple[int, ...] | None,
) -> bool:
    return observed is not None and expected is not None and set(observed) < set(expected)


__all__ = [
    "SUBMISSION_VISIBILITY_WINDOW",
    "PreparedSubmission",
    "SubmissionLookup",
    "resolve_prepared_submission",
]
