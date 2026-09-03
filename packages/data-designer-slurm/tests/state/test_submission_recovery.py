# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from data_designer.slurm.contracts import Identifier
from data_designer.slurm.launcher.models import SlurmSubmissionMatch
from data_designer.slurm.state import StateConflictError
from data_designer.slurm.state.submission_recovery import PreparedSubmission, resolve_prepared_submission


@dataclass(frozen=True)
class _SubmissionLookup:
    matches: tuple[SlurmSubmissionMatch, ...]

    def query_submissions_by_name(
        self,
        job_name: Identifier,
        *,
        submitted_after: datetime,
    ) -> tuple[SlurmSubmissionMatch, ...]:
        assert job_name == "dd-retry-0123456789abcdef0123456789abcdef"
        assert submitted_after == datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
        return self.matches


def test_prepared_submission_recovery_rejects_multiple_exact_matches() -> None:
    submitted_at = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)

    with pytest.raises(StateConflictError, match="multiple scheduler jobs"):
        resolve_prepared_submission(
            _SubmissionLookup(
                (
                    SlurmSubmissionMatch(
                        job_id=4201,
                        job_name="dd-retry-0123456789abcdef0123456789abcdef",
                        array_task_ids=(0,),
                    ),
                    SlurmSubmissionMatch(
                        job_id=4301,
                        job_name="dd-retry-0123456789abcdef0123456789abcdef",
                        array_task_ids=(0,),
                    ),
                )
            ),
            PreparedSubmission(
                job_name="dd-retry-0123456789abcdef0123456789abcdef",
                submitted_after=submitted_at,
                reconciliation_deadline=submitted_at + timedelta(minutes=5),
                expected_array_task_ids=(0,),
            ),
            observed_at=submitted_at + timedelta(minutes=1),
        )


def test_prepared_submission_recovery_returns_definitive_absence_only_after_deadline() -> None:
    submitted_at = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
    deadline = submitted_at + timedelta(minutes=5)
    lookup = _SubmissionLookup(())

    with pytest.raises(StateConflictError, match="still being reconciled"):
        resolve_prepared_submission(
            lookup,
            PreparedSubmission(
                job_name="dd-retry-0123456789abcdef0123456789abcdef",
                submitted_after=submitted_at,
                reconciliation_deadline=deadline,
                expected_array_task_ids=(0,),
            ),
            observed_at=deadline,
        )
    assert (
        resolve_prepared_submission(
            lookup,
            PreparedSubmission(
                job_name="dd-retry-0123456789abcdef0123456789abcdef",
                submitted_after=submitted_at,
                reconciliation_deadline=deadline,
                expected_array_task_ids=(0,),
            ),
            observed_at=deadline + timedelta(microseconds=1),
        )
        is None
    )


@pytest.mark.parametrize("actual_shape", [None, (0,), (0, 2)])
def test_prepared_submission_recovery_rejects_the_wrong_scheduler_shape(
    actual_shape: tuple[int, ...] | None,
) -> None:
    submitted_at = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
    match = SlurmSubmissionMatch(
        job_id=4201,
        job_name="dd-retry-0123456789abcdef0123456789abcdef",
        array_task_ids=actual_shape,
    )

    with pytest.raises(StateConflictError, match="shape"):
        resolve_prepared_submission(
            _SubmissionLookup((match,)),
            PreparedSubmission(
                job_name="dd-retry-0123456789abcdef0123456789abcdef",
                submitted_after=submitted_at,
                reconciliation_deadline=submitted_at + timedelta(minutes=5),
                expected_array_task_ids=(0, 1),
            ),
            observed_at=submitted_at + timedelta(minutes=1),
        )
