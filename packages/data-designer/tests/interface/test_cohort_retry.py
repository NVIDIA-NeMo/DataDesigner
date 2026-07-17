# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import data_designer.interface as interface
from data_designer.interface.cohort_retry import RetryExhaustion, RetryUntil, SamplerRetryMode
from data_designer.interface.errors import CohortRetryExhaustedError, DataDesignerWorkflowError


def test_retry_until_defaults_and_payload() -> None:
    policy = RetryUntil(predicate_column="is_acceptable", max_attempts=3)

    assert policy.sampler_retry_mode == SamplerRetryMode.PRESERVE
    assert policy.on_exhausted == RetryExhaustion.RAISE
    assert policy.to_dict() == {
        "predicate_column": "is_acceptable",
        "max_attempts": 3,
        "max_candidate_records": None,
        "sampler_retry_mode": "preserve",
        "on_exhausted": "raise",
    }


def test_retry_until_accepts_candidate_budget_as_only_bound() -> None:
    policy = RetryUntil(predicate_column="is_acceptable", max_candidate_records=100)

    assert policy.max_attempts is None
    assert policy.max_candidate_records == 100


def test_retry_until_coerces_enum_strings() -> None:
    policy = RetryUntil(
        predicate_column="is_acceptable",
        max_attempts=3,
        sampler_retry_mode="resample",
        on_exhausted="return_partial",
    )

    assert policy.sampler_retry_mode is SamplerRetryMode.RESAMPLE
    assert policy.on_exhausted is RetryExhaustion.RETURN_PARTIAL


@pytest.mark.parametrize("predicate_column", ["", "   ", None, 1])
def test_retry_until_rejects_invalid_predicate_column(predicate_column: object) -> None:
    with pytest.raises(DataDesignerWorkflowError, match="predicate_column must be a non-empty string"):
        RetryUntil(predicate_column=predicate_column, max_attempts=1)  # type: ignore[arg-type]


def test_retry_until_requires_at_least_one_bound() -> None:
    with pytest.raises(DataDesignerWorkflowError, match="requires at least one bound"):
        RetryUntil(predicate_column="is_acceptable")


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, "1"])
def test_retry_until_rejects_non_positive_or_non_integer_attempt_bound(value: object) -> None:
    with pytest.raises(DataDesignerWorkflowError, match="max_attempts must be a strict positive integer"):
        RetryUntil(
            predicate_column="is_acceptable",
            max_attempts=value,  # type: ignore[arg-type]
            max_candidate_records=1,
        )


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, "1"])
def test_retry_until_rejects_non_positive_or_non_integer_candidate_bound(value: object) -> None:
    with pytest.raises(DataDesignerWorkflowError, match="max_candidate_records must be a strict positive integer"):
        RetryUntil(
            predicate_column="is_acceptable",
            max_attempts=1,
            max_candidate_records=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("sampler_retry_mode", "invalid"), ("on_exhausted", "invalid")],
)
def test_retry_until_rejects_invalid_enum_values(field_name: str, value: str) -> None:
    kwargs = {field_name: value}

    with pytest.raises(DataDesignerWorkflowError, match=field_name):
        RetryUntil(predicate_column="is_acceptable", max_attempts=1, **kwargs)


def test_retry_until_is_frozen() -> None:
    policy = RetryUntil(predicate_column="is_acceptable", max_attempts=1)

    with pytest.raises(FrozenInstanceError):
        setattr(policy, "max_attempts", 2)


def test_cohort_retry_types_are_available_from_interface() -> None:
    assert interface.CohortRetryExhaustedError is CohortRetryExhaustedError
    assert interface.RetryUntil is RetryUntil
    assert interface.RetryExhaustion is RetryExhaustion
    assert interface.SamplerRetryMode is SamplerRetryMode


def test_cohort_retry_exhausted_error_exposes_structured_progress() -> None:
    error = CohortRetryExhaustedError(
        target_records=5,
        accepted_records=3,
        candidate_records=9,
        attempts=2,
        unresolved_slot_ids=[1, 4],
    )

    assert error.target_records == 5
    assert error.accepted_records == 3
    assert error.unresolved_records == 2
    assert error.candidate_records == 9
    assert error.attempts == 2
    assert error.unresolved_slot_ids == (1, 4)
    assert str(error) == "Cohort retry exhausted after 2 attempt(s): accepted 3 of 5 slots after 9 candidate records."
