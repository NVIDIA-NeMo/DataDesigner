# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pydantic import ValidationError

import data_designer.interface as interface
from data_designer.interface.errors import RecordRetryExhaustedError
from data_designer.interface.record_retry import RetryExhaustion, RetryUntil, SamplerRetryMode


def test_retry_until_defaults_and_payload() -> None:
    policy = RetryUntil(predicate_column="is_acceptable", max_attempts=3)

    assert policy.sampler_retry_mode == SamplerRetryMode.PRESERVE
    assert policy.on_exhausted == RetryExhaustion.RAISE
    assert policy.model_dump(mode="json") == {
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


@pytest.mark.parametrize(
    ("predicate_column", "error_type"),
    [
        ("", "string_too_short"),
        ("   ", "string_pattern_mismatch"),
        (None, "string_type"),
        (1, "string_type"),
        (b"is_acceptable", "string_type"),
        (bytearray(b"is_acceptable"), "string_type"),
    ],
)
def test_retry_until_rejects_invalid_predicate_column(predicate_column: object, error_type: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        RetryUntil(predicate_column=predicate_column, max_attempts=1)  # type: ignore[arg-type]

    assert exc_info.value.errors()[0]["loc"] == ("predicate_column",)
    assert exc_info.value.errors()[0]["type"] == error_type


def test_retry_until_requires_at_least_one_bound() -> None:
    with pytest.raises(ValidationError, match="requires at least one bound"):
        RetryUntil(predicate_column="is_acceptable")


@pytest.mark.parametrize(
    ("value", "error_type"),
    [
        (True, "int_type"),
        (False, "int_type"),
        (0, "greater_than"),
        (-1, "greater_than"),
        (1.0, "int_type"),
        ("1", "int_type"),
    ],
)
def test_retry_until_rejects_non_positive_or_non_integer_attempt_bound(value: object, error_type: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        RetryUntil(
            predicate_column="is_acceptable",
            max_attempts=value,  # type: ignore[arg-type]
            max_candidate_records=1,
        )

    assert exc_info.value.errors()[0]["loc"] == ("max_attempts",)
    assert exc_info.value.errors()[0]["type"] == error_type


@pytest.mark.parametrize(
    ("value", "error_type"),
    [
        (True, "int_type"),
        (False, "int_type"),
        (0, "greater_than"),
        (-1, "greater_than"),
        (1.0, "int_type"),
        ("1", "int_type"),
    ],
)
def test_retry_until_rejects_non_positive_or_non_integer_candidate_bound(value: object, error_type: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        RetryUntil(
            predicate_column="is_acceptable",
            max_attempts=1,
            max_candidate_records=value,  # type: ignore[arg-type]
        )

    assert exc_info.value.errors()[0]["loc"] == ("max_candidate_records",)
    assert exc_info.value.errors()[0]["type"] == error_type


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("sampler_retry_mode", "invalid"), ("on_exhausted", "invalid")],
)
def test_retry_until_rejects_invalid_enum_values(field_name: str, value: str) -> None:
    kwargs = {field_name: value}

    with pytest.raises(ValidationError) as exc_info:
        RetryUntil(predicate_column="is_acceptable", max_attempts=1, **kwargs)

    assert exc_info.value.errors()[0]["loc"] == (field_name,)
    assert exc_info.value.errors()[0]["type"] == "enum"


def test_retry_until_is_frozen() -> None:
    policy = RetryUntil(predicate_column="is_acceptable", max_attempts=1)

    with pytest.raises(ValidationError) as exc_info:
        setattr(policy, "max_attempts", 2)

    assert exc_info.value.errors()[0]["loc"] == ("max_attempts",)
    assert exc_info.value.errors()[0]["type"] == "frozen_instance"


def test_retry_until_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError) as exc_info:
        RetryUntil(predicate_column="is_acceptable", max_attempts=1, unknown=True)  # type: ignore[call-arg]

    assert exc_info.value.errors()[0]["loc"] == ("unknown",)
    assert exc_info.value.errors()[0]["type"] == "extra_forbidden"


def test_record_retry_types_are_available_from_interface() -> None:
    assert interface.RecordRetryExhaustedError is RecordRetryExhaustedError
    assert interface.RetryUntil is RetryUntil
    assert interface.RetryExhaustion is RetryExhaustion
    assert interface.SamplerRetryMode is SamplerRetryMode


def test_record_retry_exhausted_error_exposes_structured_progress() -> None:
    error = RecordRetryExhaustedError(
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
    assert str(error) == "Record retry exhausted after 2 attempt(s): accepted 3 of 5 slots after 9 candidate records."
