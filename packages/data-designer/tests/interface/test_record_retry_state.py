# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from data_designer.interface.record_retry import RetryUntil, SamplerRetryMode
from data_designer.interface.record_retry_state import (
    AttemptCompletion,
    AttemptManifest,
    FinalCompletion,
    RetryManifest,
)


def _policy() -> RetryUntil:
    return RetryUntil(predicate_column="accepted", max_attempts=2)


def _attempt_payload(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input_records": 2,
        "output_records": 2,
        "accepted_records": 1,
        "false_records": 1,
        "null_records": 0,
        "missing_records": 0,
    }
    return payload | updates


def _manifest_payload(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "fingerprint": "fingerprint",
        "target_records": 2,
        "policy": _policy().model_dump(mode="json"),
        "slot_column": "slot",
        "attempt_column": "attempt",
    }
    return payload | updates


def test_attempt_manifest_validates_and_serializes_accounting() -> None:
    attempt = AttemptManifest.model_validate(_attempt_payload())

    assert attempt.model_dump(mode="json") == _attempt_payload(model_usage={})


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("input_records", -1, "greater_than_equal"),
        ("output_records", -1, "greater_than_equal"),
        ("accepted_records", -1, "greater_than_equal"),
        ("false_records", -1, "greater_than_equal"),
        ("null_records", -1, "greater_than_equal"),
        ("missing_records", -1, "greater_than_equal"),
        ("input_records", True, "int_type"),
        ("input_records", 2.0, "int_type"),
        ("input_records", "2", "int_type"),
    ],
)
def test_attempt_manifest_rejects_invalid_count_fields(field_name: str, value: object, error_type: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        AttemptManifest.model_validate(_attempt_payload(**{field_name: value}))

    assert exc_info.value.errors()[0]["loc"] == (field_name,)
    assert exc_info.value.errors()[0]["type"] == error_type


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"accepted_records": 0, "false_records": 0}, "produced-row outcomes do not match output_records"),
        ({"missing_records": 1}, "attempt outcomes do not match input_records"),
    ],
)
def test_attempt_manifest_rejects_inconsistent_accounting(updates: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        AttemptManifest.model_validate(_attempt_payload(**updates))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"input_slot_ids": [0], "output_records": -1}, "greater than or equal to 0"),
        ({"input_slot_ids": [0], "output_records": 2}, "output count must fit within its input slots"),
        ({"input_slot_ids": [0, 0], "output_records": 1}, "input slot IDs must be unique"),
    ],
)
def test_attempt_completion_rejects_invalid_counts_and_slots(payload: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        AttemptCompletion.model_validate(payload)


@pytest.mark.parametrize("accepted_records", [0, -1])
def test_final_completion_requires_at_least_one_accepted_record(accepted_records: int) -> None:
    with pytest.raises(ValidationError) as exc_info:
        FinalCompletion(accepted_records=accepted_records)

    assert exc_info.value.errors()[0]["loc"] == ("accepted_records",)
    assert exc_info.value.errors()[0]["type"] == "greater_than_equal"


def test_retry_manifest_validates_nested_policy_as_retry_until() -> None:
    manifest = RetryManifest.model_validate(
        _manifest_payload(
            policy={
                "predicate_column": "accepted",
                "max_attempts": 2,
                "max_candidate_records": None,
                "sampler_retry_mode": "resample",
                "on_exhausted": "raise",
            }
        )
    )

    assert isinstance(manifest.policy, RetryUntil)
    assert manifest.policy.sampler_retry_mode is SamplerRetryMode.RESAMPLE
    assert RetryManifest.model_validate_json(manifest.model_dump_json()) == manifest


@pytest.mark.parametrize(
    ("updates", "error_location", "error_type"),
    [
        ({"target_records": 0}, ("target_records",), "greater_than_equal"),
        ({"slot_column": ""}, ("slot_column",), "string_too_short"),
        ({"attempt_column": ""}, ("attempt_column",), "string_too_short"),
        ({"unknown": True}, ("unknown",), "extra_forbidden"),
    ],
)
def test_retry_manifest_rejects_invalid_fields(
    updates: dict[str, Any],
    error_location: tuple[str, ...],
    error_type: str,
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        RetryManifest.model_validate(_manifest_payload(**updates))

    assert exc_info.value.errors()[0]["loc"] == error_location
    assert exc_info.value.errors()[0]["type"] == error_type


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"slot_column": "attempt"}, "internal slot and attempt columns must be distinct"),
        (
            {"attempts": [_attempt_payload(input_records=1, output_records=1, false_records=0)]},
            "attempt input counts must match the complete pending cohort",
        ),
        ({"unresolved_slot_ids": [1, 1]}, "unresolved slot IDs must be unique"),
        ({"unresolved_slot_ids": [2]}, "unresolved slot IDs must be unique"),
        ({"unresolved_slot_ids": [1, 0]}, "unresolved slot IDs must remain in stable cohort order"),
        ({"unresolved_slot_ids": [0]}, "unresolved slot IDs must match the pending cohort"),
        ({"status": "exhausted"}, "exhausted retry state must contain unresolved cohort slots"),
        ({"status": "complete"}, "terminal retry state must account for every unresolved cohort slot"),
    ],
)
def test_retry_manifest_rejects_inconsistent_cohort_state(updates: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        RetryManifest.model_validate(_manifest_payload(**updates))
