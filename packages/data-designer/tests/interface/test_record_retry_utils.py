# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.interface.errors import DataDesignerWorkflowError
from data_designer.interface.record_retry import RetryUntil
from data_designer.interface.record_retry_state import (
    COALESCED_ACCEPTED_PATH,
    AttemptManifest,
    RetryManifest,
    get_attempt_accepted_path,
    get_attempt_artifact_path,
    get_attempt_directory,
    get_attempt_input_path,
    get_attempt_name,
)
from data_designer.interface.record_retry_utils import coalesce_accepted, retry_bounds_exhausted


def _manifest(*attempts: AttemptManifest) -> RetryManifest:
    return RetryManifest(
        fingerprint="fingerprint",
        target_records=3,
        policy=RetryUntil(predicate_column="accepted", max_attempts=3).model_dump(mode="json"),
        slot_column="slot",
        attempt_column="attempt",
        attempts=list(attempts),
    )


def _attempt(*, input_records: int, accepted_records: int) -> AttemptManifest:
    return AttemptManifest(
        input_records=input_records,
        output_records=input_records,
        accepted_records=accepted_records,
        false_records=input_records - accepted_records,
        null_records=0,
        missing_records=0,
    )


def test_attempt_paths_are_derived_from_the_attempt_index() -> None:
    assert get_attempt_name(7) == "attempt-007"
    assert get_attempt_directory(7) == Path("attempts/attempt-007")
    assert get_attempt_input_path(7) == Path("attempts/attempt-007/input.parquet")
    assert get_attempt_artifact_path(7) == Path("attempts/attempt-007/run")
    assert get_attempt_accepted_path(7) == Path("accepted/attempt-007.parquet")


def test_coalesce_accepted_orders_one_committed_row_per_slot(tmp_path: Path) -> None:
    manifest = _manifest(
        _attempt(input_records=3, accepted_records=1),
        _attempt(input_records=2, accepted_records=1),
    )
    first_path = tmp_path / get_attempt_accepted_path(0)
    second_path = tmp_path / get_attempt_accepted_path(1)
    first_path.parent.mkdir(parents=True)
    lazy.pd.DataFrame({"slot": [2], "attempt": [0], "value": ["two"]}).to_parquet(first_path, index=False)
    lazy.pd.DataFrame({"slot": [0], "attempt": [1], "value": ["zero"]}).to_parquet(second_path, index=False)

    coalesced = coalesce_accepted(
        tmp_path,
        manifest,
        lazy.pd.DataFrame({"slot": [0, 1, 2]}),
    )

    assert coalesced[["slot", "value"]].to_dict(orient="records") == [
        {"slot": 0, "value": "zero"},
        {"slot": 2, "value": "two"},
    ]
    assert lazy.pd.read_parquet(tmp_path / COALESCED_ACCEPTED_PATH).equals(coalesced)


def test_coalesce_accepted_rejects_duplicate_slots(tmp_path: Path) -> None:
    manifest = _manifest(
        _attempt(input_records=3, accepted_records=1),
        _attempt(input_records=2, accepted_records=1),
    )
    accepted_path = tmp_path / get_attempt_accepted_path(0)
    accepted_path.parent.mkdir(parents=True)
    lazy.pd.DataFrame({"slot": [0], "attempt": [0]}).to_parquet(accepted_path, index=False)
    lazy.pd.DataFrame({"slot": [0], "attempt": [1]}).to_parquet(
        tmp_path / get_attempt_accepted_path(1),
        index=False,
    )

    with pytest.raises(DataDesignerWorkflowError, match="one-row-per-slot invariant"):
        coalesce_accepted(tmp_path, manifest, lazy.pd.DataFrame({"slot": [0, 1, 2]}))


def test_coalesce_accepted_preserves_empty_partition_schema(tmp_path: Path) -> None:
    manifest = _manifest(_attempt(input_records=3, accepted_records=0))
    accepted_path = tmp_path / get_attempt_accepted_path(0)
    accepted_path.parent.mkdir(parents=True)
    lazy.pd.DataFrame(
        {
            "slot": lazy.pd.Series(dtype="int64"),
            "attempt": lazy.pd.Series(dtype="int64"),
            "value": lazy.pd.Series(dtype="object"),
        }
    ).to_parquet(accepted_path, index=False)

    coalesced = coalesce_accepted(tmp_path, manifest, lazy.pd.DataFrame({"slot": [0, 1, 2]}))

    assert coalesced.empty
    assert list(coalesced.columns) == ["slot", "attempt", "value"]


def test_retry_bounds_require_room_for_the_complete_pending_cohort() -> None:
    manifest = _manifest(_attempt(input_records=3, accepted_records=1))

    assert retry_bounds_exhausted(
        RetryUntil(predicate_column="accepted", max_candidate_records=4),
        manifest,
        pending_records=2,
    )
    assert not retry_bounds_exhausted(
        RetryUntil(predicate_column="accepted", max_candidate_records=5),
        manifest,
        pending_records=2,
    )
