# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from data_designer.engine.dataset_builders.utils.skip_tracker import (
    SKIPPED_COLUMNS_RECORD_KEY,
    apply_skip_to_record,
    get_skipped_column_names,
    restore_skip_metadata,
    strip_skip_metadata_for_dataframe_row,
    strip_skip_metadata_from_records,
)


def test_skipped_columns_record_key_value() -> None:
    assert SKIPPED_COLUMNS_RECORD_KEY == "__internal_skipped_columns"


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        pytest.param({}, set(), id="empty"),
        pytest.param({SKIPPED_COLUMNS_RECORD_KEY: {"a", "b"}}, {"a", "b"}, id="populated"),
    ],
)
def test_get_skipped_column_names(record: dict, expected: set[str]) -> None:
    assert get_skipped_column_names(record) == expected


def test_get_skipped_column_names_returns_copy() -> None:
    inner: set[str] = {"x"}
    record = {SKIPPED_COLUMNS_RECORD_KEY: inner}
    names = get_skipped_column_names(record)
    names.add("y")
    assert record[SKIPPED_COLUMNS_RECORD_KEY] == {"x"}
    assert names == {"x", "y"}


def test_apply_skip_to_record_adds_skip_marker() -> None:
    record: dict = {}
    apply_skip_to_record(
        record,
        column_name="primary",
        cell_value=None,
        side_effect_columns=(),
    )
    assert record[SKIPPED_COLUMNS_RECORD_KEY] == {"primary"}


@pytest.mark.parametrize(
    "cell_value",
    [None, True, False, 0, 42, 3.14, "skipped"],
)
def test_apply_skip_to_record_sets_cell_value(cell_value: bool | int | float | str | None) -> None:
    record: dict = {}
    apply_skip_to_record(
        record,
        column_name="col_a",
        cell_value=cell_value,
        side_effect_columns=(),
    )
    assert record["col_a"] == cell_value


def test_apply_skip_to_record_clears_side_effects() -> None:
    record: dict = {"se1": "keep-me", "se2": 99}
    apply_skip_to_record(
        record,
        column_name="primary",
        cell_value="pv",
        side_effect_columns=("se1", "se2"),
    )
    assert record["se1"] is None
    assert record["se2"] is None
    assert record["primary"] == "pv"


def test_apply_skip_to_record_accumulates() -> None:
    record: dict = {}
    apply_skip_to_record(
        record,
        column_name="first",
        cell_value=1,
        side_effect_columns=(),
    )
    apply_skip_to_record(
        record,
        column_name="second",
        cell_value=2,
        side_effect_columns=(),
    )
    assert record[SKIPPED_COLUMNS_RECORD_KEY] == {"first", "second"}
    assert record["first"] == 1
    assert record["second"] == 2


def test_strip_skip_metadata_for_dataframe_row() -> None:
    record = {
        "a": 1,
        SKIPPED_COLUMNS_RECORD_KEY: {"x"},
        "b": 2,
    }
    stripped = strip_skip_metadata_for_dataframe_row(record)
    assert stripped == {"a": 1, "b": 2}
    assert SKIPPED_COLUMNS_RECORD_KEY not in stripped


def test_strip_skip_metadata_for_dataframe_row_no_metadata() -> None:
    record = {"a": 1, "b": [10, 20]}
    stripped = strip_skip_metadata_for_dataframe_row(record)
    assert stripped == record
    assert stripped is not record
    assert stripped["b"] is record["b"]


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        pytest.param(
            [{"k": 1, SKIPPED_COLUMNS_RECORD_KEY: {"c"}}, {"k": 2}],
            [{"k": 1}, {"k": 2}],
            id="mixed",
        ),
        pytest.param([], [], id="empty"),
    ],
)
def test_strip_skip_metadata_from_records(rows: list[dict], expected: list[dict]) -> None:
    assert strip_skip_metadata_from_records(rows) == expected


def test_restore_skip_metadata_copies_metadata() -> None:
    old = [
        {"a": 1, SKIPPED_COLUMNS_RECORD_KEY: {"col_x"}},
        {"a": 2},
        {"a": 3, SKIPPED_COLUMNS_RECORD_KEY: {"col_y", "col_z"}},
    ]
    new = [{"a": 10}, {"a": 20}, {"a": 30}]
    restore_skip_metadata(old, new)
    assert new[0][SKIPPED_COLUMNS_RECORD_KEY] == {"col_x"}
    assert SKIPPED_COLUMNS_RECORD_KEY not in new[1]
    assert new[2][SKIPPED_COLUMNS_RECORD_KEY] == {"col_y", "col_z"}


def test_restore_skip_metadata_handles_length_mismatch() -> None:
    old = [
        {"a": 1, SKIPPED_COLUMNS_RECORD_KEY: {"col_x"}},
        {"a": 2, SKIPPED_COLUMNS_RECORD_KEY: {"col_y"}},
    ]
    new = [{"a": 10}]
    restore_skip_metadata(old, new)
    assert new[0][SKIPPED_COLUMNS_RECORD_KEY] == {"col_x"}


def test_restore_skip_metadata_no_metadata() -> None:
    old = [{"a": 1}, {"a": 2}]
    new = [{"a": 10}, {"a": 20}]
    restore_skip_metadata(old, new)
    assert SKIPPED_COLUMNS_RECORD_KEY not in new[0]
    assert SKIPPED_COLUMNS_RECORD_KEY not in new[1]
