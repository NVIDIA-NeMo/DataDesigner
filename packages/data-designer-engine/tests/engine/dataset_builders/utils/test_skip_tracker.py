# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from data_designer.engine.dataset_builders.utils.skip_evaluator import get_skipped_column_names
from data_designer.engine.dataset_builders.utils.skip_tracker import (
    SKIPPED_COLUMNS_RECORD_KEY,
    apply_skip_to_record,
    prepare_records_for_skip_metadata_round_trip,
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


def test_prepare_records_for_skip_metadata_round_trip_without_metadata() -> None:
    rows = [{"a": 1}, {"a": 2}]
    prepared_rows, restore_context = prepare_records_for_skip_metadata_round_trip(rows)
    assert restore_context is None
    assert prepared_rows == rows
    assert prepared_rows is not rows


def test_prepare_records_for_skip_metadata_round_trip_injects_restore_ids() -> None:
    rows = [
        {"a": 1, SKIPPED_COLUMNS_RECORD_KEY: {"col_x"}},
        {"a": 2},
        {"a": 3, SKIPPED_COLUMNS_RECORD_KEY: {"col_y", "col_z"}},
    ]
    prepared_rows, restore_context = prepare_records_for_skip_metadata_round_trip(rows)
    assert restore_context is not None
    assert SKIPPED_COLUMNS_RECORD_KEY not in prepared_rows[0]
    assert restore_context.restore_id_column in prepared_rows[0]
    assert restore_context.skipped_columns_by_source_id == {
        "0": {"col_x"},
        "2": {"col_y", "col_z"},
    }


def test_restore_skip_metadata_uses_restore_ids_after_reorder() -> None:
    old = [
        {"a": 1, SKIPPED_COLUMNS_RECORD_KEY: {"col_x"}},
        {"a": 2},
        {"a": 3, SKIPPED_COLUMNS_RECORD_KEY: {"col_z"}},
    ]
    prepared_rows, restore_context = prepare_records_for_skip_metadata_round_trip(old)
    assert restore_context is not None
    restore_id_column = restore_context.restore_id_column

    new = [
        {"a": 30, restore_id_column: prepared_rows[2][restore_id_column]},
        {"a": 10, restore_id_column: prepared_rows[0][restore_id_column]},
        {"a": 20, restore_id_column: prepared_rows[1][restore_id_column]},
    ]
    restore_skip_metadata(new, context=restore_context, allow_resize=False)

    assert new[0][SKIPPED_COLUMNS_RECORD_KEY] == {"col_z"}
    assert new[1][SKIPPED_COLUMNS_RECORD_KEY] == {"col_x"}
    assert SKIPPED_COLUMNS_RECORD_KEY not in new[2]


def test_restore_skip_metadata_allow_resize_handles_filtered_rows() -> None:
    old = [{"a": 1}, {"a": 2}]
    prepared_rows, restore_context = prepare_records_for_skip_metadata_round_trip(old)
    assert restore_context is None

    old = [
        {"a": 1, SKIPPED_COLUMNS_RECORD_KEY: {"col_x"}},
        {"a": 2},
    ]
    prepared_rows, restore_context = prepare_records_for_skip_metadata_round_trip(old)
    assert restore_context is not None
    restore_id_column = restore_context.restore_id_column

    new = [{"a": 20, restore_id_column: prepared_rows[1][restore_id_column]}]
    restore_skip_metadata(new, context=restore_context, allow_resize=True)

    assert SKIPPED_COLUMNS_RECORD_KEY not in new[0]


def test_restore_skip_metadata_rejects_missing_restore_id_column() -> None:
    old = [{"a": 1, SKIPPED_COLUMNS_RECORD_KEY: {"col_x"}}]
    _prepared_rows, restore_context = prepare_records_for_skip_metadata_round_trip(old)
    assert restore_context is not None

    with pytest.raises(ValueError, match="must preserve the internal column"):
        restore_skip_metadata([{"a": 10}], context=restore_context, allow_resize=False)
