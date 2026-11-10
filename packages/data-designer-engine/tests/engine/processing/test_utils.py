# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.processing.utils import (
    concat_datasets,
    dataframe_to_records,
    deserialize_json_values,
    parse_list_string,
)


@pytest.fixture
def stub_sample_dataframes():
    return {
        "df1": lazy.pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]}),
        "df2": lazy.pd.DataFrame({"col3": [4, 5, 6], "col4": ["d", "e", "f"]}),
        "df_single": lazy.pd.DataFrame({"col1": [1, 2, 3]}),
    }


@pytest.fixture
def stub_overlapping_dataframes():
    return {
        "df1": lazy.pd.DataFrame({"col1": [1, 2, 3]}),
        "df2": lazy.pd.DataFrame({"col1": [4, 5, 6]}),
    }


@pytest.fixture
def stub_different_length_dataframes():
    return {
        "df1": lazy.pd.DataFrame({"col1": [1, 2, 3]}),
        "df2": lazy.pd.DataFrame({"col2": [4, 5]}),
    }


@pytest.mark.parametrize(
    "test_case,dataframes_key,expected_result,expected_error",
    [
        (
            "concat_success",
            "stub_sample_dataframes",
            {"col1": [1, 2, 3], "col2": ["a", "b", "c"], "col3": [4, 5, 6], "col4": ["d", "e", "f"]},
            None,
        ),
        ("concat_single_dataset", "stub_sample_dataframes", {"col1": [1, 2, 3]}, None),
        ("overlapping_columns_error", "stub_overlapping_dataframes", None, ValueError),
        ("different_lengths_error", "stub_different_length_dataframes", None, ValueError),
    ],
)
def test_concat_datasets_scenarios(request, test_case, dataframes_key, expected_result, expected_error):
    if dataframes_key == "stub_sample_dataframes":
        if test_case == "concat_success":
            dfs = request.getfixturevalue("stub_sample_dataframes")
            datasets = [dfs["df1"], dfs["df2"]]
        else:  # concat_single_dataset
            dfs = request.getfixturevalue("stub_sample_dataframes")
            datasets = [dfs["df_single"]]
    elif dataframes_key == "stub_overlapping_dataframes":
        dfs = request.getfixturevalue("stub_overlapping_dataframes")
        datasets = [dfs["df1"], dfs["df2"]]
    elif dataframes_key == "stub_different_length_dataframes":
        dfs = request.getfixturevalue("stub_different_length_dataframes")
        datasets = [dfs["df1"], dfs["df2"]]

    if expected_error:
        with pytest.raises(expected_error):
            concat_datasets(datasets)
    else:
        result = concat_datasets(datasets)
        lazy.pd.testing.assert_frame_equal(result, lazy.pd.DataFrame(expected_result))


@patch("data_designer.engine.processing.utils.logger", autospec=True)
def test_concat_datasets_logging(mock_logger, stub_sample_dataframes):
    datasets = [stub_sample_dataframes["df1"], stub_sample_dataframes["df2"]]
    concat_datasets(datasets)
    mock_logger.info.assert_called_once_with("(💾 + 💾) Concatenating 2 datasets")


@pytest.mark.parametrize(
    "test_case,input_data,expected_result",
    [
        ("single_string_valid_json", '{"key": "value", "number": 42}', {"key": "value", "number": 42}),
        ("single_string_invalid_json", '{"key": "value", "number": 42', '{"key": "value", "number": 42'),
        ("list_of_strings", ['{"a": 1}', '{"b": 2}', "invalid_json"], [{"a": 1}, {"b": 2}, "invalid_json"]),
        ("list_with_nested_structures", ['{"a": 1}', [2, 3], {"c": "d"}], [{"a": 1}, [2, 3], {"c": "d"}]),
        (
            "dict_with_json_strings",
            {"json_str": '{"nested": "value"}', "regular_str": "not_json", "number": 42},
            {"json_str": {"nested": "value"}, "regular_str": "not_json", "number": 42},
        ),
        (
            "dict_with_nested_structures",
            {
                "json_str": '{"nested": "value"}',
                "nested_dict": {"inner": '{"deep": "value"}'},
                "nested_list": ['{"item": 1}', 2, 3],
            },
            {
                "json_str": {"nested": "value"},
                "nested_dict": {"inner": {"deep": "value"}},
                "nested_list": [{"item": 1}, 2, 3],
            },
        ),
        ("non_string_non_dict_non_list", 42, 42),
        ("none", None, None),
        ("empty_string", "", ""),
        ("empty_list", [], []),
        ("empty_dict", {}, {}),
    ],
)
def test_deserialize_json_values_scenarios(test_case, input_data, expected_result):
    result = deserialize_json_values(input_data)
    assert result == expected_result


@pytest.mark.parametrize(
    "input_string,expected_result",
    [
        ('["a", "b", "c"]', ["a", "b", "c"]),  # valid stringified json array
        ('[" a ", " b", "c "]', ["a", "b", "c"]),  # valid stringified json array with whitespace
        ('["a", "b", "c",]', ["a", "b", "c"]),  # valid stringified json array with trailing comma
        ("['a', 'b', 'c']", ["a", "b", "c"]),  # valid python-style list with single quotes
        ("['a', 'b', 'c', ]", ["a", "b", "c"]),  # valid python-style list with trailing comma
        ("simple string   ", ["simple string"]),  # simple string with whitespace
    ],
)
def test_parse_list_string_scenarios(input_string, expected_result):
    result = parse_list_string(input_string)
    assert result == expected_result


@pytest.mark.parametrize(
    "missing,dtype",
    [(None, None), (None, "string"), (None, "string[pyarrow]"), (float("nan"), object)],
    ids=["inferred", "string", "string-pyarrow", "object-nan"],
)
def test_dataframe_to_records_gives_none_for_missing_text(missing: object, dtype: str | type | None) -> None:
    df = lazy.pd.DataFrame({"value": lazy.pd.Series(["present", missing, ""], dtype=dtype)})

    records = dataframe_to_records(df)

    assert [record["value"] for record in records] == ["present", None, ""]
    assert df["value"].isna().tolist() == [False, True, False]


def test_dataframe_to_records_keeps_numeric_datetime_and_nested_values() -> None:
    df = lazy.pd.DataFrame(
        {
            "score": [1.5, float("nan")],
            "when": [lazy.pd.Timestamp("2026-01-01"), lazy.pd.NaT],
            "nested": [{"key": None}, ["a", None]],
            "text": ["nan", "None"],
        }
    )

    records = dataframe_to_records(df)

    assert records[0] == {
        "score": 1.5,
        "when": lazy.pd.Timestamp("2026-01-01"),
        "nested": {"key": None},
        "text": "nan",
    }
    assert math.isnan(records[1]["score"])
    assert records[1]["when"] is lazy.pd.NaT
    assert records[1]["nested"] == ["a", None]
    assert records[1]["text"] == "None"
