# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.processing.utils import (
    concat_datasets,
    deserialize_json_string,
    parse_list_string,
    prepare_template_context,
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
    "value,expected_result",
    [
        pytest.param('{"key": "value", "number": 42}', {"key": "value", "number": 42}, id="object"),
        pytest.param('[{"a": 1}, 2]', [{"a": 1}, 2], id="array"),
        pytest.param('{"key": "value", "number": 42', '{"key": "value", "number": 42', id="invalid-json"),
        pytest.param(
            '{"inner": "{\\"a\\": 1}", "version": "1.10"}', {"inner": '{"a": 1}', "version": "1.10"}, id="one-pass"
        ),
        pytest.param("1.10", 1.1, id="number"),
        pytest.param("true", True, id="bool"),
        pytest.param("null", None, id="null"),
        pytest.param('"text"', "text", id="quoted-string"),
        pytest.param("", "", id="empty"),
    ],
)
def test_deserialize_json_string(value: str, expected_result: Any) -> None:
    assert deserialize_json_string(value) == expected_result


@pytest.mark.parametrize(
    "input_data,expected_result",
    [
        pytest.param('{"key": 1.10}', {"key": 1.1}, id="object-string"),
        pytest.param("[1, 2]", [1, 2], id="array-string"),
        pytest.param("1.10", "1.10", id="number-string"),
        pytest.param("true", "true", id="bool-string"),
        pytest.param("null", "null", id="null-string"),
        pytest.param('"text"', '"text"', id="quoted-string"),
        pytest.param("not_json", "not_json", id="plain-string"),
        pytest.param("", "", id="empty-string"),
        pytest.param(42, 42, id="number"),
        pytest.param(None, None, id="none"),
        pytest.param([], [], id="empty-list"),
        pytest.param({}, {}, id="empty-dict"),
        pytest.param(
            ["true", "1.10", '["a", "b"]', {"n": "007"}, 3],
            ["true", "1.10", ["a", "b"], {"n": "007"}, 3],
            id="list",
        ),
        pytest.param(
            {
                "version": "1.10",
                "flag": "true",
                "missing": "null",
                "big": "1e3",
                "quoted": '"text"',
                "padded": "007",
                "obj": '{"score": 95}',
                "arr": "[1, 2]",
                "nested": {"inner": "false", "deep": '{"ok": true}', "items": ["null", '{"x": 1}']},
                "number": 42,
            },
            {
                "version": "1.10",
                "flag": "true",
                "missing": "null",
                "big": "1e3",
                "quoted": '"text"',
                "padded": "007",
                "obj": {"score": 95},
                "arr": [1, 2],
                "nested": {"inner": "false", "deep": {"ok": True}, "items": ["null", {"x": 1}]},
                "number": 42,
            },
            id="dict",
        ),
    ],
)
def test_prepare_template_context(input_data: Any, expected_result: Any) -> None:
    assert prepare_template_context(input_data) == expected_result


def test_prepare_template_context_does_not_mutate_input() -> None:
    record = {"obj": '{"a": 1}', "nested": {"items": ['{"b": 2}']}, "native": {"c": [1]}}
    result = prepare_template_context(record)

    result["nested"]["items"][0]["b"] = 3
    result["native"]["c"].append(2)

    assert record == {"obj": '{"a": 1}', "nested": {"items": ['{"b": 2}']}, "native": {"c": [1]}}


@pytest.mark.parametrize(
    "input_string,expected_result",
    [
        ('["a", "b", "c"]', ["a", "b", "c"]),  # valid stringified json array
        ('[" a ", " b", "c "]', ["a", "b", "c"]),  # valid stringified json array with whitespace
        ('["a", "b", "c",]', ["a", "b", "c"]),  # valid stringified json array with trailing comma
        ("['a', 'b', 'c']", ["a", "b", "c"]),  # valid python-style list with single quotes
        ("['a', 'b', 'c', ]", ["a", "b", "c"]),  # valid python-style list with trailing comma
        ("simple string   ", ["simple string"]),  # simple string with whitespace
        ("[1, 2.5]", ["1", "2.5"]),  # non-string items become text, as in a native list
        ("123", ["123"]),  # JSON number stays the text the user wrote
        ("true", ["true"]),  # JSON boolean
        ("null", ["null"]),  # JSON null
        ('"text"', ['"text"']),  # JSON-quoted text is not split into characters
        ('{"a": 1,}', ['{"a": 1,}']),  # object with a trailing comma is not a list
        ("'text'", ["'text'"]),  # Python string literal is not split into characters
    ],
)
def test_parse_list_string_scenarios(input_string, expected_result):
    result = parse_list_string(input_string)
    assert result == expected_result
