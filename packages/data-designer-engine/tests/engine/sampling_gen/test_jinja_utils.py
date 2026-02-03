# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from data_designer.engine.sampling_gen.jinja_utils import JinjaDataFrame, extract_column_names_from_expression
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd


@pytest.mark.parametrize(
    ("expr", "column_names"),
    [
        ("x + y + 5", {"x", "y"}),
        ("2 * x / y + z", {"x", "y", "z"}),
        ("snake_case / 100 - 1.2", {"snake_case"}),
        ("no_space/4-x", {"no_space", "x"}),
        ("50_000 + amount.cumsum()", {"amount"}),
        ("yourColumn < 0", {"yourColumn"}),
        ("myColumn >= 100", {"myColumn"}),
        ("my_column != 0", {"my_column"}),
        ("some_dude.age + 10", {"some_dude"}),
        ("'I am a string' + i_am_a_var", {"i_am_a_var"}),
        ("some_dude.age + 1", {"some_dude"}),
        ("'I\\'m a string' + i_am_a_var", {"i_am_a_var"}),
        ('"I am a string" + i_am_a_var', {"i_am_a_var"}),
    ],
)
def test_extract_column_names_from_expression(expr: str, column_names: set[str]) -> None:
    assert extract_column_names_from_expression(expr) == column_names


@pytest.mark.parametrize(
    ("expr", "column_names"),
    [
        ("x and y", {"x", "y"}),
        ("x or y", {"x", "y"}),
        ("x in y", {"x", "y"}),
        ("i_am_awesome == True", {"i_am_awesome"}),
        ("you_are_awesome == False", {"you_are_awesome"}),
        ("this_is_none == None", {"this_is_none"}),
    ],
)
def test_extract_column_names_ignore_special_keywords(expr: str, column_names: set[str]) -> None:
    assert extract_column_names_from_expression(expr) == column_names


def test_jinja_dataframe_init():
    jdf = JinjaDataFrame("x + y")
    assert jdf.expr == "x + y"
    assert jdf._expr == "{{ x + y }}"


@pytest.mark.parametrize(
    "test_case,expr,df_data,mock_side_effect,expected_result",
    [
        ("empty_dataframe", "x > 0", {}, None, "empty_index"),
        ("ellipsis", "...", {"x": [1, 2, 3]}, None, "full_index"),
        ("with_condition", "x > 1", {"x": [1, 2, 3], "y": [4, 5, 6]}, ["False", "True", "True"], [1, 2]),
    ],
)
def test_jinja_dataframe_select_index_scenarios(test_case, expr, df_data, mock_side_effect, expected_result):
    jdf = JinjaDataFrame(expr)

    if df_data:
        df = pd.DataFrame(df_data)
    else:
        df = pd.DataFrame()

    if test_case == "with_condition":
        jdf.prepare_jinja2_template_renderer = Mock()
        jdf.render_template = Mock(side_effect=mock_side_effect)
        result = jdf.select_index(df)
        assert len(result) == len(expected_result)
        assert result.tolist() == expected_result
    else:
        result = jdf.select_index(df)
        if expected_result == "empty_index":
            assert result.equals(df.index)
        elif expected_result == "full_index":
            assert result.equals(df.index)


@pytest.mark.parametrize(
    "test_case,expr,df_data,mock_side_effect,expected_result",
    [
        ("numeric_operation", "x * 2", {"x": [1, 2, 3]}, ["2", "4", "6"], [2, 4, 6]),
        (
            "string_operation",
            "name + '_test'",
            {"name": ["John", "Jane"]},
            ["John_test", "Jane_test"],
            ["John_test", "Jane_test"],
        ),
        ("syntax_error", "x +", {"x": [1, 2]}, ["1 +", "2 +"], ["1 +", "2 +"]),
        (
            "value_error",
            "x +",
            {"x": [1, 2]},
            ["invalid_literal", "another_invalid"],
            ["invalid_literal", "another_invalid"],
        ),
    ],
)
def test_jinja_dataframe_to_column_scenarios(test_case, expr, df_data, mock_side_effect, expected_result):
    jdf = JinjaDataFrame(expr)
    df = pd.DataFrame(df_data)

    jdf.prepare_jinja2_template_renderer = Mock()
    jdf.render_template = Mock(side_effect=mock_side_effect)
    result = jdf.to_column(df)
    assert result == expected_result
