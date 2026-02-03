# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from data_designer.config.column_configs import ExpressionColumnConfig
from data_designer.engine.column_generators.generators.expression import ExpressionColumnGenerator
from data_designer.engine.column_generators.utils.errors import ExpressionTemplateRenderError
from data_designer.engine.resources.resource_provider import ResourceProvider
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd


def _create_test_config(name="test_column", expr="{{ col1 }}", dtype="str"):
    """Helper function to create test expression config."""
    return ExpressionColumnConfig(name=name, expr=expr, dtype=dtype)


def _create_test_generator(config=None, resource_provider=None):
    """Helper function to create test generator."""
    if config is None:
        config = _create_test_config()
    if resource_provider is None:
        resource_provider = Mock(spec=ResourceProvider)
    return ExpressionColumnGenerator(config=config, resource_provider=resource_provider)


def test_generator_creation():
    config = _create_test_config("test_column", "{{ col1 + col2 }}", "int")
    generator = _create_test_generator(config)
    assert generator.config == config


@patch(
    "data_designer.engine.column_generators.generators.expression.WithJinja2UserTemplateRendering.prepare_jinja2_template_renderer",
    autospec=True,
)
@patch(
    "data_designer.engine.column_generators.generators.expression.WithJinja2UserTemplateRendering.render_template",
    autospec=True,
)
@patch("data_designer.engine.column_generators.generators.expression.deserialize_json_values", autospec=True)
def test_generate_with_dataframe(mock_deserialize, mock_render, mock_prepare):
    config = _create_test_config("sum_column", "{{ col1 + col2 }}", "int")
    generator = _create_test_generator(config)

    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

    mock_prepare.return_value = None
    mock_deserialize.side_effect = lambda x: x
    mock_render.side_effect = ["5", "7", "9"]

    result = generator.generate(df)

    assert "sum_column" in result.columns
    assert result["sum_column"].tolist() == [5, 7, 9]

    mock_prepare.assert_called_once_with(generator, "{{ col1 + col2 }}", ["col1", "col2"])
    assert mock_render.call_count == 3
    assert mock_deserialize.call_count == 3


@patch(
    "data_designer.engine.column_generators.generators.expression.WithJinja2UserTemplateRendering.prepare_jinja2_template_renderer",
    autospec=True,
)
@patch(
    "data_designer.engine.column_generators.generators.expression.WithJinja2UserTemplateRendering.render_template",
    autospec=True,
)
@patch("data_designer.engine.column_generators.generators.expression.deserialize_json_values", autospec=True)
def test_generate_with_different_dtypes(mock_deserialize, mock_render, mock_prepare):
    config = _create_test_config("result_column", "{{ col1 }}", "float")
    generator = _create_test_generator(config)

    df = pd.DataFrame({"col1": [1, 2, 3]})

    mock_prepare.return_value = None
    mock_deserialize.side_effect = lambda x: x
    mock_render.side_effect = ["1.5", "2.7", "3.9"]

    result = generator.generate(df)

    assert "result_column" in result.columns
    assert result["result_column"].tolist() == [1.5, 2.7, 3.9]


@patch(
    "data_designer.engine.column_generators.generators.expression.WithJinja2UserTemplateRendering.prepare_jinja2_template_renderer",
    autospec=True,
)
@patch(
    "data_designer.engine.column_generators.generators.expression.WithJinja2UserTemplateRendering.render_template",
    autospec=True,
)
@patch("data_designer.engine.column_generators.generators.expression.deserialize_json_values", autospec=True)
def test_generate_with_bool_dtype_numeric(mock_deserialize, mock_render, mock_prepare):
    config = _create_test_config("bool_column", "{{ col1 }}", "bool")
    generator = _create_test_generator(config)

    df = pd.DataFrame({"col1": [1, 0, 1]})

    mock_prepare.return_value = None
    mock_deserialize.side_effect = lambda x: x
    mock_render.side_effect = ["1", "0", "1.0"]

    result = generator.generate(df)

    assert "bool_column" in result.columns
    assert result["bool_column"].tolist() == [True, False, True]


@patch(
    "data_designer.engine.column_generators.generators.expression.WithJinja2UserTemplateRendering.prepare_jinja2_template_renderer",
    autospec=True,
)
@patch(
    "data_designer.engine.column_generators.generators.expression.WithJinja2UserTemplateRendering.render_template",
    autospec=True,
)
@patch("data_designer.engine.column_generators.generators.expression.deserialize_json_values", autospec=True)
def test_generate_with_bool_dtype_string(mock_deserialize, mock_render, mock_prepare):
    config = _create_test_config("bool_column", "{{ col1 }}", "bool")
    generator = _create_test_generator(config)

    df = pd.DataFrame({"col1": ["true", "false", "True"]})

    mock_prepare.return_value = None
    mock_deserialize.side_effect = lambda x: x
    mock_render.side_effect = ["true", "false", "True"]

    result = generator.generate(df)

    assert "bool_column" in result.columns
    assert result["bool_column"].tolist() == [True, False, True]


def test_cast_type_invalid_dtype():
    config = _create_test_config("test_column", "{{ col1 }}", "str")
    generator = _create_test_generator(config)

    # Bypass pydantic validation to test the ValueError branch
    generator.config.dtype = "invalid"

    with pytest.raises(ValueError, match="Invalid dtype: invalid"):
        generator._cast_type("test_value")


def test_generate_with_missing_columns():
    config = _create_test_config("test_column", "{{ col1 }}", "str")
    generator = _create_test_generator(config)

    df = pd.DataFrame({"col2": [1, 2, 3]})

    with pytest.raises(
        ExpressionTemplateRenderError,
        match=r"There was an error preparing the Jinja2 expression template. The following columns \['col1'\] are missing!",
    ):
        generator.generate(df)
