# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from data_designer.config.columns import CustomColumnConfig
from data_designer.engine.column_generators.generators.custom import CustomColumnGenerator
from data_designer.engine.errors import DataDesignerRuntimeError


def test_generate_successful_custom_column(stub_resource_provider: object) -> None:
    """Test successful generation of a custom column."""

    def add_sum_column(data: pd.DataFrame) -> pd.DataFrame:
        data["sum_column"] = data["col1"] + data["other_col"]
        return data

    config = CustomColumnConfig(name="sum_column", generator_function=add_sum_column)
    generator = CustomColumnGenerator(config=config, resource_provider=stub_resource_provider)

    df = pd.DataFrame({"col1": [1, 2, 3, 4], "other_col": [10, 20, 30, 40]})
    result = generator.generate(df)

    assert "sum_column" in result.columns
    assert result["sum_column"].tolist() == [11, 22, 33, 44]
    assert len(result) == 4


def test_generate_custom_column_with_string_data(stub_resource_provider: object) -> None:
    """Test custom column generation with string manipulation."""

    def add_full_name_column(data: pd.DataFrame) -> pd.DataFrame:
        data["full_name"] = data["first_name"] + " " + data["last_name"]
        return data

    config = CustomColumnConfig(name="full_name", generator_function=add_full_name_column)
    generator = CustomColumnGenerator(config=config, resource_provider=stub_resource_provider)

    df = pd.DataFrame({"first_name": ["John", "Jane", "Bob"], "last_name": ["Doe", "Smith", "Johnson"]})
    result = generator.generate(df)

    assert "full_name" in result.columns
    assert result["full_name"].tolist() == ["John Doe", "Jane Smith", "Bob Johnson"]


def test_generate_error_when_unexpected_columns_added(stub_resource_provider: object) -> None:
    """Test that an error is raised when the generator adds unexpected columns."""

    def add_multiple_columns(data: pd.DataFrame) -> pd.DataFrame:
        data["expected_column"] = data["col1"] * 2
        data["unexpected_column"] = data["col1"] * 3  # This should cause an error
        return data

    config = CustomColumnConfig(name="expected_column", generator_function=add_multiple_columns)
    generator = CustomColumnGenerator(config=config, resource_provider=stub_resource_provider)

    df = pd.DataFrame({"col1": [1, 2, 3]})

    with pytest.raises(
        DataDesignerRuntimeError,
        match=r"Custom column generator add_multiple_columns added unexpected columns: unexpected_column",
    ):
        generator.generate(df)


def test_generate_error_when_no_column_added(stub_resource_provider: object) -> None:
    """Test that an error is raised when the generator doesn't add the expected column."""

    def add_no_columns(data: pd.DataFrame) -> pd.DataFrame:
        return data

    config = CustomColumnConfig(name="missing_column", generator_function=add_no_columns)
    generator = CustomColumnGenerator(config=config, resource_provider=stub_resource_provider)

    df = pd.DataFrame({"col1": [1, 2, 3]})

    with pytest.raises(
        DataDesignerRuntimeError,
        match=r"Custom column generator add_no_columns added no columns",
    ):
        generator.generate(df)


def test_generate_error_when_generator_function_raises_exception(stub_resource_provider: object) -> None:
    """Test that exceptions from the generator function are properly wrapped."""

    def failing_generator(data: pd.DataFrame) -> pd.DataFrame:
        raise ValueError("Something went wrong in the generator")

    config = CustomColumnConfig(name="test_column", generator_function=failing_generator)
    generator = CustomColumnGenerator(config=config, resource_provider=stub_resource_provider)

    df = pd.DataFrame({"col1": [1, 2, 3]})

    with pytest.raises(
        DataDesignerRuntimeError, match=r"Error generating custom column 'test_column': Something went wrong"
    ):
        generator.generate(df)
