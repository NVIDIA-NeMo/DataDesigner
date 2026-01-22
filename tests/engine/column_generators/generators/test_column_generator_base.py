# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

from data_designer.config.column_configs import ExpressionColumnConfig
from data_designer.engine.column_generators.generators.base import (
    ColumnGenerator,
    FromScratchColumnGenerator,
    GenerationStrategy,
)
from data_designer.engine.resources.resource_provider import ResourceProvider
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd


def _create_test_generator_class(strategy=GenerationStrategy.CELL_BY_CELL):
    """Helper function to create a test generator class."""

    class TestGenerator(ColumnGenerator[ExpressionColumnConfig]):
        @staticmethod
        def get_generation_strategy() -> GenerationStrategy:
            return strategy

        def generate(self, data):
            return data

    return TestGenerator


def _create_test_from_scratch_generator_class():
    """Helper function to create a test from-scratch generator class."""

    class TestFromScratchGenerator(FromScratchColumnGenerator[ExpressionColumnConfig]):
        @staticmethod
        def get_generation_strategy() -> GenerationStrategy:
            return GenerationStrategy.CELL_BY_CELL

        def generate(self, data):
            return data

        def generate_from_scratch(self, num_records: int):
            return pd.DataFrame({"test": [1] * num_records})

    return TestFromScratchGenerator


def _create_test_config_and_provider():
    """Helper function to create test config and resource provider."""
    config = ExpressionColumnConfig(name="test", expr="{{ col1 }}", dtype="str")
    resource_provider = Mock(spec=ResourceProvider)
    return config, resource_provider


def test_column_generator_can_generate_from_scratch_default():
    TestGenerator = _create_test_generator_class()
    config, resource_provider = _create_test_config_and_provider()
    generator = TestGenerator(config=config, resource_provider=resource_provider)
    assert generator.can_generate_from_scratch is False


def test_column_generator_generation_strategy_property():
    TestGenerator = _create_test_generator_class(GenerationStrategy.FULL_COLUMN)
    config, resource_provider = _create_test_config_and_provider()
    generator = TestGenerator(config=config, resource_provider=resource_provider)
    assert generator.get_generation_strategy() == GenerationStrategy.FULL_COLUMN


def test_column_generator_log_pre_generation():
    TestGenerator = _create_test_generator_class()
    config, resource_provider = _create_test_config_and_provider()
    generator = TestGenerator(config=config, resource_provider=resource_provider)
    generator.log_pre_generation()


def test_from_scratch_column_generator_can_generate_from_scratch():
    TestFromScratchGenerator = _create_test_from_scratch_generator_class()
    config, resource_provider = _create_test_config_and_provider()
    generator = TestFromScratchGenerator(config=config, resource_provider=resource_provider)
    assert generator.can_generate_from_scratch is True
