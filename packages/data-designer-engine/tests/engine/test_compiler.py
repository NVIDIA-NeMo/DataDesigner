# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from data_designer.config.column_configs import ExpressionColumnConfig, SamplerColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.errors import InvalidConfigError
from data_designer.config.sampler_params import CategorySamplerParams, SamplerType, UUIDSamplerParams
from data_designer.config.seed_source import HuggingFaceSeedSource
from data_designer.engine.compiler import compile_data_designer_config
from data_designer.engine.resources.resource_provider import ResourceProvider
from data_designer.engine.resources.seed_reader import SeedReader
from data_designer.engine.validation import Violation, ViolationLevel, ViolationType


@pytest.fixture
def resource_provider(stub_resource_provider: ResourceProvider, stub_seed_reader: SeedReader) -> ResourceProvider:
    stub_resource_provider.seed_reader = stub_seed_reader
    return stub_resource_provider


def test_adds_seed_columns(resource_provider: ResourceProvider):
    builder = DataDesignerConfigBuilder()
    builder.add_column(
        SamplerColumnConfig(
            name="language",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["english", "french"]),
        )
    )
    builder.with_seed_dataset(HuggingFaceSeedSource(path="hf://datasets/test/data.csv"))

    config = compile_data_designer_config(builder.build(), resource_provider)

    assert len(config.columns) == 3


def test_errors_on_seed_column_collisions(resource_provider: ResourceProvider):
    builder = DataDesignerConfigBuilder()
    builder.add_column(
        SamplerColumnConfig(
            name="city",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["new york", "los angeles"]),
        )
    )
    builder.with_seed_dataset(HuggingFaceSeedSource(path="hf://datasets/test/data.csv"))

    with pytest.raises(InvalidConfigError) as excinfo:
        compile_data_designer_config(builder.build(), resource_provider)

    assert "city" in str(excinfo)


def test_validation_errors(resource_provider: ResourceProvider):
    builder = DataDesignerConfigBuilder()
    builder.add_column(
        SamplerColumnConfig(
            name="language",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["english", "french"]),
        )
    )

    with patch("data_designer.engine.compiler.validate_data_designer_config") as patched_validate:
        patched_validate.return_value = [
            Violation(
                type=ViolationType.INVALID_COLUMN,
                message="Some error",
                level=ViolationLevel.ERROR,
            )
        ]

        with pytest.raises(InvalidConfigError) as excinfo:
            compile_data_designer_config(builder.build(), resource_provider)

    assert "validation errors" in str(excinfo)


def test_adds_id_column_when_no_sampler_and_no_seed_dataset(stub_resource_provider: ResourceProvider):
    """Test that a UUID '_internal_row_id' column is automatically added when there's no sampler column or seed dataset."""
    builder = DataDesignerConfigBuilder()
    builder.add_column(
        ExpressionColumnConfig(
            name="derived_value",
            expr="'constant_value'",
        )
    )
    stub_resource_provider.seed_reader = None

    config = compile_data_designer_config(builder.build(), stub_resource_provider)

    assert len(config.columns) == 2
    assert config.columns[0].name == "_internal_row_id"
    assert isinstance(config.columns[0], SamplerColumnConfig)
    assert config.columns[0].sampler_type == "uuid"
    assert isinstance(config.columns[0].params, UUIDSamplerParams)
    assert config.columns[0].drop is True


def test_does_not_add_id_column_when_sampler_exists(stub_resource_provider: ResourceProvider):
    """Test that no '_internal_row_id' column is added when a sampler column already exists."""
    builder = DataDesignerConfigBuilder()
    builder.add_column(
        SamplerColumnConfig(
            name="category",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["a", "b", "c"]),
        )
    )
    builder.add_column(
        ExpressionColumnConfig(
            name="derived_value",
            expr="{{ category }}_suffix",
        )
    )
    stub_resource_provider.seed_reader = None

    config = compile_data_designer_config(builder.build(), stub_resource_provider)

    assert len(config.columns) == 2
    assert config.columns[0].name == "category"
    assert config.columns[1].name == "derived_value"
    assert not any(col.name == "_internal_row_id" for col in config.columns)


def test_does_not_add_id_column_when_seed_dataset_exists(resource_provider: ResourceProvider):
    """Test that no '_internal_row_id' column is added when a seed dataset is configured."""
    builder = DataDesignerConfigBuilder()
    builder.add_column(
        ExpressionColumnConfig(
            name="derived_value",
            expr="{{ city }}_derived",
        )
    )
    builder.with_seed_dataset(HuggingFaceSeedSource(path="hf://datasets/test/data.csv"))

    config = compile_data_designer_config(builder.build(), resource_provider)

    # Should have the expression column + 2 seed columns (city, country) from the fixture
    assert len(config.columns) == 3
    assert config.columns[0].name == "derived_value"
    assert not any(col.name == "_internal_row_id" for col in config.columns)
