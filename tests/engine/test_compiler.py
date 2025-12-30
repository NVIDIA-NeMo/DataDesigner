# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import duckdb
import pytest

from data_designer.config.column_configs import SamplerColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.errors import InvalidConfigError
from data_designer.config.sampler_params import CategorySamplerParams, SamplerType
from data_designer.config.seed_dataset import HuggingFaceSeedConfig
from data_designer.engine.compiler import compile_data_designer_config
from data_designer.engine.resources.resource_provider import ResourceProvider
from data_designer.engine.resources.seed_dataset import SeedDatasetReader
from data_designer.engine.validation import Violation, ViolationLevel, ViolationType


class TestSeedDatasetReader(SeedDatasetReader):
    def get_column_names(self) -> list[str]:
        return ["age", "city"]

    def get_dataset_uri(self) -> str:
        return "unused in these tests"

    def create_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect()


@pytest.fixture
def resource_provider(stub_resource_provider: ResourceProvider) -> ResourceProvider:
    stub_resource_provider.seed_dataset_reader = TestSeedDatasetReader()
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
    builder.with_seed_dataset(HuggingFaceSeedConfig(dataset="hf://datasets/test/data.csv"))

    config = compile_data_designer_config(builder, resource_provider)

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
    builder.with_seed_dataset(HuggingFaceSeedConfig(dataset="hf://datasets/test/data.csv"))

    with pytest.raises(InvalidConfigError) as excinfo:
        compile_data_designer_config(builder, resource_provider)

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
            compile_data_designer_config(builder, resource_provider)

    assert "validation errors" in str(excinfo)
