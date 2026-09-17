# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from data_designer.config.column_configs import ExpressionColumnConfig, SamplerColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.errors import InvalidConfigError
from data_designer.config.processors import DropColumnsProcessorConfig
from data_designer.config.sampler_params import CategorySamplerParams, SamplerType, UUIDSamplerParams
from data_designer.config.seed_source import FileContentsSeedSource, HuggingFaceSeedSource
from data_designer.engine.compiler import compile_data_designer_config
from data_designer.engine.resources.resource_provider import ResourceProvider
from data_designer.engine.resources.seed_reader import FileContentsSeedReader, SeedReader, SeedReaderConfigError
from data_designer.engine.secret_resolver import PlaintextResolver
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


def test_seed_reader_config_errors_are_invalid_config_errors(resource_provider: ResourceProvider):
    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(HuggingFaceSeedSource(path="hf://datasets/test/data.csv"))
    resource_provider.seed_reader = Mock(spec=SeedReader)
    resource_provider.seed_reader.get_column_names.side_effect = SeedReaderConfigError("missing seed root")

    with pytest.raises(InvalidConfigError, match="missing seed root") as excinfo:
        compile_data_designer_config(builder.build(), resource_provider)

    assert isinstance(excinfo.value.__cause__, SeedReaderConfigError)


def test_compile_rejects_missing_fixed_schema_filesystem_seed_root(
    stub_resource_provider: ResourceProvider,
    tmp_path,
):
    missing_dir = tmp_path / "missing"
    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(FileContentsSeedSource(path=str(missing_dir), file_pattern="*.txt"))
    reader = FileContentsSeedReader()
    reader.attach(builder.build().seed_config.source, PlaintextResolver())
    stub_resource_provider.seed_reader = reader

    with pytest.raises(InvalidConfigError, match="Seed source directory .* does not exist") as excinfo:
        compile_data_designer_config(builder.build(), stub_resource_provider)

    assert isinstance(excinfo.value.__cause__, SeedReaderConfigError)


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


def test_compile_applies_processor_columns_added(resource_provider: ResourceProvider):
    """Test that columns declared in columns_added can be referenced by downstream expressions/templates."""
    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(HuggingFaceSeedSource(path="hf://datasets/test/data.csv"))
    builder.add_processor(
        DropColumnsProcessorConfig(
            name="pre_batch_add",
            column_names=[],
            columns_added=["state"],
        )
    )
    builder.add_column(
        ExpressionColumnConfig(
            name="derived_value",
            expr="{{ state }}_processed",
        )
    )

    config = compile_data_designer_config(builder.build(), resource_provider)

    column_names = [col.name for col in config.columns]
    assert "state" in column_names
    assert "city" in column_names
    assert "age" in column_names
    assert "derived_value" in column_names


def test_compile_applies_processor_columns_removed(resource_provider: ResourceProvider):
    """Test that columns declared in columns_removed are removed and cannot be referenced downstream."""
    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(HuggingFaceSeedSource(path="hf://datasets/test/data.csv"))
    builder.add_processor(
        DropColumnsProcessorConfig(
            name="pre_batch_drop",
            column_names=[],
            columns_removed=["city"],
        )
    )
    builder.add_column(
        ExpressionColumnConfig(
            name="derived_value",
            expr="{{ age }}_processed",
        )
    )

    config = compile_data_designer_config(builder.build(), resource_provider)
    column_names = [col.name for col in config.columns]
    assert "city" not in column_names
    assert "age" in column_names

    # If downstream references the removed column, compilation/validation should fail
    builder_invalid = DataDesignerConfigBuilder()
    builder_invalid.with_seed_dataset(HuggingFaceSeedSource(path="hf://datasets/test/data.csv"))
    builder_invalid.add_processor(
        DropColumnsProcessorConfig(
            name="pre_batch_drop",
            column_names=[],
            columns_removed=["city"],
        )
    )
    builder_invalid.add_column(
        ExpressionColumnConfig(
            name="derived_value",
            expr="{{ city }}_processed",
        )
    )
    with pytest.raises(InvalidConfigError, match="validation errors"):
        compile_data_designer_config(builder_invalid.build(), resource_provider)


def test_compile_processor_columns_added_collision(resource_provider: ResourceProvider):
    """Test that adding an already existing column via columns_added raises InvalidConfigError."""
    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(HuggingFaceSeedSource(path="hf://datasets/test/data.csv"))
    builder.add_processor(
        DropColumnsProcessorConfig(
            name="pre_batch_add",
            column_names=[],
            columns_added=["city"],
        )
    )

    with pytest.raises(InvalidConfigError, match="collides with an existing column"):
        compile_data_designer_config(builder.build(), resource_provider)


def test_compile_processor_columns_added_duplicate(resource_provider: ResourceProvider):
    """Test that specifying duplicate columns in columns_added raises InvalidConfigError."""
    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(HuggingFaceSeedSource(path="hf://datasets/test/data.csv"))
    builder.add_processor(
        DropColumnsProcessorConfig(
            name="pre_batch_add",
            column_names=[],
            columns_added=["state", "state"],
        )
    )

    with pytest.raises(InvalidConfigError, match="collides with an existing column"):
        compile_data_designer_config(builder.build(), resource_provider)


def test_compile_processor_columns_removed_nonexistent(resource_provider: ResourceProvider):
    """Test that removing a non-existent column via columns_removed raises InvalidConfigError."""
    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(HuggingFaceSeedSource(path="hf://datasets/test/data.csv"))
    builder.add_processor(
        DropColumnsProcessorConfig(
            name="pre_batch_drop",
            column_names=[],
            columns_removed=["non_existent"],
        )
    )

    with pytest.raises(InvalidConfigError, match="cannot remove column 'non_existent' because it does not exist"):
        compile_data_designer_config(builder.build(), resource_provider)


def test_compile_processor_columns_added_without_seed_dataset(stub_resource_provider: ResourceProvider):
    """Test that columns_added without a seed dataset raises InvalidConfigError."""
    builder = DataDesignerConfigBuilder()
    builder.add_processor(
        DropColumnsProcessorConfig(
            name="pre_batch_add",
            column_names=[],
            columns_added=["state"],
        )
    )
    stub_resource_provider.seed_reader = None

    with pytest.raises(InvalidConfigError, match="specifies 'columns_added', but no seed dataset is configured"):
        compile_data_designer_config(builder.build(), stub_resource_provider)
