# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from data_designer.config.column_configs import SamplerColumnConfig, SeedDatasetColumnConfig
from data_designer.config.column_types import DataDesignerColumnType
from data_designer.config.sampler_constraints import (
    ColumnInequalityConstraint,
    InequalityOperator,
    ScalarInequalityConstraint,
)
from data_designer.config.sampler_params import (
    CategorySamplerParams,
    GaussianSamplerParams,
    SamplerType,
)
from data_designer.config.seed_source import HuggingFaceSeedSource
from data_designer.engine.dataset_builders.multi_column_configs import (
    MultiColumnConfig,
    SamplerMultiColumnConfig,
    SeedDatasetMultiColumnConfig,
)


class StubMultiColumnConfig(MultiColumnConfig):
    pass


def test_column_type_property():
    """Test the column_type property returns the type of the first column."""

    columns = [SeedDatasetColumnConfig(name="col1")]
    config = StubMultiColumnConfig(columns=columns)
    assert config.column_type == DataDesignerColumnType.SEED_DATASET


def test_validate_column_types_same_type():
    """Test validation passes when all columns have the same type."""

    columns = [
        SeedDatasetColumnConfig(name="col1"),
        SeedDatasetColumnConfig(name="col2"),
    ]

    config = StubMultiColumnConfig(columns=columns)
    assert len(config.columns) == 2


def test_validate_column_types_different_types():
    """Test validation fails when columns have different types."""

    columns = [
        SeedDatasetColumnConfig(name="col1"),
        SamplerColumnConfig(
            name="col2",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["a", "b", "c"]),
        ),
    ]

    with pytest.raises(ValidationError, match="All column types must be of the same type"):
        StubMultiColumnConfig(columns=columns)


def test_minimum_one_column_required():
    """Test that at least one column is required."""

    with pytest.raises(ValidationError, match="at least 1 item"):
        StubMultiColumnConfig(columns=[])


def test_sampler_multi_column_config_creation():
    """Test creating a SamplerMultiColumnConfig with valid data."""
    columns = [
        SamplerColumnConfig(
            name="col1",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["a", "b", "c"]),
        ),
        SamplerColumnConfig(
            name="col2",
            sampler_type=SamplerType.GAUSSIAN,
            params=GaussianSamplerParams(mean=0.0, stddev=1.0),
        ),
    ]

    config = SamplerMultiColumnConfig(columns=columns)
    assert len(config.columns) == 2
    assert config.column_names == ["col1", "col2"]
    assert config.column_type == DataDesignerColumnType.SAMPLER


def test_sampler_multi_column_config_with_constraints():
    """Test creating a SamplerMultiColumnConfig with constraints."""
    columns = [
        SamplerColumnConfig(
            name="col1",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["a", "b", "c"]),
        ),
    ]

    constraints = [
        ScalarInequalityConstraint(
            target_column="col1",
            rhs=0.5,
            operator=InequalityOperator.GT,
        ),
        ColumnInequalityConstraint(
            target_column="col1",
            rhs="col2",
            operator=InequalityOperator.LE,
        ),
    ]

    config = SamplerMultiColumnConfig(
        columns=columns,
        constraints=constraints,
        max_rejections_factor=10,
    )

    assert len(config.constraints) == 2
    assert config.max_rejections_factor == 10


def test_sampler_multi_column_config_default_values():
    """Test default values for SamplerMultiColumnConfig."""
    columns = [
        SamplerColumnConfig(
            name="col1",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["a", "b", "c"]),
        ),
    ]

    config = SamplerMultiColumnConfig(columns=columns)
    assert config.constraints == []
    assert config.max_rejections_factor == 5


def test_seed_dataset_multi_column_config_creation():
    """Test creating a SeedDatasetMultiColumnConfig with valid data."""
    columns = [
        SeedDatasetColumnConfig(name="col1"),
        SeedDatasetColumnConfig(name="col2"),
    ]

    config = SeedDatasetMultiColumnConfig(
        source=HuggingFaceSeedSource(path="hf://datasets/test/dataset"),
        columns=columns,
    )

    assert config.source.path == "hf://datasets/test/dataset"
    assert len(config.columns) == 2
    assert config.column_names == ["col1", "col2"]
    assert config.column_type == DataDesignerColumnType.SEED_DATASET
