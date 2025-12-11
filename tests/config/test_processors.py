# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from data_designer.config.dataset_builders import BuildStage
from data_designer.config.errors import InvalidConfigError
from data_designer.config.processors import (
    DropColumnsProcessorConfig,
    ProcessorConfig,
    ProcessorType,
    SchemaTransformProcessorConfig,
    get_processor_config_from_kwargs,
)


def test_drop_columns_processor_config_creation():
    config = DropColumnsProcessorConfig(
        name="drop_columns_processor", build_stage=BuildStage.POST_BATCH, column_names=["col1", "col2"]
    )

    assert config.build_stage == BuildStage.POST_BATCH
    assert config.column_names == ["col1", "col2"]
    assert config.processor_type == ProcessorType.DROP_COLUMNS
    assert isinstance(config, ProcessorConfig)


def test_drop_columns_processor_config_validation():
    # Test unsupported stage raises error
    with pytest.raises(ValidationError, match="Invalid dataset builder stage"):
        DropColumnsProcessorConfig(
            name="drop_columns_processor", build_stage=BuildStage.PRE_BATCH, column_names=["col1"]
        )

    # Test missing required field raises error
    with pytest.raises(ValidationError, match="Field required"):
        DropColumnsProcessorConfig(name="drop_columns_processor", build_stage=BuildStage.POST_BATCH)


def test_drop_columns_processor_config_serialization():
    config = DropColumnsProcessorConfig(
        name="drop_columns_processor", build_stage=BuildStage.POST_BATCH, column_names=["col1", "col2"]
    )

    # Serialize to dict
    config_dict = config.model_dump()
    assert config_dict["build_stage"] == "post_batch"
    assert config_dict["column_names"] == ["col1", "col2"]

    # Deserialize from dict
    config_restored = DropColumnsProcessorConfig.model_validate(config_dict)
    assert config_restored.build_stage == config.build_stage
    assert config_restored.column_names == config.column_names


def test_schema_transform_processor_config_creation():
    config = SchemaTransformProcessorConfig(
        name="output_format_processor",
        build_stage=BuildStage.POST_BATCH,
        template={"text": "{{ col1 }}"},
    )

    assert config.build_stage == BuildStage.POST_BATCH
    assert config.template == {"text": "{{ col1 }}"}
    assert config.processor_type == ProcessorType.SCHEMA_TRANSFORM
    assert isinstance(config, ProcessorConfig)


def test_schema_transform_processor_config_validation():
    # Test unsupported stage raises error
    with pytest.raises(ValidationError, match="Invalid dataset builder stage"):
        SchemaTransformProcessorConfig(
            name="schema_transform_processor",
            build_stage=BuildStage.PRE_BATCH,
            template={"text": "{{ col1 }}"},
        )

    # Test missing required field raises error
    with pytest.raises(ValidationError, match="Field required"):
        SchemaTransformProcessorConfig(name="schema_transform_processor", build_stage=BuildStage.POST_BATCH)

    # Test invalid template raises error
    with pytest.raises(InvalidConfigError, match="Template must be JSON serializable"):
        SchemaTransformProcessorConfig(
            name="schema_transform_processor", build_stage=BuildStage.POST_BATCH, template={"text": {1, 2, 3}}
        )


def test_schema_transform_processor_config_serialization():
    config = SchemaTransformProcessorConfig(
        name="schema_transform_processor",
        build_stage=BuildStage.POST_BATCH,
        template={"text": "{{ col1 }}"},
    )

    # Serialize to dict
    config_dict = config.model_dump()
    assert config_dict["build_stage"] == "post_batch"
    assert config_dict["template"] == {"text": "{{ col1 }}"}

    # Deserialize from dict
    config_restored = SchemaTransformProcessorConfig.model_validate(config_dict)
    assert config_restored.build_stage == config.build_stage
    assert config_restored.template == config.template


def test_get_processor_config_from_kwargs():
    # Test successful creation
    config_drop_columns = get_processor_config_from_kwargs(
        ProcessorType.DROP_COLUMNS,
        name="drop_columns_processor",
        build_stage=BuildStage.POST_BATCH,
        column_names=["col1"],
    )
    assert isinstance(config_drop_columns, DropColumnsProcessorConfig)
    assert config_drop_columns.column_names == ["col1"]
    assert config_drop_columns.processor_type == ProcessorType.DROP_COLUMNS

    config_schema_transform = get_processor_config_from_kwargs(
        ProcessorType.SCHEMA_TRANSFORM,
        name="output_format_processor",
        build_stage=BuildStage.POST_BATCH,
        template={"text": "{{ col1 }}"},
    )
    assert isinstance(config_schema_transform, SchemaTransformProcessorConfig)
    assert config_schema_transform.template == {"text": "{{ col1 }}"}
    assert config_schema_transform.processor_type == ProcessorType.SCHEMA_TRANSFORM

    # Test with unknown processor type returns None
    from enum import Enum

    class UnknownProcessorType(str, Enum):
        UNKNOWN = "unknown"

    result = get_processor_config_from_kwargs(
        UnknownProcessorType.UNKNOWN, name="unknown_processor", build_stage=BuildStage.POST_BATCH, column_names=["col1"]
    )
    assert result is None
