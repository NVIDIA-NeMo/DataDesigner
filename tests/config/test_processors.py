# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from data_designer.config.dataset_builders import BuildStage
from data_designer.config.processors import (
    AncillaryDatasetProcessorConfig,
    DropColumnsProcessorConfig,
    ProcessorConfig,
    ProcessorType,
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


def test_output_format_processor_config_creation():
    config = AncillaryDatasetProcessorConfig(
        name="output_format_processor",
        build_stage=BuildStage.POST_BATCH,
        template='{"text": "{{ col1 }}"}',
    )

    assert config.build_stage == BuildStage.POST_BATCH
    assert config.template == '{"text": "{{ col1 }}"}'
    assert config.processor_type == ProcessorType.OUTPUT_FORMAT
    assert isinstance(config, ProcessorConfig)


def test_output_format_processor_config_validation():
    # Test unsupported stage raises error
    with pytest.raises(ValidationError, match="Invalid dataset builder stage"):
        AncillaryDatasetProcessorConfig(
            name="output_format_processor",
            build_stage=BuildStage.PRE_BATCH,
            template='{"text": "{{ col1 }}"}',
        )

    # Test missing required field raises error
    with pytest.raises(ValidationError, match="Field required"):
        AncillaryDatasetProcessorConfig(name="output_format_processor", build_stage=BuildStage.POST_BATCH)


def test_output_format_processor_config_serialization():
    config = AncillaryDatasetProcessorConfig(
        name="output_format_processor",
        build_stage=BuildStage.POST_BATCH,
        template='{"text": "{{ col1 }}"}',
    )

    # Serialize to dict
    config_dict = config.model_dump()
    assert config_dict["build_stage"] == "post_batch"
    assert config_dict["template"] == '{"text": "{{ col1 }}"}'

    # Deserialize from dict
    config_restored = AncillaryDatasetProcessorConfig.model_validate(config_dict)
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

    config_output_format = get_processor_config_from_kwargs(
        ProcessorType.OUTPUT_FORMAT,
        name="output_format_processor",
        build_stage=BuildStage.POST_BATCH,
        template='{"text": "{{ col1 }}"}',
    )
    assert isinstance(config_output_format, AncillaryDatasetProcessorConfig)
    assert config_output_format.template == '{"text": "{{ col1 }}"}'
    assert config_output_format.processor_type == ProcessorType.OUTPUT_FORMAT

    # Test with unknown processor type returns None
    from enum import Enum

    class UnknownProcessorType(str, Enum):
        UNKNOWN = "unknown"

    result = get_processor_config_from_kwargs(
        UnknownProcessorType.UNKNOWN, name="unknown_processor", build_stage=BuildStage.POST_BATCH, column_names=["col1"]
    )
    assert result is None
