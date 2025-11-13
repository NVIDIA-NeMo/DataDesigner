# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from data_designer.config.dataset_builders import BuildStage
from data_designer.config.processors import (
    DropColumnsProcessorConfig,
    JsonlExportProcessorConfig,
    ProcessorConfig,
    ProcessorType,
    get_processor_config_from_kwargs,
)


def test_drop_columns_processor_config_creation():
    config = DropColumnsProcessorConfig(build_stage=BuildStage.POST_BATCH, column_names=["col1", "col2"])

    assert config.build_stage == BuildStage.POST_BATCH
    assert config.column_names == ["col1", "col2"]
    assert config.processor_type == ProcessorType.DROP_COLUMNS
    assert isinstance(config, ProcessorConfig)


def test_drop_columns_processor_config_validation():
    # Test unsupported stage raises error
    with pytest.raises(ValidationError, match="Invalid dataset builder stage"):
        DropColumnsProcessorConfig(build_stage=BuildStage.PRE_BATCH, column_names=["col1"])

    # Test missing required field raises error
    with pytest.raises(ValidationError, match="Field required"):
        DropColumnsProcessorConfig(build_stage=BuildStage.POST_BATCH)


def test_drop_columns_processor_config_serialization():
    config = DropColumnsProcessorConfig(build_stage=BuildStage.POST_BATCH, column_names=["col1", "col2"])

    # Serialize to dict
    config_dict = config.model_dump()
    assert config_dict["build_stage"] == "post_batch"
    assert config_dict["column_names"] == ["col1", "col2"]

    # Deserialize from dict
    config_restored = DropColumnsProcessorConfig.model_validate(config_dict)
    assert config_restored.build_stage == config.build_stage
    assert config_restored.column_names == config.column_names


def test_jsonl_export_processor_config_creation():
    config = JsonlExportProcessorConfig(
        build_stage=BuildStage.POST_BATCH,
        template='{"text": "{{ col1 }}"}',
        fraction_per_file={"train.jsonl": 0.8, "validation.jsonl": 0.2},
    )

    assert config.build_stage == BuildStage.POST_BATCH
    assert config.template == '{"text": "{{ col1 }}"}'
    assert config.fraction_per_file == {"train.jsonl": 0.8, "validation.jsonl": 0.2}
    assert config.processor_type == ProcessorType.JSONL_EXPORT
    assert isinstance(config, ProcessorConfig)


def test_jsonl_export_processor_config_validation():
    # Test unsupported stage raises error
    with pytest.raises(ValidationError, match="Invalid dataset builder stage"):
        JsonlExportProcessorConfig(
            build_stage=BuildStage.PRE_BATCH,
            template='{"text": "{{ col1 }}"}',
            fraction_per_file={"train.jsonl": 0.8, "validation.jsonl": 0.2},
        )

    # Test missing required field raises error
    with pytest.raises(ValidationError, match="Field required"):
        JsonlExportProcessorConfig(build_stage=BuildStage.POST_BATCH)

    # Test invalid fraction per file raises error
    with pytest.raises(ValidationError, match="The fractions must sum to 1."):
        JsonlExportProcessorConfig(
            build_stage=BuildStage.POST_BATCH,
            template='{"text": "{{ col1 }}"}',
            fraction_per_file={"train.jsonl": 0.8, "validation.jsonl": 0.3},
        )


def test_jsonl_export_processor_config_serialization():
    config = JsonlExportProcessorConfig(
        build_stage=BuildStage.POST_BATCH,
        template='{"text": "{{ col1 }}"}',
        fraction_per_file={"train.jsonl": 0.8, "validation.jsonl": 0.2},
    )

    # Serialize to dict
    config_dict = config.model_dump()
    assert config_dict["build_stage"] == "post_batch"
    assert config_dict["template"] == '{"text": "{{ col1 }}"}'
    assert config_dict["fraction_per_file"] == {"train.jsonl": 0.8, "validation.jsonl": 0.2}

    # Deserialize from dict
    config_restored = JsonlExportProcessorConfig.model_validate(config_dict)
    assert config_restored.build_stage == config.build_stage
    assert config_restored.template == config.template
    assert config_restored.fraction_per_file == config.fraction_per_file


def test_get_processor_config_from_kwargs():
    # Test successful creation
    config_drop_columns = get_processor_config_from_kwargs(
        ProcessorType.DROP_COLUMNS, build_stage=BuildStage.POST_BATCH, column_names=["col1"]
    )
    assert isinstance(config_drop_columns, DropColumnsProcessorConfig)
    assert config_drop_columns.column_names == ["col1"]
    assert config_drop_columns.processor_type == ProcessorType.DROP_COLUMNS

    config_jsonl_export = get_processor_config_from_kwargs(
        ProcessorType.JSONL_EXPORT,
        build_stage=BuildStage.POST_BATCH,
        template='{"text": "{{ col1 }}"}',
        fraction_per_file={"train.jsonl": 0.8, "validation.jsonl": 0.2},
    )
    assert isinstance(config_jsonl_export, JsonlExportProcessorConfig)
    assert config_jsonl_export.template == '{"text": "{{ col1 }}"}'
    assert config_jsonl_export.fraction_per_file == {"train.jsonl": 0.8, "validation.jsonl": 0.2}
    assert config_jsonl_export.processor_type == ProcessorType.JSONL_EXPORT

    # Test with unknown processor type returns None
    from enum import Enum

    class UnknownProcessorType(str, Enum):
        UNKNOWN = "unknown"

    result = get_processor_config_from_kwargs(
        UnknownProcessorType.UNKNOWN, build_stage=BuildStage.POST_BATCH, column_names=["col1"]
    )
    assert result is None
