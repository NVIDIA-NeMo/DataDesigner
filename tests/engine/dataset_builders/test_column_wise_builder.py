# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pandas as pd
import pytest

from data_designer.config.column_configs import LLMTextColumnConfig, SamplerColumnConfig
from data_designer.config.dataset_builders import BuildStage
from data_designer.config.processors import DropColumnsProcessorConfig
from data_designer.engine.dataset_builders.column_wise_builder import (
    MAX_CONCURRENCY_PER_NON_LLM_GENERATOR,
    ColumnWiseDatasetBuilder,
)
from data_designer.engine.dataset_builders.errors import DatasetGenerationError
from data_designer.engine.dataset_builders.multi_column_configs import SamplerMultiColumnConfig
from data_designer.engine.registry.data_designer_registry import DataDesignerRegistry


@pytest.fixture
def stub_test_column_configs():
    return [
        LLMTextColumnConfig(name="test_column", prompt="Test prompt", model_alias="test_model"),
        LLMTextColumnConfig(name="column_to_drop", prompt="Test prompt", model_alias="test_model"),
    ]


@pytest.fixture
def stub_test_processor_configs():
    return [
        DropColumnsProcessorConfig(
            name="drop_columns_processor", build_stage=BuildStage.POST_BATCH, column_names=["column_to_drop"]
        )
    ]


@pytest.fixture
def stub_batch_manager():
    mock_batch_manager = Mock()
    mock_batch_manager.num_batches = 2
    mock_batch_manager.num_records_batch = 3
    mock_batch_manager.finish = Mock()
    mock_batch_manager.write = Mock()
    mock_batch_manager.add_records = Mock()
    mock_batch_manager.update_records = Mock()
    mock_batch_manager.update_record = Mock()
    mock_batch_manager.get_current_batch = Mock()
    mock_batch_manager.get_current_batch.side_effect = [
        pd.DataFrame({"test_column": [1, 2, 3], "column_to_drop": [1, 2, 3]}),
        pd.DataFrame({"test_column": [4, 5, 6], "column_to_drop": [4, 5, 6]}),
    ]
    mock_batch_manager.get_current_batch_number = Mock()
    mock_batch_manager.get_current_batch_number.side_effect = [1, 2]
    return mock_batch_manager


@pytest.fixture
def stub_column_wise_builder(stub_resource_provider, stub_test_column_configs, stub_test_processor_configs):
    return ColumnWiseDatasetBuilder(
        column_configs=stub_test_column_configs,
        processor_configs=stub_test_processor_configs,
        resource_provider=stub_resource_provider,
    )


def test_column_wise_dataset_builder_creation(
    stub_resource_provider, stub_test_column_configs, stub_test_processor_configs
):
    builder = ColumnWiseDatasetBuilder(
        column_configs=stub_test_column_configs,
        processor_configs=stub_test_processor_configs,
        resource_provider=stub_resource_provider,
    )

    assert builder._column_configs == stub_test_column_configs
    assert builder._resource_provider == stub_resource_provider
    assert isinstance(builder._registry, DataDesignerRegistry)


def test_column_wise_dataset_builder_creation_with_custom_registry(
    stub_resource_provider, stub_test_column_configs, stub_test_processor_configs
):
    custom_registry = Mock(spec=DataDesignerRegistry)

    builder = ColumnWiseDatasetBuilder(
        column_configs=stub_test_column_configs,
        processor_configs=stub_test_processor_configs,
        resource_provider=stub_resource_provider,
        registry=custom_registry,
    )

    assert builder._registry == custom_registry


def test_column_wise_dataset_builder_artifact_storage_property(stub_column_wise_builder, stub_resource_provider):
    assert stub_column_wise_builder.artifact_storage == stub_resource_provider.artifact_storage


def test_column_wise_dataset_builder_records_to_drop_initialization(stub_column_wise_builder):
    assert stub_column_wise_builder._records_to_drop == set()


def test_column_wise_dataset_builder_batch_manager_initialization(stub_column_wise_builder, stub_resource_provider):
    assert stub_column_wise_builder.batch_manager is not None
    assert stub_column_wise_builder.batch_manager.artifact_storage == stub_resource_provider.artifact_storage


@pytest.mark.parametrize(
    "config_type,expected_single_configs",
    [
        ("single", [LLMTextColumnConfig(name="test_column", prompt="Test prompt", model_alias="test_model")]),
        (
            "multi",
            [SamplerColumnConfig(name="sampler_col", sampler_type="category", params={"values": ["A", "B", "C"]})],
        ),
    ],
)
def test_column_wise_dataset_builder_single_column_configs_property(
    stub_resource_provider, stub_test_processor_configs, config_type, expected_single_configs
):
    if config_type == "single":
        single_config = LLMTextColumnConfig(name="test_column", prompt="Test prompt", model_alias="test_model")
        builder = ColumnWiseDatasetBuilder(
            column_configs=[single_config],
            processor_configs=stub_test_processor_configs,
            resource_provider=stub_resource_provider,
        )
        assert builder.single_column_configs == [single_config]
    else:
        sampler_config = SamplerColumnConfig(
            name="sampler_col", sampler_type="category", params={"values": ["A", "B", "C"]}
        )
        multi_config = SamplerMultiColumnConfig(columns=[sampler_config])
        builder = ColumnWiseDatasetBuilder(
            column_configs=[multi_config],
            processor_configs=stub_test_processor_configs,
            resource_provider=stub_resource_provider,
        )
        assert builder.single_column_configs == [sampler_config]


def test_column_wise_dataset_builder_build_method_basic_flow(
    stub_column_wise_builder,
    stub_batch_manager,
    stub_resource_provider,
):
    stub_resource_provider.model_registry.run_health_check = Mock()
    stub_resource_provider.model_registry.get_model_usage_stats = Mock(return_value={"test": "stats"})

    # Mock the model config to return proper max_parallel_requests
    mock_model_config = Mock()
    mock_model_config.inference_parameters.max_parallel_requests = 4
    stub_resource_provider.model_registry.get_model_config.return_value = mock_model_config

    # Mock the batch manager's iter_current_batch method
    stub_batch_manager.iter_current_batch.return_value = [(0, {"test": "data"})]

    stub_column_wise_builder.batch_manager = stub_batch_manager

    result_path = stub_column_wise_builder.build(num_records=100, buffer_size=50)

    stub_resource_provider.model_registry.run_health_check.assert_called_once()
    stub_batch_manager.finish.assert_called_once()
    assert result_path == stub_resource_provider.artifact_storage.final_dataset_path


@pytest.mark.parametrize(
    "column_configs,expected_error",
    [
        ([], "No column configs provided"),
        (
            [LLMTextColumnConfig(name="test_column", prompt="Test prompt", model_alias="test_model")],
            "The first column config must be a from-scratch column generator",
        ),
    ],
)
def test_column_wise_dataset_builder_validate_column_configs(
    stub_test_processor_configs, stub_resource_provider, column_configs, expected_error
):
    if expected_error == "The first column config must be a from-scratch column generator":
        mock_registry = Mock()
        mock_generator_class = Mock()
        mock_generator_class.can_generate_from_scratch = False
        mock_registry.column_generators.get_for_config_type.return_value = mock_generator_class

        with pytest.raises(DatasetGenerationError, match=expected_error):
            ColumnWiseDatasetBuilder(
                column_configs=column_configs,
                processor_configs=stub_test_processor_configs,
                resource_provider=stub_resource_provider,
                registry=mock_registry,
            )
    else:
        with pytest.raises(DatasetGenerationError, match=expected_error):
            ColumnWiseDatasetBuilder(
                column_configs=column_configs,
                processor_configs=stub_test_processor_configs,
                resource_provider=stub_resource_provider,
            )


def test_column_wise_dataset_builder_initialize_processors(stub_column_wise_builder):
    processors = stub_column_wise_builder._processors
    assert processors.keys() == set(BuildStage)
    assert len(processors[BuildStage.PRE_BATCH]) == 0
    assert len(processors[BuildStage.POST_BATCH]) == 1
    assert len(processors[BuildStage.PRE_GENERATION]) == 0
    assert len(processors[BuildStage.POST_GENERATION]) == 0
    assert processors[BuildStage.POST_BATCH][0].config.column_names == ["column_to_drop"]


def test_constants_max_concurrency_constant():
    assert MAX_CONCURRENCY_PER_NON_LLM_GENERATOR == 4
