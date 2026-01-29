# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from data_designer.config.column_configs import LLMTextColumnConfig, SamplerColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.dataset_builders import BuildStage
from data_designer.config.processors import DropColumnsProcessorConfig
from data_designer.config.run_config import RunConfig
from data_designer.config.sampler_params import SamplerType, UUIDSamplerParams
from data_designer.engine.column_generators.generators.base import GenerationStrategy
from data_designer.engine.dataset_builders.column_wise_builder import ColumnWiseDatasetBuilder
from data_designer.engine.dataset_builders.errors import DatasetGenerationError
from data_designer.engine.models.telemetry import InferenceEvent, NemoSourceEnum, TaskStatusEnum
from data_designer.engine.models.usage import ModelUsageStats, TokenUsageStats
from data_designer.engine.registry.data_designer_registry import DataDesignerRegistry
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd


@pytest.fixture
def stub_test_column_configs():
    return [
        SamplerColumnConfig(name="some_id", sampler_type=SamplerType.UUID, params=UUIDSamplerParams()),
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
def stub_test_config_builder(stub_test_column_configs, stub_model_configs):
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    for column_config in stub_test_column_configs:
        config_builder.add_column(column_config)
    config_builder.add_processor(
        processor_type="drop_columns",
        name="drop_columns_processor",
        build_stage=BuildStage.POST_BATCH,
        column_names=["column_to_drop"],
    )
    return config_builder


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
def stub_column_wise_builder(stub_resource_provider, stub_test_config_builder):
    return ColumnWiseDatasetBuilder(
        data_designer_config=stub_test_config_builder.build(),
        resource_provider=stub_resource_provider,
    )


def test_column_wise_dataset_builder_creation(stub_resource_provider, stub_test_config_builder):
    builder = ColumnWiseDatasetBuilder(
        data_designer_config=stub_test_config_builder.build(),
        resource_provider=stub_resource_provider,
    )
    assert len(builder._column_configs) == 3
    assert builder._resource_provider == stub_resource_provider
    assert isinstance(builder._registry, DataDesignerRegistry)


def test_column_wise_dataset_builder_creation_with_custom_registry(stub_resource_provider, stub_test_config_builder):
    custom_registry = Mock(spec=DataDesignerRegistry)

    builder = ColumnWiseDatasetBuilder(
        data_designer_config=stub_test_config_builder.build(),
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
    stub_resource_provider, stub_model_configs, config_type, expected_single_configs
):
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)

    if config_type == "single":
        # Add an LLM text column - these don't get grouped into MultiColumnConfigs
        single_config = expected_single_configs[0]
        config_builder.add_column(single_config)

        builder = ColumnWiseDatasetBuilder(
            data_designer_config=config_builder.build(),
            resource_provider=stub_resource_provider,
        )

        # Since there's no sampler, _internal_row_id is auto-added, plus the LLM column
        configs = builder.single_column_configs
        assert len(configs) == 2
        assert configs[0].name == "_internal_row_id"
        assert configs[1] == single_config

    else:
        sampler_config = expected_single_configs[0]
        config_builder.add_column(sampler_config)

        builder = ColumnWiseDatasetBuilder(
            data_designer_config=config_builder.build(),
            resource_provider=stub_resource_provider,
        )
        assert builder.single_column_configs == expected_single_configs


def test_column_wise_dataset_builder_build_method_basic_flow(
    stub_column_wise_builder,
    stub_batch_manager,
    stub_resource_provider,
):
    stub_resource_provider.run_config = RunConfig(buffer_size=50)
    stub_resource_provider.model_registry.run_health_check = Mock()
    stub_resource_provider.model_registry.get_model_usage_stats = Mock(return_value={"test": "stats"})
    stub_resource_provider.model_registry.models = {}

    # Mock the model config to return proper max_parallel_requests
    mock_model_config = Mock()
    mock_model_config.inference_parameters.max_parallel_requests = 4
    stub_resource_provider.model_registry.get_model_config.return_value = mock_model_config

    # Mock the batch manager's iter_current_batch method
    stub_batch_manager.iter_current_batch.return_value = [(0, {"test": "data"})]

    stub_column_wise_builder.batch_manager = stub_batch_manager

    result_path = stub_column_wise_builder.build(num_records=100)

    stub_resource_provider.model_registry.run_health_check.assert_called_once()
    stub_batch_manager.start.assert_called_once_with(num_records=100, buffer_size=50)
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
    stub_model_configs, stub_resource_provider, column_configs, expected_error
):
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)

    if expected_error == "The first column config must be a from-scratch column generator":
        for col_config in column_configs:
            config_builder.add_column(col_config)

        mock_registry = Mock()
        mock_generator_class = Mock()
        mock_generator_class.can_generate_from_scratch = False
        mock_registry.column_generators.get_for_config_type.return_value = mock_generator_class

        with pytest.raises(DatasetGenerationError, match=expected_error):
            ColumnWiseDatasetBuilder(
                data_designer_config=config_builder.build(),
                resource_provider=stub_resource_provider,
                registry=mock_registry,
            )
    else:
        # Empty column_configs case - config_builder will fail at build() due to validation
        with pytest.raises((DatasetGenerationError, Exception)):
            ColumnWiseDatasetBuilder(
                config_builder=config_builder,
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


def test_run_config_default_non_inference_max_parallel_workers() -> None:
    run_config = RunConfig()
    assert run_config.non_inference_max_parallel_workers == 4


@patch("data_designer.engine.dataset_builders.column_wise_builder.TelemetryHandler")
def test_emit_batch_inference_events_emits_from_deltas(
    mock_telemetry_handler_class: Mock,
    stub_resource_provider: Mock,
    stub_test_config_builder: DataDesignerConfigBuilder,
) -> None:
    usage_deltas = {"test-model": ModelUsageStats(token_usage=TokenUsageStats(input_tokens=50, output_tokens=150))}

    builder = ColumnWiseDatasetBuilder(
        data_designer_config=stub_test_config_builder.build(),
        resource_provider=stub_resource_provider,
    )

    session_id = "550e8400-e29b-41d4-a716-446655440000"

    mock_handler_instance = Mock()
    mock_telemetry_handler_class.return_value.__enter__ = Mock(return_value=mock_handler_instance)
    mock_telemetry_handler_class.return_value.__exit__ = Mock(return_value=False)

    builder._emit_batch_inference_events("batch", usage_deltas, session_id)

    mock_telemetry_handler_class.assert_called_once()
    call_kwargs = mock_telemetry_handler_class.call_args[1]
    assert call_kwargs["session_id"] == session_id

    mock_handler_instance.enqueue.assert_called_once()
    event = mock_handler_instance.enqueue.call_args[0][0]

    assert isinstance(event, InferenceEvent)
    assert event.task == "batch"
    assert event.task_status == TaskStatusEnum.SUCCESS
    assert event.nemo_source == NemoSourceEnum.DATADESIGNER
    assert event.model == "test-model"
    assert event.input_tokens == 50
    assert event.output_tokens == 150


@patch("data_designer.engine.dataset_builders.column_wise_builder.TelemetryHandler")
def test_emit_batch_inference_events_skips_when_no_deltas(
    mock_telemetry_handler_class: Mock,
    stub_resource_provider: Mock,
    stub_test_config_builder: DataDesignerConfigBuilder,
) -> None:
    usage_deltas: dict[str, ModelUsageStats] = {}

    builder = ColumnWiseDatasetBuilder(
        data_designer_config=stub_test_config_builder.build(),
        resource_provider=stub_resource_provider,
    )

    session_id = "550e8400-e29b-41d4-a716-446655440000"
    builder._emit_batch_inference_events("batch", usage_deltas, session_id)

    mock_telemetry_handler_class.assert_not_called()


@patch("data_designer.engine.dataset_builders.column_wise_builder.TelemetryHandler")
def test_emit_batch_inference_events_handles_multiple_models(
    mock_telemetry_handler_class: Mock,
    stub_resource_provider: Mock,
    stub_test_config_builder: DataDesignerConfigBuilder,
) -> None:
    usage_deltas = {
        "model-a": ModelUsageStats(token_usage=TokenUsageStats(input_tokens=100, output_tokens=200)),
        "model-b": ModelUsageStats(token_usage=TokenUsageStats(input_tokens=50, output_tokens=75)),
    }

    builder = ColumnWiseDatasetBuilder(
        data_designer_config=stub_test_config_builder.build(),
        resource_provider=stub_resource_provider,
    )

    session_id = "550e8400-e29b-41d4-a716-446655440000"
    mock_handler_instance = Mock()
    mock_telemetry_handler_class.return_value.__enter__ = Mock(return_value=mock_handler_instance)
    mock_telemetry_handler_class.return_value.__exit__ = Mock(return_value=False)

    builder._emit_batch_inference_events("preview", usage_deltas, session_id)

    assert mock_handler_instance.enqueue.call_count == 2
    events = [call[0][0] for call in mock_handler_instance.enqueue.call_args_list]
    model_names = {e.model for e in events}
    assert model_names == {"model-a", "model-b"}


@pytest.mark.parametrize(
    "disable_early_shutdown,configured_rate,expected_rate,shutdown_error_window",
    [
        (False, 0.7, 0.7, 20),  # enabled: use configured rate
        (True, 0.7, 1.0, 20),  # disabled: use 1.0 to effectively disable
        (False, 0.5, 0.5, 10),  # defaults
    ],
)
@patch("data_designer.engine.dataset_builders.column_wise_builder.ConcurrentThreadExecutor")
def test_fan_out_with_threads_uses_early_shutdown_settings_from_resource_provider(
    mock_executor_class: Mock,
    stub_resource_provider: Mock,
    stub_test_column_configs: list,
    stub_test_processor_configs: list,
    disable_early_shutdown: bool,
    configured_rate: float,
    expected_rate: float,
    shutdown_error_window: int,
) -> None:
    """Test that _fan_out_with_threads uses run settings from resource_provider."""
    from data_designer.config.run_config import RunConfig

    stub_resource_provider.run_config = RunConfig(
        disable_early_shutdown=disable_early_shutdown,
        shutdown_error_rate=configured_rate,
        shutdown_error_window=shutdown_error_window,
    )

    config_builder = DataDesignerConfigBuilder(model_configs=[])
    for column_config in stub_test_column_configs:
        config_builder.add_column(column_config)
    for processor_config in stub_test_processor_configs:
        config_builder.add_processor(processor_config)

    builder = ColumnWiseDatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )

    mock_executor_class.return_value.__enter__ = Mock(return_value=Mock())
    mock_executor_class.return_value.__exit__ = Mock(return_value=False)

    mock_generator = Mock()
    mock_generator.get_generation_strategy.return_value = GenerationStrategy.CELL_BY_CELL
    mock_generator.config.name = "test"
    mock_generator.config.column_type = "llm_text"

    builder.batch_manager = Mock()
    builder.batch_manager.num_records_batch = 10
    builder.batch_manager.iter_current_batch.return_value = []

    builder._fan_out_with_threads(mock_generator, max_workers=4)

    call_kwargs = mock_executor_class.call_args[1]
    assert call_kwargs["shutdown_error_rate"] == expected_rate
    assert call_kwargs["shutdown_error_window"] == shutdown_error_window
    assert call_kwargs["disable_early_shutdown"] == disable_early_shutdown
