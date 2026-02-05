# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from data_designer.config.column_configs import LLMTextColumnConfig, SamplerColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
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
    return [DropColumnsProcessorConfig(name="drop_columns_processor", column_names=["column_to_drop"])]


@pytest.fixture
def stub_test_config_builder(stub_test_column_configs, stub_model_configs):
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    for column_config in stub_test_column_configs:
        config_builder.add_column(column_config)
    config_builder.add_processor(
        processor_type="drop_columns",
        name="drop_columns_processor",
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
    stub_resource_provider.seed_reader = None  # No seed data for this basic flow test
    stub_resource_provider.model_registry.run_health_check = Mock()
    stub_resource_provider.model_registry.get_model_usage_stats = Mock(return_value={"test": "stats"})
    stub_resource_provider.model_registry.models = {}

    # Mock the model config to return proper max_parallel_requests
    mock_model_config = Mock()
    mock_model_config.inference_parameters.max_parallel_requests = 4
    mock_model_config.inference_parameters.get_formatted_params.return_value = []
    stub_resource_provider.model_registry.get_model_config.return_value = mock_model_config

    # Mock the batch manager's iter_current_batch method
    stub_batch_manager.iter_current_batch.return_value = [(0, {"test": "data"})]

    stub_column_wise_builder.batch_manager = stub_batch_manager
    stub_column_wise_builder._processors = []  # No processors for basic flow test

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
    assert isinstance(processors, list)
    assert len(processors) == 1
    assert processors[0].config.column_names == ["column_to_drop"]


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
    mock_generator.config.tool_alias = None  # Avoid triggering tool usage code path

    builder.batch_manager = Mock()
    builder.batch_manager.num_records_batch = 10
    builder.batch_manager.iter_current_batch.return_value = []
    builder.batch_manager.num_records_batch = 0

    builder._fan_out_with_threads(mock_generator, max_workers=4)

    call_kwargs = mock_executor_class.call_args[1]
    assert call_kwargs["shutdown_error_rate"] == expected_rate
    assert call_kwargs["shutdown_error_window"] == shutdown_error_window
    assert call_kwargs["disable_early_shutdown"] == disable_early_shutdown


def test_run_pre_generation_processors_filters_seed_data(stub_resource_provider, stub_model_configs, tmp_path):
    """Test that PRE_GENERATION processors are applied to seed data before generation."""
    from pathlib import Path

    from data_designer.config.seed_source import DataFrameSeedSource, LocalFileSeedSource
    from data_designer.engine.processing.processors.base import Processor
    from data_designer.engine.resources.seed_reader import DataFrameSeedReader

    # Set up seed reader with test data
    seed_df = pd.DataFrame({"seed_id": [1, 2, 3, 4, 5], "value": ["a", "b", "c", "d", "e"]})
    seed_source = DataFrameSeedSource(df=seed_df)
    seed_reader = DataFrameSeedReader()
    seed_reader.attach(seed_source, Mock())
    stub_resource_provider.seed_reader = seed_reader

    # Create a mock processor that filters rows during preprocess
    mock_processor = Mock(spec=Processor)
    mock_processor.name = "filter_processor"
    mock_processor.implements.side_effect = lambda m: m == "preprocess"
    mock_processor.preprocess.side_effect = lambda df: df[df["seed_id"] > 2].reset_index(drop=True)

    # Write seed file to tmp_path
    seed_path = tmp_path / "seed.parquet"
    seed_df.to_parquet(seed_path, index=False)

    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.with_seed_dataset(LocalFileSeedSource(path=str(seed_path)))
    config_builder.add_column(SamplerColumnConfig(name="uuid", sampler_type="uuid", params=UUIDSamplerParams()))

    builder = ColumnWiseDatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )
    builder._processors = [mock_processor]

    builder._run_pre_generation_processors()

    # Verify preprocess was called
    mock_processor.preprocess.assert_called_once()

    # Verify preprocessed_seed_uri was set and points to a valid file
    assert stub_resource_provider.preprocessed_seed_uri is not None
    preprocessed_path = Path(stub_resource_provider.preprocessed_seed_uri)
    assert preprocessed_path.exists()

    # Verify the preprocessed file contains filtered data (3 rows with seed_id > 2)
    preprocessed_df = pd.read_parquet(preprocessed_path)
    assert len(preprocessed_df) == 3
    assert list(preprocessed_df["seed_id"]) == [3, 4, 5]


def test_run_post_generation_processors_modifies_final_dataset(stub_resource_provider, stub_model_configs):
    """Test that postprocess callbacks are applied to the final dataset."""
    from data_designer.engine.processing.processors.base import Processor

    # Create test parquet files
    final_df = pd.DataFrame({"id": [1, 2, 3, 4, 5], "value": ["a", "b", "c", "d", "e"]})
    stub_resource_provider.artifact_storage.mkdir_if_needed(stub_resource_provider.artifact_storage.final_dataset_path)
    final_df.to_parquet(stub_resource_provider.artifact_storage.final_dataset_path / "batch_00000.parquet", index=False)

    # Create a mock processor that filters rows during postprocess
    mock_processor = Mock(spec=Processor)
    mock_processor.name = "dedup_processor"
    mock_processor.implements.side_effect = lambda m: m == "postprocess"
    mock_processor.postprocess.return_value = final_df[final_df["id"] > 2]

    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(SamplerColumnConfig(name="id", sampler_type="uuid", params=UUIDSamplerParams()))

    builder = ColumnWiseDatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )
    builder._processors = [mock_processor]

    builder._run_post_generation_processors()

    # Verify postprocess was called
    mock_processor.postprocess.assert_called_once()

    # Verify final dataset was rewritten with fewer rows
    result_df = stub_resource_provider.artifact_storage.load_dataset()
    assert len(result_df) == 3


def test_run_pre_generation_processors_skips_when_no_seed_reader(stub_resource_provider, stub_model_configs):
    """Test that preprocess is skipped when no seed reader is configured."""
    from data_designer.engine.processing.processors.base import Processor

    stub_resource_provider.seed_reader = None

    mock_processor = Mock(spec=Processor)
    mock_processor.name = "filter_processor"
    mock_processor.implements.side_effect = lambda m: m == "preprocess"

    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(SamplerColumnConfig(name="id", sampler_type="uuid", params=UUIDSamplerParams()))

    builder = ColumnWiseDatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )
    builder._processors = [mock_processor]

    builder._run_pre_generation_processors()

    # Preprocess should not be called when no seed reader
    mock_processor.preprocess.assert_not_called()


def test_build_preview_runs_pre_generation_processors(stub_resource_provider, stub_model_configs, tmp_path):
    """Test that build_preview runs PRE_GENERATION processors."""
    from data_designer.config.seed_source import DataFrameSeedSource, LocalFileSeedSource
    from data_designer.engine.resources.seed_reader import DataFrameSeedReader

    # Set up seed reader with test data
    seed_df = pd.DataFrame({"seed_id": [1, 2, 3, 4, 5], "text": ["a", "b", "c", "d", "e"]})
    seed_source = DataFrameSeedSource(df=seed_df)
    seed_reader = DataFrameSeedReader()
    seed_reader.attach(seed_source, Mock())
    stub_resource_provider.seed_reader = seed_reader

    # Write seed file to tmp_path
    seed_path = tmp_path / "seed.parquet"
    seed_df.to_parquet(seed_path, index=False)

    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.with_seed_dataset(LocalFileSeedSource(path=str(seed_path)))
    config_builder.add_column(SamplerColumnConfig(name="uuid", sampler_type="uuid", params=UUIDSamplerParams()))

    builder = ColumnWiseDatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )

    # Mock everything to isolate the test
    builder._run_model_health_check_if_needed = Mock()
    builder._run_mcp_tool_check_if_needed = Mock()
    builder._run_pre_generation_processors = Mock()
    builder._initialize_generators = Mock(return_value=[])
    builder.batch_manager.start = Mock()
    builder._run_batch = Mock()
    builder.batch_manager.get_current_batch = Mock(return_value=pd.DataFrame())
    builder.batch_manager.reset = Mock()
    builder._resource_provider.model_registry.get_model_usage_stats = Mock(return_value={})

    builder.build_preview(num_records=5)

    builder._run_pre_generation_processors.assert_called_once()


def test_process_preview_runs_both_callbacks(stub_resource_provider, stub_model_configs):
    """Test that process_preview runs process_after_batch and postprocess callbacks."""
    from data_designer.engine.processing.processors.base import Processor

    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(SamplerColumnConfig(name="id", sampler_type="uuid", params=UUIDSamplerParams()))

    builder = ColumnWiseDatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )

    # Create a mock processor with both callbacks
    mock_processor = Mock(spec=Processor)
    mock_processor.name = "test_processor"
    mock_processor.implements.side_effect = lambda m: m in ("process_after_batch", "postprocess")
    mock_processor.process_after_batch.side_effect = lambda df, **kwargs: df.assign(post_batch_applied=True)
    mock_processor.postprocess.side_effect = lambda df: df.assign(post_gen_applied=True)

    builder._processors = [mock_processor]

    input_df = pd.DataFrame({"id": [1, 2, 3]})
    result = builder.process_preview(input_df)

    # Both callbacks should have been called
    mock_processor.process_after_batch.assert_called_once()
    mock_processor.postprocess.assert_called_once()

    # Result should have both columns added
    assert "post_batch_applied" in result.columns
    assert "post_gen_applied" in result.columns


@pytest.mark.parametrize("mode", ["preview", "build"])
def test_all_processor_stages_run_in_order(stub_resource_provider, stub_model_configs, tmp_path, mode):
    """Test that all 4 processor stages run in correct order for both preview and build modes."""
    from data_designer.config.seed_source import DataFrameSeedSource, LocalFileSeedSource
    from data_designer.engine.processing.processors.base import Processor
    from data_designer.engine.resources.seed_reader import DataFrameSeedReader

    # Set up seed reader with test data
    seed_df = pd.DataFrame({"seed_id": [1, 2, 3], "text": ["a", "b", "c"]})
    seed_source = DataFrameSeedSource(df=seed_df)
    seed_reader = DataFrameSeedReader()
    seed_reader.attach(seed_source, Mock())
    stub_resource_provider.seed_reader = seed_reader

    # Write seed file to tmp_path
    seed_path = tmp_path / "seed.parquet"
    seed_df.to_parquet(seed_path, index=False)

    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.with_seed_dataset(LocalFileSeedSource(path=str(seed_path)))
    config_builder.add_column(SamplerColumnConfig(name="extra", sampler_type="uuid", params=UUIDSamplerParams()))

    builder = ColumnWiseDatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )

    # Create a processor that implements all 4 stages to track calls
    call_order = []

    mock_processor = Mock(spec=Processor)
    mock_processor.name = "all_stages_processor"
    mock_processor.implements.side_effect = lambda m: m in (
        "preprocess",
        "process_before_batch",
        "process_after_batch",
        "postprocess",
    )
    mock_processor.preprocess.side_effect = lambda df: (call_order.append("preprocess"), df)[1]
    mock_processor.process_before_batch.side_effect = lambda df: (call_order.append("process_before_batch"), df)[1]
    mock_processor.process_after_batch.side_effect = lambda df, **kw: (call_order.append("process_after_batch"), df)[1]
    mock_processor.postprocess.side_effect = lambda df: (call_order.append("postprocess"), df)[1]

    builder._processors = [mock_processor]

    if mode == "preview":
        # Preview flow: build_preview() + process_preview()
        raw_dataset = builder.build_preview(num_records=3)
        builder.process_preview(raw_dataset)
    else:
        # Build flow: build() runs all stages internally
        builder.build(num_records=3)

    # Verify all 4 stages were called
    mock_processor.preprocess.assert_called_once()
    mock_processor.process_before_batch.assert_called_once()
    mock_processor.process_after_batch.assert_called_once()
    mock_processor.postprocess.assert_called_once()

    # Verify call order matches the pipeline stages
    assert call_order == ["preprocess", "process_before_batch", "process_after_batch", "postprocess"]
