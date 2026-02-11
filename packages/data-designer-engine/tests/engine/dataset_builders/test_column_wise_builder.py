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
from data_designer.config.seed_source import DataFrameSeedSource, LocalFileSeedSource
from data_designer.engine.column_generators.generators.base import GenerationStrategy
from data_designer.engine.dataset_builders.column_wise_builder import ColumnWiseDatasetBuilder
from data_designer.engine.dataset_builders.errors import DatasetGenerationError, DatasetProcessingError
from data_designer.engine.models.telemetry import InferenceEvent, NemoSourceEnum, TaskStatusEnum
from data_designer.engine.models.usage import ModelUsageStats, TokenUsageStats
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.registry.data_designer_registry import DataDesignerRegistry
from data_designer.engine.resources.seed_reader import DataFrameSeedReader
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


@pytest.fixture
def seed_data_setup(stub_resource_provider, tmp_path):
    """Set up seed reader with test data and write seed file to disk."""
    seed_df = pd.DataFrame({"seed_id": [1, 2, 3, 4, 5], "text": ["a", "b", "c", "d", "e"]})
    seed_source = DataFrameSeedSource(df=seed_df)
    seed_reader = DataFrameSeedReader()
    seed_reader.attach(seed_source, Mock())
    stub_resource_provider.seed_reader = seed_reader

    seed_path = tmp_path / "seed.parquet"
    seed_df.to_parquet(seed_path, index=False)

    return {"seed_df": seed_df, "seed_path": seed_path}


@pytest.fixture
def builder_with_seed(stub_resource_provider, stub_model_configs, seed_data_setup):
    """Create a builder with seed dataset configured."""
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.with_seed_dataset(LocalFileSeedSource(path=str(seed_data_setup["seed_path"])))
    config_builder.add_column(SamplerColumnConfig(name="extra", sampler_type="uuid", params=UUIDSamplerParams()))

    return ColumnWiseDatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )


def create_mock_processor(name: str, stages: list[str]) -> Mock:
    """Create a mock processor that implements specified stages."""
    mock_processor = Mock(spec=Processor)
    mock_processor.name = name
    mock_processor.implements.side_effect = lambda m: m in stages
    mock_processor.process_before_batch.side_effect = lambda df: df
    mock_processor.process_after_batch.side_effect = lambda df, **kw: df
    mock_processor.process_after_generation.side_effect = lambda df: df
    return mock_processor


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
    stub_column_wise_builder.set_processor_runner([])  # No processors for basic flow test

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


# Processor tests


@pytest.fixture
def simple_builder(stub_resource_provider, stub_model_configs):
    """Minimal builder with a single UUID column and no batch files on disk."""
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(SamplerColumnConfig(name="id", sampler_type="uuid", params=UUIDSamplerParams()))
    return ColumnWiseDatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )


def test_initialize_processors(stub_column_wise_builder):
    processors = stub_column_wise_builder.processors
    assert isinstance(processors, tuple)
    assert len(processors) == 1
    assert processors[0].config.column_names == ["column_to_drop"]


@pytest.mark.parametrize(
    "processor_fn,batch_size,expected_rows,expected_files",
    [
        pytest.param(lambda df: df, 3, 9, 3, id="noop_even"),
        pytest.param(lambda df: df[df["id"] > 3], 3, 6, 2, id="filter_even"),
        pytest.param(lambda df: df[df["id"] != 3].reset_index(drop=True), 3, 8, 3, id="filter_uneven"),
        pytest.param(lambda df: df[df["id"] > 8], 3, 1, 1, id="filter_fewer_than_batch_size"),
    ],
)
def test_run_after_generation(
    stub_resource_provider, simple_builder, processor_fn, batch_size, expected_rows, expected_files
):
    """Test that process_after_generation re-chunks output by batch_size."""
    storage = stub_resource_provider.artifact_storage
    storage.mkdir_if_needed(storage.final_dataset_path)
    pd.DataFrame({"id": list(range(1, 10))}).to_parquet(storage.final_dataset_path / "batch_00000.parquet", index=False)

    mock_processor = create_mock_processor("proc", ["process_after_generation"])
    mock_processor.process_after_generation.side_effect = processor_fn

    simple_builder.set_processor_runner([mock_processor])
    simple_builder._processor_runner.run_after_generation(batch_size)

    mock_processor.process_after_generation.assert_called_once()
    batch_files = sorted(storage.final_dataset_path.glob("*.parquet"))
    assert len(batch_files) == expected_files
    assert sum(len(pd.read_parquet(f)) for f in batch_files) == expected_rows


@pytest.mark.parametrize("mode", ["preview", "build"])
def test_all_processor_stages_run_in_order(builder_with_seed, mode):
    """Test that all 3 processor stages run in correct order for both preview and build modes."""
    call_order = []
    all_stages = ["process_before_batch", "process_after_batch", "process_after_generation"]

    mock_processor = create_mock_processor("all_stages_processor", all_stages)
    mock_processor.process_before_batch.side_effect = lambda df: (call_order.append("process_before_batch"), df)[1]
    mock_processor.process_after_batch.side_effect = lambda df, **kw: (call_order.append("process_after_batch"), df)[1]
    mock_processor.process_after_generation.side_effect = lambda df: (
        call_order.append("process_after_generation"),
        df,
    )[1]

    builder_with_seed.set_processor_runner([mock_processor])

    if mode == "preview":
        raw_dataset = builder_with_seed.build_preview(num_records=3)
        builder_with_seed.process_preview(raw_dataset)
    else:
        builder_with_seed.build(num_records=3)

    mock_processor.process_before_batch.assert_called_once()
    mock_processor.process_after_batch.assert_called_once()
    mock_processor.process_after_generation.assert_called_once()

    assert call_order == all_stages


def test_processor_exception_in_process_after_batch_raises_error(simple_builder):
    """Test that processor exceptions during process_after_batch are properly wrapped."""
    mock_processor = create_mock_processor("failing_processor", ["process_after_batch"])
    mock_processor.process_after_batch.side_effect = ValueError("Post-batch processing failed")

    simple_builder.set_processor_runner([mock_processor])

    with pytest.raises(DatasetProcessingError, match="Failed in process_after_batch"):
        simple_builder._processor_runner.run_post_batch(pd.DataFrame({"id": [1, 2, 3]}), current_batch_number=0)


def test_processor_with_no_implemented_stages_is_skipped(builder_with_seed):
    """Test that a processor implementing no stages doesn't cause errors."""
    mock_processor = create_mock_processor("noop_processor", [])
    builder_with_seed.set_processor_runner([mock_processor])

    result = builder_with_seed.build_preview(num_records=3)

    assert len(result) == 3
    mock_processor.process_before_batch.assert_not_called()
    mock_processor.process_after_batch.assert_not_called()
    mock_processor.process_after_generation.assert_not_called()


def test_multiple_processors_run_in_definition_order(builder_with_seed):
    """Test that multiple processors run in the order they were defined."""
    call_order = []

    processors = []
    for label in ["a", "b", "c"]:
        p = create_mock_processor(f"processor_{label}", ["process_before_batch"])
        p.process_before_batch.side_effect = lambda df, lbl=label: (call_order.append(lbl), df)[1]
        processors.append(p)

    builder_with_seed.set_processor_runner(processors)
    builder_with_seed.build(num_records=3)

    assert call_order == ["a", "b", "c"]


def test_process_preview_with_empty_dataframe(simple_builder):
    """Test that process_preview handles empty DataFrames gracefully."""
    mock_processor = create_mock_processor("test_processor", ["process_after_batch", "process_after_generation"])
    simple_builder.set_processor_runner([mock_processor])

    result = simple_builder.process_preview(pd.DataFrame())

    assert len(result) == 0
    mock_processor.process_after_batch.assert_called_once()
    mock_processor.process_after_generation.assert_called_once()
