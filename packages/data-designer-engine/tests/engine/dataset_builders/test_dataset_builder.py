# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

import data_designer.engine.dataset_builders.dataset_builder as builder_mod
import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import CustomColumnConfig, LLMTextColumnConfig, SamplerColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.custom_column import custom_column_generator
from data_designer.config.processors import DropColumnsProcessorConfig
from data_designer.config.run_config import RunConfig
from data_designer.config.sampler_params import SamplerType, UUIDSamplerParams
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.column_generators.generators.base import GenerationStrategy
from data_designer.engine.dataset_builders.dataset_builder import DatasetBuilder
from data_designer.engine.dataset_builders.errors import DatasetGenerationError, DatasetProcessingError
from data_designer.engine.models.errors import (
    FormattedLLMErrorMessage,
    ModelGenerationValidationFailureError,
    ModelTimeoutError,
)
from data_designer.engine.models.telemetry import InferenceEvent, NemoSourceEnum, TaskStatusEnum
from data_designer.engine.models.usage import ModelUsageStats, TokenUsageStats
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.registry.data_designer_registry import DataDesignerRegistry
from data_designer.engine.resources.seed_reader import DataFrameSeedReader

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
    mock_batch_manager.replace_buffer = Mock()
    mock_batch_manager.update_record = Mock()
    mock_batch_manager.get_current_batch = Mock()
    mock_batch_manager.get_current_batch.side_effect = [
        lazy.pd.DataFrame({"test_column": [1, 2, 3], "column_to_drop": [1, 2, 3]}),
        lazy.pd.DataFrame({"test_column": [4, 5, 6], "column_to_drop": [4, 5, 6]}),
    ]
    mock_batch_manager.get_current_batch_number = Mock()
    mock_batch_manager.get_current_batch_number.side_effect = [1, 2]
    return mock_batch_manager


@pytest.fixture
def stub_dataset_builder(stub_resource_provider, stub_test_config_builder):
    return DatasetBuilder(
        data_designer_config=stub_test_config_builder.build(),
        resource_provider=stub_resource_provider,
    )


@pytest.fixture
def seed_data_setup(stub_resource_provider, tmp_path):
    """Set up seed reader with test data and write seed file to disk."""
    seed_df = lazy.pd.DataFrame({"seed_id": [1, 2, 3, 4, 5], "text": ["a", "b", "c", "d", "e"]})
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

    return DatasetBuilder(
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


def test_dataset_builder_creation(stub_resource_provider, stub_test_config_builder):
    builder = DatasetBuilder(
        data_designer_config=stub_test_config_builder.build(),
        resource_provider=stub_resource_provider,
    )
    assert len(builder._column_configs) == 3
    assert builder._resource_provider == stub_resource_provider
    assert isinstance(builder._registry, DataDesignerRegistry)


def test_dataset_builder_creation_with_custom_registry(stub_resource_provider, stub_test_config_builder):
    custom_registry = Mock(spec=DataDesignerRegistry)

    builder = DatasetBuilder(
        data_designer_config=stub_test_config_builder.build(),
        resource_provider=stub_resource_provider,
        registry=custom_registry,
    )

    assert builder._registry == custom_registry


def test_dataset_builder_artifact_storage_property(stub_dataset_builder, stub_resource_provider):
    assert stub_dataset_builder.artifact_storage == stub_resource_provider.artifact_storage


def test_dataset_builder_records_to_drop_initialization(stub_dataset_builder):
    assert stub_dataset_builder._records_to_drop == set()


def test_worker_error_callback_logs_schema_validation_detail(
    stub_dataset_builder: DatasetBuilder,
    caplog: pytest.LogCaptureFixture,
) -> None:
    exc = ModelGenerationValidationFailureError(
        FormattedLLMErrorMessage(
            cause=(
                "The model output from 'test-model' could not be parsed into the requested format while "
                "running generation for column 'test_column'. Validation detail: Response doesn't match "
                "requested <response_schema> 'name' is a required property."
            ),
            solution="Simplify the schema and retry.",
        ),
        detail="Response doesn't match requested <response_schema> 'name' is a required property.",
        failure_kind="schema_validation",
    )

    with caplog.at_level(logging.WARNING):
        stub_dataset_builder._worker_error_callback(exc, context={"index": 248, "column_name": "test_column"})

    assert "record at index 248" in caplog.text
    assert "column 'test_column'" in caplog.text
    assert "(schema validation)" in caplog.text
    assert "Response doesn't match requested <response_schema> 'name' is a required property." in caplog.text
    assert 248 in stub_dataset_builder._records_to_drop


def test_worker_error_callback_logs_timeout_detail(
    stub_dataset_builder: DatasetBuilder,
    caplog: pytest.LogCaptureFixture,
) -> None:
    exc = ModelTimeoutError(
        FormattedLLMErrorMessage(
            cause="The request to model 'test-model' timed out while running generation for column 'test_column'.",
            solution="Increase the timeout setting for the model and retry.",
        )
    )

    with caplog.at_level(logging.WARNING):
        stub_dataset_builder._worker_error_callback(exc, context={"index": 17, "column_name": "test_column"})

    assert "record at index 17" in caplog.text
    assert "column 'test_column'" in caplog.text
    assert "(timeout)" in caplog.text
    assert (
        "The request to model 'test-model' timed out while running generation for column 'test_column'." in caplog.text
    )
    assert 17 in stub_dataset_builder._records_to_drop


def test_worker_error_callback_requires_context_index(
    stub_dataset_builder: DatasetBuilder,
    caplog: pytest.LogCaptureFixture,
) -> None:
    exc = ModelTimeoutError(
        FormattedLLMErrorMessage(
            cause="The request to model 'test-model' timed out while running generation for column 'test_column'.",
            solution="Increase the timeout setting for the model and retry.",
        )
    )

    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(RuntimeError, match="Worker error callback called without a valid context index."),
    ):
        stub_dataset_builder._worker_error_callback(exc, context=None)

    assert "record at index unknown" in caplog.text
    assert len(stub_dataset_builder._records_to_drop) == 0


def test_dataset_builder_batch_manager_initialization(stub_dataset_builder, stub_resource_provider):
    assert stub_dataset_builder.batch_manager is not None
    assert stub_dataset_builder.batch_manager.artifact_storage == stub_resource_provider.artifact_storage


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
def test_dataset_builder_single_column_configs_property(
    stub_resource_provider, stub_model_configs, config_type, expected_single_configs
):
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)

    if config_type == "single":
        # Add an LLM text column - these don't get grouped into MultiColumnConfigs
        single_config = expected_single_configs[0]
        config_builder.add_column(single_config)

        builder = DatasetBuilder(
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

        builder = DatasetBuilder(
            data_designer_config=config_builder.build(),
            resource_provider=stub_resource_provider,
        )
        assert builder.single_column_configs == expected_single_configs


def test_dataset_builder_build_method_basic_flow(
    stub_dataset_builder,
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

    stub_dataset_builder.batch_manager = stub_batch_manager
    stub_dataset_builder.set_processor_runner([])  # No processors for basic flow test

    result_path = stub_dataset_builder.build(num_records=100)

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
def test_dataset_builder_validate_column_configs(
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
            DatasetBuilder(
                data_designer_config=config_builder.build(),
                resource_provider=stub_resource_provider,
                registry=mock_registry,
            )
    else:
        # Empty column_configs case - config_builder will fail at build() due to validation
        with pytest.raises((DatasetGenerationError, Exception)):
            DatasetBuilder(
                config_builder=config_builder,
                resource_provider=stub_resource_provider,
            )


def test_run_config_default_non_inference_max_parallel_workers() -> None:
    run_config = RunConfig()
    assert run_config.non_inference_max_parallel_workers == 4


@patch("data_designer.engine.dataset_builders.dataset_builder.TelemetryHandler")
def test_emit_batch_inference_events_emits_from_deltas(
    mock_telemetry_handler_class: Mock,
    stub_resource_provider: Mock,
    stub_test_config_builder: DataDesignerConfigBuilder,
) -> None:
    usage_deltas = {"test-model": ModelUsageStats(token_usage=TokenUsageStats(input_tokens=50, output_tokens=150))}

    builder = DatasetBuilder(
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


@patch("data_designer.engine.dataset_builders.dataset_builder.TelemetryHandler")
def test_emit_batch_inference_events_skips_when_no_deltas(
    mock_telemetry_handler_class: Mock,
    stub_resource_provider: Mock,
    stub_test_config_builder: DataDesignerConfigBuilder,
) -> None:
    usage_deltas: dict[str, ModelUsageStats] = {}

    builder = DatasetBuilder(
        data_designer_config=stub_test_config_builder.build(),
        resource_provider=stub_resource_provider,
    )

    session_id = "550e8400-e29b-41d4-a716-446655440000"
    builder._emit_batch_inference_events("batch", usage_deltas, session_id)

    mock_telemetry_handler_class.assert_not_called()


@patch("data_designer.engine.dataset_builders.dataset_builder.TelemetryHandler")
def test_emit_batch_inference_events_handles_multiple_models(
    mock_telemetry_handler_class: Mock,
    stub_resource_provider: Mock,
    stub_test_config_builder: DataDesignerConfigBuilder,
) -> None:
    usage_deltas = {
        "model-a": ModelUsageStats(token_usage=TokenUsageStats(input_tokens=100, output_tokens=200)),
        "model-b": ModelUsageStats(token_usage=TokenUsageStats(input_tokens=50, output_tokens=75)),
    }

    builder = DatasetBuilder(
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
@patch("data_designer.engine.dataset_builders.dataset_builder.ConcurrentThreadExecutor")
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

    builder = DatasetBuilder(
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


@patch("data_designer.engine.dataset_builders.dataset_builder.ConcurrentThreadExecutor")
def test_fan_out_with_threads_passes_column_name_in_context(
    mock_executor_class: Mock,
    stub_resource_provider: Mock,
    stub_test_config_builder: DataDesignerConfigBuilder,
) -> None:
    builder = DatasetBuilder(
        data_designer_config=stub_test_config_builder.build(),
        resource_provider=stub_resource_provider,
    )

    mock_executor = Mock()
    mock_executor_class.return_value.__enter__ = Mock(return_value=mock_executor)
    mock_executor_class.return_value.__exit__ = Mock(return_value=False)

    mock_generator = Mock()
    mock_generator.get_generation_strategy.return_value = GenerationStrategy.CELL_BY_CELL
    mock_generator.config.name = "test_column"
    mock_generator.config.column_type = "llm_text"
    mock_generator.config.tool_alias = None

    builder.batch_manager = Mock()
    builder.batch_manager.num_records_batch = 2
    builder.batch_manager.num_records_in_buffer = 2
    builder.batch_manager.iter_current_batch.return_value = [(0, {"seed": "a"}), (1, {"seed": "b"})]

    builder._fan_out_with_threads(mock_generator, max_workers=2)

    submitted_contexts = [call.kwargs["context"] for call in mock_executor.submit.call_args_list]
    assert submitted_contexts == [
        {"index": 0, "column_name": "test_column"},
        {"index": 1, "column_name": "test_column"},
    ]


@patch("data_designer.engine.dataset_builders.dataset_builder.AsyncConcurrentExecutor", create=True)
def test_fan_out_with_async_passes_column_name_in_context(
    mock_executor_class: Mock,
    stub_resource_provider: Mock,
    stub_test_config_builder: DataDesignerConfigBuilder,
) -> None:
    builder = DatasetBuilder(
        data_designer_config=stub_test_config_builder.build(),
        resource_provider=stub_resource_provider,
    )

    mock_executor = Mock()

    def _run(work_items: list[tuple[object, dict[str, int | str]]]) -> None:
        for coro, _context in work_items:
            coro.close()

    mock_executor.run.side_effect = _run
    mock_executor_class.return_value = mock_executor

    mock_generator = Mock()
    mock_generator.get_generation_strategy.return_value = GenerationStrategy.CELL_BY_CELL
    mock_generator.config.name = "test_column"
    mock_generator.config.column_type = "llm_text"
    mock_generator.config.tool_alias = None

    async def _agenerate(record: dict[str, str]) -> dict[str, str]:
        return record

    mock_generator.agenerate.side_effect = _agenerate

    builder.batch_manager = Mock()
    builder.batch_manager.num_records_batch = 2
    builder.batch_manager.iter_current_batch.return_value = [(0, {"seed": "a"}), (1, {"seed": "b"})]

    builder._fan_out_with_async(mock_generator, max_workers=2)

    work_items = mock_executor.run.call_args.args[0]
    submitted_contexts = [context for _coro, context in work_items]
    assert submitted_contexts == [
        {"index": 0, "column_name": "test_column"},
        {"index": 1, "column_name": "test_column"},
    ]


def test_full_column_custom_generator_error_is_descriptive(stub_resource_provider, stub_model_configs):
    @custom_column_generator(required_columns=["some_id"])
    def bad_fn(df: pd.DataFrame) -> pd.DataFrame:
        raise ValueError("something broke")

    config = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config.add_column(SamplerColumnConfig(name="some_id", sampler_type=SamplerType.UUID, params=UUIDSamplerParams()))
    config.add_column(CustomColumnConfig(name="col", generator_function=bad_fn, generation_strategy="full_column"))
    builder = DatasetBuilder(data_designer_config=config.build(), resource_provider=stub_resource_provider)

    with pytest.raises(DatasetGenerationError, match=r"(?s)Failed to process column 'col'.*something broke"):
        builder.build_preview(num_records=3)


def test_build_async_preview_returns_empty_dataframe_when_row_group_is_already_freed(
    stub_resource_provider,
    stub_test_config_builder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = DatasetBuilder(
        data_designer_config=stub_test_config_builder.build(),
        resource_provider=stub_resource_provider,
    )

    class StubScheduler:
        traces: list[object] = []

        async def run(self) -> None:
            return None

    class MockFuture:
        def result(self) -> None:
            return None

    def mock_run_coroutine_threadsafe(coro, loop):
        coro.close()
        return MockFuture()

    scheduler = StubScheduler()
    buffer_manager = Mock()
    buffer_manager.has_row_group.return_value = False

    monkeypatch.setattr(builder, "_prepare_async_run", Mock(return_value=(scheduler, buffer_manager)))
    monkeypatch.setattr(builder_mod, "ensure_async_engine_loop", lambda: object(), raising=False)
    monkeypatch.setattr(
        builder_mod,
        "asyncio",
        Mock(run_coroutine_threadsafe=mock_run_coroutine_threadsafe),
        raising=False,
    )

    result = builder._build_async_preview([], num_records=3)

    assert result.empty
    buffer_manager.get_dataframe.assert_not_called()
    buffer_manager.free_row_group.assert_not_called()


# Processor tests


@pytest.fixture
def simple_builder(stub_resource_provider, stub_model_configs):
    """Minimal builder with a single UUID column and no batch files on disk."""
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(SamplerColumnConfig(name="id", sampler_type="uuid", params=UUIDSamplerParams()))
    return DatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )


def test_initialize_processors(stub_dataset_builder):
    processors = stub_dataset_builder.processors
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
    lazy.pd.DataFrame({"id": list(range(1, 10))}).to_parquet(
        storage.final_dataset_path / "batch_00000.parquet", index=False
    )

    mock_processor = create_mock_processor("proc", ["process_after_generation"])
    mock_processor.process_after_generation.side_effect = processor_fn

    simple_builder.set_processor_runner([mock_processor])
    simple_builder._processor_runner.run_after_generation(batch_size)

    mock_processor.process_after_generation.assert_called_once()
    batch_files = sorted(storage.final_dataset_path.glob("*.parquet"))
    assert len(batch_files) == expected_files
    assert sum(len(lazy.pd.read_parquet(f)) for f in batch_files) == expected_rows


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
        simple_builder._processor_runner.run_post_batch(lazy.pd.DataFrame({"id": [1, 2, 3]}), current_batch_number=0)


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

    result = simple_builder.process_preview(lazy.pd.DataFrame())

    assert len(result) == 0
    mock_processor.process_after_batch.assert_called_once()
    mock_processor.process_after_generation.assert_called_once()


# allow_resize integration tests
#
# Factory: _make_resize_full_expand. Stubs: _resize_full_keep_first, _resize_cell_*.


def _make_resize_full_expand(n: int, primary_col: str, side_effect_col: str):
    """FULL_COLUMN: expand n times per seed_id."""

    @custom_column_generator(required_columns=["seed_id"], side_effect_columns=[side_effect_col])
    def fn(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in df.iterrows():
            for i in range(n):
                rows.append({**row.to_dict(), primary_col: f"{row['seed_id']}_v{i}", side_effect_col: i})
        return lazy.pd.DataFrame(rows)

    return fn


@custom_column_generator(required_columns=["seed_id"])
def _resize_full_keep_first(df: pd.DataFrame) -> pd.DataFrame:
    """FULL_COLUMN: keep first row per seed_id (retraction)."""
    return df.drop_duplicates(subset="seed_id").assign(filtered=True)


@custom_column_generator(required_columns=["seed_id"])
def _resize_cell_expand(row: dict) -> list[dict]:
    """CELL_BY_CELL: one row -> two rows (doubled)."""
    return [
        {**row, "doubled": f"{row['seed_id']}_a"},
        {**row, "doubled": f"{row['seed_id']}_b"},
    ]


@custom_column_generator(required_columns=["seed_id"])
def _resize_cell_filter_odd(row: dict) -> dict | list[dict]:
    """CELL_BY_CELL: drop even seed_id, keep odd."""
    if row["seed_id"] % 2 == 0:
        return []
    return {**row, "kept": row["seed_id"]}


@custom_column_generator(required_columns=["seed_id"])
def _resize_cell_drop_all(row: dict) -> list[dict]:
    """CELL_BY_CELL: return [] for every row (drop all)."""
    return []


_RESIZE_SPECS: dict[str, list[tuple[str, object, GenerationStrategy]]] = {
    "cell_filter_odd": [("kept", _resize_cell_filter_odd, GenerationStrategy.CELL_BY_CELL)],
    "cell_x2": [("doubled", _resize_cell_expand, GenerationStrategy.CELL_BY_CELL)],
    "cell_drop_all": [("dropped", _resize_cell_drop_all, GenerationStrategy.CELL_BY_CELL)],
    "full_x3": [("expanded", _make_resize_full_expand(3, "expanded", "copy"), GenerationStrategy.FULL_COLUMN)],
    "full_chain": [
        ("expanded", _make_resize_full_expand(3, "expanded", "copy"), GenerationStrategy.FULL_COLUMN),
        ("filtered", _resize_full_keep_first, GenerationStrategy.FULL_COLUMN),
        ("expanded_again", _make_resize_full_expand(3, "expanded_again", "copy2"), GenerationStrategy.FULL_COLUMN),
    ],
    "cell_plus_full_chain": [
        ("doubled", _resize_cell_expand, GenerationStrategy.CELL_BY_CELL),
        ("filtered", _resize_full_keep_first, GenerationStrategy.FULL_COLUMN),
        ("expanded_again", _make_resize_full_expand(3, "expanded_again", "copy2"), GenerationStrategy.FULL_COLUMN),
    ],
}


def _resize_columns(spec: str) -> list[CustomColumnConfig]:
    """Return column configs for a given allow_resize recipe."""
    return [
        CustomColumnConfig(
            name=name,
            generator_function=fn,
            generation_strategy=strat,
            allow_resize=True,
        )
        for name, fn, strat in _RESIZE_SPECS[spec]
    ]


def _build_resize_builder(stub_resource_provider, stub_model_configs, seed_data_setup, columns):
    """Build a DatasetBuilder with the given resize column configs."""
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.with_seed_dataset(LocalFileSeedSource(path=str(seed_data_setup["seed_path"])))
    for col in columns:
        config_builder.add_column(col)
    return DatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )


@pytest.mark.parametrize(
    "spec,num_records,expected_len,check_doubled_order",
    [
        ("cell_filter_odd", 5, 3, False),
        ("cell_x2", 5, 10, True),
        ("cell_drop_all", 5, 0, False),
        ("full_x3", 5, 15, False),
        ("full_chain", 5, 15, False),
        ("cell_plus_full_chain", 5, 15, False),
    ],
    ids=[
        "cell_filter_odd_preview",
        "cell_x2_preview",
        "cell_drop_all_preview",
        "full_x3_preview",
        "full_chain_preview",
        "cell_plus_full_chain_preview",
    ],
)
def test_allow_resize_preview(
    stub_resource_provider,
    stub_model_configs,
    seed_data_setup,
    spec,
    num_records,
    expected_len,
    check_doubled_order,
):
    """Preview with allow_resize columns (FULL_COLUMN and/or CELL_BY_CELL) yields expected length."""
    columns = _resize_columns(spec)
    builder = _build_resize_builder(stub_resource_provider, stub_model_configs, seed_data_setup, columns)
    result = builder.build_preview(num_records=num_records)
    assert len(result) == expected_len
    if check_doubled_order:
        expected = [x for i in range(1, 6) for x in (f"{i}_a", f"{i}_b")]
        assert result["doubled"].tolist() == expected


@pytest.mark.parametrize(
    "spec,num_records,buffer_size,expected_total_rows",
    [
        ("cell_x2", 5, 2, 10),  # batches [2,2,1] -> each x2 -> 4+4+2
        ("cell_filter_odd", 5, 2, 3),  # batches [2,2,1] -> keep odd -> 1+1+1
        ("cell_drop_all", 5, 2, 0),  # each batch -> 0 rows
        ("full_x3", 5, 2, 15),  # batches [2,2,1] -> each x3 -> 6+6+3
        ("full_x3", 4, 2, 12),  # batches [2,2] -> 6+6
        ("full_chain", 5, 2, 15),  # batches [2,2,1] -> x3, dedup, x3 -> 15
    ],
    ids=[
        "cell_x2_multibatch",
        "cell_filter_odd_multibatch",
        "cell_drop_all_multibatch",
        "full_x3_multibatch_5_2",
        "full_x3_multibatch_4_2",
        "full_chain_multibatch",
    ],
)
def test_allow_resize_multiple_batches(
    stub_resource_provider,
    stub_model_configs,
    seed_data_setup,
    spec,
    num_records,
    buffer_size,
    expected_total_rows,
):
    """Resized batches are written independently and combine to expected total rows."""
    stub_resource_provider.run_config = RunConfig(buffer_size=buffer_size)
    columns = _resize_columns(spec)
    builder = _build_resize_builder(stub_resource_provider, stub_model_configs, seed_data_setup, columns)
    builder.build(num_records=num_records)
    final_path = builder.artifact_storage.final_dataset_path
    if expected_total_rows == 0 and not final_path.exists():
        df = lazy.pd.DataFrame()
    else:
        df = lazy.pd.read_parquet(final_path)
    assert len(df) == expected_total_rows


# ---------------------------------------------------------------------------
# Resume mechanism tests
# ---------------------------------------------------------------------------


import json as _json
from pathlib import Path as _Path

from data_designer.engine.storage.artifact_storage import ArtifactStorage as _ArtifactStorage


def _write_metadata(dataset_dir: _Path, **fields) -> None:
    """Write a metadata.json into an existing dataset folder."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "sentinel.txt").write_text("x")  # make folder non-empty for resolved_dataset_name
    (dataset_dir / "metadata.json").write_text(_json.dumps(fields))


def _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, *, buffer_size: int = 2):
    """Return a DatasetBuilder whose ArtifactStorage has resume=True."""
    storage = _ArtifactStorage(artifact_path=tmp_path, resume=True)
    stub_resource_provider.artifact_storage = storage
    stub_resource_provider.run_config = RunConfig(buffer_size=buffer_size)
    return DatasetBuilder(
        data_designer_config=stub_test_config_builder.build(),
        resource_provider=stub_resource_provider,
    )


def test_build_resume_raises_without_metadata(stub_resource_provider, stub_test_config_builder, tmp_path):
    """resume=True when only the folder exists (no metadata.json) raises DatasetGenerationError.

    This covers the case where a run was interrupted before any batch completed — the
    folder was created by _write_builder_config but metadata.json was never written.
    """
    # Pre-create the folder with content so resolved_dataset_name(resume=True) returns "dataset"
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "builder_config.json").write_text("{}")  # non-empty, no metadata

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path)
    with pytest.raises(DatasetGenerationError, match="metadata.json not found"):
        builder.build(num_records=4, resume=True)


def test_build_resume_raises_on_num_records_mismatch(stub_resource_provider, stub_test_config_builder, tmp_path):
    """resume=True raises when num_records differs from the original run."""
    dataset_dir = tmp_path / "dataset"
    _write_metadata(
        dataset_dir,
        target_num_records=10,
        buffer_size=2,
        num_completed_batches=2,
        actual_num_records=4,
    )

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)
    with pytest.raises(DatasetGenerationError, match="num_records=4 does not match"):
        builder.build(num_records=4, resume=True)


def test_build_resume_raises_on_buffer_size_mismatch(stub_resource_provider, stub_test_config_builder, tmp_path):
    """resume=True raises when buffer_size differs from the original run."""
    dataset_dir = tmp_path / "dataset"
    _write_metadata(
        dataset_dir,
        target_num_records=4,
        buffer_size=2,
        num_completed_batches=1,
        actual_num_records=2,
    )

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=3)
    with pytest.raises(DatasetGenerationError, match="buffer_size=3 does not match"):
        builder.build(num_records=4, resume=True)


def test_build_resume_logs_warning_when_already_complete(
    stub_resource_provider, stub_test_config_builder, tmp_path, caplog
):
    """resume=True on a fully-complete dataset logs a warning and returns without generating."""
    dataset_dir = tmp_path / "dataset"
    # 4 records, 2 per batch = 2 batches; num_completed_batches == 2 → already done
    _write_metadata(
        dataset_dir,
        target_num_records=4,
        buffer_size=2,
        num_completed_batches=2,
        actual_num_records=4,
    )

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)
    with caplog.at_level(logging.WARNING):
        builder.build(num_records=4, resume=True)

    assert any("already complete" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# _find_completed_row_group_ids tests
# ---------------------------------------------------------------------------


def test_find_completed_row_group_ids_empty_dir(stub_resource_provider, stub_test_config_builder, tmp_path):
    """Returns empty set when final_dataset_path does not exist."""
    dataset_dir = tmp_path / "dataset"
    _write_metadata(dataset_dir, target_num_records=4, buffer_size=2, num_completed_batches=0, actual_num_records=0)

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path)
    assert builder._find_completed_row_group_ids() == set()


def test_find_completed_row_group_ids_with_files(stub_resource_provider, stub_test_config_builder, tmp_path):
    """Returns correct IDs from batch_*.parquet files in parquet-files/."""
    dataset_dir = tmp_path / "dataset"
    _write_metadata(dataset_dir, target_num_records=6, buffer_size=2, num_completed_batches=2, actual_num_records=4)

    parquet_dir = dataset_dir / "parquet-files"
    parquet_dir.mkdir(parents=True)
    (parquet_dir / "batch_00000.parquet").write_text("")
    (parquet_dir / "batch_00002.parquet").write_text("")

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)
    assert builder._find_completed_row_group_ids() == {0, 2}


def test_find_completed_row_group_ids_ignores_non_batch_files(
    stub_resource_provider, stub_test_config_builder, tmp_path
):
    """Non-batch files in parquet-files/ are silently ignored."""
    dataset_dir = tmp_path / "dataset"
    _write_metadata(dataset_dir, target_num_records=4, buffer_size=2, num_completed_batches=1, actual_num_records=2)

    parquet_dir = dataset_dir / "parquet-files"
    parquet_dir.mkdir(parents=True)
    (parquet_dir / "batch_00001.parquet").write_text("")
    (parquet_dir / "unrelated.parquet").write_text("")

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)
    assert builder._find_completed_row_group_ids() == {1}


# ---------------------------------------------------------------------------
# Async resume via _build_async tests
# ---------------------------------------------------------------------------


def _write_parquet_files(parquet_dir: _Path, row_group_ids: list[int]) -> None:
    """Create stub batch_*.parquet files for the given row group IDs."""
    parquet_dir.mkdir(parents=True, exist_ok=True)
    for rg_id in row_group_ids:
        (parquet_dir / f"batch_{rg_id:05d}.parquet").write_text("")


def test_build_async_resume_logs_warning_when_already_complete(
    stub_resource_provider, stub_test_config_builder, tmp_path, caplog
):
    """Async resume on a fully-complete dataset logs a warning and returns without running."""
    dataset_dir = tmp_path / "dataset"
    # 4 records at buffer_size=2 → 2 row groups (IDs 0 and 1)
    _write_metadata(dataset_dir, target_num_records=4, buffer_size=2, num_completed_batches=2, actual_num_records=4)
    _write_parquet_files(dataset_dir / "parquet-files", [0, 1])

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)

    with caplog.at_level(logging.WARNING):
        with patch.object(builder_mod, "DATA_DESIGNER_ASYNC_ENGINE", True):
            with patch.object(builder, "_run_model_health_check_if_needed"):
                builder.build(num_records=4, resume=True)

    assert any("already complete" in record.message for record in caplog.records)


def test_build_async_resume_raises_without_metadata(stub_resource_provider, stub_test_config_builder, tmp_path):
    """Async resume raises DatasetGenerationError when metadata.json is missing."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "builder_config.json").write_text("{}")

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path)

    with patch.object(builder_mod, "DATA_DESIGNER_ASYNC_ENGINE", True):
        with patch.object(builder, "_run_model_health_check_if_needed"):
            with pytest.raises(DatasetGenerationError, match="metadata.json not found"):
                builder.build(num_records=4, resume=True)
