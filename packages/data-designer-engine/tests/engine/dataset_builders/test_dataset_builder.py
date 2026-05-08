# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

import data_designer.engine.dataset_builders.dataset_builder as builder_mod
import data_designer.lazy_heavy_imports as lazy
from data_designer.config.base import SkipConfig
from data_designer.config.column_configs import CustomColumnConfig, LLMTextColumnConfig, SamplerColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.custom_column import custom_column_generator
from data_designer.config.processors import DropColumnsProcessorConfig
from data_designer.config.run_config import RunConfig
from data_designer.config.sampler_params import SamplerType, UUIDSamplerParams
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.column_generators.generators.base import GenerationStrategy
from data_designer.engine.dataset_builders.dataset_builder import DatasetBuilder, _ConfigCompatibility
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
from data_designer.engine.storage.artifact_storage import ResumeMode

if TYPE_CHECKING:
    import pandas as pd


@pytest.fixture(autouse=True)
def _force_sync_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin tests in this file to the legacy sync engine.

    These tests use Mock-based stub resource providers that don't satisfy the
    contracts expected by the async task-queue scheduler (e.g. the registry's
    ``get_aggregate_max_parallel_requests()`` returns a Mock instead of an int).
    They cover sync-engine behavior; the async path has dedicated coverage in
    ``test_async_builder_integration.py`` and ``test_async_scheduler.py``.
    """
    monkeypatch.setattr(builder_mod, "DATA_DESIGNER_ASYNC_ENGINE", False)


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
    stub_model_configs: dict[str, object],
) -> None:
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(
        SamplerColumnConfig(name="some_id", sampler_type=SamplerType.UUID, params=UUIDSamplerParams())
    )
    builder = DatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )
    builder.build_preview(num_records=1)

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
    stub_model_configs: dict[str, object],
) -> None:
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(
        SamplerColumnConfig(name="some_id", sampler_type=SamplerType.UUID, params=UUIDSamplerParams())
    )
    builder = DatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )
    builder.build_preview(num_records=1)

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
        early_shutdown: bool = False
        partial_row_groups: tuple[int, ...] = ()
        first_non_retryable_error: Exception | None = None

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
    buffer_manager.actual_num_records = 0

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


def test_reset_run_state_clears_per_run_signals(stub_resource_provider, stub_test_config_builder) -> None:
    """``_reset_run_state`` must clear all per-run state so reused builders don't leak."""
    builder = DatasetBuilder(
        data_designer_config=stub_test_config_builder.build(),
        resource_provider=stub_resource_provider,
    )
    # Simulate prior-run state.
    builder._early_shutdown = True
    builder._partial_row_groups = (0, 1)
    builder._actual_num_records = 42
    builder._task_traces = ["trace"]  # type: ignore[list-item]

    builder._reset_run_state()

    assert builder.early_shutdown is False
    assert builder.partial_row_groups == ()
    assert builder.actual_num_records == -1
    assert builder.task_traces == []


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
def _resize_full_drop_seed_one(df: pd.DataFrame) -> pd.DataFrame:
    """FULL_COLUMN: drop the row with seed_id == 1."""
    return df[df["seed_id"] != 1].reset_index(drop=True).assign(filtered=True)


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


def test_resume_rejects_allow_resize_columns(stub_resource_provider, stub_model_configs, seed_data_setup, tmp_path):
    """Resume is rejected when allow_resize=True would make batch boundaries ambiguous."""
    artifact_path = tmp_path / "artifacts"
    artifact_path.mkdir()
    _write_metadata(
        artifact_path / "dataset",
        target_num_records=5,
        buffer_size=2,
        num_completed_batches=1,
        actual_num_records=2,
    )

    stub_resource_provider.artifact_storage = _ArtifactStorage(artifact_path=artifact_path, resume=ResumeMode.ALWAYS)
    columns = _resize_columns("cell_x2")
    builder = _build_resize_builder(stub_resource_provider, stub_model_configs, seed_data_setup, columns)

    with pytest.raises(DatasetGenerationError, match="allow_resize=True"):
        builder.build(num_records=5, resume=ResumeMode.ALWAYS)


def test_if_possible_allows_allow_resize_when_starting_fresh(
    stub_resource_provider, stub_model_configs, seed_data_setup
):
    """IF_POSSIBLE with allow_resize=True starts fresh when there is no checkpoint to resume."""
    columns = _resize_columns("cell_x2")
    builder = _build_resize_builder(stub_resource_provider, stub_model_configs, seed_data_setup, columns)

    final_path = builder.build(num_records=5, resume=ResumeMode.IF_POSSIBLE)

    df = lazy.pd.read_parquet(final_path)
    assert len(df) == 10


def test_if_possible_allows_allow_resize_when_config_is_incompatible(
    stub_resource_provider, stub_model_configs, seed_data_setup, tmp_path
):
    """IF_POSSIBLE with allow_resize=True starts fresh when an existing dataset is incompatible."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    sentinel = dataset_dir / "important_file.txt"
    sentinel.write_text("precious data")

    storage = _ArtifactStorage(artifact_path=tmp_path, resume=ResumeMode.IF_POSSIBLE)
    stub_resource_provider.artifact_storage = storage
    columns = _resize_columns("cell_x2")
    builder = _build_resize_builder(stub_resource_provider, stub_model_configs, seed_data_setup, columns)

    with patch.object(builder, "_check_resume_config_compatibility", return_value=_ConfigCompatibility.INCOMPATIBLE):
        final_path = builder.build(num_records=5, resume=ResumeMode.IF_POSSIBLE)

    assert storage.resume == ResumeMode.NEVER
    assert sentinel.exists()
    assert final_path != dataset_dir / "parquet-files"
    df = lazy.pd.read_parquet(final_path)
    assert len(df) == 10


# skip metadata preservation tests


def _make_label_generator(label: str, *required: str):
    """FULL_COLUMN generator that adds a column with a constant label value."""

    @custom_column_generator(required_columns=list(required))
    def fn(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(**{label: f"generated_{label}"})

    return fn


def _make_label_generator_with_side_effect(label: str, side_effect_label: str, *required: str):
    """FULL_COLUMN generator that adds a column plus one side-effect column."""

    @custom_column_generator(required_columns=list(required), side_effect_columns=[side_effect_label])
    def fn(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(
            **{
                label: f"generated_{label}",
                side_effect_label: f"generated_{side_effect_label}",
            }
        )

    return fn


def test_skip_metadata_preserved_across_non_skip_aware_full_column(
    stub_resource_provider, stub_model_configs, seed_data_setup
):
    """Skip metadata must survive when a non-skip-aware FULL_COLUMN column runs
    between a skip-setting column and a downstream propagating column.

    Scenario: rating(seed) -> review(skip.when) -> summary(no skip) -> complaint(propagate_skip)
    Before the fix, summary's replace_buffer erased __internal_skipped_columns,
    causing complaint to generate for rows that should have been skipped.
    """
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.with_seed_dataset(LocalFileSeedSource(path=str(seed_data_setup["seed_path"])))

    config_builder.add_column(
        CustomColumnConfig(
            name="review",
            generator_function=_make_label_generator("review", "seed_id"),
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            skip=SkipConfig(when="{{ seed_id < 3 }}"),
        )
    )
    config_builder.add_column(
        CustomColumnConfig(
            name="summary",
            generator_function=_make_label_generator("summary", "text"),
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            propagate_skip=False,
        )
    )
    config_builder.add_column(
        CustomColumnConfig(
            name="complaint",
            generator_function=_make_label_generator("complaint", "review"),
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            propagate_skip=True,
        )
    )

    builder = DatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )
    result = builder.build_preview(num_records=5)

    skipped_ids = {1, 2}
    for _, row in result.iterrows():
        if row["seed_id"] in skipped_ids:
            assert row["review"] is None or lazy.pd.isna(row["review"]), (
                f"seed_id={row['seed_id']}: review should be skipped"
            )
            assert row["complaint"] is None or lazy.pd.isna(row["complaint"]), (
                f"seed_id={row['seed_id']}: complaint should propagate skip from review"
            )
        else:
            assert row["complaint"] == "generated_complaint", f"seed_id={row['seed_id']}: complaint should be generated"


def test_skip_metadata_preserved_when_no_rows_skipped_for_current_column(
    stub_resource_provider, stub_model_configs, seed_data_setup
):
    """The has_skipped=False fallthrough must preserve sibling skip metadata.

    Scenario: review(skip.when seed_id<3) -> analysis(propagate_skip, required_columns=[review])
    analysis can_skip=True (via propagation) but no rows are skipped by analysis's
    own expression (it has none). The has_skipped=False fallthrough must still
    preserve review's skip metadata so propagation works.
    """
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.with_seed_dataset(LocalFileSeedSource(path=str(seed_data_setup["seed_path"])))

    config_builder.add_column(
        CustomColumnConfig(
            name="review",
            generator_function=_make_label_generator("review", "seed_id"),
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            skip=SkipConfig(when="{{ seed_id < 3 }}"),
        )
    )
    config_builder.add_column(
        CustomColumnConfig(
            name="analysis",
            generator_function=_make_label_generator("analysis", "review"),
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            propagate_skip=True,
        )
    )

    builder = DatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )
    result = builder.build_preview(num_records=5)

    skipped_ids = {1, 2}
    for _, row in result.iterrows():
        if row["seed_id"] in skipped_ids:
            assert row["analysis"] is None or lazy.pd.isna(row["analysis"]), (
                f"seed_id={row['seed_id']}: analysis should propagate skip from review"
            )
        else:
            assert row["analysis"] == "generated_analysis", f"seed_id={row['seed_id']}: analysis should be generated"


def test_skip_propagation_resolves_side_effect_dependencies_in_sync_builder(
    stub_resource_provider, stub_model_configs, seed_data_setup
):
    """A downstream dependency on a skipped side-effect should auto-skip.

    Scenario: review(skip.when, produces review_side_effect) ->
    analysis(required_columns=[review_side_effect], propagate_skip=True).
    """
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.with_seed_dataset(LocalFileSeedSource(path=str(seed_data_setup["seed_path"])))

    config_builder.add_column(
        CustomColumnConfig(
            name="review",
            generator_function=_make_label_generator_with_side_effect("review", "review_side_effect", "seed_id"),
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            skip=SkipConfig(when="{{ seed_id < 3 }}"),
        )
    )
    config_builder.add_column(
        CustomColumnConfig(
            name="analysis",
            generator_function=_make_label_generator("analysis", "review_side_effect"),
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            propagate_skip=True,
        )
    )

    builder = DatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )
    result = builder.build_preview(num_records=5)

    skipped_ids = {1, 2}
    for _, row in result.iterrows():
        if row["seed_id"] in skipped_ids:
            assert row["review_side_effect"] is None or lazy.pd.isna(row["review_side_effect"]), (
                f"seed_id={row['seed_id']}: review_side_effect should be cleared when review is skipped"
            )
            assert row["analysis"] is None or lazy.pd.isna(row["analysis"]), (
                f"seed_id={row['seed_id']}: analysis should propagate skip from review"
            )
        else:
            assert row["analysis"] == "generated_analysis", f"seed_id={row['seed_id']}: analysis should be generated"


def test_skip_metadata_restore_preserves_row_identity_across_allow_resize_full_column(
    stub_resource_provider, stub_model_configs, seed_data_setup
):
    """Filtering out a skipped row must not transfer its skip provenance to surviving rows."""
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.with_seed_dataset(LocalFileSeedSource(path=str(seed_data_setup["seed_path"])))

    config_builder.add_column(
        CustomColumnConfig(
            name="review",
            generator_function=_make_label_generator("review", "seed_id"),
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            skip=SkipConfig(when="{{ seed_id == 1 }}"),
        )
    )
    config_builder.add_column(
        CustomColumnConfig(
            name="filtered",
            generator_function=_resize_full_drop_seed_one,
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            allow_resize=True,
            propagate_skip=False,
        )
    )
    config_builder.add_column(
        CustomColumnConfig(
            name="analysis",
            generator_function=_make_label_generator("analysis", "review"),
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            propagate_skip=True,
        )
    )

    builder = DatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )
    result = builder.build_preview(num_records=5)

    assert result["seed_id"].tolist() == [2, 3, 4, 5]
    assert result["analysis"].tolist() == ["generated_analysis"] * 4


def test_allow_resize_column_not_blocked_by_upstream_skip(stub_resource_provider, stub_model_configs, seed_data_setup):
    """An allow_resize=True column depending on a skippable upstream must not
    enter the skip-aware branch (which enforces 1:1 row counts).

    Before the fix, _column_can_skip returned True for allow_resize columns
    with propagate_skip=True and required_columns pointing to a skippable
    upstream, causing a DatasetGenerationError on the row-count check.
    """
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.with_seed_dataset(LocalFileSeedSource(path=str(seed_data_setup["seed_path"])))

    config_builder.add_column(
        CustomColumnConfig(
            name="review",
            generator_function=_make_label_generator("review", "seed_id"),
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            skip=SkipConfig(when="{{ seed_id < 3 }}"),
        )
    )
    config_builder.add_column(
        CustomColumnConfig(
            name="expanded",
            generator_function=_make_resize_full_expand(2, "expanded", "copy"),
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            allow_resize=True,
        )
    )

    builder = DatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )
    result = builder.build_preview(num_records=5)
    assert len(result) == 10


def test_skip_chained_transitive_propagation_through_three_levels(
    stub_resource_provider, stub_model_configs, seed_data_setup
) -> None:
    """Skip at level 1 must propagate transitively through levels 2, 3, and 4.

    Pipeline: seed_id(seed) -> L1(skip.when) -> L2(propagate) -> L3(propagate) -> L4(propagate)
    """
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.with_seed_dataset(LocalFileSeedSource(path=str(seed_data_setup["seed_path"])))

    config_builder.add_column(
        CustomColumnConfig(
            name="L1",
            generator_function=_make_label_generator("L1", "seed_id"),
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            skip=SkipConfig(when="{{ seed_id < 3 }}"),
        )
    )
    config_builder.add_column(
        CustomColumnConfig(
            name="L2",
            generator_function=_make_label_generator("L2", "L1"),
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            propagate_skip=True,
        )
    )
    config_builder.add_column(
        CustomColumnConfig(
            name="L3",
            generator_function=_make_label_generator("L3", "L2"),
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            propagate_skip=True,
        )
    )
    config_builder.add_column(
        CustomColumnConfig(
            name="L4",
            generator_function=_make_label_generator("L4", "L3"),
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            propagate_skip=True,
        )
    )

    builder = DatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )
    result = builder.build_preview(num_records=5)

    assert len(result) == 5
    skipped_ids = {1, 2}
    for _, row in result.iterrows():
        if row["seed_id"] in skipped_ids:
            for col in ("L1", "L2", "L3", "L4"):
                assert row[col] is None or lazy.pd.isna(row[col]), (
                    f"seed_id={row['seed_id']}: {col} should be skipped transitively"
                )
        else:
            for col in ("L1", "L2", "L3", "L4"):
                assert row[col] == f"generated_{col}", f"seed_id={row['seed_id']}: {col} should be generated"


def test_skip_two_independent_gates_in_same_pipeline(
    stub_resource_provider, stub_model_configs, seed_data_setup
) -> None:
    """Two columns with independent skip.when expressions; downstream propagates from both.

    Pipeline: seed_id(seed) -> gate_a(skip seed_id<3) -> gate_b(skip seed_id>4) -> merge(propagate)
    merge should be skipped when *either* gate_a or gate_b was skipped.
    """
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.with_seed_dataset(LocalFileSeedSource(path=str(seed_data_setup["seed_path"])))

    config_builder.add_column(
        CustomColumnConfig(
            name="gate_a",
            generator_function=_make_label_generator("gate_a", "seed_id"),
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            skip=SkipConfig(when="{{ seed_id < 3 }}"),
        )
    )
    config_builder.add_column(
        CustomColumnConfig(
            name="gate_b",
            generator_function=_make_label_generator("gate_b", "seed_id"),
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            skip=SkipConfig(when="{{ seed_id > 4 }}"),
        )
    )
    config_builder.add_column(
        CustomColumnConfig(
            name="merge",
            generator_function=_make_label_generator("merge", "gate_a", "gate_b"),
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            propagate_skip=True,
        )
    )

    builder = DatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )
    result = builder.build_preview(num_records=5)

    assert len(result) == 5
    for _, row in result.iterrows():
        sid = row["seed_id"]
        if sid < 3 or sid > 4:
            assert row["merge"] is None or lazy.pd.isna(row["merge"]), (
                f"seed_id={sid}: merge should be skipped (gate_a or gate_b skipped)"
            )
        else:
            assert row["merge"] == "generated_merge", f"seed_id={sid}: merge should be generated"


def test_skip_custom_value_preserved_in_output(stub_resource_provider, stub_model_configs, seed_data_setup) -> None:
    """Custom skip.value should appear in the final DataFrame instead of None."""
    sentinel = "__SKIPPED__"
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.with_seed_dataset(LocalFileSeedSource(path=str(seed_data_setup["seed_path"])))

    config_builder.add_column(
        CustomColumnConfig(
            name="review",
            generator_function=_make_label_generator("review", "seed_id"),
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            skip=SkipConfig(when="{{ seed_id < 3 }}", value=sentinel),
        )
    )

    builder = DatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )
    result = builder.build_preview(num_records=5)

    assert len(result) == 5
    skipped_ids = {1, 2}
    for _, row in result.iterrows():
        if row["seed_id"] in skipped_ids:
            assert row["review"] == sentinel, f"seed_id={row['seed_id']}: review should have custom skip value"
        else:
            assert row["review"] == "generated_review", f"seed_id={row['seed_id']}: review should be generated"


def test_skip_row_count_preserved_across_pipeline(stub_resource_provider, stub_model_configs, seed_data_setup) -> None:
    """Skip must never change the row count — all 5 seed rows must survive."""
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.with_seed_dataset(LocalFileSeedSource(path=str(seed_data_setup["seed_path"])))

    config_builder.add_column(
        CustomColumnConfig(
            name="review",
            generator_function=_make_label_generator("review", "seed_id"),
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            skip=SkipConfig(when="{{ seed_id < 3 }}"),
        )
    )
    config_builder.add_column(
        CustomColumnConfig(
            name="analysis",
            generator_function=_make_label_generator("analysis", "review"),
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            propagate_skip=True,
        )
    )
    config_builder.add_column(
        CustomColumnConfig(
            name="summary",
            generator_function=_make_label_generator("summary", "analysis"),
            generation_strategy=GenerationStrategy.FULL_COLUMN,
            propagate_skip=True,
        )
    )

    builder = DatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )
    result = builder.build_preview(num_records=5)

    assert len(result) == 5, "Skip must not change the row count"
    assert result["seed_id"].tolist() == [1, 2, 3, 4, 5]


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


def _write_parquet_files(
    parquet_dir: _Path, row_group_ids: list[int], row_counts: dict[int, int] | None = None
) -> None:
    """Create batch_*.parquet files for the given row group IDs.

    Both engines now derive ``num_completed_batches`` and ``actual_num_records`` from
    these files at resume time, so any test that exercises the resume progress path
    must seed the dataset directory with matching parquet files in addition to
    ``metadata.json``.
    """
    parquet_dir.mkdir(parents=True, exist_ok=True)
    for rg_id in row_group_ids:
        count = row_counts.get(rg_id, 2) if row_counts is not None else 2
        lazy.pd.DataFrame({"value": list(range(count))}).to_parquet(
            parquet_dir / f"batch_{rg_id:05d}.parquet", index=False
        )


def _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, *, buffer_size: int = 2):
    """Return a DatasetBuilder whose ArtifactStorage has resume=ResumeMode.ALWAYS."""
    storage = _ArtifactStorage(artifact_path=tmp_path, resume=ResumeMode.ALWAYS)
    stub_resource_provider.artifact_storage = storage
    stub_resource_provider.run_config = RunConfig(buffer_size=buffer_size)
    return DatasetBuilder(
        data_designer_config=stub_test_config_builder.build(),
        resource_provider=stub_resource_provider,
    )


def test_build_resume_starts_fresh_without_metadata(stub_resource_provider, stub_test_config_builder, tmp_path, caplog):
    """resume=True when only the folder exists (no metadata.json) logs an info message and starts fresh.

    This covers the case where a run was interrupted before any batch completed — the
    folder was created by _write_builder_config but metadata.json was never written.
    Previously this raised DatasetGenerationError; now it silently restarts from batch 0.
    """
    # Pre-create the folder with content so resolved_dataset_name(resume=True) returns "dataset"
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "builder_config.json").write_text("{}")  # non-empty, no metadata

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path)
    with caplog.at_level(logging.INFO):
        with patch.object(builder, "_run_model_health_check_if_needed"):
            with patch.object(builder, "_run_batch"):
                with patch.object(builder.batch_manager, "finish"):
                    # resume=False is set internally; build dispatches to the normal (non-resume) path
                    builder.build(num_records=4, resume=ResumeMode.ALWAYS)

    assert any("interrupted before any batch completed" in record.message for record in caplog.records)


def test_build_resume_raises_when_num_records_below_actual(stub_resource_provider, stub_test_config_builder, tmp_path):
    """resume=ALWAYS raises when num_records is less than what has already been generated."""
    dataset_dir = tmp_path / "dataset"
    _write_metadata(
        dataset_dir,
        target_num_records=10,
        buffer_size=2,
        num_completed_batches=3,
        actual_num_records=6,
    )
    # Six records on disk drive the actual_num_records check; metadata is informational only.
    _write_parquet_files(dataset_dir / "parquet-files", [0, 1, 2])

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)
    with pytest.raises(DatasetGenerationError, match="num_records=4 is less than the 6 records already generated"):
        builder.build(num_records=4, resume=ResumeMode.ALWAYS)


def test_build_resume_raises_when_num_records_below_original_target(
    stub_resource_provider, stub_test_config_builder, tmp_path
):
    """resume=ALWAYS raises when num_records is between actual and original target (negative extension_records)."""
    dataset_dir = tmp_path / "dataset"
    _write_metadata(
        dataset_dir,
        target_num_records=10,
        buffer_size=2,
        num_completed_batches=2,
        actual_num_records=4,
    )

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)
    with pytest.raises(DatasetGenerationError, match="num_records=7 is less than the original target"):
        builder.build(num_records=7, resume=ResumeMode.ALWAYS)


def test_build_resume_allows_larger_num_records(stub_resource_provider, stub_test_config_builder, tmp_path, caplog):
    """resume=ALWAYS succeeds when num_records > original target (extending the dataset)."""
    dataset_dir = tmp_path / "dataset"
    _write_metadata(
        dataset_dir,
        target_num_records=4,
        buffer_size=2,
        num_completed_batches=2,
        actual_num_records=4,
    )

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)
    with caplog.at_level(logging.WARNING):
        with patch.object(builder, "_run_model_health_check_if_needed"):
            # 6 > 4 already generated → not already complete, should start generating
            # Here we just verify it does NOT raise on the num_records check
            with patch.object(builder, "_build_with_resume", return_value=True):
                builder.build(num_records=6, resume=ResumeMode.ALWAYS)


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
        builder.build(num_records=4, resume=ResumeMode.ALWAYS)


def test_build_resume_raises_on_corrupt_metadata(stub_resource_provider, stub_test_config_builder, tmp_path):
    """resume=ALWAYS raises clearly when metadata.json was partially written."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "sentinel.txt").write_text("x")
    (dataset_dir / "metadata.json").write_text("{not valid json")

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)
    with pytest.raises(DatasetGenerationError, match="metadata.json is corrupt"):
        builder.build(num_records=4, resume=ResumeMode.ALWAYS)


def test_build_resume_always_raises_on_config_mismatch(stub_resource_provider, stub_test_config_builder, tmp_path):
    """resume=ALWAYS raises DatasetGenerationError when the stored config fingerprint differs."""
    dataset_dir = tmp_path / "dataset"
    _write_metadata(dataset_dir, target_num_records=4, buffer_size=2, num_completed_batches=1, actual_num_records=2)

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path)
    with patch.object(builder, "_check_resume_config_compatibility", return_value=_ConfigCompatibility.INCOMPATIBLE):
        with pytest.raises(DatasetGenerationError, match="does not match the config used"):
            builder.build(num_records=4, resume=ResumeMode.ALWAYS)


def test_build_resume_logs_warning_when_already_complete(
    stub_resource_provider, stub_test_config_builder, tmp_path, caplog
):
    """resume=True on a fully-complete dataset logs a warning and returns without generating."""
    dataset_dir = tmp_path / "dataset"
    # 4 records, 2 per batch = 2 batches; both row groups on disk → already done
    _write_metadata(
        dataset_dir,
        target_num_records=4,
        buffer_size=2,
        num_completed_batches=2,
        actual_num_records=4,
    )
    _write_parquet_files(dataset_dir / "parquet-files", [0, 1])

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)
    with caplog.at_level(logging.WARNING):
        builder.build(num_records=4, resume=ResumeMode.ALWAYS)

    assert any("already complete" in record.message for record in caplog.records)


def test_build_resume_already_complete_does_not_run_after_generation_processors(
    stub_resource_provider, stub_test_config_builder, tmp_path
):
    """When already complete, run_after_generation must NOT be called (would destroy the dataset)."""
    dataset_dir = tmp_path / "dataset"
    _write_metadata(
        dataset_dir,
        target_num_records=4,
        buffer_size=2,
        num_completed_batches=2,
        actual_num_records=4,
    )
    _write_parquet_files(dataset_dir / "parquet-files", [0, 1])

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)
    with patch.object(builder._processor_runner, "run_after_generation") as mock_after:
        builder.build(num_records=4, resume=ResumeMode.ALWAYS)

    mock_after.assert_not_called()


def test_build_resume_post_generation_processed_same_target_returns_existing_path(
    stub_resource_provider, stub_test_config_builder, tmp_path
):
    """A post-processed completed dataset is a no-op when requested target matches."""
    dataset_dir = tmp_path / "dataset"
    _write_metadata(
        dataset_dir,
        target_num_records=4,
        buffer_size=2,
        num_completed_batches=2,
        actual_num_records=4,
        post_generation_processed=True,
    )

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)
    with patch.object(builder, "_initialize_generators_and_graph") as mock_initialize:
        result = builder.build(num_records=4, resume=ResumeMode.ALWAYS)

    assert result == builder.artifact_storage.final_dataset_path
    mock_initialize.assert_not_called()


def test_build_resume_post_generation_processed_extension_raises(
    stub_resource_provider, stub_test_config_builder, tmp_path
):
    """A post-processed dataset cannot be extended via resume."""
    dataset_dir = tmp_path / "dataset"
    _write_metadata(
        dataset_dir,
        target_num_records=4,
        buffer_size=2,
        num_completed_batches=2,
        actual_num_records=4,
        post_generation_processed=True,
    )

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)
    with pytest.raises(DatasetGenerationError, match="Extending would mix pre- and post-processor records"):
        builder.build(num_records=6, resume=ResumeMode.ALWAYS)


def test_build_resume_post_generation_processed_smaller_target_raises(
    stub_resource_provider, stub_test_config_builder, tmp_path
):
    """A post-processed dataset cannot be resumed with a smaller target than already generated."""
    dataset_dir = tmp_path / "dataset"
    _write_metadata(
        dataset_dir,
        target_num_records=4,
        buffer_size=2,
        num_completed_batches=2,
        actual_num_records=4,
        post_generation_processed=True,
    )

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)
    with pytest.raises(DatasetGenerationError, match="num_records=2 is less than the 4 records"):
        builder.build(num_records=2, resume=ResumeMode.ALWAYS)


def test_build_resume_post_generation_started_raises(stub_resource_provider, stub_test_config_builder, tmp_path):
    """A dataset with an interrupted after-generation processor cannot be resumed."""
    dataset_dir = tmp_path / "dataset"
    _write_metadata(
        dataset_dir,
        target_num_records=4,
        buffer_size=2,
        num_completed_batches=2,
        actual_num_records=4,
        post_generation_state="started",
        post_generation_processed=False,
    )

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)
    with pytest.raises(DatasetGenerationError, match="started but did not complete"):
        builder.build(num_records=4, resume=ResumeMode.ALWAYS)


def test_build_marks_post_generation_started_before_running_processors(
    stub_resource_provider, stub_test_config_builder, tmp_path
):
    """The crash-recovery marker is durable before after-generation processors mutate final parquet files."""
    dataset_dir = tmp_path / "dataset"
    _write_metadata(
        dataset_dir,
        target_num_records=4,
        buffer_size=2,
        num_completed_batches=1,
        actual_num_records=2,
    )

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)
    with patch.object(builder, "_initialize_generators_and_graph", return_value=([], None)):
        with patch.object(builder, "_build_with_resume", return_value=True):
            with patch.object(builder._processor_runner, "has_processors_for", return_value=True):
                with patch.object(builder._processor_runner, "run_after_generation", side_effect=RuntimeError("boom")):
                    with pytest.raises(RuntimeError, match="boom"):
                        builder.build(num_records=4, resume=ResumeMode.ALWAYS)

    metadata = _json.loads((dataset_dir / "metadata.json").read_text())
    assert metadata["post_generation_state"] == "started"
    assert metadata["post_generation_processed"] is False


def test_build_resume_not_already_complete_when_extension_fits_in_slack(
    stub_resource_provider, stub_test_config_builder, tmp_path
):
    """Non-aligned extension fitting in the last group's slack must not falsely trigger 'already complete'.

    original_target=5, buffer_size=2 → 3 batches [2,2,1]; extending to num_records=6:
    ceil(6/2)=3 == num_completed_batches=3 used to trigger the false 'already complete' branch.
    Correct total_batches = 3 + ceil(1/2) = 4, so batch 3 (1 record) must be scheduled.
    """
    dataset_dir = tmp_path / "dataset"
    _write_metadata(dataset_dir, target_num_records=5, buffer_size=2, num_completed_batches=3, actual_num_records=5)
    # Three row groups [2, 2, 1] on disk so the unified resume path sees 3 completed batches.
    _write_parquet_files(dataset_dir / "parquet-files", [0, 1, 2], row_counts={2: 1})

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)

    with patch.object(builder, "_run_batch") as mock_run_batch:
        with patch.object(builder.batch_manager, "finish"):
            with patch.object(builder, "_run_model_health_check_if_needed"):
                builder.build(num_records=6, resume=ResumeMode.ALWAYS)

    mock_run_batch.assert_called_once()
    assert mock_run_batch.call_args.kwargs["current_batch_number"] == 3


def test_build_resume_recovers_progress_from_disk_when_metadata_lags(
    stub_resource_provider, stub_test_config_builder, tmp_path, caplog
):
    """Sync resume uses parquet files on disk as the source of truth for progress.

    Crash window: ``move_partial_result_to_final_file_path`` succeeded for batch 1 but
    ``write_metadata`` had not yet committed the matching ``num_completed_batches`` /
    ``actual_num_records`` update. Before unification, sync took the stale metadata
    counters at face value and re-generated batch 1, double-counting records. After
    unification, both engines derive progress from ``parquet-files/batch_*.parquet``,
    so this scenario resolves to "already complete" and skips redundant generation.
    """
    dataset_dir = tmp_path / "dataset"
    # Metadata lags — claims only 1 batch / 2 records committed.
    _write_metadata(dataset_dir, target_num_records=4, buffer_size=2, num_completed_batches=1, actual_num_records=2)
    # Filesystem truth — both row groups written before the crash.
    _write_parquet_files(dataset_dir / "parquet-files", [0, 1])

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)
    with caplog.at_level(logging.WARNING):
        with patch.object(builder, "_run_batch") as mock_run_batch:
            with patch.object(builder.batch_manager, "finish"):
                with patch.object(builder, "_run_model_health_check_if_needed"):
                    builder.build(num_records=4, resume=ResumeMode.ALWAYS)

    mock_run_batch.assert_not_called()
    assert any("already complete" in record.message for record in caplog.records)


def test_build_resume_raises_on_non_contiguous_batch_ids_under_sync(
    stub_resource_provider, stub_test_config_builder, tmp_path
):
    """Sync resume rejects non-contiguous parquet IDs (likely written by an incompatible engine).

    The sync engine writes batches sequentially, so a hole between batch 0 and batch 2
    can only mean external mutation or data written by a different engine (e.g. the
    async engine, which can complete row groups out of order). Letting sync proceed
    would silently re-generate batch 1 with stale row counters; raising surfaces the
    inconsistency loudly.
    """
    dataset_dir = tmp_path / "dataset"
    _write_metadata(dataset_dir, target_num_records=6, buffer_size=2, num_completed_batches=2, actual_num_records=4)
    _write_parquet_files(dataset_dir / "parquet-files", [0, 2])

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)
    with pytest.raises(DatasetGenerationError, match="non-contiguous"):
        builder.build(num_records=6, resume=ResumeMode.ALWAYS)


# ---------------------------------------------------------------------------
# Async resume via _build_async tests
# ---------------------------------------------------------------------------


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
                builder.build(num_records=4, resume=ResumeMode.ALWAYS)

    assert any("already complete" in record.message for record in caplog.records)


def test_build_async_resume_starts_fresh_without_metadata(
    stub_resource_provider, stub_test_config_builder, tmp_path, caplog
):
    """Async resume with no metadata.json logs an info message and starts fresh.

    Previously this raised DatasetGenerationError; now it silently restarts from row group 0.
    The log is emitted in build() before dispatching to _build_async, so mocking _build_async
    does not suppress the message.
    """
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "builder_config.json").write_text("{}")

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path)

    with caplog.at_level(logging.INFO):
        with patch.object(builder_mod, "DATA_DESIGNER_ASYNC_ENGINE", True):
            with patch.object(builder, "_run_model_health_check_if_needed"):
                with patch.object(builder, "_build_async", return_value=True) as mock_async:
                    builder.build(num_records=4, resume=ResumeMode.ALWAYS)

    # _build_async is called with resume=NEVER because the no-metadata path resets the mode
    _, kwargs = mock_async.call_args
    assert kwargs.get("resume") == ResumeMode.NEVER
    assert any("interrupted before any batch completed" in record.message for record in caplog.records)


def test_build_async_resume_already_complete_does_not_run_after_generation_processors(
    stub_resource_provider, stub_test_config_builder, tmp_path
):
    """Async resume: when already complete, run_after_generation must NOT be called."""
    dataset_dir = tmp_path / "dataset"
    _write_metadata(dataset_dir, target_num_records=4, buffer_size=2, num_completed_batches=2, actual_num_records=4)
    _write_parquet_files(dataset_dir / "parquet-files", [0, 1])

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)

    with patch.object(builder_mod, "DATA_DESIGNER_ASYNC_ENGINE", True):
        with patch.object(builder, "_run_model_health_check_if_needed"):
            with patch.object(builder._processor_runner, "run_after_generation") as mock_after:
                builder.build(num_records=4, resume=ResumeMode.ALWAYS)

    mock_after.assert_not_called()


def test_find_completed_row_groups_used_for_initial_total_batches(
    stub_resource_provider, stub_test_config_builder, tmp_path
):
    """initial_total_num_batches uses filesystem count, not metadata count.

    Simulates the crash window: 2 parquet files exist on disk but metadata still
    records num_completed_batches=1 (write_metadata crashed after the second
    row group was moved to parquet-files/ but before metadata was updated).
    Verifies that _find_completed_row_groups() (= 2) is used, not metadata (= 1).
    """
    dataset_dir = tmp_path / "dataset"
    # Metadata lags — says only 1 batch completed
    _write_metadata(dataset_dir, target_num_records=4, buffer_size=2, num_completed_batches=1, actual_num_records=2)
    # Filesystem truth — 2 row groups already written
    _write_parquet_files(dataset_dir / "parquet-files", [0, 1])

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)
    # Both row groups are on disk → dataset is already complete → generated=False
    with patch.object(builder_mod, "DATA_DESIGNER_ASYNC_ENGINE", True):
        with patch.object(builder, "_run_model_health_check_if_needed"):
            with patch.object(builder._processor_runner, "run_after_generation") as mock_after:
                builder.build(num_records=4, resume=ResumeMode.ALWAYS)

    # Already complete based on filesystem count (2 files ≥ 2 row groups) — no generation needed
    mock_after.assert_not_called()


def test_initial_actual_num_records_from_filesystem_in_crash_window(
    stub_resource_provider, stub_test_config_builder, tmp_path
):
    """initial_actual_num_records is derived from filesystem, not stale metadata.

    Crash window scenario: row groups 0 and 1 are on disk but metadata only records
    num_completed_batches=1 / actual_num_records=2 (write_metadata crashed after
    the second row group was written but before it updated the file).

    With 6 records and buffer_size=2 (3 row groups total), the correct
    initial_actual_num_records is 4 (groups 0+1), not 2 (stale metadata value).
    """
    import asyncio as stdlib_asyncio

    dataset_dir = tmp_path / "dataset"
    # Metadata lags — says only 1 batch completed with 2 records
    _write_metadata(dataset_dir, target_num_records=6, buffer_size=2, num_completed_batches=1, actual_num_records=2)
    # Filesystem truth — 2 row groups already written (ids 0 and 1)
    _write_parquet_files(dataset_dir / "parquet-files", [0, 1])

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)

    captured: dict = {}

    def capturing_prepare(*args, **kwargs):
        captured["initial_actual_num_records"] = kwargs.get("initial_actual_num_records", 0)
        captured["initial_total_num_batches"] = kwargs.get("initial_total_num_batches", 0)
        mock_scheduler = Mock()
        mock_scheduler.traces = []
        mock_buffer_manager = Mock()
        mock_buffer_manager.actual_num_records = 6
        return mock_scheduler, mock_buffer_manager

    mock_future = Mock()
    mock_future.result = Mock(return_value=None)

    # asyncio and ensure_async_engine_loop are lazy-imported in dataset_builder only when
    # DATA_DESIGNER_ASYNC_ENGINE=True at module load time.  Inject them for the duration
    # of this test so _build_async can proceed past the early-return path.
    with patch.object(builder_mod, "DATA_DESIGNER_ASYNC_ENGINE", True):
        with patch.object(builder_mod, "asyncio", stdlib_asyncio, create=True):
            with patch.object(builder_mod, "ensure_async_engine_loop", Mock(return_value=Mock()), create=True):
                with patch.object(stdlib_asyncio, "run_coroutine_threadsafe", return_value=mock_future):
                    with patch.object(builder, "_run_model_health_check_if_needed"):
                        with patch.object(builder, "_prepare_async_run", side_effect=capturing_prepare):
                            builder.build(num_records=6, resume=ResumeMode.ALWAYS)

    # Filesystem says 2 groups done (IDs 0+1) → 2+2 = 4 records, not stale metadata value 2
    assert captured["initial_actual_num_records"] == 4
    assert captured["initial_total_num_batches"] == 2


def test_initial_actual_num_records_uses_actual_parquet_rows_for_partial_row_group(
    stub_resource_provider, stub_test_config_builder, tmp_path
):
    """Partial salvaged row groups count persisted parquet rows, not requested group size."""
    import asyncio as stdlib_asyncio

    dataset_dir = tmp_path / "dataset"
    _write_metadata(dataset_dir, target_num_records=6, buffer_size=2, num_completed_batches=1, actual_num_records=2)
    # Row group 1 was salvaged with only one surviving row.
    _write_parquet_files(dataset_dir / "parquet-files", [0, 1], row_counts={1: 1})

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)
    captured: dict = {}

    def capturing_prepare(*args, **kwargs):
        captured["initial_actual_num_records"] = kwargs.get("initial_actual_num_records", 0)
        captured["initial_total_num_batches"] = kwargs.get("initial_total_num_batches", 0)
        mock_scheduler = Mock()
        mock_scheduler.traces = []
        mock_scheduler.early_shutdown = False
        mock_scheduler.partial_row_groups = ()
        mock_scheduler.first_non_retryable_error = None
        mock_buffer_manager = Mock()
        mock_buffer_manager.actual_num_records = 6
        return mock_scheduler, mock_buffer_manager

    mock_future = Mock()
    mock_future.result = Mock(return_value=None)

    with patch.object(builder_mod, "DATA_DESIGNER_ASYNC_ENGINE", True):
        with patch.object(builder_mod, "asyncio", stdlib_asyncio, create=True):
            with patch.object(builder_mod, "ensure_async_engine_loop", Mock(return_value=Mock()), create=True):
                with patch.object(stdlib_asyncio, "run_coroutine_threadsafe", return_value=mock_future):
                    with patch.object(builder, "_run_model_health_check_if_needed"):
                        with patch.object(builder, "_prepare_async_run", side_effect=capturing_prepare):
                            builder.build(num_records=6, resume=ResumeMode.ALWAYS)

    assert captured["initial_actual_num_records"] == 3
    assert captured["initial_total_num_batches"] == 2


def test_build_async_resume_initial_actual_num_records_uses_original_target(
    stub_resource_provider, stub_test_config_builder, tmp_path
):
    """initial_actual_num_records uses the original target_num_records, not the new num_records.

    When extending a non-aligned run (original num_records=5, buffer_size=2 → row groups [2,2,1]),
    all 3 row groups completed. Resuming with num_records=7 must not use the new target in the
    formula: min(2, 7-2*2)=min(2,3)=2 would give 6, but the actual data is 5 records.
    """
    import asyncio as stdlib_asyncio

    dataset_dir = tmp_path / "dataset"
    # Original run: 5 records, buffer_size=2, all 3 row groups done
    _write_metadata(dataset_dir, target_num_records=5, buffer_size=2, num_completed_batches=3, actual_num_records=5)
    _write_parquet_files(dataset_dir / "parquet-files", [0, 1, 2], row_counts={2: 1})

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)

    captured: dict = {}

    def capturing_prepare(*args, **kwargs):
        captured["initial_actual_num_records"] = kwargs.get("initial_actual_num_records", 0)
        mock_scheduler = Mock()
        mock_scheduler.traces = []
        mock_buffer_manager = Mock()
        mock_buffer_manager.actual_num_records = 7
        return mock_scheduler, mock_buffer_manager

    mock_future = Mock()
    mock_future.result = Mock(return_value=None)

    with patch.object(builder_mod, "DATA_DESIGNER_ASYNC_ENGINE", True):
        with patch.object(builder_mod, "asyncio", stdlib_asyncio, create=True):
            with patch.object(builder_mod, "ensure_async_engine_loop", Mock(return_value=Mock()), create=True):
                with patch.object(stdlib_asyncio, "run_coroutine_threadsafe", return_value=mock_future):
                    with patch.object(builder, "_run_model_health_check_if_needed"):
                        with patch.object(builder, "_prepare_async_run", side_effect=capturing_prepare):
                            # Extend the dataset: new target is 7, original was 5
                            builder.build(num_records=7, resume=ResumeMode.ALWAYS)

    # Row groups [2, 2, 1] from original 5-record run: 2+2+1=5, not 2+2+2=6
    assert captured["initial_actual_num_records"] == 5


def test_build_async_resume_initial_actual_num_records_extension_crash_window(
    stub_resource_provider, stub_test_config_builder, tmp_path
):
    """Extension row groups on disk use new num_records in the size formula, not original target.

    Crash window: original run had num_records=5, buffer_size=2 (row groups [2,2,1], all done).
    Extension starts with num_records=9; row group 3 (2 records) is written to disk but
    write_metadata crashes before updating the file. On resume, completed_ids={0,1,2,3}
    while metadata still reports target_num_records=5.

    Correct count: groups 0,1 → 2+2; group 2 (last original, non-aligned) → 1; group 3
    (extension) → min(2, 9-6)=2. Total = 7, not 4 (which the unguarded formula gives,
    since min(2, 5-6) = -1).
    """
    import asyncio as stdlib_asyncio

    dataset_dir = tmp_path / "dataset"
    _write_metadata(dataset_dir, target_num_records=5, buffer_size=2, num_completed_batches=3, actual_num_records=5)
    _write_parquet_files(dataset_dir / "parquet-files", [0, 1, 2, 3], row_counts={2: 1})

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)

    captured: dict = {}

    def capturing_prepare(*args, **kwargs):
        captured["initial_actual_num_records"] = kwargs.get("initial_actual_num_records", 0)
        mock_scheduler = Mock()
        mock_scheduler.traces = []
        mock_buffer_manager = Mock()
        mock_buffer_manager.actual_num_records = 9
        return mock_scheduler, mock_buffer_manager

    mock_future = Mock()
    mock_future.result = Mock(return_value=None)

    with patch.object(builder_mod, "DATA_DESIGNER_ASYNC_ENGINE", True):
        with patch.object(builder_mod, "asyncio", stdlib_asyncio, create=True):
            with patch.object(builder_mod, "ensure_async_engine_loop", Mock(return_value=Mock()), create=True):
                with patch.object(stdlib_asyncio, "run_coroutine_threadsafe", return_value=mock_future):
                    with patch.object(builder, "_run_model_health_check_if_needed"):
                        with patch.object(builder, "_prepare_async_run", side_effect=capturing_prepare):
                            builder.build(num_records=9, resume=ResumeMode.ALWAYS)

    # 2+2+1 (original) + 2 (extension group 3) = 7, not 4 (which unguarded formula gives)
    assert captured["initial_actual_num_records"] == 7


def test_build_async_resume_stale_original_target_after_incremental_metadata_write(
    stub_resource_provider, stub_test_config_builder, tmp_path
):
    """original_target_num_records stays immutable even after an incremental metadata write.

    Scenario: original run had num_records=5, buffer_size=2 (row groups [2,2,1], all done).
    Extension to num_records=9 starts; row group 3 (2 records) completes and finalize_row_group
    writes metadata with target_num_records=9. Crash before row group 4.

    On second resume, metadata now shows target_num_records=9. Without the fix, original_target
    would be read as 9, making num_original_groups=5 and producing wrong _rg_size values.
    With the fix, original_target_num_records=5 is preserved in metadata, giving the correct
    initial_actual_num_records=7 (2+2+1 original + 2 extension).
    """
    import asyncio as stdlib_asyncio

    dataset_dir = tmp_path / "dataset"
    # Metadata reflects a post-incremental-write state: target updated to 9, original still 5
    _write_metadata(
        dataset_dir,
        target_num_records=9,
        original_target_num_records=5,
        buffer_size=2,
        num_completed_batches=4,
        actual_num_records=7,
    )
    _write_parquet_files(dataset_dir / "parquet-files", [0, 1, 2, 3], row_counts={2: 1})

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)

    captured: dict = {}

    def capturing_prepare(*args, **kwargs):
        captured["initial_actual_num_records"] = kwargs.get("initial_actual_num_records", 0)
        mock_scheduler = Mock()
        mock_scheduler.traces = []
        mock_buffer_manager = Mock()
        mock_buffer_manager.actual_num_records = 9
        return mock_scheduler, mock_buffer_manager

    mock_future = Mock()
    mock_future.result = Mock(return_value=None)

    with patch.object(builder_mod, "DATA_DESIGNER_ASYNC_ENGINE", True):
        with patch.object(builder_mod, "asyncio", stdlib_asyncio, create=True):
            with patch.object(builder_mod, "ensure_async_engine_loop", Mock(return_value=Mock()), create=True):
                with patch.object(stdlib_asyncio, "run_coroutine_threadsafe", return_value=mock_future):
                    with patch.object(builder, "_run_model_health_check_if_needed"):
                        with patch.object(builder, "_prepare_async_run", side_effect=capturing_prepare):
                            builder.build(num_records=9, resume=ResumeMode.ALWAYS)

    # original_target=5 → groups 0,1 → 2+2; group 2 → 1; group 3 (ext) → min(2,9-6)=2. Total=7
    assert captured["initial_actual_num_records"] == 7


def test_build_async_resume_skip_row_groups_contains_completed_ids(
    stub_resource_provider, stub_test_config_builder, tmp_path
):
    """precomputed_row_groups passed to _prepare_async_run excludes already-completed row groups.

    Verifies the skip mechanism so the scheduler never re-generates a row group that
    already has a parquet file on disk.  6 records, buffer_size=2 → 3 row groups total;
    row groups 0 and 2 already on disk → only row group 1 should be scheduled.
    """
    import asyncio as stdlib_asyncio

    dataset_dir = tmp_path / "dataset"
    # 6 records, buffer_size=2 → 3 row groups total; row groups 0 and 2 already on disk
    _write_metadata(dataset_dir, target_num_records=6, buffer_size=2, num_completed_batches=2, actual_num_records=4)
    _write_parquet_files(dataset_dir / "parquet-files", [0, 2])

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)

    captured: dict = {}

    def capturing_prepare(*args, **kwargs):
        captured["precomputed_row_groups"] = kwargs.get("precomputed_row_groups")
        mock_scheduler = Mock()
        mock_scheduler.traces = []
        mock_buffer_manager = Mock()
        mock_buffer_manager.actual_num_records = 6
        return mock_scheduler, mock_buffer_manager

    mock_future = Mock()
    mock_future.result = Mock(return_value=None)

    with patch.object(builder_mod, "DATA_DESIGNER_ASYNC_ENGINE", True):
        with patch.object(builder_mod, "asyncio", stdlib_asyncio, create=True):
            with patch.object(builder_mod, "ensure_async_engine_loop", Mock(return_value=Mock()), create=True):
                with patch.object(stdlib_asyncio, "run_coroutine_threadsafe", return_value=mock_future):
                    with patch.object(builder, "_run_model_health_check_if_needed"):
                        with patch.object(builder, "_prepare_async_run", side_effect=capturing_prepare):
                            builder.build(num_records=6, resume=ResumeMode.ALWAYS)

    # Only rg_id=1 remains; rg_id=0 and rg_id=2 are already on disk
    assert captured["precomputed_row_groups"] == [(1, 2)]


def test_build_async_resume_extension_non_aligned_row_group_sizes(
    stub_resource_provider, stub_test_config_builder, tmp_path
):
    """Extension row groups get the correct size when the original run was non-aligned.

    Original run: num_records=5, buffer_size=2 → row groups [2, 2, 1], all completed.
    Extending to num_records=7: the loop previously deducted 2 for rg_id=2 (instead of 1),
    leaving remaining=1 so rg_id=3 received size 1 instead of 2.  7 records were never
    generated; only 6 reached the dataset and a false partial-completion warning fired.

    After the fix, precomputed_row_groups must be [(3, 2)], not [(3, 1)].
    """
    import asyncio as stdlib_asyncio

    dataset_dir = tmp_path / "dataset"
    _write_metadata(dataset_dir, target_num_records=5, buffer_size=2, num_completed_batches=3, actual_num_records=5)
    _write_parquet_files(dataset_dir / "parquet-files", [0, 1, 2], row_counts={2: 1})

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)

    captured: dict = {}

    def capturing_prepare(*args, **kwargs):
        captured["precomputed_row_groups"] = kwargs.get("precomputed_row_groups")
        mock_scheduler = Mock()
        mock_scheduler.traces = []
        mock_buffer_manager = Mock()
        mock_buffer_manager.actual_num_records = 7
        return mock_scheduler, mock_buffer_manager

    mock_future = Mock()
    mock_future.result = Mock(return_value=None)

    with patch.object(builder_mod, "DATA_DESIGNER_ASYNC_ENGINE", True):
        with patch.object(builder_mod, "asyncio", stdlib_asyncio, create=True):
            with patch.object(builder_mod, "ensure_async_engine_loop", Mock(return_value=Mock()), create=True):
                with patch.object(stdlib_asyncio, "run_coroutine_threadsafe", return_value=mock_future):
                    with patch.object(builder, "_run_model_health_check_if_needed"):
                        with patch.object(builder, "_prepare_async_run", side_effect=capturing_prepare):
                            builder.build(num_records=7, resume=ResumeMode.ALWAYS)

    # rg_id=3 should have 2 records (7-5=2 extension records, buffer_size=2), not 1
    assert captured["precomputed_row_groups"] == [(3, 2)]


def test_build_async_resume_not_already_complete_when_extension_fits_in_slack(
    stub_resource_provider, stub_test_config_builder, tmp_path
):
    """Non-aligned extension fitting in the last group's slack must not falsely trigger 'already complete'.

    original_target=5, buffer_size=2 → 3 row groups; extending to num_records=6:
    ceil(6/2)=3 == len(completed_ids)=3 used to trigger the false 'already complete' branch.
    Correct total_row_groups = 3 + ceil(1/2) = 4, so _prepare_async_run must be called.
    """
    import asyncio as stdlib_asyncio

    dataset_dir = tmp_path / "dataset"
    _write_metadata(dataset_dir, target_num_records=5, buffer_size=2, num_completed_batches=3, actual_num_records=5)
    _write_parquet_files(dataset_dir / "parquet-files", [0, 1, 2], row_counts={2: 1})

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)

    def capturing_prepare(*args, **kwargs):
        mock_scheduler = Mock()
        mock_scheduler.traces = []
        mock_buffer_manager = Mock()
        mock_buffer_manager.actual_num_records = 6
        return mock_scheduler, mock_buffer_manager

    mock_future = Mock()
    mock_future.result = Mock(return_value=None)

    with patch.object(builder_mod, "DATA_DESIGNER_ASYNC_ENGINE", True):
        with patch.object(builder_mod, "asyncio", stdlib_asyncio, create=True):
            with patch.object(builder_mod, "ensure_async_engine_loop", Mock(return_value=Mock()), create=True):
                with patch.object(stdlib_asyncio, "run_coroutine_threadsafe", return_value=mock_future):
                    with patch.object(builder, "_run_model_health_check_if_needed"):
                        with patch.object(builder, "_prepare_async_run", side_effect=capturing_prepare) as mock_prepare:
                            builder.build(num_records=6, resume=ResumeMode.ALWAYS)

    # _prepare_async_run must be called — the dataset is NOT already complete
    mock_prepare.assert_called_once()


def test_if_possible_incompatible_config_does_not_overwrite_existing_dataset(
    stub_resource_provider, stub_test_config_builder, tmp_path
):
    """IF_POSSIBLE + incompatible config must NOT resolve to the existing dataset directory.

    Bug: _check_resume_config_compatibility() used base_dataset_path, triggering the
    resolved_dataset_name cached_property while artifact_storage.resume was still IF_POSSIBLE.
    The property cached the existing directory name; after resume was reset to NEVER locally,
    artifact_storage.resume was never updated, so _write_builder_config() still wrote into the
    old directory.

    Fix: _check_resume_config_compatibility() uses artifact_path/dataset_name directly and
    build() syncs artifact_storage.resume = NEVER before the first real access to base_dataset_path.
    """
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    sentinel = dataset_dir / "important_file.txt"
    sentinel.write_text("precious data")

    storage = _ArtifactStorage(artifact_path=tmp_path, resume=ResumeMode.IF_POSSIBLE)
    stub_resource_provider.artifact_storage = storage

    builder = DatasetBuilder(
        data_designer_config=stub_test_config_builder.build(),
        resource_provider=stub_resource_provider,
    )

    # Simulate incompatible config and mock out all I/O so build() does not actually generate data
    with patch.object(builder, "_check_resume_config_compatibility", return_value=_ConfigCompatibility.INCOMPATIBLE):
        with patch.object(builder, "_run_model_health_check_if_needed"):
            with patch.object(builder, "_run_mcp_tool_check_if_needed"):
                with patch.object(builder, "_write_builder_config"):
                    with patch.object(builder, "_initialize_generators_and_graph", return_value=([], None)):
                        with patch.object(builder.batch_manager, "start"):
                            with patch.object(builder.batch_manager, "finish"):
                                with patch.object(builder._processor_runner, "run_after_generation"):
                                    builder.build(num_records=2, resume=ResumeMode.IF_POSSIBLE)

    # artifact_storage.resume must be downgraded to NEVER so resolved_dataset_name uses NEVER semantics
    assert storage.resume == ResumeMode.NEVER

    # resolved_dataset_name has not been cached yet (compat check bypassed base_dataset_path,
    # _write_builder_config was mocked). Accessing it now must give a timestamped name.
    assert sentinel.exists(), "Existing dataset directory must not be touched"
    assert storage.resolved_dataset_name != "dataset", (
        "resolved_dataset_name must be a new timestamped directory, not the existing one"
    )


def test_if_possible_incompatible_config_refreshes_media_storage_path(
    stub_resource_provider, stub_test_config_builder, tmp_path
):
    """After IF_POSSIBLE → NEVER downgrade, _media_storage must point to the new timestamped dir.

    Bug: validate_folder_names initialises MediaStorage with base_dataset_path at Pydantic
    construction time (while resume=IF_POSSIBLE), caching the original directory name.
    After the cache pop and resume=NEVER, base_dataset_path resolves to a new timestamped
    directory, but _media_storage.base_path still holds the old path — producing broken
    image references for image-column datasets.

    Fix: refresh_media_storage_path() is called after the cache pop.
    """
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "existing_file.parquet").write_text("data")  # non-empty dir triggers NEVER→timestamp

    storage = _ArtifactStorage(artifact_path=tmp_path, resume=ResumeMode.IF_POSSIBLE)
    stub_resource_provider.artifact_storage = storage

    # Trigger validate_folder_names so _media_storage is initialised with IF_POSSIBLE semantics
    # (non-empty dir + IF_POSSIBLE → resolved_dataset_name returns "dataset", not timestamped)
    original_media_base = storage.media_storage.base_path

    builder = DatasetBuilder(
        data_designer_config=stub_test_config_builder.build(),
        resource_provider=stub_resource_provider,
    )

    with patch.object(builder, "_check_resume_config_compatibility", return_value=_ConfigCompatibility.INCOMPATIBLE):
        with patch.object(builder, "_run_model_health_check_if_needed"):
            with patch.object(builder, "_run_mcp_tool_check_if_needed"):
                with patch.object(builder, "_write_builder_config"):
                    with patch.object(builder, "_initialize_generators_and_graph", return_value=([], None)):
                        with patch.object(builder.batch_manager, "start"):
                            with patch.object(builder.batch_manager, "finish"):
                                with patch.object(builder._processor_runner, "run_after_generation"):
                                    builder.build(num_records=2, resume=ResumeMode.IF_POSSIBLE)

    new_media_base = storage.media_storage.base_path
    assert new_media_base != original_media_base, (
        "media_storage.base_path must be updated to the new timestamped directory after IF_POSSIBLE → NEVER downgrade"
    )
    assert new_media_base == storage.base_dataset_path, (
        "media_storage.base_path must match base_dataset_path after downgrade"
    )


def test_if_possible_starts_fresh_when_no_existing_directory(
    stub_resource_provider, stub_test_config_builder, tmp_path
):
    """IF_POSSIBLE on a first-ever run (no dataset directory) must start fresh, not raise.

    Bug: _check_resume_config_compatibility returned True when config_path did not exist,
    which caused IF_POSSIBLE to upgrade to ALWAYS. resolved_dataset_name then raised
    ArtifactStorageError because ALWAYS requires an existing directory.

    Fix: return False when the dataset directory itself is absent.
    """
    storage = _ArtifactStorage(artifact_path=tmp_path, resume=ResumeMode.IF_POSSIBLE)
    stub_resource_provider.artifact_storage = storage

    builder = DatasetBuilder(
        data_designer_config=stub_test_config_builder.build(),
        resource_provider=stub_resource_provider,
    )

    with patch.object(builder, "_run_model_health_check_if_needed"):
        with patch.object(builder, "_run_mcp_tool_check_if_needed"):
            with patch.object(builder, "_write_builder_config"):
                with patch.object(builder, "_initialize_generators_and_graph", return_value=([], None)):
                    with patch.object(builder.batch_manager, "start"):
                        with patch.object(builder.batch_manager, "finish"):
                            with patch.object(builder._processor_runner, "run_after_generation"):
                                builder.build(num_records=2, resume=ResumeMode.IF_POSSIBLE)

    assert storage.resume == ResumeMode.NEVER


def test_if_possible_starts_fresh_when_directory_is_empty(stub_resource_provider, stub_test_config_builder, tmp_path):
    """IF_POSSIBLE on an empty dataset directory must start fresh, not raise.

    Edge case: a prior run crashed in the window between mkdir and the first file write
    inside _write_builder_config, leaving an empty directory. _check_resume_config_compatibility
    previously returned True (config file absent → assume compatible), causing IF_POSSIBLE to
    upgrade to ALWAYS, which then raised ArtifactStorageError because the directory is empty.

    Fix: treat an empty directory the same as a missing one — return False.
    """
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()  # empty — no files written yet

    storage = _ArtifactStorage(artifact_path=tmp_path, resume=ResumeMode.IF_POSSIBLE)
    stub_resource_provider.artifact_storage = storage

    builder = DatasetBuilder(
        data_designer_config=stub_test_config_builder.build(),
        resource_provider=stub_resource_provider,
    )

    with patch.object(builder, "_run_model_health_check_if_needed"):
        with patch.object(builder, "_run_mcp_tool_check_if_needed"):
            with patch.object(builder, "_write_builder_config"):
                with patch.object(builder, "_initialize_generators_and_graph", return_value=([], None)):
                    with patch.object(builder.batch_manager, "start"):
                        with patch.object(builder.batch_manager, "finish"):
                            with patch.object(builder._processor_runner, "run_after_generation"):
                                builder.build(num_records=2, resume=ResumeMode.IF_POSSIBLE)

    assert storage.resume == ResumeMode.NEVER
