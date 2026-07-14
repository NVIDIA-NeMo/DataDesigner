# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import concurrent.futures
import json
import logging
import tracemalloc
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from unittest.mock import Mock, patch

import pytest

import data_designer.engine.dataset_builders.dataset_builder as builder_mod
import data_designer.lazy_heavy_imports as lazy
from data_designer.config.base import SkipConfig
from data_designer.config.column_configs import (
    CustomColumnConfig,
    ExpressionColumnConfig,
    LLMTextColumnConfig,
    SamplerColumnConfig,
)
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.custom_column import custom_column_generator
from data_designer.config.processors import DropColumnsProcessorConfig, SchemaTransformProcessorConfig
from data_designer.config.record_selection import RecordSelectionConfig, RecordSelectionExhaustion
from data_designer.config.run_config import RunConfig
from data_designer.config.sampler_params import (
    CategorySamplerParams,
    SamplerType,
    SubcategorySamplerParams,
    UUIDSamplerParams,
)
from data_designer.config.seed import IndexRange, PartitionBlock, SamplingStrategy
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.column_generators.generators.base import GenerationStrategy
from data_designer.engine.column_generators.generators.samplers import SamplerColumnGenerator
from data_designer.engine.dataset_builders.acceptance import (
    AcceptanceController,
    CandidateBatch,
    SelectionBatchMarker,
    SelectionDecision,
)
from data_designer.engine.dataset_builders.dataset_builder import DatasetBuilder, build_row_group_resume_plan
from data_designer.engine.dataset_builders.errors import (
    DatasetGenerationError,
    DatasetProcessingError,
    RecordSelectionEarlyShutdownError,
)
from data_designer.engine.dataset_builders.row_group_plan import CompactRowGroupPlan
from data_designer.engine.dataset_builders.utils.processor_runner import ProcessorRunner
from data_designer.engine.dataset_builders.utils.row_group_buffer import RowGroupBufferManager
from data_designer.engine.errors import DataDesignerRuntimeError
from data_designer.engine.models.telemetry import InferenceEvent, NemoSourceEnum, TaskStatusEnum
from data_designer.engine.models.usage import ModelUsageStats, TokenUsageStats
from data_designer.engine.observability import SchedulerAdmissionEvent
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.registry.data_designer_registry import DataDesignerRegistry
from data_designer.engine.resources.seed_reader import DataFrameSeedReader
from data_designer.engine.storage.artifact_storage import ArtifactStorage, ResumeMode
from data_designer.engine.testing import InMemoryAdmissionEventSink

if TYPE_CHECKING:
    import pandas as pd


def _replace_processors(builder: DatasetBuilder, processors: list[Processor]) -> None:
    builder._processor_runner = ProcessorRunner(processors=processors, artifact_storage=builder.artifact_storage)


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


def _create_boundary_selection_builder(
    stub_resource_provider: Any,
    *,
    after_generation: bool,
) -> tuple[DatasetBuilder, Mock | None]:
    """Build a two-row completed selection publication with six-digit candidate names."""
    stub_resource_provider.run_config = RunConfig(
        buffer_size=1,
        display_tui=False,
        preserve_dropped_columns=True,
    )
    config = DataDesignerConfigBuilder()
    config.add_column(
        SamplerColumnConfig(
            name="keep",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=[True]),
        )
    )
    config.add_column(
        SamplerColumnConfig(
            name="payload",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["accepted"]),
            drop=True,
        )
    )
    config.add_processor(
        SchemaTransformProcessorConfig(
            name="formatted",
            template={"value": "{{ payload }}"},
        )
    )
    config.with_record_selection(RecordSelectionConfig(predicate_column="keep", max_candidate_records=100_001))
    builder = DatasetBuilder(data_designer_config=config.build(), resource_provider=stub_resource_provider)
    after_generation_processor: Mock | None = None
    if after_generation:
        after_generation_processor = create_mock_processor("identity", ["process_after_generation"])
        _replace_processors(
            builder,
            [*builder._processor_runner.processors, after_generation_processor],
        )

    builder.build(num_records=2)
    if after_generation_processor is not None:
        after_generation_processor.process_after_generation.reset_mock()
    return builder, after_generation_processor


def _rewrite_completed_selection_at_filename_boundary(
    builder: DatasetBuilder,
    *,
    after_generation: bool,
    valid_publication: bool = True,
) -> str:
    """Rewrite two small artifacts to model legacy names around the five-digit boundary."""
    storage = builder.artifact_storage
    legacy_batch_ids = (99_999, 100_000)
    for checkpoint_id, legacy_batch_id in enumerate(legacy_batch_ids):
        accepted_source = storage.selection_accepted_path / f"batch_{checkpoint_id:06d}.parquet"
        accepted_target = storage.selection_accepted_path / f"batch_{checkpoint_id:05d}.parquet"
        accepted_source.unlink()
        lazy.pd.DataFrame({"value": [legacy_batch_id]}).to_parquet(accepted_target, index=False)

        checkpoint_source = storage.selection_checkpoints_path / f"batch_{checkpoint_id:06d}.json"
        checkpoint = json.loads(checkpoint_source.read_text(encoding="utf-8"))
        checkpoint["accepted_partition"] = accepted_target.relative_to(storage.base_dataset_path).as_posix()
        checkpoint_path = storage.selection_checkpoints_path / f"batch_{checkpoint_id:05d}.json"
        checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
        checkpoint_source.unlink()

        dropped_source = storage.dropped_columns_dataset_path / f"batch_{checkpoint_id:06d}.parquet"
        dropped_source.unlink()
        lazy.pd.DataFrame({"dropped": [legacy_batch_id]}).to_parquet(
            storage.dropped_columns_dataset_path / f"batch_{legacy_batch_id}.parquet",
            index=False,
        )

        processor_source = storage.processors_outputs_path / "formatted" / f"batch_{checkpoint_id:06d}.parquet"
        processor_source.unlink()
        lazy.pd.DataFrame({"value": [legacy_batch_id]}).to_parquet(
            storage.processors_outputs_path / "formatted" / f"batch_{legacy_batch_id}.parquet",
            index=False,
        )

    for path in storage.final_dataset_path.glob("batch_*.parquet"):
        path.unlink()
    if after_generation:
        publication_values = legacy_batch_ids[::-1]
        publication_names = ("batch_00000.parquet", "batch_00001.parquet")
    else:
        publication_values = legacy_batch_ids
        publication_names = ("batch_99999.parquet", "batch_100000.parquet")
    if not valid_publication:
        publication_values = publication_values[:1]
        publication_names = publication_names[:1]
    for name, value in zip(publication_names, publication_values, strict=True):
        lazy.pd.DataFrame({"value": [value]}).to_parquet(storage.final_dataset_path / name, index=False)

    metadata = storage.read_metadata()
    publication_id = metadata["record_selection"]["publication_id"]
    metadata["file_paths"] = storage.get_file_paths()
    storage.write_metadata(metadata)
    return publication_id


def test_dataset_builder_creation(stub_resource_provider, stub_test_config_builder):
    builder = DatasetBuilder(
        data_designer_config=stub_test_config_builder.build(),
        resource_provider=stub_resource_provider,
    )
    assert len(builder._column_configs) == 3
    assert builder._resource_provider == stub_resource_provider
    assert isinstance(builder._registry, DataDesignerRegistry)


def test_record_selection_rejects_known_non_boolean_builtin_predicate(stub_resource_provider) -> None:
    config = DataDesignerConfigBuilder()
    config.add_column(
        SamplerColumnConfig(
            name="numeric_predicate",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=[0, 1]),
        )
    )
    config.with_record_selection(RecordSelectionConfig(predicate_column="numeric_predicate", max_candidate_records=1))

    with pytest.raises(DatasetGenerationError, match="built-in column type.*does not produce boolean"):
        DatasetBuilder(data_designer_config=config.build(), resource_provider=stub_resource_provider)


def test_record_selection_accepts_known_boolean_category_predicate(stub_resource_provider) -> None:
    config = DataDesignerConfigBuilder()
    config.add_column(
        SamplerColumnConfig(
            name="boolean_predicate",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=[True, False]),
            conditional_params={"true": CategorySamplerParams(values=[False, True])},
        )
    )
    config.with_record_selection(RecordSelectionConfig(predicate_column="boolean_predicate", max_candidate_records=1))

    DatasetBuilder(data_designer_config=config.build(), resource_provider=stub_resource_provider)


def test_record_selection_runs_with_boolean_category_predicate(stub_resource_provider) -> None:
    stub_resource_provider.run_config = RunConfig(buffer_size=1, display_tui=False)
    config = DataDesignerConfigBuilder()
    config.add_column(
        SamplerColumnConfig(
            name="boolean_predicate",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=[True]),
        )
    )
    config.with_record_selection(RecordSelectionConfig(predicate_column="boolean_predicate", max_candidate_records=1))
    builder = DatasetBuilder(data_designer_config=config.build(), resource_provider=stub_resource_provider)

    output_path = builder.build(num_records=1)

    assert lazy.pd.read_parquet(output_path)["boolean_predicate"].tolist() == [True]


def test_record_selection_runs_with_boolean_subcategory_predicate(stub_resource_provider) -> None:
    stub_resource_provider.run_config = RunConfig(buffer_size=1, display_tui=False)
    config = DataDesignerConfigBuilder()
    config.add_column(
        SamplerColumnConfig(
            name="parent",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["group"]),
        )
    )
    config.add_column(
        SamplerColumnConfig(
            name="boolean_predicate",
            sampler_type=SamplerType.SUBCATEGORY,
            params=SubcategorySamplerParams(category="parent", values={"group": [True]}),
        )
    )
    config.with_record_selection(RecordSelectionConfig(predicate_column="boolean_predicate", max_candidate_records=1))
    builder = DatasetBuilder(data_designer_config=config.build(), resource_provider=stub_resource_provider)

    output_path = builder.build(num_records=1)

    assert lazy.pd.read_parquet(output_path)["boolean_predicate"].tolist() == [True]


def test_record_selection_rejects_subcategory_with_non_boolean_conditional_values(stub_resource_provider) -> None:
    config = DataDesignerConfigBuilder()
    config.add_column(
        SamplerColumnConfig(
            name="parent",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["group"]),
        )
    )
    config.add_column(
        SamplerColumnConfig(
            name="mixed_predicate",
            sampler_type=SamplerType.SUBCATEGORY,
            params=SubcategorySamplerParams(category="parent", values={"group": [True, False]}),
            conditional_params={"true": SubcategorySamplerParams(category="parent", values={"group": ["not-boolean"]})},
        )
    )
    config.with_record_selection(RecordSelectionConfig(predicate_column="mixed_predicate", max_candidate_records=1))

    with pytest.raises(DatasetGenerationError, match="does not produce boolean"):
        DatasetBuilder(data_designer_config=config.build(), resource_provider=stub_resource_provider)


def test_record_selection_rejects_converted_boolean_subcategory(stub_resource_provider) -> None:
    config = DataDesignerConfigBuilder()
    config.add_column(
        SamplerColumnConfig(
            name="parent",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["group"]),
        )
    )
    config.add_column(
        SamplerColumnConfig(
            name="converted_predicate",
            sampler_type=SamplerType.SUBCATEGORY,
            params=SubcategorySamplerParams(category="parent", values={"group": [True]}),
            convert_to="str",
        )
    )
    config.with_record_selection(RecordSelectionConfig(predicate_column="converted_predicate", max_candidate_records=1))

    with pytest.raises(DatasetGenerationError, match="does not produce boolean"):
        DatasetBuilder(data_designer_config=config.build(), resource_provider=stub_resource_provider)


def test_record_selection_uses_fixed_width_for_all_candidate_artifacts(stub_resource_provider) -> None:
    stub_resource_provider.run_config = RunConfig(
        buffer_size=1,
        display_tui=False,
        preserve_dropped_columns=True,
    )
    config = DataDesignerConfigBuilder()
    config.add_column(
        SamplerColumnConfig(
            name="keep",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=[True]),
        )
    )
    config.add_column(
        SamplerColumnConfig(
            name="payload",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["accepted"]),
        )
    )
    config.add_processor(
        SchemaTransformProcessorConfig(
            name="formatted",
            template={"value": "{{ payload }}"},
        )
    )
    config.add_processor(
        DropColumnsProcessorConfig(
            name="drop_payload",
            column_names=["payload"],
        )
    )
    config.with_record_selection(RecordSelectionConfig(predicate_column="keep", max_candidate_records=100_001))
    builder = DatasetBuilder(data_designer_config=config.build(), resource_provider=stub_resource_provider)

    builder.build(num_records=1)

    storage = builder.artifact_storage
    expected_batch_name = "batch_000000.parquet"
    assert [path.name for path in storage.selection_accepted_path.glob("batch_*.parquet")] == [expected_batch_name]
    assert [path.name for path in storage.selection_checkpoints_path.glob("*.json")] == ["batch_000000.json"]
    assert [path.name for path in storage.dropped_columns_dataset_path.glob("*.parquet")] == [expected_batch_name]
    assert [path.name for path in (storage.processors_outputs_path / "formatted").glob("*.parquet")] == [
        expected_batch_name
    ]

    metadata = storage.read_metadata()
    assert metadata["file_paths"]["processor-files"]["formatted"] == [
        f"processors-files/formatted/{expected_batch_name}"
    ]
    assert storage.load_processor_dataset("formatted")["value"].tolist() == ["accepted"]
    assert storage.load_dataset_with_dropped_columns()["payload"].tolist() == ["accepted"]


def test_record_selection_after_generation_uses_publication_width(stub_resource_provider) -> None:
    stub_resource_provider.run_config = RunConfig(buffer_size=1, display_tui=False)
    config = DataDesignerConfigBuilder()
    config.add_column(
        SamplerColumnConfig(
            name="keep",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=[True]),
        )
    )
    config.with_record_selection(RecordSelectionConfig(predicate_column="keep", max_candidate_records=100_001))
    builder = DatasetBuilder(data_designer_config=config.build(), resource_provider=stub_resource_provider)
    processor = create_mock_processor("identity", ["process_after_generation"])
    _replace_processors(builder, [processor])

    output_path = builder.build(num_records=1)

    storage = builder.artifact_storage
    processor.process_after_generation.assert_called_once()
    assert [path.name for path in storage.selection_accepted_path.glob("batch_*.parquet")] == ["batch_000000.parquet"]
    assert [path.name for path in output_path.glob("*.parquet")] == ["batch_00000.parquet"]
    assert storage.read_metadata()["file_paths"]["parquet-files"] == ["parquet-files/batch_00000.parquet"]


@pytest.mark.parametrize(
    ("stage", "message"),
    [
        ("process_after_batch", "Post-batch processors must retain at least one column"),
        ("process_after_generation", "After-generation processors must retain at least one column"),
    ],
)
def test_record_selection_rejects_nonempty_zero_column_processor_output(
    stub_resource_provider,
    stage: str,
    message: str,
) -> None:
    stub_resource_provider.run_config = RunConfig(buffer_size=1, display_tui=False)
    config = DataDesignerConfigBuilder()
    config.add_column(
        SamplerColumnConfig(
            name="keep",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=[True]),
        )
    )
    config.with_record_selection(RecordSelectionConfig(predicate_column="keep", max_candidate_records=1))
    builder = DatasetBuilder(data_designer_config=config.build(), resource_provider=stub_resource_provider)
    processor = create_mock_processor("drop_all_columns", [stage])
    getattr(processor, stage).side_effect = lambda dataframe, **_kwargs: dataframe.drop(columns=dataframe.columns)
    _replace_processors(builder, [processor])

    with pytest.raises((DatasetGenerationError, DatasetProcessingError), match=message):
        builder.build(num_records=1)


def test_record_selection_refreshes_file_manifest_only_for_publication(stub_resource_provider) -> None:
    stub_resource_provider.run_config = RunConfig(buffer_size=1, display_tui=False)
    config = DataDesignerConfigBuilder()
    config.add_column(
        SamplerColumnConfig(
            name="keep",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=[True]),
        )
    )
    config.add_column(
        SamplerColumnConfig(
            name="payload",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["accepted"]),
        )
    )
    config.add_processor(
        SchemaTransformProcessorConfig(
            name="formatted",
            template={"value": "{{ payload }}"},
        )
    )
    config.with_record_selection(RecordSelectionConfig(predicate_column="keep", max_candidate_records=5))
    builder = DatasetBuilder(data_designer_config=config.build(), resource_provider=stub_resource_provider)

    original_get_file_paths = ArtifactStorage.get_file_paths
    get_file_paths_calls = 0

    def count_get_file_paths(storage: ArtifactStorage) -> dict[str, list[str] | dict[str, list[str]]]:
        nonlocal get_file_paths_calls
        get_file_paths_calls += 1
        return original_get_file_paths(storage)

    with patch.object(ArtifactStorage, "get_file_paths", count_get_file_paths):
        builder.build(num_records=5)

    assert get_file_paths_calls == 1


def test_record_selection_rejects_category_with_non_boolean_conditional_values(stub_resource_provider) -> None:
    config = DataDesignerConfigBuilder()
    config.add_column(
        SamplerColumnConfig(
            name="mixed_predicate",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=[True, False]),
            conditional_params={"true": CategorySamplerParams(values=[0, 1])},
        )
    )
    config.with_record_selection(RecordSelectionConfig(predicate_column="mixed_predicate", max_candidate_records=1))

    with pytest.raises(DatasetGenerationError, match="does not produce boolean"):
        DatasetBuilder(data_designer_config=config.build(), resource_provider=stub_resource_provider)


def test_record_selection_defers_custom_predicate_type_validation_to_runtime(stub_resource_provider) -> None:
    @custom_column_generator()
    def make_predicate(row: dict) -> bool:
        return bool(row)

    config = DataDesignerConfigBuilder()
    config.add_column(CustomColumnConfig(name="custom_predicate", generator_function=make_predicate))
    config.with_record_selection(RecordSelectionConfig(predicate_column="custom_predicate", max_candidate_records=1))

    DatasetBuilder(data_designer_config=config.build(), resource_provider=stub_resource_provider)


def test_record_selection_persists_early_shutdown_across_exhausted_resume(stub_resource_provider) -> None:
    stub_resource_provider.run_config = RunConfig(
        buffer_size=1,
        shutdown_error_rate=0.5,
        shutdown_error_window=1,
        display_tui=False,
    )
    config = DataDesignerConfigBuilder()
    config.add_column(
        SamplerColumnConfig(
            name="value",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["constant"]),
        )
    )
    config.add_column(ExpressionColumnConfig(name="keep", expr="{{ true }}", dtype="bool", drop=True))
    config.with_record_selection(
        RecordSelectionConfig(
            predicate_column="keep",
            max_candidate_records=1,
            on_exhausted=RecordSelectionExhaustion.RETURN_PARTIAL,
        )
    )
    builder = DatasetBuilder(data_designer_config=config.build(), resource_provider=stub_resource_provider)

    with (
        patch.object(
            SamplerColumnGenerator,
            "generate_from_scratch",
            side_effect=DataDesignerRuntimeError("candidate generation failed"),
        ),
        pytest.raises(RecordSelectionEarlyShutdownError) as exc_info,
    ):
        builder.build(num_records=1)

    assert "resume=ResumeMode.ALWAYS" not in str(exc_info.value)
    terminal_error = builder.artifact_storage.read_metadata()["record_selection"]["terminal_error"]
    assert terminal_error["kind"] == "early_shutdown"
    marker = json.loads(builder.artifact_storage.selection_checkpoint_path(0).read_text())
    assert marker["terminal_error_kind"] == "early_shutdown"

    # Simulate the crash window where the committed marker reaches disk before
    # the best-effort metadata mirror. Resume must still replay the typed error.
    metadata = builder.artifact_storage.read_metadata()
    metadata["record_selection"].pop("terminal_error")
    builder.artifact_storage.write_metadata(metadata)

    with pytest.raises(RecordSelectionEarlyShutdownError):
        builder.build(num_records=1, resume=ResumeMode.ALWAYS)


def test_record_selection_persists_fatal_post_checkpoint_error_across_resume(stub_resource_provider) -> None:
    stub_resource_provider.run_config = RunConfig(buffer_size=1, display_tui=False)
    config = DataDesignerConfigBuilder()
    config.add_column(
        SamplerColumnConfig(
            name="value",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["constant"]),
        )
    )
    config.add_column(ExpressionColumnConfig(name="keep", expr="{{ false }}", dtype="bool", drop=True))
    config.with_record_selection(
        RecordSelectionConfig(
            predicate_column="keep",
            max_candidate_records=2,
            on_exhausted=RecordSelectionExhaustion.RETURN_PARTIAL,
        )
    )
    builder = DatasetBuilder(data_designer_config=config.build(), resource_provider=stub_resource_provider)

    def commit_then_fail(
        _generators: list[object],
        *,
        controller: AcceptanceController,
        batch: CandidateBatch,
        **_kwargs: object,
    ) -> None:
        builder.artifact_storage.write_selection_schema(builder._derive_empty_selection_schema())
        marker = controller.record_checkpoint(
            batch=batch,
            decision=SelectionDecision(
                accepted_indices=(),
                candidate_records=1,
                accepted_records=0,
                rejected_records=1,
                null_predicate_records=0,
                failed_generation_records=0,
                trimmed_accepted_records=0,
            ),
            accepted_partition=None,
        )
        builder.artifact_storage.write_selection_checkpoint(batch.candidate_batch_id, marker.to_dict())
        raise DatasetGenerationError("fatal generation error after checkpoint")

    with (
        patch.object(builder, "_run_candidate_batch", side_effect=commit_then_fail),
        pytest.raises(DatasetGenerationError, match="fatal generation error after checkpoint"),
    ):
        builder.build(num_records=1)

    marker = json.loads(builder.artifact_storage.selection_checkpoint_path(0).read_text())
    assert marker["terminal_error_kind"] == "generation_error"
    metadata = builder.artifact_storage.read_metadata()
    metadata["record_selection"].pop("terminal_error")
    builder.artifact_storage.write_metadata(metadata)

    with pytest.raises(DatasetGenerationError, match="fatal generation error after checkpoint"):
        builder.build(num_records=1, resume=ResumeMode.ALWAYS)


def test_record_selection_post_checkpoint_metadata_failure_is_resumable(stub_resource_provider) -> None:
    stub_resource_provider.run_config = RunConfig(buffer_size=1, display_tui=False)
    config = DataDesignerConfigBuilder()
    config.add_column(
        SamplerColumnConfig(
            name="keep",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=[True]),
        )
    )
    config.with_record_selection(RecordSelectionConfig(predicate_column="keep", max_candidate_records=1))
    builder = DatasetBuilder(data_designer_config=config.build(), resource_provider=stub_resource_provider)
    storage = builder.artifact_storage
    original_update_metadata = ArtifactStorage.update_metadata
    original_write_selection_schema = ArtifactStorage.write_selection_schema
    failed_once = False
    schema_write_count = 0

    def fail_first_post_checkpoint_metadata_write(self: ArtifactStorage, updates: dict[str, Any]) -> Path:
        nonlocal failed_once
        if storage.selection_checkpoint_path(0).is_file() and not failed_once:
            failed_once = True
            raise OSError("transient metadata failure")
        return original_update_metadata(self, updates)

    def record_schema_write(self: ArtifactStorage, dataframe: pd.DataFrame) -> Path:
        nonlocal schema_write_count
        schema_write_count += 1
        return original_write_selection_schema(self, dataframe)

    with (
        patch.object(
            ArtifactStorage,
            "update_metadata",
            autospec=True,
            side_effect=fail_first_post_checkpoint_metadata_write,
        ),
        patch.object(
            ArtifactStorage,
            "write_selection_schema",
            autospec=True,
            side_effect=record_schema_write,
        ),
    ):
        with pytest.raises(DatasetGenerationError, match="Failed after committing.*transient metadata failure"):
            builder.build(num_records=1)

        marker = json.loads(storage.selection_checkpoint_path(0).read_text())
        assert marker["terminal_error_kind"] is None
        assert marker["schema_materialized"] is True

        output_path = builder.build(num_records=1, resume=ResumeMode.ALWAYS)

    assert failed_once
    assert schema_write_count == 1
    assert lazy.pd.read_parquet(output_path)["keep"].tolist() == [True]
    assert storage.read_metadata()["post_generation_state"] == "complete"


def test_record_selection_callback_precedes_simultaneous_early_shutdown(stub_resource_provider) -> None:
    stub_resource_provider.run_config = RunConfig(buffer_size=2, display_tui=False)
    config = DataDesignerConfigBuilder()
    config.add_column(
        SamplerColumnConfig(
            name="value",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["constant"]),
        )
    )
    config.add_column(ExpressionColumnConfig(name="keep", expr="{{ true }}", dtype="bool"))
    selection_config = RecordSelectionConfig(predicate_column="keep", max_candidate_records=4)
    config.with_record_selection(selection_config)
    builder = DatasetBuilder(data_designer_config=config.build(), resource_provider=stub_resource_provider)
    buffer_manager = RowGroupBufferManager(builder.artifact_storage)
    buffer_manager.init_row_group(0, 2)

    class StubScheduler:
        traces: list[Any] = []
        early_shutdown = True
        partial_row_groups: tuple[int, ...] = ()
        first_non_retryable_error: Exception | None = None

        def __init__(
            self,
            *,
            select_dataframe: Callable[[int, int, pd.DataFrame], pd.DataFrame],
            on_finalize_row_group: Callable[[int], None],
        ) -> None:
            self._select_dataframe = select_dataframe
            self._on_finalize_row_group = on_finalize_row_group

        async def run(self) -> None:
            dataframe = lazy.pd.DataFrame(
                {
                    "value": ["accepted", "rejected"],
                    "keep": [True, False],
                }
            )
            selected = self._select_dataframe(0, 2, dataframe)
            buffer_manager.replace_dataframe(0, selected)
            self._on_finalize_row_group(0)

    def prepare_async_run(*_args: Any, **kwargs: Any) -> tuple[StubScheduler, RowGroupBufferManager]:
        return (
            StubScheduler(
                select_dataframe=kwargs["select_dataframe"],
                on_finalize_row_group=kwargs["on_finalize_row_group"],
            ),
            buffer_manager,
        )

    def fail_callback(_path: Path) -> None:
        raise RuntimeError("callback sentinel")

    with patch.object(builder, "_prepare_async_run", side_effect=prepare_async_run):
        with pytest.raises(DatasetGenerationError, match="callback sentinel") as exc_info:
            builder._build_with_record_selection(
                [],
                target_num_records=2,
                buffer_size=2,
                on_batch_complete=fail_callback,
                resume=ResumeMode.NEVER,
            )

    assert not isinstance(exc_info.value, RecordSelectionEarlyShutdownError)
    marker = SelectionBatchMarker.from_dict(builder.artifact_storage.read_selection_checkpoints()[0])
    assert marker.terminal_error_kind == "early_shutdown"
    assert "resume=ResumeMode.ALWAYS" in marker.terminal_error_message
    resumed_controller = AcceptanceController(
        config=selection_config,
        target_records=2,
        buffer_size=2,
        markers=(marker,),
    )
    builder._raise_unrecoverable_selection_terminal_error(resumed_controller, resumed_controller.terminal_error)


def test_record_selection_newer_checkpoint_clears_stale_terminal_metadata(stub_resource_provider) -> None:
    stub_resource_provider.run_config = RunConfig(buffer_size=1, display_tui=False)
    config = DataDesignerConfigBuilder()
    config.add_column(
        SamplerColumnConfig(
            name="value",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["constant"]),
        )
    )
    config.add_column(ExpressionColumnConfig(name="keep", expr="{{ false }}", dtype="bool", drop=True))
    config.with_record_selection(
        RecordSelectionConfig(
            predicate_column="keep",
            max_candidate_records=2,
            on_exhausted=RecordSelectionExhaustion.RETURN_PARTIAL,
        )
    )
    builder = DatasetBuilder(data_designer_config=config.build(), resource_provider=stub_resource_provider)
    markers = (
        SelectionBatchMarker(
            candidate_batch_id=0,
            row_group_id=0,
            candidate_start_offset=0,
            candidate_records=1,
            accepted_records=0,
            rejected_records=1,
            null_predicate_records=0,
            failed_generation_records=0,
            trimmed_accepted_records=0,
            accepted_partition=None,
            terminal_error_kind="early_shutdown",
            terminal_error_message="stale early shutdown",
        ),
        SelectionBatchMarker(
            candidate_batch_id=1,
            row_group_id=1,
            candidate_start_offset=1,
            candidate_records=1,
            accepted_records=0,
            rejected_records=1,
            null_predicate_records=0,
            failed_generation_records=0,
            trimmed_accepted_records=0,
            accepted_partition=None,
        ),
    )
    for marker in markers:
        builder.artifact_storage.write_selection_checkpoint(marker.candidate_batch_id, marker.to_dict())
    builder.artifact_storage.write_selection_schema(builder._derive_empty_selection_schema())
    builder.artifact_storage.write_metadata(
        {
            "target_num_records": 1,
            "original_target_num_records": 1,
            "actual_num_records": 0,
            "buffer_size": 1,
            "record_selection": {
                "candidate_batches_completed": 1,
                "run_buffer_size": 1,
                "terminal_error": {
                    "kind": "early_shutdown",
                    "message": "stale early shutdown",
                },
            },
        }
    )

    builder._build_with_record_selection(
        [],
        target_num_records=1,
        buffer_size=1,
        on_batch_complete=None,
        resume=ResumeMode.ALWAYS,
    )

    assert lazy.pd.read_parquet(builder.artifact_storage.final_dataset_path).empty
    assert "terminal_error" not in builder.artifact_storage.read_metadata()["record_selection"]


def test_record_selection_replaces_schema_from_completely_failed_first_batch(stub_resource_provider) -> None:
    stub_resource_provider.run_config = RunConfig(buffer_size=1, disable_early_shutdown=True, display_tui=False)
    config = DataDesignerConfigBuilder()
    config.add_column(
        SamplerColumnConfig(
            name="value",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["constant"]),
        )
    )
    config.add_column(ExpressionColumnConfig(name="keep", expr="{{ value == 'never' }}", dtype="bool", drop=True))
    config.with_record_selection(
        RecordSelectionConfig(
            predicate_column="keep",
            max_candidate_records=3,
            on_exhausted=RecordSelectionExhaustion.RETURN_PARTIAL,
        )
    )
    builder = DatasetBuilder(data_designer_config=config.build(), resource_provider=stub_resource_provider)
    original_write_selection_schema = ArtifactStorage.write_selection_schema
    schema_write_count = 0

    def record_schema_write(self: ArtifactStorage, dataframe: pd.DataFrame) -> Path:
        nonlocal schema_write_count
        schema_write_count += 1
        return original_write_selection_schema(self, dataframe)

    with (
        patch.object(
            ArtifactStorage,
            "write_selection_schema",
            autospec=True,
            side_effect=record_schema_write,
        ),
        patch.object(
            SamplerColumnGenerator,
            "generate_from_scratch",
            side_effect=[
                DataDesignerRuntimeError("first candidate batch failed"),
                lazy.pd.DataFrame({"value": ["constant"]}),
                DataDesignerRuntimeError("final candidate batch failed"),
            ],
        ),
        pytest.raises(DatasetGenerationError, match="first candidate batch failed"),
    ):
        builder.build(num_records=1)

    schema = lazy.pd.read_parquet(builder.artifact_storage.selection_schema_path)
    assert schema.empty
    assert schema.columns.tolist() == ["value"]
    assert schema_write_count == 2
    assert [marker["schema_materialized"] for marker in builder.artifact_storage.read_selection_checkpoints()] == [
        False,
        True,
        True,
    ]


def test_record_selection_publication_id_is_stable_per_attempt_and_changes_on_republish(
    stub_resource_provider,
) -> None:
    stub_resource_provider.run_config = RunConfig(buffer_size=1, display_tui=False)
    config = DataDesignerConfigBuilder()
    config.add_column(
        SamplerColumnConfig(
            name="keep",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=[True]),
        )
    )
    selection_config = RecordSelectionConfig(predicate_column="keep", max_candidate_records=1)
    config.with_record_selection(selection_config)
    builder = DatasetBuilder(data_designer_config=config.build(), resource_provider=stub_resource_provider)
    storage = builder.artifact_storage
    original_update_metadata = ArtifactStorage.update_metadata
    publication_events: list[tuple[str, str]] = []

    def record_publication_metadata(self: ArtifactStorage, updates: dict[str, Any]) -> Path:
        path = original_update_metadata(self, updates)
        state = updates.get("post_generation_state")
        if isinstance(state, str):
            publication_id = storage.read_metadata()["record_selection"]["publication_id"]
            publication_events.append((state, publication_id))
        return path

    with patch.object(
        ArtifactStorage,
        "update_metadata",
        autospec=True,
        side_effect=record_publication_metadata,
    ):
        builder.build(num_records=1)
        controller = AcceptanceController(
            config=selection_config,
            target_records=1,
            buffer_size=1,
            markers=builder._load_selection_markers(),
        )
        builder._publish_selection_result(controller, buffer_size=1)

    assert [state for state, _publication_id in publication_events] == [
        "pending",
        "started",
        "complete",
        "pending",
        "started",
        "complete",
    ]
    first_attempt_ids = {publication_id for _state, publication_id in publication_events[:3]}
    second_attempt_ids = {publication_id for _state, publication_id in publication_events[3:]}
    assert len(first_attempt_ids) == 1
    assert len(second_attempt_ids) == 1
    assert first_attempt_ids != second_attempt_ids


def test_record_selection_completed_legacy_artifact_migration_recovers_without_reprocessing(
    stub_resource_provider,
) -> None:
    stub_resource_provider.run_config = RunConfig(
        buffer_size=1,
        display_tui=False,
        preserve_dropped_columns=True,
    )
    config = DataDesignerConfigBuilder()
    config.add_column(
        SamplerColumnConfig(
            name="keep",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=[True]),
        )
    )
    config.add_column(
        SamplerColumnConfig(
            name="payload",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["accepted"]),
            drop=True,
        )
    )
    config.add_processor(
        SchemaTransformProcessorConfig(
            name="formatted",
            template={"value": "{{ payload }}"},
        )
    )
    config.with_record_selection(RecordSelectionConfig(predicate_column="keep", max_candidate_records=100_001))
    builder = DatasetBuilder(data_designer_config=config.build(), resource_provider=stub_resource_provider)

    output_path = builder.build(num_records=1)
    storage = builder.artifact_storage
    metadata = storage.read_metadata()
    previous_publication_id = metadata["record_selection"]["publication_id"]

    accepted = storage.selection_accepted_path / "batch_000000.parquet"
    accepted.rename(accepted.with_name("batch_00000.parquet"))
    checkpoint = storage.selection_checkpoints_path / "batch_000000.json"
    legacy_checkpoint = checkpoint.with_name("batch_00000.json")
    checkpoint.rename(legacy_checkpoint)
    checkpoint_payload = json.loads(legacy_checkpoint.read_text(encoding="utf-8"))
    checkpoint_payload["accepted_partition"] = "selection-accepted/batch_00000.parquet"
    legacy_checkpoint.write_text(json.dumps(checkpoint_payload), encoding="utf-8")
    dropped = storage.dropped_columns_dataset_path / "batch_000000.parquet"
    dropped.rename(dropped.with_name("batch_00000.parquet"))
    processor = storage.processors_outputs_path / "formatted" / "batch_000000.parquet"
    processor.rename(processor.with_name("batch_00000.parquet"))
    metadata["file_paths"]["processor-files"]["formatted"] = ["processors-files/formatted/batch_00000.parquet"]
    storage.write_metadata(metadata)

    with (
        patch.object(builder, "_load_selection_markers", side_effect=RuntimeError("simulated post-migration crash")),
        pytest.raises(RuntimeError, match="simulated post-migration crash"),
    ):
        builder.build(num_records=1, resume=ResumeMode.ALWAYS)

    interrupted_metadata = storage.read_metadata()
    assert interrupted_metadata["post_generation_state"] == "started"
    assert interrupted_metadata["record_selection_artifact_migration"] == {
        "previously_complete": True,
        "side_artifacts_deferred": False,
    }
    assert not storage.selection_artifact_migration_path.exists()
    assert storage.selection_accepted_path.joinpath("batch_000000.parquet").is_file()
    assert storage.selection_checkpoints_path.joinpath("batch_000000.json").is_file()

    with patch.object(builder._processor_runner, "run_after_generation") as run_after_generation:
        resumed_output = builder.build(num_records=1, resume=ResumeMode.ALWAYS)

    refreshed_metadata = storage.read_metadata()
    assert resumed_output == output_path
    run_after_generation.assert_not_called()
    assert refreshed_metadata["post_generation_state"] == "complete"
    assert "record_selection_artifact_migration" not in refreshed_metadata
    assert refreshed_metadata["record_selection"]["publication_id"] != previous_publication_id
    assert refreshed_metadata["file_paths"]["processor-files"]["formatted"] == [
        "processors-files/formatted/batch_000000.parquet"
    ]


def test_record_selection_completed_migration_keeps_unprocessed_boundary_artifacts_aligned(
    stub_resource_provider: Any,
) -> None:
    builder, _after_generation_processor = _create_boundary_selection_builder(
        stub_resource_provider,
        after_generation=False,
    )
    storage = builder.artifact_storage
    previous_publication_id = _rewrite_completed_selection_at_filename_boundary(
        builder,
        after_generation=False,
    )

    output_path = builder.build(num_records=2, resume=ResumeMode.ALWAYS)

    metadata = storage.read_metadata()
    assert output_path == storage.final_dataset_path
    assert sorted(path.name for path in storage.final_dataset_path.glob("batch_*.parquet")) == [
        "batch_00000.parquet",
        "batch_00001.parquet",
    ]
    assert sorted(path.name for path in storage.selection_accepted_path.glob("batch_*.parquet")) == [
        "batch_000000.parquet",
        "batch_000001.parquet",
    ]
    assert sorted(path.name for path in storage.dropped_columns_dataset_path.glob("batch_*.parquet")) == [
        "batch_099999.parquet",
        "batch_100000.parquet",
    ]
    assert storage.load_dataset_with_dropped_columns().to_dict(orient="records") == [
        {"value": 99_999, "dropped": 99_999},
        {"value": 100_000, "dropped": 100_000},
    ]
    assert storage.load_processor_dataset("formatted")["value"].tolist() == [99_999, 100_000]
    assert metadata["post_generation_state"] == "complete"
    assert metadata["record_selection"]["publication_id"] != previous_publication_id
    assert "record_selection_artifact_migration" not in metadata


def test_record_selection_completed_migration_defers_processed_side_artifacts_without_reprocessing(
    stub_resource_provider: Any,
) -> None:
    builder, after_generation_processor = _create_boundary_selection_builder(
        stub_resource_provider,
        after_generation=True,
    )
    assert after_generation_processor is not None
    storage = builder.artifact_storage
    previous_publication_id = _rewrite_completed_selection_at_filename_boundary(
        builder,
        after_generation=True,
    )

    with (
        patch.object(builder, "_load_selection_markers", side_effect=RuntimeError("simulated post-migration crash")),
        pytest.raises(RuntimeError, match="simulated post-migration crash"),
    ):
        builder.build(num_records=2, resume=ResumeMode.ALWAYS)

    interrupted_metadata = storage.read_metadata()
    assert interrupted_metadata["record_selection_artifact_migration"] == {
        "previously_complete": True,
        "side_artifacts_deferred": True,
    }
    assert sorted(path.name for path in storage.selection_accepted_path.glob("batch_*.parquet")) == [
        "batch_000000.parquet",
        "batch_000001.parquet",
    ]
    assert sorted(path.name for path in storage.dropped_columns_dataset_path.glob("batch_*.parquet")) == [
        "batch_100000.parquet",
        "batch_99999.parquet",
    ]

    output_path = builder.build(num_records=2, resume=ResumeMode.ALWAYS)

    metadata = storage.read_metadata()
    assert output_path == storage.final_dataset_path
    after_generation_processor.process_after_generation.assert_not_called()
    assert storage.load_dataset_with_dropped_columns().to_dict(orient="records") == [
        {"value": 100_000, "dropped": 100_000},
        {"value": 99_999, "dropped": 99_999},
    ]
    assert storage.load_processor_dataset("formatted")["value"].tolist() == [100_000, 99_999]
    assert metadata["record_selection"]["publication_id"] != previous_publication_id
    assert "record_selection_artifact_migration" not in metadata

    current_publication_id = metadata["record_selection"]["publication_id"]
    builder.build(num_records=2, resume=ResumeMode.ALWAYS)

    after_generation_processor.process_after_generation.assert_not_called()
    assert storage.read_metadata()["record_selection"]["publication_id"] == current_publication_id
    assert sorted(path.name for path in storage.dropped_columns_dataset_path.glob("batch_*.parquet")) == [
        "batch_100000.parquet",
        "batch_99999.parquet",
    ]


def test_record_selection_failed_processed_reuse_persists_full_migration_before_republication(
    stub_resource_provider: Any,
) -> None:
    builder, after_generation_processor = _create_boundary_selection_builder(
        stub_resource_provider,
        after_generation=True,
    )
    assert after_generation_processor is not None
    storage = builder.artifact_storage
    _rewrite_completed_selection_at_filename_boundary(
        builder,
        after_generation=True,
        valid_publication=False,
    )
    original_normalize = storage.normalize_selection_candidate_artifact_width
    normalization_scopes: list[bool] = []

    def interrupt_deferred_migration(
        _storage: ArtifactStorage,
        *,
        include_side_artifacts: bool = True,
    ) -> bool:
        normalization_scopes.append(include_side_artifacts)
        if include_side_artifacts:
            raise OSError("simulated deferred migration interruption")
        return original_normalize(include_side_artifacts=include_side_artifacts)

    with (
        patch.object(
            ArtifactStorage,
            "normalize_selection_candidate_artifact_width",
            autospec=True,
            side_effect=interrupt_deferred_migration,
        ),
        pytest.raises(OSError, match="simulated deferred migration interruption"),
    ):
        builder.build(num_records=2, resume=ResumeMode.ALWAYS)

    interrupted_metadata = storage.read_metadata()
    assert normalization_scopes == [False, True]
    assert interrupted_metadata["record_selection_artifact_migration"] == {
        "previously_complete": True,
        "side_artifacts_deferred": False,
    }
    assert sorted(path.name for path in storage.dropped_columns_dataset_path.glob("batch_*.parquet")) == [
        "batch_100000.parquet",
        "batch_99999.parquet",
    ]
    after_generation_processor.process_after_generation.assert_not_called()

    output_path = builder.build(num_records=2, resume=ResumeMode.ALWAYS)

    metadata = storage.read_metadata()
    assert output_path == storage.final_dataset_path
    after_generation_processor.process_after_generation.assert_called_once()
    assert sorted(path.name for path in storage.dropped_columns_dataset_path.glob("batch_*.parquet")) == [
        "batch_099999.parquet",
        "batch_100000.parquet",
    ]
    assert storage.load_dataset_with_dropped_columns().to_dict(orient="records") == [
        {"value": 99_999, "dropped": 99_999},
        {"value": 100_000, "dropped": 100_000},
    ]
    assert storage.load_processor_dataset("formatted")["value"].tolist() == [99_999, 100_000]
    assert metadata["post_generation_state"] == "complete"
    assert "record_selection_artifact_migration" not in metadata


@pytest.mark.parametrize("stored_publication_id", [None, "", "   "])
def test_record_selection_resume_migrates_completed_publication_without_valid_id(
    stub_resource_provider,
    stored_publication_id: str | None,
) -> None:
    stub_resource_provider.run_config = RunConfig(buffer_size=1, display_tui=False)
    config = DataDesignerConfigBuilder()
    config.add_column(
        SamplerColumnConfig(
            name="keep",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=[True]),
        )
    )
    config.with_record_selection(RecordSelectionConfig(predicate_column="keep", max_candidate_records=1))
    builder = DatasetBuilder(data_designer_config=config.build(), resource_provider=stub_resource_provider)

    output_path = builder.build(num_records=1)
    metadata = builder.artifact_storage.read_metadata()
    previous_publication_id = metadata["record_selection"]["publication_id"]
    if stored_publication_id is None:
        metadata["record_selection"].pop("publication_id")
    else:
        metadata["record_selection"]["publication_id"] = stored_publication_id
    builder.artifact_storage.write_metadata(metadata)

    resumed_output_path = builder.build(num_records=1, resume=ResumeMode.ALWAYS)

    migrated_publication_id = builder.artifact_storage.read_metadata()["record_selection"]["publication_id"]
    assert resumed_output_path == output_path
    assert isinstance(migrated_publication_id, str)
    assert migrated_publication_id
    assert migrated_publication_id != previous_publication_id


def test_record_selection_normalizes_corrupt_accepted_partition_error(stub_resource_provider) -> None:
    config = DataDesignerConfigBuilder()
    config.add_column(
        SamplerColumnConfig(
            name="value",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["constant"]),
        )
    )
    config.add_column(ExpressionColumnConfig(name="keep", expr="{{ true }}", dtype="bool", drop=True))
    config.with_record_selection(RecordSelectionConfig(predicate_column="keep", max_candidate_records=1))
    builder = DatasetBuilder(data_designer_config=config.build(), resource_provider=stub_resource_provider)
    partition = builder.artifact_storage.selection_partition_path(0)
    partition.parent.mkdir(parents=True)
    partition.write_bytes(b"not parquet")
    marker = SelectionBatchMarker(
        candidate_batch_id=0,
        row_group_id=0,
        candidate_start_offset=0,
        candidate_records=1,
        accepted_records=1,
        rejected_records=0,
        null_predicate_records=0,
        failed_generation_records=0,
        trimmed_accepted_records=0,
        accepted_partition=str(partition.relative_to(builder.artifact_storage.base_dataset_path)),
    )

    with pytest.raises(DatasetGenerationError, match="missing or unreadable accepted partition"):
        builder._validate_selection_partitions((marker,))


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


def test_dataset_builder_combines_scheduler_event_sinks(
    stub_resource_provider,
    stub_test_config_builder,
) -> None:
    provider_sink = InMemoryAdmissionEventSink()
    local_sink = InMemoryAdmissionEventSink()
    stub_resource_provider.scheduler_event_sink = provider_sink
    builder = DatasetBuilder(
        data_designer_config=stub_test_config_builder.build(),
        resource_provider=stub_resource_provider,
    )
    generators, _graph = builder._initialize_generators_and_graph()
    for generator in generators:
        generator.log_pre_generation = Mock()

    with patch.object(builder_mod, "AsyncTaskScheduler", return_value=Mock()) as scheduler_factory:
        builder._prepare_async_run(generators, num_records=1, buffer_size=1, scheduler_event_sink=local_sink)

    sink = scheduler_factory.call_args.kwargs["scheduler_event_sink"]
    assert sink is not None
    event = SchedulerAdmissionEvent.capture("scheduler_job_started", sequence=1)
    sink.emit_scheduler_event(event)

    assert local_sink.scheduler_events == [event]
    assert provider_sink.scheduler_events == [event]


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
    assert run_config.max_in_flight_tasks == 1024
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


def test_full_column_custom_generator_failure_sets_first_error(stub_resource_provider, stub_model_configs):
    @custom_column_generator(required_columns=["some_id"])
    def bad_fn(df: pd.DataFrame) -> pd.DataFrame:
        raise ValueError("something broke")

    config = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config.add_column(SamplerColumnConfig(name="some_id", sampler_type=SamplerType.UUID, params=UUIDSamplerParams()))
    config.add_column(CustomColumnConfig(name="col", generator_function=bad_fn, generation_strategy="full_column"))
    builder = DatasetBuilder(data_designer_config=config.build(), resource_provider=stub_resource_provider)

    result = builder.build_preview(num_records=3)

    assert result.empty
    assert builder.first_non_retryable_error is not None
    assert "Custom generator function failed for column 'col': something broke" in str(
        builder.first_non_retryable_error
    )


def test_expression_column_row_drops_shrink_sync_batch(
    stub_resource_provider: Mock,
    stub_model_configs: dict[str, object],
    caplog: pytest.LogCaptureFixture,
) -> None:
    seed_source = DataFrameSeedSource(df=lazy.pd.DataFrame({"seed_id": [1, 2, 3, 4], "text": ["a", "", "c", "d"]}))
    seed_reader = DataFrameSeedReader()
    seed_reader.attach(seed_source, Mock())
    stub_resource_provider.seed_reader = seed_reader

    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.with_seed_dataset(seed_source)
    config_builder.add_column(ExpressionColumnConfig(name="copy", expr="{{ text }}"))
    builder = DatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )

    with caplog.at_level(logging.WARNING):
        result = builder.build_preview(num_records=4)

    assert result["seed_id"].tolist() == [1, 3, 4]
    assert result["copy"].tolist() == ["a", "c", "d"]
    assert "Expression column 'copy' dropped 1/4 rows after render: EmptyRenderedExpression=1." in caplog.text


def test_expression_column_row_drops_shrink_sync_skip_aware_batch(
    stub_resource_provider: Mock,
    stub_model_configs: dict[str, object],
) -> None:
    seed_source = DataFrameSeedSource(df=lazy.pd.DataFrame({"seed_id": [1, 2, 3], "text": ["skip-me", "", "keep"]}))
    seed_reader = DataFrameSeedReader()
    seed_reader.attach(seed_source, Mock())
    stub_resource_provider.seed_reader = seed_reader

    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.with_seed_dataset(seed_source)
    config_builder.add_column(
        ExpressionColumnConfig(
            name="copy",
            expr="{{ text }}",
            skip=SkipConfig(when="{{ seed_id == 1 }}", value="skipped"),
        )
    )
    builder = DatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )

    result = builder.build_preview(num_records=3)

    assert result["seed_id"].tolist() == [1, 3]
    assert result["copy"].tolist() == ["skipped", "keep"]


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


def test_await_async_scheduler_result_waits_for_scheduler_cleanup_on_keyboard_interrupt() -> None:
    class MockFuture:
        def __init__(self) -> None:
            self.result_calls = 0

        def result(self) -> None:
            self.result_calls += 1
            if self.result_calls == 1:
                raise KeyboardInterrupt
            assert scheduler.request_cancel.called
            raise concurrent.futures.CancelledError

    scheduler = Mock()
    future = MockFuture()

    with pytest.raises(KeyboardInterrupt):
        builder_mod._await_async_scheduler_result(future, scheduler)

    scheduler.request_cancel.assert_called_once_with()
    assert future.result_calls == 2


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

    _replace_processors(simple_builder, [mock_processor])
    simple_builder._processor_runner.run_after_generation(batch_size)

    mock_processor.process_after_generation.assert_called_once()
    batch_files = sorted(storage.final_dataset_path.glob("*.parquet"))
    assert len(batch_files) == expected_files
    assert sum(len(lazy.pd.read_parquet(f)) for f in batch_files) == expected_rows


def test_run_after_generation_selection_publication_names_use_output_partition_count(
    stub_resource_provider,
    simple_builder,
) -> None:
    storage = stub_resource_provider.artifact_storage
    storage.configure_selection_batch_file_width(max_candidate_records=100_001, candidate_batch_size=1)
    storage.mkdir_if_needed(storage.final_dataset_path)
    lazy.pd.DataFrame({"id": list(range(5))}).to_parquet(
        storage.final_dataset_path / "batch_00000.parquet",
        index=False,
    )
    processor = create_mock_processor("proc", ["process_after_generation"])
    _replace_processors(simple_builder, [processor])

    simple_builder._processor_runner.run_after_generation(2, selection_publication=True)

    batch_files = sorted(storage.final_dataset_path.glob("*.parquet"))
    assert [path.name for path in batch_files] == [
        "batch_00000.parquet",
        "batch_00001.parquet",
        "batch_00002.parquet",
    ]
    assert [value for path in batch_files for value in lazy.pd.read_parquet(path)["id"]] == list(range(5))


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

    _replace_processors(builder_with_seed, [mock_processor])

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

    _replace_processors(simple_builder, [mock_processor])

    with pytest.raises(DatasetProcessingError, match="Failed in process_after_batch"):
        simple_builder._processor_runner.run_post_batch(lazy.pd.DataFrame({"id": [1, 2, 3]}), current_batch_number=0)


def test_processor_with_no_implemented_stages_is_skipped(builder_with_seed):
    """Test that a processor implementing no stages doesn't cause errors."""
    mock_processor = create_mock_processor("noop_processor", [])
    _replace_processors(builder_with_seed, [mock_processor])

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

    _replace_processors(builder_with_seed, processors)
    builder_with_seed.build(num_records=3)

    assert call_order == ["a", "b", "c"]


def test_pre_batch_processor_row_count_change_rejected(builder_with_seed, caplog):
    mock_processor = create_mock_processor("filtering_processor", ["process_before_batch"])
    mock_processor.process_before_batch.side_effect = lambda df: df.iloc[:2].reset_index(drop=True)
    _replace_processors(builder_with_seed, [mock_processor])

    with caplog.at_level(logging.INFO):
        with pytest.raises(DatasetGenerationError, match="Pre-batch processor changed row count"):
            builder_with_seed.build(num_records=3)

    assert not any("PRE_BATCH processors changed the record count" in record.message for record in caplog.records)


def test_process_preview_with_empty_dataframe(simple_builder):
    """Test that process_preview handles empty DataFrames gracefully."""
    mock_processor = create_mock_processor("test_processor", ["process_after_batch", "process_after_generation"])
    _replace_processors(simple_builder, [mock_processor])

    result = simple_builder.process_preview(lazy.pd.DataFrame())

    assert len(result) == 0
    mock_processor.process_after_batch.assert_called_once()
    mock_processor.process_after_generation.assert_called_once()


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


def _write_metadata(dataset_dir: Path, **fields) -> None:
    """Write a metadata.json into an existing dataset folder."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "sentinel.txt").write_text("x")  # make folder non-empty for resolved_dataset_name
    (dataset_dir / "metadata.json").write_text(json.dumps(fields))


def _write_incompatible_config_metadata(dataset_dir: Path, config_hash_version: str, **fields) -> None:
    _write_metadata(
        dataset_dir,
        **fields,
        config_hash="different-config",
        config_hash_version=config_hash_version,
    )


def _write_parquet_files(parquet_dir: Path, row_group_ids: list[int], row_counts: dict[int, int] | None = None) -> None:
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
    storage = ArtifactStorage(artifact_path=tmp_path, resume=ResumeMode.ALWAYS)
    stub_resource_provider.artifact_storage = storage
    stub_resource_provider.run_config = RunConfig(buffer_size=buffer_size)
    return DatasetBuilder(
        data_designer_config=stub_test_config_builder.build(),
        resource_provider=stub_resource_provider,
    )


def _make_sampler_only_builder(
    stub_resource_provider: Mock,
    tmp_path: Path,
    *,
    resume: ResumeMode = ResumeMode.IF_POSSIBLE,
    write_scheduler_events: bool = False,
) -> tuple[DatasetBuilder, ArtifactStorage]:
    """Create a builder that can run end-to-end without model or MCP stubs."""
    storage = ArtifactStorage(artifact_path=tmp_path, resume=resume)
    stub_resource_provider.artifact_storage = storage
    stub_resource_provider.run_config = RunConfig(
        buffer_size=2,
        write_scheduler_events=write_scheduler_events,
    )

    config_builder = DataDesignerConfigBuilder()
    config_builder.add_column(
        SamplerColumnConfig(name="some_id", sampler_type=SamplerType.UUID, params=UUIDSamplerParams())
    )
    return (
        DatasetBuilder(
            data_designer_config=config_builder.build(),
            resource_provider=stub_resource_provider,
        ),
        storage,
    )


@pytest.mark.parametrize("write_scheduler_events", [False, True])
def test_build_writes_scheduler_events_when_enabled(
    stub_resource_provider: Mock,
    tmp_path: Path,
    write_scheduler_events: bool,
) -> None:
    builder, _storage = _make_sampler_only_builder(
        stub_resource_provider,
        tmp_path,
        resume=ResumeMode.NEVER,
        write_scheduler_events=write_scheduler_events,
    )

    final_path = builder.build(num_records=2, resume=ResumeMode.NEVER)
    event_path = final_path.parent / "scheduler_events.jsonl"

    assert event_path.exists() is write_scheduler_events
    if write_scheduler_events:
        events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
        assert events[0]["event_kind"] == "scheduler_job_started"
        assert events[-1]["event_kind"] == "scheduler_job_completed"


def test_preview_does_not_write_scheduler_events(stub_resource_provider: Mock, tmp_path: Path) -> None:
    builder, _storage = _make_sampler_only_builder(
        stub_resource_provider,
        tmp_path,
        resume=ResumeMode.NEVER,
        write_scheduler_events=True,
    )

    builder.build_preview(num_records=1)

    assert list(tmp_path.rglob("scheduler_events.jsonl")) == []


def test_resumed_build_appends_scheduler_event_segment(stub_resource_provider: Mock, tmp_path: Path) -> None:
    builder, _storage = _make_sampler_only_builder(
        stub_resource_provider,
        tmp_path,
        resume=ResumeMode.NEVER,
        write_scheduler_events=True,
    )
    final_path = builder.build(num_records=1, resume=ResumeMode.NEVER)
    event_path = final_path.parent / "scheduler_events.jsonl"

    resumed_builder, _storage = _make_sampler_only_builder(
        stub_resource_provider,
        tmp_path,
        resume=ResumeMode.ALWAYS,
        write_scheduler_events=True,
    )
    resumed_builder.build(num_records=3, resume=ResumeMode.ALWAYS)

    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    starts = [event for event in events if event["event_kind"] == "scheduler_job_started"]
    assert len(starts) == 2
    assert starts[0]["diagnostics"]["run_id"] != starts[1]["diagnostics"]["run_id"]


def test_build_resume_ordered_seed_dataset_continues_from_next_planned_row(stub_resource_provider, tmp_path):
    """Regression for issue #709: resume must not replay ordered seed rows."""

    seed_source = DataFrameSeedSource(df=lazy.pd.DataFrame({"name": ["alpha", "beta", "gamma"]}))
    seed_reader = DataFrameSeedReader()
    seed_reader.attach(seed_source, Mock())

    config_builder = DataDesignerConfigBuilder()
    config_builder.with_seed_dataset(
        seed_source,
        sampling_strategy=SamplingStrategy.ORDERED,
        selection_strategy=IndexRange(start=0, end=2),
    )
    config_builder.add_column(ExpressionColumnConfig(name="copy", expr="{{ name }}"))

    storage = ArtifactStorage(artifact_path=tmp_path, dataset_name="dataset", resume=ResumeMode.NEVER)
    stub_resource_provider.artifact_storage = storage
    stub_resource_provider.seed_reader = seed_reader
    stub_resource_provider.run_config = RunConfig(disable_early_shutdown=True, buffer_size=1)

    builder = DatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )

    builder.build(num_records=1, resume=ResumeMode.NEVER)

    resumed_seed_reader = DataFrameSeedReader()
    resumed_seed_reader.attach(seed_source, Mock())
    stub_resource_provider.seed_reader = resumed_seed_reader
    stub_resource_provider.artifact_storage = ArtifactStorage(
        artifact_path=tmp_path,
        dataset_name="dataset",
        resume=ResumeMode.ALWAYS,
    )

    resumed_builder = DatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )

    final_path = resumed_builder.build(num_records=3, resume=ResumeMode.ALWAYS)
    result = lazy.pd.concat(
        [lazy.pd.read_parquet(path) for path in sorted(final_path.glob("batch_*.parquet"))],
        ignore_index=True,
    )

    assert result["name"].tolist() == ["alpha", "beta", "gamma"]
    assert result["copy"].tolist() == ["alpha", "beta", "gamma"]


def test_build_resume_ordered_seed_dataset_extension_wraps_at_cycle_boundary(stub_resource_provider, tmp_path):
    """Resume that extends past a full seed cycle hits the modulo == 0 branch.

    Companion to the basic #709 regression: when the resumed run's first new
    row group starts at an offset that is a non-zero multiple of the seed
    selection size, ``_index_range_at_offset`` returns the full original
    ``_index_range`` so reads restart at ``_index_range.start`` like a fresh
    cycle (instead of producing a degenerate empty range).
    """
    seed_source = DataFrameSeedSource(df=lazy.pd.DataFrame({"name": ["alpha", "beta", "gamma"]}))
    seed_reader = DataFrameSeedReader()
    seed_reader.attach(seed_source, Mock())

    config_builder = DataDesignerConfigBuilder()
    config_builder.with_seed_dataset(
        seed_source,
        sampling_strategy=SamplingStrategy.ORDERED,
        selection_strategy=IndexRange(start=0, end=2),
    )
    config_builder.add_column(ExpressionColumnConfig(name="copy", expr="{{ name }}"))

    storage = ArtifactStorage(artifact_path=tmp_path, dataset_name="dataset", resume=ResumeMode.NEVER)
    stub_resource_provider.artifact_storage = storage
    stub_resource_provider.seed_reader = seed_reader
    stub_resource_provider.run_config = RunConfig(disable_early_shutdown=True, buffer_size=1)

    builder = DatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )
    # Run 1: target=3 fills exactly one full cycle through the 3-row selection.
    builder.build(num_records=3, resume=ResumeMode.NEVER)

    resumed_seed_reader = DataFrameSeedReader()
    resumed_seed_reader.attach(seed_source, Mock())
    stub_resource_provider.seed_reader = resumed_seed_reader
    stub_resource_provider.artifact_storage = ArtifactStorage(
        artifact_path=tmp_path,
        dataset_name="dataset",
        resume=ResumeMode.ALWAYS,
    )

    resumed_builder = DatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )
    # Run 2: extend to 6. The first extension row group has start offset 3,
    # which is exactly one selection cycle (3 % 3 == 0) — the wrap branch.
    final_path = resumed_builder.build(num_records=6, resume=ResumeMode.ALWAYS)
    result = lazy.pd.concat(
        [lazy.pd.read_parquet(path) for path in sorted(final_path.glob("batch_*.parquet"))],
        ignore_index=True,
    )

    assert result["name"].tolist() == ["alpha", "beta", "gamma", "alpha", "beta", "gamma"]
    assert result["copy"].tolist() == ["alpha", "beta", "gamma", "alpha", "beta", "gamma"]


def test_build_resume_ordered_seed_dataset_with_partition_block_continues_within_partition(
    stub_resource_provider, tmp_path
):
    """Resume must seek into the partition slice, not just the full dataset.

    Companion to the basic #709 regression that uses ``IndexRange``: this exercises
    the same offset machinery with ``PartitionBlock``, which resolves to a
    contiguous ``IndexRange`` only because of ``PartitionBlock.to_index_range``.
    The resumed run also crosses a cycle boundary inside the partition, hitting
    both the offset-into-partition branch and the wraparound (``relative_offset == 0``)
    branch end-to-end.
    """

    seed_source = DataFrameSeedSource(df=lazy.pd.DataFrame({"name": ["a", "b", "c", "d", "e", "f"]}))
    seed_reader = DataFrameSeedReader()
    seed_reader.attach(seed_source, Mock())

    # PartitionBlock(index=1, num_partitions=3) over 6 rows -> IndexRange(2, 3),
    # i.e. a 2-row cycle of ["c", "d"]. With buffer_size=1 and num_records=4, a
    # full continuous run would emit ["c", "d", "c", "d"].
    config_builder = DataDesignerConfigBuilder()
    config_builder.with_seed_dataset(
        seed_source,
        sampling_strategy=SamplingStrategy.ORDERED,
        selection_strategy=PartitionBlock(index=1, num_partitions=3),
    )
    config_builder.add_column(ExpressionColumnConfig(name="copy", expr="{{ name }}"))

    storage = ArtifactStorage(artifact_path=tmp_path, dataset_name="dataset", resume=ResumeMode.NEVER)
    stub_resource_provider.artifact_storage = storage
    stub_resource_provider.seed_reader = seed_reader
    stub_resource_provider.run_config = RunConfig(disable_early_shutdown=True, buffer_size=1)

    builder = DatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )

    builder.build(num_records=1, resume=ResumeMode.NEVER)

    resumed_seed_reader = DataFrameSeedReader()
    resumed_seed_reader.attach(seed_source, Mock())
    stub_resource_provider.seed_reader = resumed_seed_reader
    stub_resource_provider.artifact_storage = ArtifactStorage(
        artifact_path=tmp_path,
        dataset_name="dataset",
        resume=ResumeMode.ALWAYS,
    )

    resumed_builder = DatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )
    final_path = resumed_builder.build(num_records=4, resume=ResumeMode.ALWAYS)
    result = lazy.pd.concat(
        [lazy.pd.read_parquet(path) for path in sorted(final_path.glob("batch_*.parquet"))],
        ignore_index=True,
    )

    assert result["name"].tolist() == ["c", "d", "c", "d"]
    assert result["copy"].tolist() == ["c", "d", "c", "d"]


def test_build_resume_starts_fresh_without_metadata(stub_resource_provider, stub_test_config_builder, tmp_path, caplog):
    """resume=True when only the folder exists (no metadata.json) logs an info message and starts fresh.

    This covers the case where a run was interrupted before any row group completed - the
    folder was created by _write_builder_config but metadata.json was never written.
    Previously this raised DatasetGenerationError; now it silently restarts from row group 0.
    """
    # Pre-create the folder with content so resolved_dataset_name(resume=True) returns "dataset"
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "builder_config.json").write_text("{}")  # non-empty, no metadata

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path)
    with caplog.at_level(logging.INFO):
        with patch.object(builder_mod, "run_readiness_check"):
            with patch.object(builder, "_build_async", return_value=True) as mock_async:
                builder.build(num_records=4, resume=ResumeMode.ALWAYS)

    _, kwargs = mock_async.call_args
    assert kwargs.get("resume") == ResumeMode.NEVER
    assert any("interrupted before any row group completed" in record.message for record in caplog.records)


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


def test_build_resume_raises_when_original_target_metadata_exceeds_target(
    stub_resource_provider, stub_test_config_builder, tmp_path
):
    """resume=ALWAYS rejects corrupt metadata where original_target exceeds target."""
    dataset_dir = tmp_path / "dataset"
    _write_metadata(
        dataset_dir,
        target_num_records=10,
        original_target_num_records=20,
        buffer_size=2,
        num_completed_batches=2,
        actual_num_records=4,
    )
    _write_parquet_files(dataset_dir / "parquet-files", [0, 1])

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)
    with pytest.raises(DatasetGenerationError, match="original_target_num_records=20.*target_num_records=10"):
        builder.build(num_records=10, resume=ResumeMode.ALWAYS)


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
        with patch.object(builder_mod, "run_readiness_check"):
            # 6 > 4 already generated → not already complete, should start generating
            # Here we just verify it does NOT raise on the num_records check
            with patch.object(builder, "_build_async", return_value=True):
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


def test_build_resume_always_raises_on_dropped_column_artifact_policy_mismatch(
    stub_resource_provider,
    stub_test_config_builder,
    tmp_path,
    caplog,
):
    """resume=ALWAYS rejects runs that would mix dropped-column artifact policies."""
    dataset_dir = tmp_path / "dataset"
    _write_metadata(
        dataset_dir,
        target_num_records=4,
        buffer_size=2,
        num_completed_batches=1,
        actual_num_records=2,
        preserve_dropped_columns=True,
    )

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)
    stub_resource_provider.run_config = RunConfig(buffer_size=2, preserve_dropped_columns=False)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(DatasetGenerationError, match="does not match the config used"):
            builder.build(num_records=4, resume=ResumeMode.ALWAYS)

    assert any("preserve_dropped_columns changed from True to False" in record.message for record in caplog.records)


def test_build_if_possible_starts_fresh_on_dropped_column_artifact_policy_mismatch(
    stub_resource_provider,
    stub_test_config_builder,
    tmp_path,
):
    """resume=IF_POSSIBLE starts fresh when dropped-column artifact policy differs."""
    dataset_dir = tmp_path / "dataset"
    _write_metadata(
        dataset_dir,
        target_num_records=4,
        buffer_size=2,
        num_completed_batches=1,
        actual_num_records=2,
        preserve_dropped_columns=True,
    )

    storage = ArtifactStorage(artifact_path=tmp_path, resume=ResumeMode.IF_POSSIBLE)
    stub_resource_provider.artifact_storage = storage
    stub_resource_provider.run_config = RunConfig(buffer_size=2, preserve_dropped_columns=False)
    builder = DatasetBuilder(
        data_designer_config=stub_test_config_builder.build(),
        resource_provider=stub_resource_provider,
    )

    with patch.object(builder_mod, "run_readiness_check"):
        with patch.object(builder, "_build_async", return_value=True):
            final_path = builder.build(num_records=4, resume=ResumeMode.IF_POSSIBLE)

    assert storage.resume == ResumeMode.NEVER
    assert (dataset_dir / "sentinel.txt").exists()
    assert final_path != dataset_dir / "parquet-files"


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
    _write_incompatible_config_metadata(
        dataset_dir,
        stub_test_config_builder.build().fingerprint()["config_hash_version"],
        target_num_records=4,
        buffer_size=2,
        num_completed_batches=1,
        actual_num_records=2,
    )
    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path)
    with pytest.raises(DatasetGenerationError, match="does not match the config used"):
        builder.build(num_records=4, resume=ResumeMode.ALWAYS)


def test_build_resume_always_raises_on_unreadable_stored_config(
    stub_resource_provider, stub_test_config_builder, tmp_path
):
    """resume=ALWAYS rejects legacy stored configs that fail schema validation."""
    dataset_dir = tmp_path / "dataset"
    _write_metadata(
        dataset_dir,
        target_num_records=4,
        buffer_size=2,
        num_completed_batches=1,
        actual_num_records=2,
    )
    (dataset_dir / "builder_config.json").write_text('{"data_designer": {"columns": [{"allow_resize": true}]}}')

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path)
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
        with patch.object(builder, "_build_async", return_value=True):
            with patch.object(builder._processor_runner, "has_processors_for", return_value=True):
                with patch.object(builder._processor_runner, "run_after_generation", side_effect=RuntimeError("boom")):
                    with pytest.raises(RuntimeError, match="boom"):
                        builder.build(num_records=4, resume=ResumeMode.ALWAYS)

    metadata = json.loads((dataset_dir / "metadata.json").read_text())
    assert metadata["post_generation_state"] == "started"
    assert metadata["post_generation_processed"] is False


def test_build_resume_complete_dataset_runs_after_generation_when_no_marker(
    stub_resource_provider, stub_test_config_builder, tmp_path
):
    """Crash window: complete parquet files on disk + after-gen processors + no ``post_generation_state``.

    Reproduces the gap between the final batch parquet write and the
    ``post_generation_state="started"`` write: if the process crashes there, the
    next ``resume=ALWAYS`` previously saw the dataset as ``already complete``,
    set ``generated=False`` and skipped after-generation entirely. After this
    fix, after-generation runs unconditionally on the on-disk dataset whenever
    after-generation processors are configured, leaving "started" and "complete"
    markers behind so the next resume short-circuits correctly.
    """
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
    after_gen_processor = create_mock_processor("after_gen", ["process_after_generation"])
    _replace_processors(builder, [after_gen_processor])

    builder.build(num_records=4, resume=ResumeMode.ALWAYS)

    after_gen_processor.process_after_generation.assert_called_once()
    metadata = json.loads((dataset_dir / "metadata.json").read_text())
    assert metadata["post_generation_state"] == "complete"
    assert metadata["post_generation_processed"] is True


def test_build_resume_post_generation_processed_missing_target_raises_clearly(
    stub_resource_provider, stub_test_config_builder, tmp_path
):
    """A post-processed dataset whose metadata lacks ``target_num_records`` raises a clear error.

    Without the explicit guard, ``prior_target`` is ``None`` and the ``num_records <
    prior_target`` comparison raises a raw ``TypeError``. Mirror ``_load_resume_state``
    and surface a clear ``DatasetGenerationError`` for the missing required field.
    """
    dataset_dir = tmp_path / "dataset"
    _write_metadata(
        dataset_dir,
        buffer_size=2,
        post_generation_processed=True,
    )

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)
    with pytest.raises(DatasetGenerationError, match="missing required field 'target_num_records'"):
        builder.build(num_records=4, resume=ResumeMode.ALWAYS)


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
        with patch.object(builder_mod, "run_readiness_check"):
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
        with patch.object(builder_mod, "run_readiness_check"):
            with patch.object(builder, "_build_async", return_value=True) as mock_async:
                builder.build(num_records=4, resume=ResumeMode.ALWAYS)

    # _build_async is called with resume=NEVER because the no-metadata path resets the mode
    _, kwargs = mock_async.call_args
    assert kwargs.get("resume") == ResumeMode.NEVER
    assert any("interrupted before any row group completed" in record.message for record in caplog.records)


def test_build_async_resume_already_complete_does_not_run_after_generation_processors(
    stub_resource_provider, stub_test_config_builder, tmp_path
):
    """Async resume: when already complete, run_after_generation must NOT be called."""
    dataset_dir = tmp_path / "dataset"
    _write_metadata(dataset_dir, target_num_records=4, buffer_size=2, num_completed_batches=2, actual_num_records=4)
    _write_parquet_files(dataset_dir / "parquet-files", [0, 1])

    builder = _make_resume_builder(stub_resource_provider, stub_test_config_builder, tmp_path, buffer_size=2)

    with patch.object(builder_mod, "run_readiness_check"):
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
    with patch.object(builder_mod, "run_readiness_check"):
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

    with patch.object(builder_mod, "asyncio", stdlib_asyncio, create=True):
        with patch.object(builder_mod, "ensure_async_engine_loop", Mock(return_value=Mock()), create=True):
            with patch.object(stdlib_asyncio, "run_coroutine_threadsafe", return_value=mock_future):
                with patch.object(builder_mod, "run_readiness_check"):
                    with patch.object(builder, "_prepare_async_run", side_effect=capturing_prepare):
                        builder.build(num_records=6, resume=ResumeMode.ALWAYS)

    # Filesystem says 2 groups done (IDs 0+1) → 2+2 = 4 records, not stale metadata value 2
    assert captured["initial_actual_num_records"] == 4
    assert captured["initial_total_num_batches"] == 2


def test_row_group_resume_plan_keeps_original_offsets_for_remaining_groups() -> None:
    """Async resume uses these planned offsets when completed row-group IDs have holes."""
    plan = build_row_group_resume_plan(
        original_target=4,
        num_records=4,
        buffer_size=1,
        completed_ids={0, 2},
    )

    assert plan.total_row_groups == 4
    assert not isinstance(plan.remaining_row_groups, list)
    assert list(plan.remaining_row_groups) == [(1, 1), (3, 1)]
    assert plan.remaining_row_groups.row_group_start_offset(1) == 1
    assert plan.remaining_row_groups.row_group_start_offset(3) == 3


def test_compact_row_group_plan_rejects_negative_extension() -> None:
    with pytest.raises(ValueError, match="num_records must be greater than or equal to original_target"):
        CompactRowGroupPlan.resume(
            original_target=10,
            num_records=8,
            buffer_size=2,
            completed_ids=set(),
        )


def test_row_group_resume_plan_tracks_completed_ids_at_half_complete_boundary() -> None:
    plan = CompactRowGroupPlan.resume(
        original_target=12,
        num_records=12,
        buffer_size=2,
        completed_ids={1, 2, 4},
    )

    assert getattr(plan, "_filter_includes_scheduled") is False
    assert getattr(plan, "_scheduled_ids") is None
    assert list(plan) == [(0, 2), (3, 2), (5, 2)]
    assert plan.scheduled_total_rows == 6
    assert plan.row_group_start_offset(5) == 10
    with pytest.raises(KeyError):
        plan.row_group_size(1)


def test_row_group_resume_plan_stays_sparse_when_almost_complete() -> None:
    completed_ids = set(range(999_998))

    tracemalloc.start()
    try:
        plan = CompactRowGroupPlan.resume(
            original_target=2_000_000,
            num_records=2_000_000,
            buffer_size=2,
            completed_ids=completed_ids,
        )
        remaining = list(plan)
        current_bytes, _peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert remaining == [(999_998, 2), (999_999, 2)]
    assert plan.scheduled_total_rows == 4
    assert current_bytes < 5 * 1024 * 1024


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

    with patch.object(builder_mod, "asyncio", stdlib_asyncio, create=True):
        with patch.object(builder_mod, "ensure_async_engine_loop", Mock(return_value=Mock()), create=True):
            with patch.object(stdlib_asyncio, "run_coroutine_threadsafe", return_value=mock_future):
                with patch.object(builder_mod, "run_readiness_check"):
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

    with patch.object(builder_mod, "asyncio", stdlib_asyncio, create=True):
        with patch.object(builder_mod, "ensure_async_engine_loop", Mock(return_value=Mock()), create=True):
            with patch.object(stdlib_asyncio, "run_coroutine_threadsafe", return_value=mock_future):
                with patch.object(builder_mod, "run_readiness_check"):
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

    with patch.object(builder_mod, "asyncio", stdlib_asyncio, create=True):
        with patch.object(builder_mod, "ensure_async_engine_loop", Mock(return_value=Mock()), create=True):
            with patch.object(stdlib_asyncio, "run_coroutine_threadsafe", return_value=mock_future):
                with patch.object(builder_mod, "run_readiness_check"):
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

    with patch.object(builder_mod, "asyncio", stdlib_asyncio, create=True):
        with patch.object(builder_mod, "ensure_async_engine_loop", Mock(return_value=Mock()), create=True):
            with patch.object(stdlib_asyncio, "run_coroutine_threadsafe", return_value=mock_future):
                with patch.object(builder_mod, "run_readiness_check"):
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

    with patch.object(builder_mod, "asyncio", stdlib_asyncio, create=True):
        with patch.object(builder_mod, "ensure_async_engine_loop", Mock(return_value=Mock()), create=True):
            with patch.object(stdlib_asyncio, "run_coroutine_threadsafe", return_value=mock_future):
                with patch.object(builder_mod, "run_readiness_check"):
                    with patch.object(builder, "_prepare_async_run", side_effect=capturing_prepare):
                        builder.build(num_records=6, resume=ResumeMode.ALWAYS)

    # Only rg_id=1 remains; rg_id=0 and rg_id=2 are already on disk
    assert list(captured["precomputed_row_groups"]) == [(1, 2)]


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

    with patch.object(builder_mod, "asyncio", stdlib_asyncio, create=True):
        with patch.object(builder_mod, "ensure_async_engine_loop", Mock(return_value=Mock()), create=True):
            with patch.object(stdlib_asyncio, "run_coroutine_threadsafe", return_value=mock_future):
                with patch.object(builder_mod, "run_readiness_check"):
                    with patch.object(builder, "_prepare_async_run", side_effect=capturing_prepare):
                        builder.build(num_records=7, resume=ResumeMode.ALWAYS)

    # rg_id=3 should have 2 records (7-5=2 extension records, buffer_size=2), not 1
    assert list(captured["precomputed_row_groups"]) == [(3, 2)]


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

    with patch.object(builder_mod, "asyncio", stdlib_asyncio, create=True):
        with patch.object(builder_mod, "ensure_async_engine_loop", Mock(return_value=Mock()), create=True):
            with patch.object(stdlib_asyncio, "run_coroutine_threadsafe", return_value=mock_future):
                with patch.object(builder_mod, "run_readiness_check"):
                    with patch.object(builder, "_prepare_async_run", side_effect=capturing_prepare) as mock_prepare:
                        builder.build(num_records=6, resume=ResumeMode.ALWAYS)

    # _prepare_async_run must be called — the dataset is NOT already complete
    mock_prepare.assert_called_once()


def test_if_possible_incompatible_config_does_not_overwrite_existing_dataset(stub_resource_provider, tmp_path):
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

    builder, storage = _make_sampler_only_builder(stub_resource_provider, tmp_path)
    _write_incompatible_config_metadata(
        dataset_dir,
        builder.data_designer_config.fingerprint()["config_hash_version"],
    )

    final_path = builder.build(num_records=2, resume=ResumeMode.IF_POSSIBLE)

    # artifact_storage.resume must be downgraded to NEVER so resolved_dataset_name uses NEVER semantics
    assert storage.resume == ResumeMode.NEVER

    assert sentinel.exists(), "Existing dataset directory must not be touched"
    assert final_path != dataset_dir / "parquet-files"
    assert storage.resolved_dataset_name != "dataset", (
        "resolved_dataset_name must be a new timestamped directory, not the existing one"
    )


def test_if_possible_incompatible_config_refreshes_media_storage_path(stub_resource_provider, tmp_path):
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

    builder, storage = _make_sampler_only_builder(stub_resource_provider, tmp_path)

    # Trigger validate_folder_names so _media_storage is initialised with IF_POSSIBLE semantics
    # (non-empty dir + IF_POSSIBLE → resolved_dataset_name returns "dataset", not timestamped)
    original_media_base = storage.media_storage.base_path

    _write_incompatible_config_metadata(
        dataset_dir,
        builder.data_designer_config.fingerprint()["config_hash_version"],
    )

    builder.build(num_records=2, resume=ResumeMode.IF_POSSIBLE)

    new_media_base = storage.media_storage.base_path
    assert new_media_base != original_media_base, (
        "media_storage.base_path must be updated to the new timestamped directory after IF_POSSIBLE → NEVER downgrade"
    )
    assert new_media_base == storage.base_dataset_path, (
        "media_storage.base_path must match base_dataset_path after downgrade"
    )


def test_if_possible_starts_fresh_when_no_existing_directory(stub_resource_provider, tmp_path):
    """IF_POSSIBLE on a first-ever run (no dataset directory) must start fresh, not raise.

    Bug: _check_resume_config_compatibility returned True when config_path did not exist,
    which caused IF_POSSIBLE to upgrade to ALWAYS. resolved_dataset_name then raised
    ArtifactStorageError because ALWAYS requires an existing directory.

    Fix: return False when the dataset directory itself is absent.
    """
    builder, storage = _make_sampler_only_builder(stub_resource_provider, tmp_path)
    final_path = builder.build(num_records=2, resume=ResumeMode.IF_POSSIBLE)

    assert storage.resume == ResumeMode.NEVER
    assert final_path.exists()


def test_if_possible_starts_fresh_when_directory_is_empty(stub_resource_provider, tmp_path):
    """IF_POSSIBLE on an empty dataset directory must start fresh, not raise.

    Edge case: a prior run crashed in the window between mkdir and the first file write
    inside _write_builder_config, leaving an empty directory. _check_resume_config_compatibility
    previously returned True (config file absent → assume compatible), causing IF_POSSIBLE to
    upgrade to ALWAYS, which then raised ArtifactStorageError because the directory is empty.

    Fix: treat an empty directory the same as a missing one — return False.
    """
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()  # empty — no files written yet

    builder, storage = _make_sampler_only_builder(stub_resource_provider, tmp_path)
    final_path = builder.build(num_records=2, resume=ResumeMode.IF_POSSIBLE)

    assert storage.resume == ResumeMode.NEVER
    assert final_path.exists()
