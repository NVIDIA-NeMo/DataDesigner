# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from data_designer.config.analysis.column_profilers import JudgeScoreProfilerConfig
from data_designer.config.column_configs import (
    CustomColumnConfig,
    ExpressionColumnConfig,
    ImageColumnConfig,
    LLMStructuredColumnConfig,
    LLMTextColumnConfig,
    SamplerColumnConfig,
)
from data_designer.config.column_types import ColumnConfigT
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.custom_column import custom_column_generator
from data_designer.config.models import ModelConfig
from data_designer.config.processors import DropColumnsProcessorConfig, ProcessorType
from data_designer.config.sampler_constraints import ColumnInequalityConstraint, ScalarInequalityConstraint
from data_designer.config.seed import IndexRange, SamplingStrategy
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.interface.cohort_retry import RetryUntil, SamplerRetryMode
from data_designer.interface.cohort_retry_builders import CohortRetryBuilderProjection
from data_designer.interface.errors import DataDesignerWorkflowError


@custom_column_generator(side_effect_columns=["custom_accept"])
def _custom_predicate(row: dict[str, object]) -> dict[str, object]:
    return {**row, "custom_accept": True}


def _touch_parquet(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.touch()
    return path


def _policy(
    predicate_column: str = "accepted",
    *,
    mode: SamplerRetryMode = SamplerRetryMode.PRESERVE,
) -> RetryUntil:
    return RetryUntil(
        predicate_column=predicate_column,
        max_attempts=3,
        sampler_retry_mode=mode,
    )


def _seeded_projection_builder(
    stub_model_configs: list[ModelConfig],
    seed_path: Path,
) -> DataDesignerConfigBuilder:
    builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    builder.with_seed_dataset(
        LocalFileSeedSource(path=str(seed_path)),
        sampling_strategy=SamplingStrategy.SHUFFLE,
        selection_strategy=IndexRange(start=1, end=2),
    )
    builder.add_column(
        SamplerColumnConfig(
            name="category",
            sampler_type="category",
            params={"values": ["a", "b"]},
            drop=True,
        )
    )
    builder.add_column(
        SamplerColumnConfig(
            name="score",
            sampler_type="uniform",
            params={"low": 0, "high": 1},
        )
    )
    builder.add_constraint(ScalarInequalityConstraint(target_column="score", operator="ge", rhs=0))
    builder.add_column(ExpressionColumnConfig(name="accepted", expr="{{ score > 0.5 }}", dtype="bool"))
    builder.add_column(
        ImageColumnConfig(
            name="image",
            prompt="Draw {{ category }}",
            model_alias="stub-model",
            drop=True,
        )
    )
    return builder


def test_preserve_base_keeps_seed_sampler_constraints_and_selection(
    stub_model_configs: list[ModelConfig],
    tmp_path: Path,
) -> None:
    original = _seeded_projection_builder(stub_model_configs, _touch_parquet(tmp_path, "seed.parquet"))
    projection = CohortRetryBuilderProjection(original, _policy())

    base = projection.build_base_builder()

    assert projection.original_dropped_names == ("category", "image")
    assert projection.requires_base_materialization is True
    assert [column.name for column in base.get_column_configs()] == ["category", "score"]
    assert all(not column.drop for column in base.get_column_configs())
    assert base.build().constraints == original.build().constraints
    assert base.get_processor_configs() == []
    assert base.get_profilers() == []

    base_seed = base.get_seed_config()
    original_seed = original.get_seed_config()
    assert base_seed is not None and original_seed is not None
    assert base_seed is not original_seed
    assert base_seed.source is not original_seed.source
    assert base_seed.sampling_strategy == SamplingStrategy.SHUFFLE
    assert base_seed.selection_strategy == IndexRange(start=1, end=2)
    assert base.model_configs[0] is not original.model_configs[0]
    assert original.get_column_config("category").drop is True


def test_resample_base_is_seed_only_and_uses_passthrough_processor(
    stub_model_configs: list[ModelConfig],
    tmp_path: Path,
) -> None:
    original = _seeded_projection_builder(stub_model_configs, _touch_parquet(tmp_path, "seed.parquet"))
    projection = CohortRetryBuilderProjection(original, _policy(mode=SamplerRetryMode.RESAMPLE))

    base = projection.build_base_builder()

    assert base.get_column_configs() == []
    assert base.build().constraints is None
    assert len(base.get_processor_configs()) == 1
    passthrough = base.get_processor_configs()[0]
    assert passthrough.processor_type == ProcessorType.DROP_COLUMNS
    assert passthrough.column_names == []
    seed = base.get_seed_config()
    assert seed is not None
    assert seed.sampling_strategy == SamplingStrategy.SHUFFLE
    assert seed.selection_strategy == IndexRange(start=1, end=2)


def test_resample_seedless_projection_skips_base_materialization(
    stub_model_configs: list[ModelConfig],
) -> None:
    builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    builder.add_column(SamplerColumnConfig(name="source", sampler_type="category", params={"values": ["a", "b"]}))
    builder.add_column(ExpressionColumnConfig(name="accepted", expr="{{ source == 'a' }}", dtype="bool"))

    projection = CohortRetryBuilderProjection(builder, _policy(mode=SamplerRetryMode.RESAMPLE))

    assert projection.requires_base_materialization is False


@pytest.mark.parametrize(
    ("mode", "expected_names", "expects_constraints"),
    [
        (SamplerRetryMode.PRESERVE, ["accepted", "image"], False),
        (SamplerRetryMode.RESAMPLE, ["category", "score", "accepted", "image"], True),
    ],
)
def test_attempt_builder_projects_mode_and_resets_drop_flags(
    stub_model_configs: list[ModelConfig],
    tmp_path: Path,
    mode: SamplerRetryMode,
    expected_names: list[str],
    expects_constraints: bool,
) -> None:
    original = _seeded_projection_builder(stub_model_configs, _touch_parquet(tmp_path, "seed.parquet"))
    attempt_input = _touch_parquet(tmp_path, f"attempt-{mode.value}.parquet")
    projection = CohortRetryBuilderProjection(original, _policy(mode=mode))

    attempt = projection.build_attempt_builder(attempt_input)

    assert [column.name for column in attempt.get_column_configs()] == expected_names
    assert all(not column.drop for column in attempt.get_column_configs())
    assert bool(attempt.build().constraints) is expects_constraints
    assert attempt.get_processor_configs() == []
    assert attempt.get_profilers() == []
    seed = attempt.get_seed_config()
    assert seed is not None
    assert seed.source.path == str(attempt_input)
    assert seed.sampling_strategy == SamplingStrategy.ORDERED
    assert seed.selection_strategy is None
    assert original.get_column_config("category").drop is True
    assert original.get_column_config("image").drop is True


def test_final_builder_restores_explicit_and_implicit_drop_semantics_and_profilers(
    stub_model_configs: list[ModelConfig],
    tmp_path: Path,
) -> None:
    builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    builder.add_column(SamplerColumnConfig(name="direct_hidden", sampler_type="uuid", params={}, drop=True))
    builder.add_column(SamplerColumnConfig(name="processor_hidden", sampler_type="uuid", params={}))
    builder.add_column(ExpressionColumnConfig(name="accepted", expr="{{ true }}", dtype="bool"))
    explicit_drop = DropColumnsProcessorConfig(name="cleanup", column_names=["processor_*"])
    profiler = JudgeScoreProfilerConfig(model_alias="stub-model", summary_score_sample_size=5)
    builder.add_processor(explicit_drop)
    builder.add_profiler(profiler)
    projection = CohortRetryBuilderProjection(builder, _policy())
    final_input = _touch_parquet(tmp_path, "accepted.parquet")

    final = projection.build_final_builder(final_input)

    assert final.get_column_configs() == []
    assert [processor.name for processor in final.get_processor_configs()][0] == "cleanup"
    assert len(final.get_processor_configs()) == 2
    implicit_drop = final.get_processor_configs()[1]
    assert implicit_drop.column_names == ["direct_hidden"]
    assert final.get_processor_configs()[0] is not explicit_drop
    assert final.get_profilers() == [profiler]
    assert final.get_profilers()[0] is not profiler
    seed = final.get_seed_config()
    assert seed is not None
    assert seed.source.path == str(final_input)
    assert seed.sampling_strategy == SamplingStrategy.ORDERED
    assert projection.original_dropped_names == ("direct_hidden", "processor_hidden")


def test_final_builder_adds_seed_passthrough_when_no_user_processors_or_drops(
    stub_model_configs: list[ModelConfig],
    tmp_path: Path,
) -> None:
    builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    builder.add_column(ExpressionColumnConfig(name="accepted", expr="{{ true }}", dtype="bool"))
    projection = CohortRetryBuilderProjection(builder, _policy())

    final = projection.build_final_builder(_touch_parquet(tmp_path, "accepted.parquet"))

    assert len(final.get_processor_configs()) == 1
    assert final.get_processor_configs()[0].column_names == []


def test_projection_rejects_preserved_sampler_dependency_on_generated_column(
    stub_model_configs: list[ModelConfig],
) -> None:
    builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    builder.add_column(
        SamplerColumnConfig(
            name="category",
            sampler_type="category",
            params={"values": ["a", "b"]},
            conditional_params={"accepted": {"values": ["a"]}},
        )
    )
    builder.add_column(ExpressionColumnConfig(name="accepted", expr="{{ true }}", dtype="bool"))

    with pytest.raises(DataDesignerWorkflowError, match="depend on regenerated columns"):
        CohortRetryBuilderProjection(builder, _policy())


def test_projection_rejects_constraint_dependency_on_generated_column(
    stub_model_configs: list[ModelConfig],
) -> None:
    builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    builder.add_column(SamplerColumnConfig(name="score", sampler_type="uniform", params={"low": 0, "high": 1}))
    builder.add_column(ExpressionColumnConfig(name="accepted", expr="{{ score > 0.5 }}", dtype="bool"))
    builder.add_constraint(ColumnInequalityConstraint(target_column="score", operator="gt", rhs="accepted"))

    with pytest.raises(DataDesignerWorkflowError, match="'score'.*'accepted'"):
        CohortRetryBuilderProjection(builder, _policy())


@pytest.mark.parametrize(
    ("column", "predicate", "error"),
    [
        (ExpressionColumnConfig(name="accepted", expr="{{ true }}", dtype="str"), "accepted", "dtype='bool'"),
        (
            SamplerColumnConfig(name="accepted", sampler_type="category", params={"values": ["yes", "no"]}),
            "accepted",
            "unsupported column type",
        ),
        (
            LLMStructuredColumnConfig(
                name="accepted",
                prompt="Classify",
                model_alias="stub-model",
                output_format={"type": "object", "properties": {"accepted": {"type": "boolean"}}},
            ),
            "accepted",
            "extract the Boolean into an expression",
        ),
    ],
)
def test_projection_rejects_invalid_predicate_producers(
    stub_model_configs: list[ModelConfig],
    column: ColumnConfigT,
    predicate: str,
    error: str,
) -> None:
    builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    builder.add_column(column)

    with pytest.raises(DataDesignerWorkflowError, match=error):
        CohortRetryBuilderProjection(builder, _policy(predicate))


def test_projection_accepts_custom_side_effect_and_llm_text_predicates(
    stub_model_configs: list[ModelConfig],
) -> None:
    custom_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    custom_builder.add_column(CustomColumnConfig(name="custom", generator_function=_custom_predicate))
    text_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    text_builder.add_column(LLMTextColumnConfig(name="llm_accept", prompt="Return a Boolean", model_alias="stub-model"))

    custom_projection = CohortRetryBuilderProjection(custom_builder, _policy("custom_accept"))
    text_projection = CohortRetryBuilderProjection(text_builder, _policy("llm_accept"))

    assert custom_projection.predicate_column == "custom_accept"
    assert text_projection.predicate_column == "llm_accept"


def test_projection_rejects_missing_predicate(stub_model_configs: list[ModelConfig]) -> None:
    builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    builder.add_column(ExpressionColumnConfig(name="other", expr="{{ true }}", dtype="bool"))

    with pytest.raises(DataDesignerWorkflowError, match="not declared"):
        CohortRetryBuilderProjection(builder, _policy("accepted"))
