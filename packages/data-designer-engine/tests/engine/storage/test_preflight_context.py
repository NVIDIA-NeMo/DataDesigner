# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal
from unittest.mock import Mock

import pytest

from data_designer.config.base import ProcessorConfig, SkipConfig
from data_designer.config.column_configs import ExpressionColumnConfig, LLMStructuredColumnConfig, LLMTextColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.errors import InvalidConfigError
from data_designer.config.models import ModelConfig
from data_designer.config.processors import DropColumnsProcessorConfig, SchemaTransformProcessorConfig
from data_designer.config.run_config import RunConfig
from data_designer.engine.dataset_builders.dataset_builder import DatasetBuilder
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.registry.data_designer_registry import DataDesignerRegistry
from data_designer.engine.validation import (
    ViolationLevel,
    validate_parquet_storage,
    validate_schema_transform_processor,
)

if TYPE_CHECKING:
    import pandas as pd


class FutureEncoderConfig(ProcessorConfig):
    processor_type: Literal["future_encoder"] = "future_encoder"


class FutureEncoder(Processor[FutureEncoderConfig]):
    def process_after_batch(self, data: pd.DataFrame, *, current_batch_number: int | None) -> pd.DataFrame:
        return data.assign(temporary=data["temporary"].map(json.dumps))


class FutureWorkload(DataDesignerConfig):
    processors: list[FutureEncoderConfig]


def _column(*, drop: bool = False, skip: SkipConfig | None = None) -> LLMStructuredColumnConfig:
    return LLMStructuredColumnConfig(
        name="temporary", model_alias="test_model", prompt="Generate", output_format={"const": {}}, drop=drop, skip=skip
    )


@pytest.mark.parametrize("processor_drop", [False, True])
def test_discarded_values_are_not_checked_for_native_storage(processor_drop: bool) -> None:
    column = _column(drop=not processor_drop)
    processors = [DropColumnsProcessorConfig(name="drop", column_names=["temp*"])] if processor_drop else []
    assert validate_parquet_storage([column], processor_configs=processors, preserve_dropped_columns=False) == []
    assert any(
        violation.level == ViolationLevel.ERROR
        for violation in validate_parquet_storage([column], processor_configs=processors, preserve_dropped_columns=True)
    )


def test_unknown_processor_turns_schema_proofs_into_warnings() -> None:
    violations = validate_parquet_storage([_column()], processor_configs=[FutureEncoderConfig(name="encoder")])
    assert violations and all(violation.level == ViolationLevel.WARNING for violation in violations)


def test_known_non_mutating_processor_does_not_hide_native_failure() -> None:
    violations = validate_parquet_storage(
        [_column()], processor_configs=[SchemaTransformProcessorConfig(name="copy", template={"copy": "ok"})]
    )
    assert any(violation.level == ViolationLevel.ERROR for violation in violations)


def test_skipped_generation_can_store_a_native_compatible_value() -> None:
    violations = validate_parquet_storage([_column(skip=SkipConfig(when="{{ condition }}", value="skipped"))])
    assert violations and all(violation.level == ViolationLevel.WARNING for violation in violations)


@pytest.mark.parametrize("template_key", ["actual", "{{ output_name }}", "{# comment #}actual"])
def test_processor_storage_names_are_checked_without_guessing_dynamic_outputs(template_key: str) -> None:
    processor = SchemaTransformProcessorConfig(
        name="copy", template={template_key: "ok"}, json_storage_columns=["typo"]
    )
    violations = validate_schema_transform_processor(columns=[], processor_configs=[processor])
    assert len(violations) == 1
    expected = ViolationLevel.ERROR if template_key == "actual" else ViolationLevel.WARNING
    assert violations[0].level == expected
    assert "copy" in violations[0].message and "typo" in violations[0].message


def test_literal_processor_storage_name_is_valid_with_other_dynamic_keys() -> None:
    processor = SchemaTransformProcessorConfig(
        name="copy", template={"actual": "ok", "{{ output_name }}": "ok"}, json_storage_columns=["actual"]
    )
    assert validate_schema_transform_processor(columns=[], processor_configs=[processor]) == []


def test_impossible_processor_storage_names_fail_before_generation(stub_resource_provider: Mock) -> None:
    stub_resource_provider.seed_reader = None
    config = DataDesignerConfigBuilder()
    config.add_column(ExpressionColumnConfig(name="source", expr="ok", dtype="str"))
    config.add_processor(
        SchemaTransformProcessorConfig(name="copy", template={"actual": "{{ source }}"}, json_storage_columns=["typo"])
    )
    with pytest.raises(InvalidConfigError, match="validation errors"):
        DatasetBuilder(config.build(), stub_resource_provider)
    stub_resource_provider.model_registry.arun_health_check.assert_not_called()


@pytest.mark.parametrize("preserve", [False, True])
def test_compiler_uses_the_artifact_policy(
    stub_resource_provider: Mock, stub_model_configs: list, preserve: bool
) -> None:
    provider = stub_resource_provider
    provider.seed_reader = None
    provider.run_config = RunConfig(preserve_dropped_columns=preserve)
    provider.model_registry.get_model.return_value.agenerate.return_value = ({}, [])
    provider.model_registry.get_model_config.return_value = ModelConfig(
        alias="test_model", model="test", provider="test_provider", skip_health_check=True
    )
    provider.model_registry.get_model_provider.return_value.name = "test_provider"
    config = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config.add_column(_column(drop=True))
    config.add_column(ExpressionColumnConfig(name="output", expr="ok", dtype="str"))
    if preserve:
        with pytest.raises(InvalidConfigError, match="validation errors"):
            DatasetBuilder(config.build(), provider)
    else:
        builder = DatasetBuilder(config.build(), provider)
        builder.build(num_records=2)
        assert builder.artifact_storage.load_dataset().to_dict("records") == [{"output": "ok"}] * 2


def test_post_batch_plugin_can_repair_generated_values_before_storage(
    stub_resource_provider: Mock, stub_model_configs: list
) -> None:
    provider = stub_resource_provider
    provider.seed_reader = None
    provider.run_config = RunConfig(buffer_size=2, max_concurrent_row_groups=1)
    provider.model_registry.get_model.return_value.agenerate.return_value = ({}, [])
    provider.model_registry.get_model_config.return_value = ModelConfig(
        alias="test_model", model="test", provider="test_provider", skip_health_check=True
    )
    provider.model_registry.get_model_provider.return_value.name = "test_provider"
    config = FutureWorkload(
        columns=[_column()], model_configs=stub_model_configs, processors=[FutureEncoderConfig(name="encoder")]
    )
    registry = DataDesignerRegistry()
    registry.processors.register("future_encoder", FutureEncoder, FutureEncoderConfig, False)
    builder = DatasetBuilder(config, provider, registry)
    # This test declares its plugin on a typed workload subclass rather than
    # changing the process-global plugin discovery union used by BuilderConfig.
    with pytest.warns(UserWarning, match="Pydantic serializer warnings"):
        builder.build(num_records=2)
    assert builder.artifact_storage.load_dataset()["temporary"].tolist() == ["{}", "{}"]


@pytest.mark.parametrize(
    "prompt,propagate_skip,expected",
    [
        ("Generate", True, ViolationLevel.ERROR),
        ("{{ source }}", True, ViolationLevel.WARNING),
        ("{{ source }}", False, ViolationLevel.ERROR),
    ],
    ids=["unrelated", "propagated", "propagation-disabled"],
)
def test_storage_certainty_only_changes_for_skippable_columns(
    prompt: str, propagate_skip: bool, expected: ViolationLevel
) -> None:
    source = ExpressionColumnConfig(name="source", expr="ok", dtype="str", skip=SkipConfig(when="{{ condition }}"))
    output = _column().model_copy(update={"prompt": prompt, "propagate_skip": propagate_skip})
    violations = validate_parquet_storage([output, source])
    assert violations and all(violation.level == expected for violation in violations)


@pytest.mark.parametrize("propagate_bridge", [False, True])
def test_storage_skip_uncertainty_follows_transitive_dependencies(propagate_bridge: bool) -> None:
    source = ExpressionColumnConfig(name="source", expr="ok", dtype="str", skip=SkipConfig(when="{{ condition }}"))
    bridge = ExpressionColumnConfig(name="bridge", expr="{{ source }}", dtype="str", propagate_skip=propagate_bridge)
    output = _column().model_copy(update={"prompt": "{{ bridge }}"})
    violations = validate_parquet_storage([output, bridge, source])
    expected = ViolationLevel.WARNING if propagate_bridge else ViolationLevel.ERROR
    assert violations and all(violation.level == expected for violation in violations)


@pytest.mark.parametrize("use_side_effect", [False, True])
def test_storage_skip_uncertainty_resolves_producer_side_effects(use_side_effect: bool) -> None:
    source = LLMTextColumnConfig(
        name="source",
        model_alias="test_model",
        prompt="Generate",
        extract_reasoning_content=True,
        skip=SkipConfig(when="{{ condition }}"),
    )
    dependency = "source__reasoning_content" if use_side_effect else "source"
    output = _column().model_copy(update={"prompt": "{{ " + dependency + " }}"})
    violations = validate_parquet_storage([output, source])
    assert violations and all(violation.level == ViolationLevel.WARNING for violation in violations)
