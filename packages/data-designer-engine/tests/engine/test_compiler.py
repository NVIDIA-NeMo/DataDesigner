# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING, Annotated, Literal
from unittest.mock import Mock, patch

import pytest
from pydantic import Field

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.base import ProcessorConfig
from data_designer.config.column_configs import ExpressionColumnConfig, LLMTextColumnConfig, SamplerColumnConfig
from data_designer.config.config_builder import BuilderConfig, DataDesignerConfigBuilder
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.errors import InvalidConfigError
from data_designer.config.processor_types import ProcessorConfigT
from data_designer.config.processors import DropColumnsProcessorConfig
from data_designer.config.sampler_params import CategorySamplerParams, SamplerType, UUIDSamplerParams
from data_designer.config.seed_source import FileContentsSeedSource, HuggingFaceSeedSource
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.config.utils.trace_type import TraceType
from data_designer.engine.compiler import compile_data_designer_config
from data_designer.engine.dataset_builders.dataset_builder import DatasetBuilder
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.processing.processors.registry import create_default_processor_registry
from data_designer.engine.registry.data_designer_registry import DataDesignerRegistry
from data_designer.engine.resources.resource_provider import ResourceProvider
from data_designer.engine.resources.seed_reader import (
    DataFrameSeedReader,
    FileContentsSeedReader,
    SeedReader,
    SeedReaderConfigError,
)
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.engine.validation import Violation, ViolationLevel, ViolationType

if TYPE_CHECKING:
    import pandas as pd


class SeedTransformConfig(ProcessorConfig):
    processor_type: Literal["test_seed_transform"] = "test_seed_transform"


class LateTransformConfig(ProcessorConfig):
    processor_type: Literal["test_late_transform"] = "test_late_transform"


class LateTransformProcessor(Processor[LateTransformConfig]):
    def process_after_generation(self, data: pd.DataFrame) -> pd.DataFrame:
        return data


class ProcessorTestConfig(DataDesignerConfig):
    processors: (
        list[
            Annotated[
                ProcessorConfigT | SeedTransformConfig | LateTransformConfig,
                Field(discriminator="processor_type"),
            ]
        ]
        | None
    ) = None


class ProcessorTestBuilderConfig(BuilderConfig):
    data_designer: ProcessorTestConfig


class SeedTransformProcessor(Processor[SeedTransformConfig]):
    def process_before_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        for name in self.config.columns_added:
            result[name] = data.iloc[:, 0] * 2
        return result.drop(columns=self.config.columns_removed)


@pytest.fixture(autouse=True)
def register_seed_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("data_designer.config.config_builder.DataDesignerConfig", ProcessorTestConfig)
    monkeypatch.setattr("data_designer.config.config_builder.BuilderConfig", ProcessorTestBuilderConfig)
    monkeypatch.setattr(
        "data_designer.engine.dataset_builders.dataset_builder.BuilderConfig", ProcessorTestBuilderConfig
    )
    registry = create_default_processor_registry()
    for attr in ("_registry", "_reverse_registry", "_config_registry", "_reverse_config_registry"):
        monkeypatch.setattr(type(registry), attr, getattr(registry, attr).copy())
    registry.register("test_seed_transform", SeedTransformProcessor, SeedTransformConfig)
    registry.register("test_late_transform", LateTransformProcessor, LateTransformConfig)


@pytest.fixture
def resource_provider(stub_resource_provider: ResourceProvider, stub_seed_reader: SeedReader) -> ResourceProvider:
    stub_resource_provider.seed_reader = stub_seed_reader
    return stub_resource_provider


def test_processor_plugin_union_roundtrip() -> None:
    # Plugin config unions are fixed at import time, so exercise injection in a fresh interpreter.
    script = """
from typing import Literal

from data_designer.config.base import ProcessorConfig
from data_designer.plugins.plugin import Plugin, PluginType
from data_designer.plugins.registry import PluginRegistry

class ExtensionConfig(ProcessorConfig):
    processor_type: Literal["test_extension"] = "test_extension"
    plugin_payload: dict[str, list[str]]

plugin = Plugin.model_construct(
    impl_qualified_name="example.ExtensionProcessor",
    config_qualified_name="__main__.ExtensionConfig",
    plugin_type=PluginType.PROCESSOR,
)
plugin.__dict__["config_cls"] = ExtensionConfig
PluginRegistry()._plugins["test_extension"] = plugin

from data_designer.config.data_designer_config import DataDesignerConfig

processor = ExtensionConfig(
    name="extension",
    columns_added=["new_column"],
    columns_removed=["raw_column"],
    plugin_payload={"columns_added": ["opaque_payload"]},
)
config = DataDesignerConfig(columns=[], processors=[processor])
restored = DataDesignerConfig.model_validate_json(config.model_dump_json())
assert type(restored.processors[0]) is ExtensionConfig
assert restored.processors[0].model_dump() == processor.model_dump()
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_adds_seed_columns(resource_provider: ResourceProvider) -> None:
    builder = DataDesignerConfigBuilder()
    builder.add_column(
        SamplerColumnConfig(
            name="language",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["english", "french"]),
        )
    )
    builder.with_seed_dataset(HuggingFaceSeedSource(path="hf://datasets/test/data.csv"))

    config = compile_data_designer_config(builder.build(), resource_provider)

    assert len(config.columns) == 3


def test_errors_on_seed_column_collisions(resource_provider: ResourceProvider) -> None:
    builder = DataDesignerConfigBuilder()
    builder.add_column(
        SamplerColumnConfig(
            name="city",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["new york", "los angeles"]),
        )
    )
    builder.with_seed_dataset(HuggingFaceSeedSource(path="hf://datasets/test/data.csv"))

    with pytest.raises(InvalidConfigError) as excinfo:
        compile_data_designer_config(builder.build(), resource_provider)

    assert "city" in str(excinfo)


def test_seed_reader_config_errors_are_invalid_config_errors(resource_provider: ResourceProvider) -> None:
    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(HuggingFaceSeedSource(path="hf://datasets/test/data.csv"))
    resource_provider.seed_reader = Mock(spec=SeedReader)
    resource_provider.seed_reader.get_column_names.side_effect = SeedReaderConfigError("missing seed root")

    with pytest.raises(InvalidConfigError, match="missing seed root") as excinfo:
        compile_data_designer_config(builder.build(), resource_provider)

    assert isinstance(excinfo.value.__cause__, SeedReaderConfigError)


def test_compile_rejects_missing_fixed_schema_filesystem_seed_root(
    stub_resource_provider: ResourceProvider,
    tmp_path,
):
    missing_dir = tmp_path / "missing"
    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(FileContentsSeedSource(path=str(missing_dir), file_pattern="*.txt"))
    reader = FileContentsSeedReader()
    reader.attach(builder.build().seed_config.source, PlaintextResolver())
    stub_resource_provider.seed_reader = reader

    with pytest.raises(InvalidConfigError, match="Seed source directory .* does not exist") as excinfo:
        compile_data_designer_config(builder.build(), stub_resource_provider)

    assert isinstance(excinfo.value.__cause__, SeedReaderConfigError)


def test_validation_errors(resource_provider: ResourceProvider) -> None:
    builder = DataDesignerConfigBuilder()
    builder.add_column(
        SamplerColumnConfig(
            name="language",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["english", "french"]),
        )
    )

    with patch("data_designer.engine.compiler.validate_data_designer_config") as patched_validate:
        patched_validate.return_value = [
            Violation(
                type=ViolationType.INVALID_COLUMN,
                message="Some error",
                level=ViolationLevel.ERROR,
            )
        ]

        with pytest.raises(InvalidConfigError) as excinfo:
            compile_data_designer_config(builder.build(), resource_provider)

    assert "validation errors" in str(excinfo)


def test_adds_id_column_when_no_sampler_and_no_seed_dataset(stub_resource_provider: ResourceProvider) -> None:
    """Test that a UUID '_internal_row_id' column is automatically added when there's no sampler column or seed dataset."""
    builder = DataDesignerConfigBuilder()
    builder.add_column(
        ExpressionColumnConfig(
            name="derived_value",
            expr="'constant_value'",
        )
    )
    stub_resource_provider.seed_reader = None

    config = compile_data_designer_config(builder.build(), stub_resource_provider)

    assert len(config.columns) == 2
    assert config.columns[0].name == "_internal_row_id"
    assert isinstance(config.columns[0], SamplerColumnConfig)
    assert config.columns[0].sampler_type == "uuid"
    assert isinstance(config.columns[0].params, UUIDSamplerParams)
    assert config.columns[0].drop is True


def test_does_not_add_id_column_when_sampler_exists(stub_resource_provider: ResourceProvider) -> None:
    """Test that no '_internal_row_id' column is added when a sampler column already exists."""
    builder = DataDesignerConfigBuilder()
    builder.add_column(
        SamplerColumnConfig(
            name="category",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["a", "b", "c"]),
        )
    )
    builder.add_column(
        ExpressionColumnConfig(
            name="derived_value",
            expr="{{ category }}_suffix",
        )
    )
    stub_resource_provider.seed_reader = None

    config = compile_data_designer_config(builder.build(), stub_resource_provider)

    assert len(config.columns) == 2
    assert config.columns[0].name == "category"
    assert config.columns[1].name == "derived_value"
    assert not any(col.name == "_internal_row_id" for col in config.columns)


def test_does_not_add_id_column_when_seed_dataset_exists(resource_provider: ResourceProvider) -> None:
    """Test that no '_internal_row_id' column is added when a seed dataset is configured."""
    builder = DataDesignerConfigBuilder()
    builder.add_column(
        ExpressionColumnConfig(
            name="derived_value",
            expr="{{ city }}_derived",
        )
    )
    builder.with_seed_dataset(HuggingFaceSeedSource(path="hf://datasets/test/data.csv"))

    config = compile_data_designer_config(builder.build(), resource_provider)

    # Should have the expression column + 2 seed columns (city, country) from the fixture
    assert len(config.columns) == 3
    assert config.columns[0].name == "derived_value"
    assert not any(col.name == "_internal_row_id" for col in config.columns)


def test_compile_applies_processor_columns_added(resource_provider: ResourceProvider) -> None:
    """Test that columns declared in columns_added can be referenced by downstream expressions/templates."""
    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(HuggingFaceSeedSource(path="hf://datasets/test/data.csv"))
    builder.add_processor(
        SeedTransformConfig(
            name="pre_batch_add",
            columns_added=["state"],
        )
    )
    builder.add_column(
        ExpressionColumnConfig(
            name="derived_value",
            expr="{{ state }}_processed",
        )
    )

    config = compile_data_designer_config(builder.build(), resource_provider)

    column_names = [col.name for col in config.columns]
    assert "state" in column_names
    assert "city" in column_names
    assert "age" in column_names
    assert "derived_value" in column_names


def test_compile_applies_processor_columns_removed(resource_provider: ResourceProvider) -> None:
    """Test that columns declared in columns_removed are removed and cannot be referenced downstream."""
    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(HuggingFaceSeedSource(path="hf://datasets/test/data.csv"))
    builder.add_processor(
        SeedTransformConfig(
            name="pre_batch_drop",
            columns_removed=["city"],
        )
    )
    builder.add_column(
        ExpressionColumnConfig(
            name="derived_value",
            expr="{{ age }}_processed",
        )
    )

    config = compile_data_designer_config(builder.build(), resource_provider)
    column_names = [col.name for col in config.columns]
    assert "city" not in column_names
    assert "age" in column_names

    # If downstream references the removed column, compilation/validation should fail
    builder_invalid = DataDesignerConfigBuilder()
    builder_invalid.with_seed_dataset(HuggingFaceSeedSource(path="hf://datasets/test/data.csv"))
    builder_invalid.add_processor(
        SeedTransformConfig(
            name="pre_batch_drop",
            columns_removed=["city"],
        )
    )
    builder_invalid.add_column(
        ExpressionColumnConfig(
            name="derived_value",
            expr="{{ city }}_processed",
        )
    )
    with pytest.raises(InvalidConfigError, match="validation errors"):
        compile_data_designer_config(builder_invalid.build(), resource_provider)


def test_compile_processor_columns_added_collision(resource_provider: ResourceProvider) -> None:
    """Test that adding an already existing column via columns_added raises InvalidConfigError."""
    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(HuggingFaceSeedSource(path="hf://datasets/test/data.csv"))
    builder.add_processor(
        SeedTransformConfig(
            name="pre_batch_add",
            columns_added=["city"],
        )
    )

    with pytest.raises(InvalidConfigError, match="collides with an existing column"):
        compile_data_designer_config(builder.build(), resource_provider)


@pytest.mark.parametrize("side_effect_name", ["response__reasoning_content", "response__trace"])
def test_compile_processor_columns_added_side_effect_collision(
    resource_provider: ResourceProvider, side_effect_name: str
) -> None:
    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(HuggingFaceSeedSource(path="hf://datasets/test/data.csv"))
    builder.add_column(
        LLMTextColumnConfig(
            name="response",
            model_alias="test_model",
            prompt="Describe {{ city }}",
            extract_reasoning_content=True,
            with_trace=TraceType.ALL_MESSAGES,
        )
    )
    builder.add_processor(SeedTransformConfig(name="pre_batch_add", columns_added=[side_effect_name]))

    with pytest.raises(InvalidConfigError, match=f"'{side_effect_name}' which collides with an existing column"):
        compile_data_designer_config(builder.build(), resource_provider)


def test_compile_processor_columns_added_duplicate(resource_provider: ResourceProvider) -> None:
    """Test that specifying duplicate columns in columns_added raises InvalidConfigError."""
    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(HuggingFaceSeedSource(path="hf://datasets/test/data.csv"))
    builder.add_processor(
        SeedTransformConfig(
            name="pre_batch_add",
            columns_added=["state", "state"],
        )
    )

    with pytest.raises(InvalidConfigError, match="collides with an existing column"):
        compile_data_designer_config(builder.build(), resource_provider)


def test_compile_processor_columns_removed_nonexistent(resource_provider: ResourceProvider) -> None:
    """Test that removing a non-existent column via columns_removed raises InvalidConfigError."""
    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(HuggingFaceSeedSource(path="hf://datasets/test/data.csv"))
    builder.add_processor(
        SeedTransformConfig(
            name="pre_batch_drop",
            columns_removed=["non_existent"],
        )
    )

    with pytest.raises(InvalidConfigError, match="cannot remove column 'non_existent' because it does not exist"):
        compile_data_designer_config(builder.build(), resource_provider)


def test_compile_processor_columns_added_without_seed_dataset(stub_resource_provider: ResourceProvider) -> None:
    """Test that columns_added without a seed dataset raises InvalidConfigError."""
    builder = DataDesignerConfigBuilder()
    builder.add_processor(
        SeedTransformConfig(
            name="pre_batch_add",
            columns_added=["state"],
        )
    )
    stub_resource_provider.seed_reader = None

    with pytest.raises(InvalidConfigError, match="specifies 'columns_added', but no seed dataset is configured"):
        compile_data_designer_config(builder.build(), stub_resource_provider)


@pytest.mark.parametrize("removed", [[], ["city"], ["missing"]])
@pytest.mark.parametrize("after_generation", [False, True], ids=["post-batch", "after-generation"])
def test_later_stage_declarations_do_not_change_generation_schema(
    resource_provider: ResourceProvider, removed: list[str], after_generation: bool
) -> None:
    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(HuggingFaceSeedSource(path="hf://datasets/test/data.csv"))
    processor = (
        LateTransformConfig(name="late", columns_added=["state"], columns_removed=removed)
        if after_generation
        else DropColumnsProcessorConfig(name="late", column_names=[], columns_added=["state"], columns_removed=removed)
    )
    builder.add_processor(processor)
    builder.add_column(ExpressionColumnConfig(name="derived", expr="{{ city }}"))
    config = compile_data_designer_config(builder.build(), resource_provider)
    assert {column.name for column in config.columns} == {"city", "age", "derived"}

    builder.add_column(ExpressionColumnConfig(name="invalid", expr="{{ state }}"))
    with pytest.raises(InvalidConfigError, match="validation errors"):
        compile_data_designer_config(builder.build(), resource_provider)


@pytest.mark.parametrize("added", [[], ["transformed"]])
@pytest.mark.parametrize("preview", [True, False], ids=["preview", "build"])
@pytest.mark.parametrize("remove_raw", [True, False], ids=["remove-seed", "keep-seed"])
def test_pre_batch_consumes_all_raw_seed_columns(
    stub_resource_provider: ResourceProvider, added: list[str], preview: bool, remove_raw: bool
) -> None:
    source = DataFrameSeedSource(df=lazy.pd.DataFrame({"raw": [1, 2, 3]}))
    reader = DataFrameSeedReader()
    reader.attach(source, PlaintextResolver())
    stub_resource_provider.seed_reader = reader
    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(source, sampling_strategy="ordered")
    builder.add_processor(
        SeedTransformConfig(name="transform", columns_added=added, columns_removed=["raw"] if remove_raw else [])
    )
    builder.add_column(ExpressionColumnConfig(name="output", expr="{{ transformed + 1 }}" if added else "42"))
    dataset_builder = DatasetBuilder(
        data_designer_config=builder.build(),
        resource_provider=stub_resource_provider,
        registry=DataDesignerRegistry(),
    )
    result = dataset_builder.build_preview(num_records=3) if preview else dataset_builder.build(num_records=3)
    if not preview:
        result = dataset_builder.artifact_storage.load_dataset()
    assert ("raw" in result.columns) is (not remove_raw)
    assert result["output"].tolist() == (["3", "5", "7"] if added else ["42", "42", "42"])


def test_pre_batch_cannot_remove_generated_columns(resource_provider: ResourceProvider) -> None:
    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(HuggingFaceSeedSource(path="hf://datasets/test/data.csv"))
    builder.add_column(ExpressionColumnConfig(name="output", expr="42"))
    builder.add_processor(SeedTransformConfig(name="transform", columns_removed=["output"]))
    with pytest.raises(InvalidConfigError, match="can only remove seed columns"):
        compile_data_designer_config(builder.build(), resource_provider)
