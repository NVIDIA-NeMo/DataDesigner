# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pyarrow.parquet as pq
import pytest
from pydantic import BaseModel, Field, JsonValue

from data_designer.config.base import ConfigBase
from data_designer.config.column_configs import CustomColumnConfig, ExpressionColumnConfig, LLMStructuredColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.custom_column import custom_column_generator
from data_designer.config.processors import SchemaTransformProcessorConfig
from data_designer.config.run_config import ResumeMode, RunConfig
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.config.utils.storage_schema import JSON_STORAGE_KEY, JSON_STORAGE_RUN_KEY, JsonStoragePlan
from data_designer.engine.column_generators.utils.prompt_renderer import create_response_recipe
from data_designer.engine.dataset_builders.dataset_builder import DatasetBuilder
from data_designer.engine.dataset_builders.errors import ArtifactStorageError, DatasetGenerationError
from data_designer.engine.dataset_builders.utils.processor_runner import ProcessorRunner
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.processing.processors.schema_transform import SchemaTransformProcessor
from data_designer.engine.resources.seed_reader import DataFrameSeedReader, LocalFileSeedReader
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.engine.storage.artifact_storage import ArtifactStorage, BatchStage

PAYLOAD = {"alpha": 1, "nested": {"values": [1, "two", True, None]}, "explicit_null": None}


@custom_column_generator()
def generate_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {**row, "payload": PAYLOAD}


class ToolCall(BaseModel):
    tool: str
    arguments: dict[str, JsonValue] = Field(json_schema_extra={JSON_STORAGE_KEY: "json"})


class Oracle(BaseModel):
    calls: list[ToolCall]


class InspectAfterGeneration(Processor[ConfigBase]):
    def process_after_generation(self, data: pd.DataFrame) -> pd.DataFrame:
        assert all(value == PAYLOAD for value in data["payload"])
        return data.assign(processed=True)


def test_schema_recipe_generates_logical_values_without_storage_metadata(
    stub_resource_provider: Mock, stub_model_configs: list
) -> None:
    config = LLMStructuredColumnConfig(name="oracle", model_alias="test_model", prompt="Generate", output_format=Oracle)
    recipe = create_response_recipe(config)
    example = {"calls": [{"tool": "search", "arguments": PAYLOAD}]}
    assert JSON_STORAGE_KEY not in recipe.schema
    assert recipe.parse(json.dumps(example)) == example
    builder_config = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    builder_config.add_column(config)
    builder = DatasetBuilder(builder_config.build(), stub_resource_provider)
    assert builder.artifact_storage.json_storage.columns == {"oracle": [["calls", None, "arguments"]]}
    saved_column = next(column for column in builder.data_designer_config.columns if column.name == "oracle")
    assert JSON_STORAGE_KEY in json.dumps(saved_column.output_format)


def test_custom_pipeline_preview_create_resume_and_dropped_columns(stub_resource_provider: Mock) -> None:
    provider = stub_resource_provider
    provider.seed_reader = None
    provider.run_config = RunConfig(buffer_size=2, max_concurrent_row_groups=1)
    config = DataDesignerConfigBuilder()
    config.add_column(CustomColumnConfig(name="payload", generator_function=generate_payload, storage_encoding="json"))
    builder = DatasetBuilder(config.build(), provider)
    preview = builder.build_preview(num_records=2)
    assert preview["payload"].tolist() == [PAYLOAD, PAYLOAD]
    builder.build(num_records=3)
    assert builder.artifact_storage.load_dataset()["payload"].tolist() == [PAYLOAD] * 3
    raw = pq.read_table(builder.artifact_storage.final_dataset_path).to_pylist()
    assert all(isinstance(row["payload"], str) for row in raw)
    assert (
        builder.artifact_storage.read_metadata()[JSON_STORAGE_RUN_KEY]
        == builder.artifact_storage.json_storage.model_dump()
    )
    builder.build(num_records=5, resume=ResumeMode.ALWAYS)
    assert builder.artifact_storage.load_dataset()["payload"].tolist() == [PAYLOAD] * 5
    # Internal dropped UUID columns remain readable alongside encoded main data.
    assert len(builder.artifact_storage.load_dataset_with_dropped_columns()) == 5


def test_after_generation_reads_logical_objects_and_reencodes(stub_resource_provider: Mock) -> None:
    storage = stub_resource_provider.artifact_storage
    storage.json_storage = JsonStoragePlan(columns={"payload": [[]]})
    for batch in range(2):
        storage.write_batch_to_parquet_file(batch, pd.DataFrame({"payload": [PAYLOAD]}), BatchStage.FINAL_RESULT)
    processor = InspectAfterGeneration(ConfigBase(), stub_resource_provider)
    ProcessorRunner([processor], storage).run_after_generation(batch_size=1)
    result = storage.load_dataset()
    assert result["payload"].tolist() == [PAYLOAD, PAYLOAD]
    assert result["processed"].tolist() == [True, True]


@pytest.mark.parametrize("source_kind", ["file", "glob", "dataframe"])
@pytest.mark.parametrize("nested", [False, True])
def test_seed_readers_decode_and_preserve_contract(tmp_path: Path, source_kind: str, nested: bool) -> None:
    paths = [["calls", None, "arguments"]] if nested else [[]]
    value = {"calls": [{"arguments": PAYLOAD}]} if nested else PAYLOAD
    storage = ArtifactStorage(artifact_path=tmp_path, json_storage=JsonStoragePlan(columns={"payload": paths}))
    file = storage.write_batch_to_parquet_file(0, pd.DataFrame({"payload": [value]}), BatchStage.FINAL_RESULT)
    if source_kind == "dataframe":
        reader = DataFrameSeedReader()
        reader.attach(DataFrameSeedSource(df=storage.load_dataset()), PlaintextResolver())
    else:
        reader = LocalFileSeedReader()
        reader.attach(
            LocalFileSeedSource(path=str(file if source_kind == "file" else file.parent / "*.parquet")),
            PlaintextResolver(),
        )
    assert reader.get_storage_plan() == storage.json_storage
    batch = reader.create_batch_reader(batch_size=1, index_range=None, shuffle=False).read_next_batch().to_pandas()
    assert batch.iloc[0]["payload"] == value
    assert reader.get_column_names() == ["payload"]


def test_seed_handoff_reencodes_in_next_pipeline(stub_resource_provider: Mock, tmp_path: Path) -> None:
    source = ArtifactStorage(
        artifact_path=tmp_path, dataset_name="source", json_storage=JsonStoragePlan(columns={"payload": [[]]})
    )
    path = source.write_batch_to_parquet_file(0, pd.DataFrame({"payload": [PAYLOAD]}), BatchStage.FINAL_RESULT)
    reader = LocalFileSeedReader()
    reader.attach(LocalFileSeedSource(path=str(path)), PlaintextResolver())
    provider = stub_resource_provider
    provider.seed_reader = reader
    provider.artifact_storage = ArtifactStorage(artifact_path=tmp_path, dataset_name="next")
    config = DataDesignerConfigBuilder().with_seed_dataset(LocalFileSeedSource(path=str(path)))
    config.add_column(ExpressionColumnConfig(name="copied", expr="{{ payload.alpha }}", dtype="int"))
    builder = DatasetBuilder(config.build(), provider)
    builder.build(num_records=1)
    assert builder.artifact_storage.load_dataset().iloc[0]["payload"] == PAYLOAD
    assert builder.artifact_storage.load_dataset().iloc[0]["copied"] == 1


@pytest.mark.parametrize("value", [[1, "two", True, None], {}])
def test_unannotated_unserializable_column_has_actionable_error(tmp_path: Path, value: Any) -> None:
    storage = ArtifactStorage(artifact_path=tmp_path)
    with pytest.raises(
        ArtifactStorageError, match="column 'payload'.*storage_encoding='json'.*x-data-designer-storage"
    ):
        storage.write_batch_to_parquet_file(0, pd.DataFrame({"payload": [value]}), BatchStage.FINAL_RESULT)


def test_valid_legacy_writes_keep_native_structure(tmp_path: Path) -> None:
    storage = ArtifactStorage(artifact_path=tmp_path)
    path = storage.write_batch_to_parquet_file(
        0, pd.DataFrame({"payload": [{"alpha": 1}, {"beta": 2}]}), BatchStage.FINAL_RESULT
    )
    assert pq.read_table(path).to_pylist() == [
        {"payload": {"alpha": 1, "beta": None}},
        {"payload": {"alpha": None, "beta": 2}},
    ]


def test_resume_rejects_changed_encoding_before_writes(stub_resource_provider: Mock) -> None:
    provider = stub_resource_provider
    provider.seed_reader = None
    config = DataDesignerConfigBuilder()
    config.add_column(CustomColumnConfig(name="payload", generator_function=generate_payload, storage_encoding="json"))
    builder = DatasetBuilder(config.build(), provider)
    builder.build(num_records=1)
    original = list(builder.artifact_storage.final_dataset_path.glob("*.parquet"))
    builder.artifact_storage.json_storage = JsonStoragePlan()
    with pytest.raises(DatasetGenerationError):
        builder.build(num_records=2, resume=ResumeMode.ALWAYS)
    assert list(builder.artifact_storage.final_dataset_path.glob("*.parquet")) == original


def test_processor_outputs_have_independent_encoding(stub_resource_provider: Mock) -> None:
    config = SchemaTransformProcessorConfig(
        name="transformed",
        template={"new_payload": {"values": [1, "two"], "alpha": "{{ payload.alpha }}"}},
        json_storage_columns=["new_payload"],
    )
    processor = SchemaTransformProcessor(config, stub_resource_provider)
    frame = pd.DataFrame({"payload": [PAYLOAD]})
    result = processor.process_after_batch(frame, current_batch_number=0)
    assert result is frame
    saved = stub_resource_provider.artifact_storage.load_processor_dataset("transformed")
    assert saved.iloc[0]["new_payload"] == {"values": [1, "two"], "alpha": "1"}


@pytest.mark.parametrize("encoded", [False, True])
def test_failed_write_preserves_existing_artifact_and_cleans_temporary_files(tmp_path: Path, encoded: bool) -> None:
    storage = ArtifactStorage(
        artifact_path=tmp_path,
        json_storage=JsonStoragePlan(columns={"payload": [[]]}) if encoded else JsonStoragePlan(),
    )
    good = {"alpha": 1}
    file = storage.write_batch_to_parquet_file(0, pd.DataFrame({"payload": [good]}), BatchStage.FINAL_RESULT)
    bad = {1, 2} if encoded else [1, "two"]
    with pytest.raises(ArtifactStorageError):
        storage.write_batch_to_parquet_file(0, pd.DataFrame({"payload": [bad]}), BatchStage.FINAL_RESULT)
    assert storage.load_dataset().iloc[0]["payload"] == good
    assert list(file.parent.iterdir()) == [file]


@pytest.mark.parametrize("encoded", ["main", "dropped", "both", "neither"])
def test_combined_main_and_dropped_load_preserves_contract_for_dataframe_seeds(tmp_path: Path, encoded: str) -> None:
    columns = {}
    if encoded in ("main", "both"):
        columns["main_value"] = [[]]
    if encoded in ("dropped", "both"):
        columns["dropped_value"] = [[]]
    plan = JsonStoragePlan(columns=columns)
    main_values = [None, {"args": [1, "two", True, None]}, {}] if "main_value" in columns else ["one", None, "three"]
    dropped_values = (
        [None, {"nested": {"alpha": 1, "mixed": [True, "two", None]}}, {}]
        if "dropped_value" in columns
        else ["id0", None, "id2"]
    )
    main = pd.DataFrame({"main_value": pd.Series(main_values, dtype=object)})
    dropped = pd.DataFrame({"dropped_value": pd.Series(dropped_values, dtype=object)})
    expected = [
        {"main_value": main_value, "dropped_value": dropped_value}
        for main_value, dropped_value in zip(main_values, dropped_values)
    ]
    storage = ArtifactStorage(artifact_path=tmp_path, json_storage=plan)
    storage.write_batch_to_parquet_file(0, main, BatchStage.FINAL_RESULT)
    storage.write_batch_to_parquet_file(0, dropped, BatchStage.DROPPED_COLUMNS)

    combined = storage.load_dataset_with_dropped_columns()
    assert combined.to_dict("records") == expected
    if columns:
        assert combined.attrs[JSON_STORAGE_RUN_KEY] == plan.model_dump()
    else:
        assert JSON_STORAGE_RUN_KEY not in combined.attrs

    reader = DataFrameSeedReader()
    reader.attach(DataFrameSeedSource(df=combined), PlaintextResolver())
    assert reader.get_storage_plan() == plan
    restored = reader.create_batch_reader(batch_size=3, index_range=None, shuffle=False).read_next_batch().to_pandas()
    assert restored.astype(object).where(restored.notna(), None).to_dict("records") == expected
    for name in columns:
        assert restored[name].tolist() == combined[name].tolist()
    assert main["main_value"].tolist() == main_values
    assert dropped["dropped_value"].tolist() == dropped_values


@pytest.mark.parametrize("source_kind", ["file", "dataframe"])
@pytest.mark.parametrize("paths", [[[]], [["arguments"]], [["calls", None, "arguments"]]])
def test_seed_readers_preserve_null_encoded_values(
    tmp_path: Path, source_kind: str, paths: list[list[str | None]]
) -> None:
    storage = ArtifactStorage(artifact_path=tmp_path, json_storage=JsonStoragePlan(columns={"payload": paths}))
    if paths == [[]]:
        values = [None, PAYLOAD, None]
    elif paths == [["arguments"]]:
        values = [None, {"arguments": None}, {"arguments": PAYLOAD}]
    else:
        values = [None, {"calls": None}, {"calls": [None, {"arguments": None}, {"arguments": PAYLOAD}]}]
    file = storage.write_batch_to_parquet_file(
        0, pd.DataFrame({"payload": pd.Series(values, dtype=object)}), BatchStage.FINAL_RESULT
    )
    if source_kind == "file":
        reader = LocalFileSeedReader()
        reader.attach(LocalFileSeedSource(path=str(file)), PlaintextResolver())
    else:
        reader = DataFrameSeedReader()
        reader.attach(DataFrameSeedSource(df=storage.load_dataset()), PlaintextResolver())
    batches = reader.create_batch_reader(batch_size=1, index_range=None, shuffle=False)
    restored = [batches.read_next_batch().to_pandas().iloc[0]["payload"] for _ in values]
    assert restored == values


@pytest.mark.parametrize("source_kind", ["file", "dataframe"])
@pytest.mark.parametrize("nested", [False, True])
def test_seed_readers_preserve_all_null_encoded_columns(tmp_path: Path, source_kind: str, nested: bool) -> None:
    storage = ArtifactStorage(
        artifact_path=tmp_path,
        json_storage=JsonStoragePlan(columns={"payload": [["arguments"]] if nested else [[]]}),
    )
    file = storage.write_batch_to_parquet_file(
        0, pd.DataFrame({"payload": pd.Series([None], dtype=object)}), BatchStage.FINAL_RESULT
    )
    if source_kind == "file":
        reader = LocalFileSeedReader()
        reader.attach(LocalFileSeedSource(path=str(file)), PlaintextResolver())
    else:
        reader = DataFrameSeedReader()
        reader.attach(DataFrameSeedSource(df=storage.load_dataset()), PlaintextResolver())
    batch = reader.create_batch_reader(batch_size=1, index_range=None, shuffle=False).read_next_batch().to_pandas()
    assert batch["payload"].tolist() == [None]


@pytest.mark.parametrize("encoded_sibling", [False, True])
@pytest.mark.parametrize("value", [2**64, 2**100, -(2**100)])
def test_native_integer_overflow_has_actionable_column_error(tmp_path: Path, encoded_sibling: bool, value: int) -> None:
    storage = ArtifactStorage(
        artifact_path=tmp_path,
        json_storage=JsonStoragePlan(columns={"encoded": [[]]}) if encoded_sibling else JsonStoragePlan(),
    )
    frame = pd.DataFrame({"payload": pd.Series([value], dtype=object), "encoded": [{"value": 1}]})
    with pytest.raises(ArtifactStorageError, match="column 'payload'.*storage_encoding='json'"):
        storage.write_batch_to_parquet_file(0, frame, BatchStage.FINAL_RESULT)
    assert not list(storage.final_dataset_path.iterdir())
