# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pyarrow.parquet as pq
import pytest
from pydantic import BaseModel, Field, JsonValue

from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults
from data_designer.config.column_configs import (
    CustomColumnConfig,
    ExpressionColumnConfig,
    GenerationStrategy,
    LLMStructuredColumnConfig,
)
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.custom_column import custom_column_generator
from data_designer.config.dataset_metadata import DatasetMetadata
from data_designer.config.errors import InvalidFileFormatError
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.config.run_config import RunConfig
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.config.utils.io_helpers import read_parquet_dataset
from data_designer.config.utils.storage_schema import JSON_STORAGE_METADATA_KEY, JSON_STORAGE_RUN_KEY, JsonStoragePlan
from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.engine.storage.artifact_storage import ArtifactStorage, BatchStage
from data_designer.interface.data_designer import DataDesigner
from data_designer.interface.results import DatasetCreationResults


@pytest.fixture
def json_results(tmp_path: Path, stub_dataset_profiler_results: DatasetProfilerResults) -> DatasetCreationResults:
    storage = ArtifactStorage(artifact_path=tmp_path, json_storage=JsonStoragePlan(columns={"payload": [[]]}))
    for batch, value in enumerate([{"alpha": 1}, {"beta": None}, {"mixed": [1, "two", True, None]}]):
        storage.write_batch_to_parquet_file(batch, pd.DataFrame({"payload": [value]}), BatchStage.FINAL_RESULT)
    return DatasetCreationResults(
        artifact_storage=storage,
        analysis=stub_dataset_profiler_results,
        config_builder=DataDesignerConfigBuilder(),
        dataset_metadata=DatasetMetadata(),
        requested_num_records=3,
    )


def test_result_loading_and_dataframe_seed_keep_logical_values(json_results: DatasetCreationResults) -> None:
    values = json_results.load_dataset()["payload"].tolist()
    assert values == [{"alpha": 1}, {"beta": None}, {"mixed": [1, "two", True, None]}]
    assert json_results.count_records() == 3
    seed = json_results.to_config_builder()
    assert seed.build().seed_config.source.df["payload"].tolist() == values


def test_result_seed_projection_does_not_encode_reused_column_name(
    json_results: DatasetCreationResults, tmp_path: Path, stub_model_providers: list[ModelProvider]
) -> None:
    storage = json_results.artifact_storage
    for file in storage.final_dataset_path.glob("*.parquet"):
        file.unlink()
    storage.write_batch_to_parquet_file(
        0, pd.DataFrame({"payload": [{"alpha": 1}], "question": ["hi"]}), BatchStage.FINAL_RESULT
    )
    config = json_results.to_config_builder(columns=["question"])
    assert config.build().seed_config.source.df.attrs[JSON_STORAGE_RUN_KEY] == JsonStoragePlan().model_dump()
    config.add_column(ExpressionColumnConfig(name="payload", expr="new", dtype="str"))
    managed = tmp_path / "managed"
    managed.mkdir()
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=managed,
    )
    designer.set_run_config(RunConfig(otel_metrics_port=None))

    results = designer.create(config, num_records=1, dataset_name="projected")
    assert pq.read_table(results.artifact_storage.final_dataset_path).to_pylist() == [
        {"question": "hi", "payload": "new"}
    ]
    assert not results.artifact_storage.json_storage.columns
    assert JSON_STORAGE_RUN_KEY not in results.artifact_storage.read_metadata()


def test_jsonl_export_restores_nested_json(json_results: DatasetCreationResults, tmp_path: Path) -> None:
    output = json_results.export(tmp_path / "export.jsonl")
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records == [{"payload": value} for value in json_results.load_dataset()["payload"]]
    assert all(isinstance(record["payload"], dict) for record in records)


@pytest.mark.parametrize(
    "separator", ["\u0085", "\u2028", "\u2029"], ids=["nel", "line-separator", "paragraph-separator"]
)
@pytest.mark.parametrize("nested", [False, True], ids=["root", "nested"])
def test_jsonl_export_preserves_unicode_separators_in_protected_and_native_strings(
    json_results: DatasetCreationResults, tmp_path: Path, separator: str, nested: bool
) -> None:
    storage = json_results.artifact_storage
    for file in storage.final_dataset_path.glob("*.parquet"):
        file.unlink()
    storage.json_storage = JsonStoragePlan(columns={"payload": [["arguments"]] if nested else [[]]})
    text = f"before{separator}after\nnewline"
    payload = {"arguments": {"text": text}} if nested else {"text": text}
    expected = [{"payload": payload, "native": text}, {"payload": payload, "native": "last"}]
    for batch, record in enumerate(expected):
        storage.write_batch_to_parquet_file(batch, pd.DataFrame([record]), BatchStage.FINAL_RESULT)

    output = json_results.export(tmp_path / "unicode.jsonl")

    with output.open(encoding="utf-8") as stream:
        assert [json.loads(line) for line in stream] == expected


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("value", [None, {"precise": 1.2345678901234567, "large": 2**100, "small": 1e-100}])
def test_jsonl_export_preserves_json_numbers_and_nulls(
    json_results: DatasetCreationResults, tmp_path: Path, nested: bool, value: Any
) -> None:
    storage = json_results.artifact_storage
    for file in storage.final_dataset_path.glob("*.parquet"):
        file.unlink()
    storage.json_storage = JsonStoragePlan(columns={"payload": [["arguments"]] if nested else [[]]})
    payload = {"arguments": value} if nested else value
    storage.write_batch_to_parquet_file(
        0,
        pd.DataFrame({"payload": [payload], "date": [pd.Timestamp("2026-10-20")], "missing": [float("nan")]}),
        BatchStage.FINAL_RESULT,
    )
    record = json.loads(json_results.export(tmp_path / "precise.jsonl").read_text())
    assert record == {"payload": payload, "date": "2026-10-20T00:00:00.000", "missing": None}


def test_csv_export_has_json_text(json_results: DatasetCreationResults, tmp_path: Path) -> None:
    output = json_results.export(tmp_path / "export.csv")
    # CSV is a text export; structured cells must be valid JSON, not Python reprs.
    values = [json.loads(value) for value in pd.read_csv(output)["payload"]]
    assert values == json_results.load_dataset()["payload"].tolist()


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
def test_csv_structured_cells_normalize_native_nonfinite_siblings_without_rounding_json(
    json_results: DatasetCreationResults, tmp_path: Path, nonfinite: float
) -> None:
    storage = json_results.artifact_storage
    for file in storage.final_dataset_path.glob("*.parquet"):
        file.unlink()
    storage.json_storage = JsonStoragePlan(columns={"payload": [["arguments"]]})
    exact = {"precise": 1.2345678901234567, "large": 2**100, "small": 1e-100, "text": "NaN"}
    storage.write_batch_to_parquet_file(
        0,
        pd.DataFrame(
            {
                "payload": [
                    {"arguments": exact, "native": {"value": nonfinite, "items": [nonfinite]}},
                    {"arguments": {}, "native": {"value": 1.0, "items": [1.0]}},
                ],
                "ordinary": ["NaN", "Infinity"],
                "native_scalar": [nonfinite, 1.0],
            }
        ),
        BatchStage.FINAL_RESULT,
    )
    output = json_results.export(tmp_path / "strict.csv")
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    restored = [json.loads(row["payload"], parse_constant=_reject_nonfinite_json_constant) for row in rows]
    assert restored == [
        {"arguments": exact, "native": {"value": None, "items": [None]}},
        {"arguments": {}, "native": {"value": 1.0, "items": [1.0]}},
    ]
    assert [row["ordinary"] for row in rows] == ["NaN", "Infinity"]
    baseline = tmp_path / "native-baseline.csv"
    json_results.load_dataset()[["native_scalar"]].to_csv(baseline, index=False)
    with baseline.open(newline="") as stream:
        expected_native = [row["native_scalar"] for row in csv.DictReader(stream)]
    assert [row["native_scalar"] for row in rows] == expected_native


def test_parquet_export_preserves_contract(json_results: DatasetCreationResults, tmp_path: Path) -> None:
    output = json_results.export(tmp_path / "export.parquet")
    assert JSON_STORAGE_METADATA_KEY in pq.read_schema(output).metadata
    assert read_parquet_dataset(output)["payload"].tolist() == json_results.load_dataset()["payload"].tolist()
    assert all(isinstance(row["payload"], str) for row in pq.read_table(output).to_pylist())


def test_parquet_export_rejects_mixed_contracts(json_results: DatasetCreationResults, tmp_path: Path) -> None:
    pd.DataFrame({"payload": ['{"legacy":1}']}).to_parquet(
        json_results.artifact_storage.final_dataset_path / "batch_99999.parquet"
    )
    with pytest.raises(InvalidFileFormatError, match="Incompatible"):
        json_results.export(tmp_path / "export.parquet")


@custom_column_generator()
def create_tool_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {**row, "payload": {"args": {"mixed": [1, "two", True, None]}}}


class AnnotatedCall(BaseModel):
    tool: str
    arguments: dict[str, JsonValue] = Field(json_schema_extra={"x-data-designer-storage": "json"})


class AnnotatedOracle(BaseModel):
    calls: list[AnnotatedCall]


@pytest.mark.parametrize("structured", [False, True], ids=["custom", "structured"])
def test_public_create_profiles_and_exports_json_storage(
    tmp_path: Path, stub_model_providers: list[ModelProvider], structured: bool
) -> None:
    managed = tmp_path / "managed"
    managed.mkdir()
    designer = DataDesigner(
        artifact_path=tmp_path / "artifacts",
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=managed,
    )
    designer.set_run_config(RunConfig(otel_metrics_port=None, buffer_size=2))
    config = DataDesignerConfigBuilder()
    expected: dict[str, Any]
    if structured:
        config.add_model_config(
            ModelConfig(
                alias="test",
                model="test",
                provider=stub_model_providers[0].name,
                skip_health_check=True,
            )
        )
        config.add_column(
            LLMStructuredColumnConfig(
                name="payload", prompt="Generate", model_alias="test", output_format=AnnotatedOracle
            )
        )
        expected = {"calls": [{"tool": "search", "arguments": {"query": "weather", "values": [1, "two", None]}}]}
    else:
        config.add_column(
            CustomColumnConfig(name="payload", generator_function=create_tool_payload, storage_encoding="json")
        )
        expected = {"args": {"mixed": [1, "two", True, None]}}

    async def model_response(facade: ModelFacade, **kwargs: Any) -> tuple[dict, list]:
        assert "x-data-designer-storage" not in str(kwargs)
        return kwargs["parser"](json.dumps(expected)), []

    with patch.object(ModelFacade, "agenerate", autospec=True, side_effect=model_response):
        results = designer.create(config, num_records=3, dataset_name="tool_calls")
        preview = designer.preview(config, num_records=3)
    assert results.load_dataset()["payload"].tolist() == [expected] * 3
    assert isinstance(results.load_analysis(), DatasetProfilerResults)
    output = results.export(tmp_path / "logical.jsonl")
    assert [json.loads(line)["payload"] for line in output.read_text().splitlines()] == [expected] * 3
    assert preview.dataset["payload"].tolist() == [expected] * 3
    assert preview.dataset.attrs[JSON_STORAGE_RUN_KEY] == results.artifact_storage.json_storage.model_dump()
    seeded = preview.to_config_builder()
    seeded.add_column(ExpressionColumnConfig(name="question", expr="hi", dtype="str"))
    reseeded = designer.create(seeded, num_records=3, dataset_name="preview_seed")
    assert reseeded.load_dataset()["payload"].tolist() == [expected] * 3
    assert reseeded.artifact_storage.json_storage == results.artifact_storage.json_storage
    inherited_preview = designer.preview(seeded, num_records=3)
    assert inherited_preview.dataset["payload"].tolist() == [expected] * 3
    assert inherited_preview.dataset.attrs[JSON_STORAGE_RUN_KEY] == preview.dataset.attrs[JSON_STORAGE_RUN_KEY]


@pytest.mark.parametrize("drop_payload", [False, True], ids=["projected", "dropped"])
def test_preview_seed_projection_preserves_source_and_does_not_encode_reused_name(
    tmp_path: Path, stub_model_providers: list[ModelProvider], drop_payload: bool
) -> None:
    managed = tmp_path / "managed"
    managed.mkdir()
    designer = DataDesigner(
        artifact_path=tmp_path / "artifacts",
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=managed,
    )
    designer.set_run_config(RunConfig(otel_metrics_port=None))
    config = DataDesignerConfigBuilder()
    config.add_column(
        CustomColumnConfig(
            name="payload", generator_function=create_tool_payload, storage_encoding="json", drop=drop_payload
        )
    )
    config.add_column(ExpressionColumnConfig(name="question", expr="hi", dtype="str"))

    preview = designer.preview(config, num_records=1)
    expected_plan = JsonStoragePlan(columns={} if drop_payload else {"payload": [[]]})
    if drop_payload:
        assert JSON_STORAGE_RUN_KEY not in preview.dataset.attrs
        assert "payload" not in preview.dataset.columns
    else:
        assert preview.dataset.attrs[JSON_STORAGE_RUN_KEY] == expected_plan.model_dump()
    original_attrs = preview.dataset.attrs.copy()
    seeded = preview.to_config_builder(columns=["question"])
    assert preview.dataset.attrs == original_attrs
    seeded_plan = seeded.build().seed_config.source.df.attrs.get(JSON_STORAGE_RUN_KEY, {})
    assert JsonStoragePlan.model_validate(seeded_plan) == JsonStoragePlan()
    seeded.add_column(ExpressionColumnConfig(name="payload", expr="new", dtype="str"))

    results = designer.create(seeded, num_records=1, dataset_name="projected_preview")
    assert not results.artifact_storage.json_storage.columns
    assert pq.read_table(results.artifact_storage.final_dataset_path).to_pylist() == [
        {"question": "hi", "payload": "new"}
    ]


def _reject_nonfinite_json_constant(value: str) -> None:
    pytest.fail(f"Structured CSV cell contains invalid JSON constant {value}")


class ScalarPayloadParams(BaseModel):
    values: list[JsonValue]


@custom_column_generator(required_columns=["id"])
def create_scalar_payload(row: dict[str, Any], generator_params: ScalarPayloadParams) -> dict[str, Any]:
    return {**row, "payload": generator_params.values[row["id"]]}


@custom_column_generator(required_columns=["payload"])
def inspect_scalar_payload(df: pd.DataFrame) -> pd.DataFrame:
    df["observed"] = [json.dumps(value, allow_nan=False) for value in df["payload"]]
    return df


@pytest.mark.parametrize(
    "values",
    [[None, 1], [1.25, None], [2**60 + 1, 1.25], [2**1024, 1.25], [None, None]],
    ids=["null-integer", "float-null", "precise-integer-float", "huge-integer-float", "all-null"],
)
def test_public_generation_preserves_json_scalar_types_before_batch_generators_and_storage(
    tmp_path: Path, stub_model_providers: list[ModelProvider], values: list[JsonValue]
) -> None:
    managed = tmp_path / "managed"
    managed.mkdir()
    designer = DataDesigner(
        artifact_path=tmp_path / "artifacts",
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=managed,
    )
    designer.set_run_config(RunConfig(otel_metrics_port=None, buffer_size=len(values)))
    config = DataDesignerConfigBuilder().with_seed_dataset(DataFrameSeedSource(df=pd.DataFrame({"id": [0, 1]})))
    config.add_column(
        CustomColumnConfig(
            name="payload",
            generator_function=create_scalar_payload,
            generator_params=ScalarPayloadParams(values=values),
            storage_encoding="json",
        )
    )
    config.add_column(
        CustomColumnConfig(
            name="observed",
            generator_function=inspect_scalar_payload,
            generation_strategy=GenerationStrategy.FULL_COLUMN,
        )
    )

    preview = designer.preview(config, num_records=len(values))
    results = designer.create(config, num_records=len(values), dataset_name="scalar_json")
    for frame in (preview.dataset, results.load_dataset()):
        assert [(type(value), value) for value in frame["payload"]] == [(type(value), value) for value in values]
        assert frame["observed"].tolist() == [json.dumps(value, allow_nan=False) for value in values]
    with results.export(tmp_path / "scalars.csv").open(newline="") as stream:
        assert [row["payload"] for row in csv.DictReader(stream)] == [
            "" if value is None else str(value) for value in values
        ]
    exported_jsonl = results.export(tmp_path / "scalars.jsonl")
    json_values = [json.loads(line)["payload"] for line in exported_jsonl.read_text().splitlines()]
    assert [(type(value), value) for value in json_values] == [(type(value), value) for value in values]
    preview_seed = preview.to_config_builder()
    preview_seed.add_column(ExpressionColumnConfig(name="question", expr="hi", dtype="str"))
    handoff = designer.create(preview_seed, num_records=len(values), dataset_name="scalar_preview_seed")
    assert [(type(value), value) for value in handoff.load_dataset()["payload"]] == [
        (type(value), value) for value in values
    ]
    exported = results.export(tmp_path / "scalars.parquet")
    file_config = DataDesignerConfigBuilder().with_seed_dataset(LocalFileSeedSource(path=str(exported)))
    file_config.add_column(ExpressionColumnConfig(name="question", expr="hi", dtype="str"))
    file_preview = designer.preview(file_config, num_records=len(values))
    file_results = designer.create(file_config, num_records=len(values), dataset_name="scalar_file_seed")
    for frame in (file_preview.dataset, file_results.load_dataset()):
        assert [(type(value), value) for value in frame["payload"]] == [(type(value), value) for value in values]
