# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import CustomColumnConfig, ExpressionColumnConfig, SamplerColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.custom_column import custom_column_generator
from data_designer.config.dataset_metadata import DatasetMetadata
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.config.preview_results import PreviewResults
from data_designer.config.processors import DropColumnsProcessorConfig, SchemaTransformProcessorConfig
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.engine.storage.artifact_storage import ArtifactStorage, BatchStage
from data_designer.interface.composite_workflow import SkippedStageResult
from data_designer.interface.data_designer import DataDesigner
from data_designer.interface.errors import DataDesignerWorkflowError
from data_designer.interface.results import DatasetCreationResults


@pytest.fixture
def stub_artifact_path(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


def _data_designer(artifact_path: Path, model_providers: list[ModelProvider]) -> DataDesigner:
    return DataDesigner(artifact_path=artifact_path, model_providers=model_providers)


def _real_data_designer(artifact_path: Path, model_providers: list[ModelProvider]) -> DataDesigner:
    return DataDesigner(
        artifact_path=artifact_path,
        model_providers=model_providers,
        secret_resolver=PlaintextResolver(),
    )


def _result_from_df(
    artifact_path: Path,
    dataset_name: str,
    df: lazy.pd.DataFrame,
    config_builder: DataDesignerConfigBuilder,
    stub_dataset_profiler_results,
) -> DatasetCreationResults:
    ArtifactStorage.mkdir_if_needed(artifact_path)
    artifact_storage = ArtifactStorage(artifact_path=artifact_path, dataset_name=dataset_name)
    artifact_storage.write_batch_to_parquet_file(0, df, BatchStage.FINAL_RESULT)
    return DatasetCreationResults(
        artifact_storage=artifact_storage,
        analysis=stub_dataset_profiler_results,
        config_builder=config_builder,
        dataset_metadata=DatasetMetadata(),
    )


def _patch_create(data_designer: DataDesigner, stub_dataset_profiler_results) -> MagicMock:
    def fake_create(
        config_builder: DataDesignerConfigBuilder,
        *,
        num_records: int,
        dataset_name: str,
        **kwargs,
    ) -> DatasetCreationResults:
        del kwargs
        df = lazy.pd.DataFrame({"category": ["alpha"] * num_records, "category_copy": ["alpha"] * num_records})
        return _result_from_df(
            data_designer._artifact_path,
            dataset_name,
            df,
            config_builder,
            stub_dataset_profiler_results,
        )

    data_designer.create = MagicMock(side_effect=fake_create)
    return data_designer.create


def _category_builder(model_configs: list[ModelConfig]) -> DataDesignerConfigBuilder:
    builder = DataDesignerConfigBuilder(model_configs=model_configs)
    builder.add_column(
        SamplerColumnConfig(
            name="category",
            sampler_type="category",
            params={"values": ["alpha", "beta", "gamma"]},
        )
    )
    return builder


def _copy_builder(model_configs: list[ModelConfig]) -> DataDesignerConfigBuilder:
    builder = DataDesignerConfigBuilder(model_configs=model_configs)
    builder.add_column(ExpressionColumnConfig(name="category_copy", expr="{{ category }}"))
    return builder


def _seeded_builder(model_configs: list[ModelConfig], rows: list[dict]) -> DataDesignerConfigBuilder:
    builder = DataDesignerConfigBuilder(model_configs=model_configs)
    builder.with_seed_dataset(DataFrameSeedSource(df=lazy.pd.DataFrame(rows)))
    return builder


def test_dataset_creation_results_to_config_builder_columns(
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
    tmp_path: Path,
) -> None:
    results = _result_from_df(
        tmp_path / "artifacts",
        "dataset",
        lazy.pd.DataFrame({"category": ["alpha", "beta", "gamma"], "other": [1, 2, 3]}),
        _category_builder(stub_model_configs),
        stub_dataset_profiler_results,
    )

    builder = results.to_config_builder(columns=["category"])

    seed_config = builder.get_seed_config()
    assert isinstance(seed_config.source, DataFrameSeedSource)
    assert builder.model_configs == stub_model_configs
    assert list(seed_config.source.df.columns) == ["category"]
    assert len(seed_config.source.df) == 3


def test_preview_results_to_config_builder_columns(
    stub_model_configs: list[ModelConfig],
) -> None:
    results = PreviewResults(
        config_builder=_category_builder(stub_model_configs),
        dataset=lazy.pd.DataFrame({"category": ["alpha", "beta"], "other": [1, 2]}),
        dataset_metadata=DatasetMetadata(),
    )

    builder = results.to_config_builder(columns=["category"])

    seed_config = builder.get_seed_config()
    assert isinstance(seed_config.source, DataFrameSeedSource)
    assert list(seed_config.source.df.columns) == ["category"]
    assert len(seed_config.source.df) == 2


def test_composite_workflow_runs_linear_stages_with_disk_handoff(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
) -> None:
    data_designer = _data_designer(stub_artifact_path, stub_model_providers)
    create_mock = _patch_create(data_designer, stub_dataset_profiler_results)
    workflow = data_designer.compose_workflow(name="linear-chain")
    workflow.add_stage("base", _category_builder(stub_model_configs), num_records=3)
    workflow.add_stage("copy", _copy_builder(stub_model_configs))

    results = workflow.run()

    assert set(results.keys()) == {"base", "copy"}
    assert results["base"].count_records() == 3
    assert results["copy"].count_records() == 3
    final_df = results.load_dataset()
    assert "category_copy" in final_df.columns
    assert (stub_artifact_path / "linear-chain" / "stage-0-base").is_dir()
    assert (stub_artifact_path / "linear-chain" / "stage-1-copy").is_dir()

    metadata = json.loads((stub_artifact_path / "linear-chain" / "workflow-metadata.json").read_text())
    assert [stage["status"] for stage in metadata["stages"]] == ["completed", "completed"]
    assert metadata["stages"][1]["seeded_from_stage"] == "base"
    assert metadata["stages"][1]["depends_on"] == ["base"]
    assert metadata["stages"][1]["num_records_requested"] == 3
    assert create_mock.call_args_list[1].kwargs["num_records"] == 3
    second_stage_builder = create_mock.call_args_list[1].args[0]
    seed_config = second_stage_builder.get_seed_config()
    assert isinstance(seed_config.source, LocalFileSeedSource)
    assert seed_config.source.path.endswith("stage-0-base/parquet-files/*.parquet")


def test_composite_workflow_callback_output_controls_next_stage_default_count(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
) -> None:
    data_designer = _data_designer(stub_artifact_path, stub_model_providers)
    _patch_create(data_designer, stub_dataset_profiler_results)

    def keep_first(stage_path: Path) -> Path:
        df = lazy.pd.read_parquet(stage_path / "parquet-files")
        output_path = stage_path / "callback-outputs" / "first-row"
        output_path.mkdir(parents=True)
        df.head(1).to_parquet(output_path / "data.parquet", index=False)
        return output_path

    workflow = data_designer.compose_workflow(name="callback-chain")
    workflow.add_stage(
        "base",
        _category_builder(stub_model_configs),
        num_records=3,
        on_success=keep_first,
        on_success_version="first-row",
    )
    workflow.add_stage("copy", _copy_builder(stub_model_configs))

    results = workflow.run()

    assert results["base"].count_records() == 3
    assert results["copy"].count_records() == 1
    metadata = json.loads((stub_artifact_path / "callback-chain" / "workflow-metadata.json").read_text())
    assert metadata["stages"][0]["output_records"] == 1
    assert metadata["stages"][1]["num_records_requested"] == 1


def test_composite_workflow_explicit_downstream_num_records_supports_explode(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
) -> None:
    data_designer = _data_designer(stub_artifact_path, stub_model_providers)
    create_mock = _patch_create(data_designer, stub_dataset_profiler_results)
    workflow = data_designer.compose_workflow(name="explode-chain")
    workflow.add_stage("base", _category_builder(stub_model_configs), num_records=2)
    workflow.add_stage("expanded", _copy_builder(stub_model_configs), num_records=7)

    results = workflow.run()

    assert results["base"].count_records() == 2
    assert results["expanded"].count_records() == 7
    assert create_mock.call_args_list[1].kwargs["num_records"] == 7


def test_composite_workflow_empty_callback_can_skip_downstream_stages(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
) -> None:
    data_designer = _data_designer(stub_artifact_path, stub_model_providers)
    _patch_create(data_designer, stub_dataset_profiler_results)

    def empty_output(stage_path: Path) -> Path:
        output_path = stage_path / "callback-outputs" / "empty"
        output_path.mkdir(parents=True)
        lazy.pd.DataFrame({"category": []}).to_parquet(output_path / "data.parquet", index=False)
        return output_path

    workflow = data_designer.compose_workflow(name="empty-chain")
    workflow.add_stage(
        "base",
        _category_builder(stub_model_configs),
        num_records=2,
        on_success=empty_output,
        on_success_version="empty",
        allow_empty=True,
    )
    workflow.add_stage("copy", _copy_builder(stub_model_configs))

    results = workflow.run()

    assert isinstance(results["copy"], SkippedStageResult)
    assert results["copy"].upstream_stage == "base"
    with pytest.raises(DataDesignerWorkflowError, match="Final stage 'copy' was skipped"):
        results.load_dataset()


def test_composite_workflow_empty_workflow_fails_before_artifacts(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
) -> None:
    workflow = _data_designer(stub_artifact_path, stub_model_providers).compose_workflow(name="empty-workflow")

    with pytest.raises(DataDesignerWorkflowError, match="has no stages"):
        workflow.run()

    assert not (stub_artifact_path / "empty-workflow").exists()


@pytest.mark.parametrize("name", ["bad/name", "bad*name", ""])
def test_composite_workflow_rejects_invalid_workflow_names(
    name: str,
    stub_model_providers: list[ModelProvider],
) -> None:
    with pytest.raises(DataDesignerWorkflowError):
        DataDesigner(model_providers=stub_model_providers).compose_workflow(name=name)


@pytest.mark.parametrize("name", ["bad/name", "bad*name", ""])
def test_composite_workflow_rejects_invalid_stage_names(
    name: str,
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    workflow = _data_designer(stub_artifact_path, stub_model_providers).compose_workflow(name="invalid-stage")

    with pytest.raises(DataDesignerWorkflowError):
        workflow.add_stage(name, _category_builder(stub_model_configs))


def test_composite_workflow_rejects_duplicate_stage_names(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    data_designer = _data_designer(stub_artifact_path, stub_model_providers)
    workflow = data_designer.compose_workflow(name="duplicate-chain")
    workflow.add_stage("base", _category_builder(stub_model_configs))

    with pytest.raises(DataDesignerWorkflowError, match="already used"):
        workflow.add_stage("base", _copy_builder(stub_model_configs))


def test_composite_workflow_runs_three_real_async_stages(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage_1 = _seeded_builder(stub_model_configs, [{"name": "Ada"}, {"name": "Linus"}])
    stage_1.add_column(ExpressionColumnConfig(name="persona", expr="{{ name }} persona"))

    stage_2 = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    stage_2.add_column(ExpressionColumnConfig(name="prompt_seed", expr="{{ persona }} prompt"))

    stage_3 = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    stage_3.add_column(ExpressionColumnConfig(name="final_label", expr="{{ prompt_seed }} final"))

    workflow = _real_data_designer(tmp_path / "artifacts", stub_model_providers).compose_workflow(name="three-stage")
    workflow.add_stage("personas", stage_1, num_records=2)
    workflow.add_stage("prompts", stage_2)
    workflow.add_stage("final", stage_3)

    df = workflow.run().load_dataset().sort_values("name").reset_index(drop=True)

    assert df[["name", "persona", "prompt_seed", "final_label"]].to_dict(orient="records") == [
        {
            "name": "Ada",
            "persona": "Ada persona",
            "prompt_seed": "Ada persona prompt",
            "final_label": "Ada persona prompt final",
        },
        {
            "name": "Linus",
            "persona": "Linus persona",
            "prompt_seed": "Linus persona prompt",
            "final_label": "Linus persona prompt final",
        },
    ]


def test_composite_workflow_callback_can_expand_rows_between_real_async_stages(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage_1 = _seeded_builder(stub_model_configs, [{"name": "Ada"}, {"name": "Linus"}])
    stage_1.add_column(ExpressionColumnConfig(name="persona", expr="{{ name }}"))

    def expand(stage_path: Path) -> Path:
        df = lazy.pd.read_parquet(stage_path / "parquet-files")
        expanded = lazy.pd.DataFrame([{**row, "turn": turn} for row in df.to_dict(orient="records") for turn in (1, 2)])
        output_path = stage_path / "callback-outputs" / "expand-turns"
        output_path.mkdir(parents=True)
        expanded.to_parquet(output_path / "data.parquet", index=False)
        return output_path

    stage_2 = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    stage_2.add_column(ExpressionColumnConfig(name="message", expr="{{ persona }} turn {{ turn }}"))

    workflow = _real_data_designer(tmp_path / "artifacts", stub_model_providers).compose_workflow(name="expand")
    workflow.add_stage("personas", stage_1, num_records=2, on_success=expand, on_success_version="expand-turns")
    workflow.add_stage("messages", stage_2)

    df = workflow.run().load_dataset().sort_values(["name", "turn"]).reset_index(drop=True)

    assert df["message"].tolist() == [
        "Ada turn 1",
        "Ada turn 2",
        "Linus turn 1",
        "Linus turn 2",
    ]


def test_composite_workflow_does_not_forward_dropped_processor_columns(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage_1 = _seeded_builder(stub_model_configs, [{"name": "Ada", "secret": "hidden"}])
    stage_1.add_column(ExpressionColumnConfig(name="public_name", expr="{{ name }}"))
    stage_1.add_processor(DropColumnsProcessorConfig(name="drop_secret", column_names=["secret"]))

    stage_2 = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    stage_2.add_column(ExpressionColumnConfig(name="copied_name", expr="{{ public_name }}"))

    workflow = _real_data_designer(tmp_path / "artifacts", stub_model_providers).compose_workflow(name="drop-processor")
    workflow.add_stage("redacted", stage_1, num_records=1)
    workflow.add_stage("downstream", stage_2)

    df = workflow.run().load_dataset()

    assert df.to_dict(orient="records") == [{"name": "Ada", "public_name": "Ada", "copied_name": "Ada"}]
    assert "secret" not in df.columns


def test_composite_workflow_can_seed_from_processor_output_callback(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage_1 = _seeded_builder(stub_model_configs, [{"name": "Ada"}, {"name": "Linus"}])
    stage_1.add_column(ExpressionColumnConfig(name="persona", expr="{{ name }}"))
    stage_1.add_processor(SchemaTransformProcessorConfig(name="compact", template={"compact_name": "{{ persona }}"}))

    def use_processor_output(stage_path: Path) -> Path:
        return stage_path / "processors-files" / "compact"

    stage_2 = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    stage_2.add_column(ExpressionColumnConfig(name="final", expr="{{ compact_name }} final"))

    workflow = _real_data_designer(tmp_path / "artifacts", stub_model_providers).compose_workflow(
        name="processor-callback"
    )
    workflow.add_stage("compact", stage_1, num_records=2, on_success=use_processor_output)
    workflow.add_stage("final", stage_2)

    df = workflow.run().load_dataset().sort_values("compact_name").reset_index(drop=True)

    assert df.to_dict(orient="records") == [
        {"compact_name": "Ada", "final": "Ada final"},
        {"compact_name": "Linus", "final": "Linus final"},
    ]


def test_composite_workflow_runs_custom_generator_in_downstream_stage(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    @custom_column_generator(required_columns=["name"])
    def slug(row: dict) -> dict:
        return {**row, "slug": row["name"].lower()}

    stage_1 = _seeded_builder(stub_model_configs, [{"name": "Ada"}, {"name": "Linus"}])
    stage_1.add_column(ExpressionColumnConfig(name="name_copy", expr="{{ name }}"))

    stage_2 = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    stage_2.add_column(CustomColumnConfig(name="slug", generator_function=slug))

    workflow = _real_data_designer(tmp_path / "artifacts", stub_model_providers).compose_workflow(
        name="custom-generator"
    )
    workflow.add_stage("base", stage_1, num_records=2)
    workflow.add_stage("custom", stage_2)

    df = workflow.run().load_dataset().sort_values("name").reset_index(drop=True)

    assert df[["name", "slug"]].to_dict(orient="records") == [
        {"name": "Ada", "slug": "ada"},
        {"name": "Linus", "slug": "linus"},
    ]


def test_composite_workflow_export_defaults_to_final_stage(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
) -> None:
    data_designer = _data_designer(tmp_path / "artifacts", stub_model_providers)

    def fake_create(
        config_builder: DataDesignerConfigBuilder,
        *,
        num_records: int,
        dataset_name: str,
        **kwargs,
    ) -> DatasetCreationResults:
        del num_records, kwargs
        value = "first" if dataset_name == "stage-0-first" else "final"
        return _result_from_df(
            data_designer._artifact_path,
            dataset_name,
            lazy.pd.DataFrame({"stage": [value]}),
            config_builder,
            stub_dataset_profiler_results,
        )

    data_designer.create = MagicMock(side_effect=fake_create)
    first = _seeded_builder(stub_model_configs, [{"stage": "first"}])
    first.add_column(ExpressionColumnConfig(name="stage_copy", expr="{{ stage }}"))
    last = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    last.add_column(ExpressionColumnConfig(name="stage_final", expr="{{ stage }}"))

    workflow = data_designer.compose_workflow(name="export-final")
    workflow.add_stage("first", first, num_records=1)
    workflow.add_stage("last", last)

    output = workflow.run().export(tmp_path / "out.jsonl")

    assert output.read_text(encoding="utf-8").strip() == '{"stage":"final"}'


def test_composite_workflow_push_to_hub_defaults_to_final_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
) -> None:
    data_designer = _data_designer(tmp_path / "artifacts", stub_model_providers)

    def fake_create(
        config_builder: DataDesignerConfigBuilder,
        *,
        num_records: int,
        dataset_name: str,
        **kwargs,
    ) -> DatasetCreationResults:
        del num_records, kwargs
        return _result_from_df(
            data_designer._artifact_path,
            dataset_name,
            lazy.pd.DataFrame({"stage": [dataset_name]}),
            config_builder,
            stub_dataset_profiler_results,
        )

    upload_calls = []

    class StubHubClient:
        def __init__(self, token: str | None = None):
            self.token = token

        def upload_dataset(self, **kwargs):
            upload_calls.append(kwargs)
            return "https://huggingface.co/datasets/user/final"

    data_designer.create = MagicMock(side_effect=fake_create)
    monkeypatch.setattr("data_designer.interface.results.HuggingFaceHubClient", StubHubClient)
    first = _seeded_builder(stub_model_configs, [{"stage": "first"}])
    first.add_column(ExpressionColumnConfig(name="stage_copy", expr="{{ stage }}"))
    last = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    last.add_column(ExpressionColumnConfig(name="stage_final", expr="{{ stage }}"))

    workflow = data_designer.compose_workflow(name="push-final")
    workflow.add_stage("first", first, num_records=1)
    workflow.add_stage("last", last)

    url = workflow.run().push_to_hub("user/final", "description", token="token", private=True, tags=["tag"])

    assert url == "https://huggingface.co/datasets/user/final"
    assert upload_calls == [
        {
            "repo_id": "user/final",
            "base_dataset_path": tmp_path / "artifacts" / "push-final" / "stage-1-last",
            "private": True,
            "description": "description",
            "tags": ["tag"],
        }
    ]
