# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shutil
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
from data_designer.engine.storage.artifact_storage import ArtifactStorage, BatchStage, ResumeMode
from data_designer.interface.composite_workflow import RepeatUntil, SkippedStageResult, SkippedStageStatus
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
        artifact_path = Path(kwargs.pop("artifact_path", data_designer.artifact_path))
        del kwargs
        df = lazy.pd.DataFrame({"category": ["alpha"] * num_records, "category_copy": ["alpha"] * num_records})
        return _result_from_df(
            artifact_path,
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


def _expression_builder(model_configs: list[ModelConfig], name: str, expr: str) -> DataDesignerConfigBuilder:
    builder = DataDesignerConfigBuilder(model_configs=model_configs)
    builder.add_column(ExpressionColumnConfig(name=name, expr=expr))
    return builder


def _copy_builder(model_configs: list[ModelConfig]) -> DataDesignerConfigBuilder:
    return _expression_builder(model_configs, "category_copy", "{{ category }}")


def _seeded_builder(model_configs: list[ModelConfig], rows: list[dict]) -> DataDesignerConfigBuilder:
    builder = DataDesignerConfigBuilder(model_configs=model_configs)
    builder.with_seed_dataset(DataFrameSeedSource(df=lazy.pd.DataFrame(rows)))
    return builder


def _load_workflow_metadata(artifact_path: Path, workflow_name: str) -> dict:
    return json.loads((artifact_path / workflow_name / "workflow-metadata.json").read_text())


def _mark_stage_resumable(metadata: dict, index: int, status: str) -> None:
    metadata["stages"][index]["status"] = status
    for key in (
        "num_records_actual",
        "output_records",
        "output_seed_path",
        "callback_output_path",
        "output_processor_output_path",
        "duration_sec",
    ):
        metadata["stages"][index].pop(key, None)


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
    assert data_designer.artifact_path == stub_artifact_path

    metadata = _load_workflow_metadata(stub_artifact_path, "linear-chain")
    assert [stage["status"] for stage in metadata["stages"]] == ["completed", "completed"]
    assert metadata["stages"][1]["seeded_from_stage"] == "base"
    assert metadata["stages"][1]["depends_on"] == ["base"]
    assert metadata["stages"][1]["num_records_requested"] == 3
    assert create_mock.call_args_list[1].kwargs["num_records"] == 3
    assert create_mock.call_args_list[1].kwargs["artifact_path"] == stub_artifact_path / "linear-chain"
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
    metadata = _load_workflow_metadata(stub_artifact_path, "callback-chain")
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
    assert results["copy"].status == SkippedStageStatus.SKIPPED_EMPTY_UPSTREAM
    assert results["copy"].upstream_stage == "base"
    with pytest.raises(DataDesignerWorkflowError, match="Final stage 'copy' was skipped"):
        results.load_dataset()


def test_composite_workflow_empty_callback_fails_by_default(
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

    workflow = data_designer.compose_workflow(name="empty-default")
    workflow.add_stage("base", _category_builder(stub_model_configs), num_records=2, on_success=empty_output)

    with pytest.raises(DataDesignerWorkflowError, match="produced an empty output"):
        workflow.run()


def test_composite_workflow_empty_workflow_fails_before_artifacts(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
) -> None:
    workflow = _data_designer(stub_artifact_path, stub_model_providers).compose_workflow(name="empty-workflow")

    with pytest.raises(DataDesignerWorkflowError, match="has no stages"):
        workflow.run()

    assert not (stub_artifact_path / "empty-workflow").exists()


@pytest.mark.parametrize("name", ["bad/name", "bad*name", "", ".", ".."])
def test_composite_workflow_rejects_invalid_workflow_names(
    name: str,
    stub_model_providers: list[ModelProvider],
) -> None:
    with pytest.raises(DataDesignerWorkflowError):
        DataDesigner(model_providers=stub_model_providers).compose_workflow(name=name)


@pytest.mark.parametrize("name", ["bad/name", "bad*name", "", ".", ".."])
def test_composite_workflow_rejects_invalid_stage_names(
    name: str,
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    workflow = _data_designer(stub_artifact_path, stub_model_providers).compose_workflow(name="invalid-stage")

    with pytest.raises(DataDesignerWorkflowError):
        workflow.add_stage(name, _category_builder(stub_model_configs))


@pytest.mark.parametrize("output", ["processor:", "callback", "processor:bad/name"])
def test_composite_workflow_rejects_invalid_stage_outputs(
    output: str,
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    workflow = _data_designer(stub_artifact_path, stub_model_providers).compose_workflow(name="invalid-output")

    with pytest.raises(DataDesignerWorkflowError):
        workflow.add_stage("base", _category_builder(stub_model_configs), output=output)


def test_composite_workflow_rejects_unknown_processor_stage_output(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    workflow = _data_designer(stub_artifact_path, stub_model_providers).compose_workflow(name="unknown-output")

    with pytest.raises(DataDesignerWorkflowError, match="not configured"):
        workflow.add_stage("base", _category_builder(stub_model_configs), output="processor:missing")


def test_composite_workflow_rejects_duplicate_output_processor_names(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage = _category_builder(stub_model_configs)
    stage.add_processor(SchemaTransformProcessorConfig(name="compact", template={"category": "{{ category }}"}))
    workflow = _data_designer(stub_artifact_path, stub_model_providers).compose_workflow(name="duplicate-processor")

    with pytest.raises(DataDesignerWorkflowError, match="distinct"):
        workflow.add_stage(
            "base",
            stage,
            output_processors=[SchemaTransformProcessorConfig(name="compact", template={"text": "{{ category }}"})],
        )


def test_composite_workflow_rejects_duplicate_names_within_output_processors(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage = _category_builder(stub_model_configs)
    workflow = _data_designer(stub_artifact_path, stub_model_providers).compose_workflow(
        name="duplicate-within-output-processors"
    )

    with pytest.raises(DataDesignerWorkflowError, match="distinct within output_processors"):
        workflow.add_stage(
            "base",
            stage,
            output_processors=[
                DropColumnsProcessorConfig(name="drop_scratch", column_names=["scratch"]),
                DropColumnsProcessorConfig(name="drop_scratch", column_names=["other_scratch"]),
            ],
        )


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


def test_composite_workflow_clones_stage_builders_on_add(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
) -> None:
    data_designer = _data_designer(stub_artifact_path, stub_model_providers)
    create_mock = _patch_create(data_designer, stub_dataset_profiler_results)
    builder = _category_builder(stub_model_configs)
    workflow = data_designer.compose_workflow(name="clone-builder")
    workflow.add_stage("base", builder, num_records=1)
    builder.add_column(ExpressionColumnConfig(name="late_column", expr="{{ category }}"))

    workflow.run()

    stage_builder = create_mock.call_args.args[0]
    assert [column.name for column in stage_builder.get_column_configs()] == ["category"]


def test_composite_workflow_targets_materializes_requested_stage(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
) -> None:
    data_designer = _data_designer(stub_artifact_path, stub_model_providers)
    create_mock = _patch_create(data_designer, stub_dataset_profiler_results)
    workflow = data_designer.compose_workflow(name="target-chain")
    workflow.add_stage("base", _category_builder(stub_model_configs), num_records=3)
    workflow.add_stage("copy", _copy_builder(stub_model_configs))
    workflow.add_stage("final", _expression_builder(stub_model_configs, "final", "{{ category_copy }}"))

    results = workflow.run(targets="copy")

    assert [call.kwargs["dataset_name"] for call in create_mock.call_args_list] == ["stage-0-base", "stage-1-copy"]
    assert list(results.keys()) == ["base", "copy"]
    assert results.final_stage_name == "copy"
    assert results.count_records() == 3
    metadata = _load_workflow_metadata(stub_artifact_path, "target-chain")
    assert [stage["name"] for stage in metadata["stages"]] == ["base", "copy"]


def test_composite_workflow_rerun_from_forces_stage_and_descendants(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
) -> None:
    data_designer = _data_designer(stub_artifact_path, stub_model_providers)
    create_mock = _patch_create(data_designer, stub_dataset_profiler_results)
    workflow = data_designer.compose_workflow(name="rerun-from")
    workflow.add_stage("base", _category_builder(stub_model_configs), num_records=2)
    workflow.add_stage("copy", _copy_builder(stub_model_configs))
    workflow.add_stage("final", _expression_builder(stub_model_configs, "final", "{{ category_copy }}"))
    workflow.run()
    create_mock.reset_mock()

    resumed = data_designer.compose_workflow(name="rerun-from")
    resumed.add_stage("base", _category_builder(stub_model_configs), num_records=2)
    resumed.add_stage("copy", _copy_builder(stub_model_configs))
    resumed.add_stage("final", _expression_builder(stub_model_configs, "final", "{{ category_copy }}"))
    resumed.run(resume=ResumeMode.IF_POSSIBLE, rerun_from="copy")

    assert [call.kwargs["dataset_name"] for call in create_mock.call_args_list] == ["stage-1-copy", "stage-2-final"]
    assert [call.kwargs["resume"] for call in create_mock.call_args_list] == [ResumeMode.NEVER, ResumeMode.NEVER]


def test_composite_workflow_rerun_from_requires_resume(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
) -> None:
    data_designer = _data_designer(stub_artifact_path, stub_model_providers)
    create_mock = _patch_create(data_designer, stub_dataset_profiler_results)
    workflow = data_designer.compose_workflow(name="rerun-from-no-resume")
    workflow.add_stage("base", _category_builder(stub_model_configs), num_records=2)
    workflow.add_stage("copy", _copy_builder(stub_model_configs))

    with pytest.raises(DataDesignerWorkflowError, match="rerun_from requires resume"):
        workflow.run(rerun_from="copy")

    assert create_mock.call_count == 0


def test_composite_workflow_stage_output_override_seeds_descendants(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage_1 = _seeded_builder(stub_model_configs, [{"name": "Ada"}, {"name": "Linus"}])
    stage_1.add_column(ExpressionColumnConfig(name="persona", expr="{{ name }}"))
    stage_2 = _expression_builder(stub_model_configs, "final", "{{ persona }} {{ approval }}")

    data_designer = _real_data_designer(tmp_path / "artifacts", stub_model_providers)
    workflow = data_designer.compose_workflow(name="hitl-override")
    workflow.add_stage("drafts", stage_1, num_records=2)
    workflow.add_stage("expanded", stage_2)
    draft_results = workflow.run(targets="drafts")
    assert draft_results.count_records() == 2

    approved_path = tmp_path / "approved.parquet"
    lazy.pd.DataFrame([{"name": "Grace", "persona": "Grace", "approval": "approved"}]).to_parquet(
        approved_path,
        index=False,
    )

    resumed = data_designer.compose_workflow(name="hitl-override")
    resumed.add_stage("drafts", stage_1, num_records=2)
    resumed.add_stage("expanded", stage_2)
    results = resumed.run(
        resume=ResumeMode.IF_POSSIBLE,
        stage_output_overrides={"drafts": approved_path},
    )

    assert results.get_stage_output_path("drafts") == approved_path.resolve()
    assert results.load_dataset().to_dict(orient="records") == [
        {"name": "Grace", "persona": "Grace", "approval": "approved", "final": "Grace approved"}
    ]
    metadata = _load_workflow_metadata(tmp_path / "artifacts", "hitl-override")
    assert metadata["stages"][0]["stage_output_override_path"] == str(approved_path.resolve())
    assert metadata["stages"][1]["seed_path"] == str(approved_path.resolve())


def test_composite_workflow_stage_output_override_path_must_exist(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
) -> None:
    data_designer = _data_designer(stub_artifact_path, stub_model_providers)
    create_mock = _patch_create(data_designer, stub_dataset_profiler_results)
    workflow = data_designer.compose_workflow(name="missing-override")
    workflow.add_stage("base", _category_builder(stub_model_configs), num_records=2)
    workflow.add_stage("copy", _copy_builder(stub_model_configs))

    with pytest.raises(DataDesignerWorkflowError, match="Invalid stage output override"):
        workflow.run(targets="copy", stage_output_overrides={"base": stub_artifact_path / "missing.parquet"})

    assert create_mock.call_count == 0


def test_composite_workflow_resume_if_possible_skips_completed_stages(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
) -> None:
    data_designer = _data_designer(stub_artifact_path, stub_model_providers)
    create_mock = _patch_create(data_designer, stub_dataset_profiler_results)
    workflow = data_designer.compose_workflow(name="resume-skip")
    workflow.add_stage("base", _category_builder(stub_model_configs), num_records=3)
    workflow.add_stage("copy", _copy_builder(stub_model_configs))
    workflow.run()
    create_mock.reset_mock()

    resumed = data_designer.compose_workflow(name="resume-skip")
    resumed.add_stage("base", _category_builder(stub_model_configs), num_records=3)
    resumed.add_stage("copy", _copy_builder(stub_model_configs))
    results = resumed.run(resume=ResumeMode.IF_POSSIBLE)

    assert create_mock.call_count == 0
    assert results.count_records() == 3
    assert results.load_dataset()["category"].tolist() == ["alpha", "alpha", "alpha"]


def test_composite_workflow_resume_if_possible_skips_stage_with_output_processors(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage = _seeded_builder(stub_model_configs, [{"name": "Ada", "secret": "hidden"}])
    stage.add_column(ExpressionColumnConfig(name="public_name", expr="{{ name }}"))

    data_designer = _real_data_designer(tmp_path / "artifacts", stub_model_providers)
    workflow = data_designer.compose_workflow(name="resume-output-processors")
    workflow.add_stage(
        "base",
        stage,
        num_records=1,
        output_processors=[DropColumnsProcessorConfig(name="drop_secret", column_names=["secret"])],
    )
    workflow.add_stage("final", _expression_builder(stub_model_configs, "final", "{{ public_name }} final"))
    first = workflow.run()
    output_processor_dir = first["base"].artifact_storage.base_dataset_path
    output_processor_dir_mtime = output_processor_dir.stat().st_mtime_ns
    output_processor_file = first["base"].artifact_storage.final_dataset_path / "batch_00000.parquet"
    output_processor_mtime = output_processor_file.stat().st_mtime_ns

    resumed = data_designer.compose_workflow(name="resume-output-processors")
    resumed.add_stage(
        "base",
        stage,
        num_records=1,
        output_processors=[DropColumnsProcessorConfig(name="drop_secret", column_names=["secret"])],
    )
    resumed.add_stage("final", _expression_builder(stub_model_configs, "final", "{{ public_name }} final"))
    results = resumed.run(resume=ResumeMode.IF_POSSIBLE)

    assert "secret" not in results["base"].load_dataset().columns
    assert results.load_dataset().to_dict(orient="records") == [
        {"name": "Ada", "public_name": "Ada", "final": "Ada final"}
    ]
    assert output_processor_dir.stat().st_mtime_ns == output_processor_dir_mtime
    assert output_processor_file.stat().st_mtime_ns == output_processor_mtime


def test_composite_workflow_resume_if_possible_uses_relative_metadata_paths_after_move(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
) -> None:
    source_artifacts = tmp_path / "source" / "artifacts"
    moved_artifacts = tmp_path / "moved" / "artifacts"
    data_designer = _data_designer(source_artifacts, stub_model_providers)
    _patch_create(data_designer, stub_dataset_profiler_results)
    workflow = data_designer.compose_workflow(name="resume-moved")
    workflow.add_stage("base", _category_builder(stub_model_configs), num_records=2)
    workflow.run()
    metadata = _load_workflow_metadata(source_artifacts, "resume-moved")
    assert metadata["stages"][0]["output_seed_path"] == "stage-0-base/parquet-files"

    shutil.copytree(source_artifacts, moved_artifacts)
    moved_data_designer = _data_designer(moved_artifacts, stub_model_providers)
    create_mock = _patch_create(moved_data_designer, stub_dataset_profiler_results)
    resumed = moved_data_designer.compose_workflow(name="resume-moved")
    resumed.add_stage("base", _category_builder(stub_model_configs), num_records=2)
    results = resumed.run(resume=ResumeMode.IF_POSSIBLE)

    assert create_mock.call_count == 0
    assert results.count_records() == 2


def test_composite_workflow_resume_if_possible_preserves_completed_empty_skip(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
) -> None:
    data_designer = _data_designer(stub_artifact_path, stub_model_providers)
    create_mock = _patch_create(data_designer, stub_dataset_profiler_results)

    def empty_output(stage_path: Path) -> Path:
        output_path = stage_path / "callback-outputs" / "empty"
        output_path.mkdir(parents=True)
        lazy.pd.DataFrame({"category": []}).to_parquet(output_path / "data.parquet", index=False)
        return output_path

    workflow = data_designer.compose_workflow(name="resume-empty")
    workflow.add_stage(
        "base",
        _category_builder(stub_model_configs),
        num_records=2,
        on_success=empty_output,
        on_success_version="empty",
        allow_empty=True,
    )
    workflow.add_stage("copy", _copy_builder(stub_model_configs))
    workflow.run()
    create_mock.reset_mock()

    resumed = data_designer.compose_workflow(name="resume-empty")
    resumed.add_stage(
        "base",
        _category_builder(stub_model_configs),
        num_records=2,
        on_success=empty_output,
        on_success_version="empty",
        allow_empty=True,
    )
    resumed.add_stage("copy", _copy_builder(stub_model_configs))
    results = resumed.run(resume=ResumeMode.IF_POSSIBLE)

    assert create_mock.call_count == 0
    assert isinstance(results["copy"], SkippedStageResult)
    assert results["copy"].upstream_stage == "base"


def test_composite_workflow_resume_if_possible_reruns_changed_stage_only(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
) -> None:
    data_designer = _data_designer(stub_artifact_path, stub_model_providers)
    create_mock = _patch_create(data_designer, stub_dataset_profiler_results)
    workflow = data_designer.compose_workflow(name="resume-changed")
    workflow.add_stage("base", _category_builder(stub_model_configs), num_records=2)
    workflow.add_stage("copy", _copy_builder(stub_model_configs))
    workflow.run()
    sentinel = stub_artifact_path / "resume-changed" / "stage-0-base" / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    create_mock.reset_mock()

    resumed = data_designer.compose_workflow(name="resume-changed")
    resumed.add_stage("base", _category_builder(stub_model_configs), num_records=2)
    resumed.add_stage("copy", _expression_builder(stub_model_configs, "category_copy", "{{ category }} v2"))
    resumed.run(resume=ResumeMode.IF_POSSIBLE)

    assert [call.kwargs["dataset_name"] for call in create_mock.call_args_list] == ["stage-1-copy"]
    assert sentinel.exists()


def test_composite_workflow_resume_if_possible_missing_callback_output_reruns_descendants(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
) -> None:
    data_designer = _data_designer(stub_artifact_path, stub_model_providers)
    create_mock = _patch_create(data_designer, stub_dataset_profiler_results)

    def keep_first(stage_path: Path) -> Path:
        df = lazy.pd.read_parquet(stage_path / "parquet-files")
        output_path = stage_path / "callback-outputs" / "first-row"
        output_path.mkdir(parents=True)
        df.head(1).to_parquet(output_path / "data.parquet", index=False)
        return output_path

    workflow = data_designer.compose_workflow(name="resume-callback")
    workflow.add_stage(
        "base",
        _category_builder(stub_model_configs),
        num_records=3,
        on_success=keep_first,
        on_success_version="first-row",
    )
    workflow.add_stage("copy", _copy_builder(stub_model_configs))
    workflow.run()
    callback_output = stub_artifact_path / "resume-callback" / "stage-0-base" / "callback-outputs" / "first-row"
    for parquet_file in callback_output.glob("*.parquet"):
        parquet_file.unlink()
    create_mock.reset_mock()

    resumed = data_designer.compose_workflow(name="resume-callback")
    resumed.add_stage(
        "base",
        _category_builder(stub_model_configs),
        num_records=3,
        on_success=keep_first,
        on_success_version="first-row",
    )
    resumed.add_stage("copy", _copy_builder(stub_model_configs))
    resumed.run(resume=ResumeMode.IF_POSSIBLE)

    assert [call.kwargs["dataset_name"] for call in create_mock.call_args_list] == ["stage-0-base", "stage-1-copy"]


def test_composite_workflow_resume_if_possible_corrupt_metadata_starts_fresh(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
) -> None:
    data_designer = _data_designer(stub_artifact_path, stub_model_providers)
    create_mock = _patch_create(data_designer, stub_dataset_profiler_results)
    workflow = data_designer.compose_workflow(name="resume-corrupt")
    workflow.add_stage("base", _category_builder(stub_model_configs), num_records=2)
    workflow.add_stage("copy", _copy_builder(stub_model_configs))
    workflow.run()
    metadata_path = stub_artifact_path / "resume-corrupt" / "workflow-metadata.json"
    metadata_path.write_text("{", encoding="utf-8")
    create_mock.reset_mock()

    resumed = data_designer.compose_workflow(name="resume-corrupt")
    resumed.add_stage("base", _category_builder(stub_model_configs), num_records=2)
    resumed.add_stage("copy", _copy_builder(stub_model_configs))
    resumed.run(resume=ResumeMode.IF_POSSIBLE)

    assert [call.kwargs["dataset_name"] for call in create_mock.call_args_list] == ["stage-0-base", "stage-1-copy"]


@pytest.mark.parametrize("metadata_payload", [[], None, "oops"])
def test_composite_workflow_resume_if_possible_invalid_metadata_shape_starts_fresh(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
    metadata_payload,
) -> None:
    data_designer = _data_designer(stub_artifact_path, stub_model_providers)
    create_mock = _patch_create(data_designer, stub_dataset_profiler_results)
    workflow = data_designer.compose_workflow(name="resume-invalid-shape")
    workflow.add_stage("base", _category_builder(stub_model_configs), num_records=2)
    workflow.add_stage("copy", _copy_builder(stub_model_configs))
    workflow.run()
    metadata_path = stub_artifact_path / "resume-invalid-shape" / "workflow-metadata.json"
    metadata_path.write_text(json.dumps(metadata_payload), encoding="utf-8")
    create_mock.reset_mock()

    resumed = data_designer.compose_workflow(name="resume-invalid-shape")
    resumed.add_stage("base", _category_builder(stub_model_configs), num_records=2)
    resumed.add_stage("copy", _copy_builder(stub_model_configs))
    resumed.run(resume=ResumeMode.IF_POSSIBLE)

    assert [call.kwargs["dataset_name"] for call in create_mock.call_args_list] == ["stage-0-base", "stage-1-copy"]


@pytest.mark.parametrize("status", ["running", "failed"])
def test_composite_workflow_resume_if_possible_delegates_matching_resumable_stage(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
    status: str,
) -> None:
    data_designer = _data_designer(stub_artifact_path, stub_model_providers)
    create_mock = _patch_create(data_designer, stub_dataset_profiler_results)
    workflow = data_designer.compose_workflow(name="resume-partial")
    workflow.add_stage("base", _category_builder(stub_model_configs), num_records=2)
    workflow.add_stage("copy", _copy_builder(stub_model_configs))
    workflow.run()
    metadata_path = stub_artifact_path / "resume-partial" / "workflow-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _mark_stage_resumable(metadata, 0, status)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    create_mock.reset_mock()

    resumed = data_designer.compose_workflow(name="resume-partial")
    resumed.add_stage("base", _category_builder(stub_model_configs), num_records=2)
    resumed.add_stage("copy", _copy_builder(stub_model_configs))
    resumed.run(resume=ResumeMode.IF_POSSIBLE)

    assert [call.kwargs["dataset_name"] for call in create_mock.call_args_list] == ["stage-0-base", "stage-1-copy"]
    assert [call.kwargs["resume"] for call in create_mock.call_args_list] == [ResumeMode.ALWAYS, ResumeMode.NEVER]


def test_composite_workflow_resume_always_reruns_descendants_after_partial_stage(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
) -> None:
    data_designer = _data_designer(stub_artifact_path, stub_model_providers)
    create_mock = _patch_create(data_designer, stub_dataset_profiler_results)
    workflow = data_designer.compose_workflow(name="resume-always-partial")
    workflow.add_stage("base", _category_builder(stub_model_configs), num_records=2)
    workflow.add_stage("copy", _copy_builder(stub_model_configs))
    workflow.run()
    metadata_path = stub_artifact_path / "resume-always-partial" / "workflow-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _mark_stage_resumable(metadata, 0, "running")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    create_mock.reset_mock()

    resumed = data_designer.compose_workflow(name="resume-always-partial")
    resumed.add_stage("base", _category_builder(stub_model_configs), num_records=2)
    resumed.add_stage("copy", _expression_builder(stub_model_configs, "category_copy", "{{ category }} v2"))
    resumed.run(resume=ResumeMode.ALWAYS)

    assert [call.kwargs["dataset_name"] for call in create_mock.call_args_list] == ["stage-0-base", "stage-1-copy"]
    assert [call.kwargs["resume"] for call in create_mock.call_args_list] == [ResumeMode.ALWAYS, ResumeMode.NEVER]


def test_composite_workflow_resume_always_requires_metadata(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    data_designer = _data_designer(stub_artifact_path, stub_model_providers)
    workflow = data_designer.compose_workflow(name="resume-missing")
    workflow.add_stage("base", _category_builder(stub_model_configs), num_records=2)

    with pytest.raises(DataDesignerWorkflowError, match="no workflow metadata found"):
        workflow.run(resume=ResumeMode.ALWAYS)


def test_composite_workflow_resume_always_rejects_corrupt_metadata(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
) -> None:
    data_designer = _data_designer(stub_artifact_path, stub_model_providers)
    _patch_create(data_designer, stub_dataset_profiler_results)
    workflow = data_designer.compose_workflow(name="resume-corrupt-always")
    workflow.add_stage("base", _category_builder(stub_model_configs), num_records=2)
    workflow.run()
    metadata_path = stub_artifact_path / "resume-corrupt-always" / "workflow-metadata.json"
    metadata_path.write_text("{", encoding="utf-8")

    resumed = data_designer.compose_workflow(name="resume-corrupt-always")
    resumed.add_stage("base", _category_builder(stub_model_configs), num_records=2)
    with pytest.raises(DataDesignerWorkflowError, match="workflow metadata is corrupt"):
        resumed.run(resume=ResumeMode.ALWAYS)


@pytest.mark.parametrize("metadata_payload", [[], None, "oops"])
def test_composite_workflow_resume_always_rejects_invalid_metadata_shape(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
    metadata_payload,
) -> None:
    data_designer = _data_designer(stub_artifact_path, stub_model_providers)
    _patch_create(data_designer, stub_dataset_profiler_results)
    workflow = data_designer.compose_workflow(name="resume-invalid-shape-always")
    workflow.add_stage("base", _category_builder(stub_model_configs), num_records=2)
    workflow.run()
    metadata_path = stub_artifact_path / "resume-invalid-shape-always" / "workflow-metadata.json"
    metadata_path.write_text(json.dumps(metadata_payload), encoding="utf-8")

    resumed = data_designer.compose_workflow(name="resume-invalid-shape-always")
    resumed.add_stage("base", _category_builder(stub_model_configs), num_records=2)
    with pytest.raises(DataDesignerWorkflowError, match="workflow metadata has invalid shape"):
        resumed.run(resume=ResumeMode.ALWAYS)


def test_composite_workflow_resume_always_rejects_changed_stage(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
) -> None:
    data_designer = _data_designer(stub_artifact_path, stub_model_providers)
    create_mock = _patch_create(data_designer, stub_dataset_profiler_results)
    workflow = data_designer.compose_workflow(name="resume-always")
    workflow.add_stage("base", _category_builder(stub_model_configs), num_records=2)
    workflow.add_stage("copy", _copy_builder(stub_model_configs))
    workflow.run()
    create_mock.reset_mock()

    resumed = data_designer.compose_workflow(name="resume-always")
    resumed.add_stage("base", _category_builder(stub_model_configs), num_records=2)
    resumed.add_stage("copy", _expression_builder(stub_model_configs, "category_copy", "{{ category }} v2"))
    with pytest.raises(DataDesignerWorkflowError, match="not reusable"):
        resumed.run(resume=ResumeMode.ALWAYS)

    assert create_mock.call_count == 0


def test_composite_workflow_runs_three_real_async_stages(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage_1 = _seeded_builder(stub_model_configs, [{"name": "Ada"}, {"name": "Linus"}])
    stage_1.add_column(ExpressionColumnConfig(name="persona", expr="{{ name }} persona"))

    stage_2 = _expression_builder(stub_model_configs, "prompt_seed", "{{ persona }} prompt")
    stage_3 = _expression_builder(stub_model_configs, "final_label", "{{ prompt_seed }} final")

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

    stage_2 = _expression_builder(stub_model_configs, "message", "{{ persona }} turn {{ turn }}")

    workflow = _real_data_designer(tmp_path / "artifacts", stub_model_providers).compose_workflow(name="expand")
    workflow.add_stage("personas", stage_1, num_records=2, on_success=expand, on_success_version="expand-turns")
    workflow.add_stage("messages", stage_2)

    results = workflow.run()
    df = results.load_dataset().sort_values(["name", "turn"]).reset_index(drop=True)
    stage_output = results.load_stage_output("personas").sort_values(["name", "turn"]).reset_index(drop=True)

    assert df["message"].tolist() == [
        "Ada turn 1",
        "Ada turn 2",
        "Linus turn 1",
        "Linus turn 2",
    ]
    assert stage_output["turn"].tolist() == [1, 2, 1, 2]
    assert results["personas"].count_records() == 2
    assert results.count_stage_output_records("personas") == 4


def test_composite_workflow_repeat_until_append_accumulates_and_trims(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage_1 = _seeded_builder(
        stub_model_configs,
        [
            {"name": "Ada", "keep": False},
            {"name": "Grace", "keep": True},
            {"name": "Linus", "keep": True},
            {"name": "Margaret", "keep": True},
        ],
    )
    stage_1.add_column(ExpressionColumnConfig(name="candidate", expr="{{ name }} candidate"))

    def keep_rows(stage_path: Path) -> Path:
        df = lazy.pd.read_parquet(stage_path / "parquet-files")
        output_path = stage_path / "callback-outputs" / "kept"
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True)
        df[df["keep"]].to_parquet(output_path / "data.parquet", index=False)
        return output_path

    stage_2 = _expression_builder(stub_model_configs, "final", "{{ candidate }} final")
    workflow = _real_data_designer(tmp_path / "artifacts", stub_model_providers).compose_workflow(name="repeat-append")
    workflow.add_stage(
        "candidates",
        stage_1,
        num_records=2,
        on_success=keep_rows,
        on_success_version="kept-v1",
        repeat_until=RepeatUntil(output_records=2, max_iterations=3),
    )
    workflow.add_stage("final", stage_2)

    results = workflow.run()
    final = results.load_dataset().sort_values("name").reset_index(drop=True)
    metadata = _load_workflow_metadata(tmp_path / "artifacts", "repeat-append")

    assert final[["name", "candidate", "final"]].to_dict(orient="records") == [
        {"name": "Grace", "candidate": "Grace candidate", "final": "Grace candidate final"},
        {"name": "Linus", "candidate": "Linus candidate", "final": "Linus candidate final"},
    ]
    assert results["candidates"].count_records() == 4
    assert results.count_stage_output_records("candidates") == 2
    assert metadata["stages"][0]["num_records_requested"] == 4
    assert metadata["stages"][0]["repeat_iterations"] == 2
    assert metadata["stages"][0]["repeat_generated_records"] == 4
    assert metadata["stages"][0]["repeat_satisfied"] is True
    assert metadata["stages"][0]["repeat_until_output_path"].endswith("stage-0-candidates/repeat-until/selected-output")


def test_composite_workflow_repeat_until_returns_partial_when_exhausted(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage_1 = _seeded_builder(
        stub_model_configs,
        [
            {"name": "Ada", "keep": False},
            {"name": "Grace", "keep": True},
            {"name": "Linus", "keep": False},
            {"name": "Margaret", "keep": False},
        ],
    )
    stage_1.add_column(ExpressionColumnConfig(name="candidate", expr="{{ name }} candidate"))

    def keep_rows(stage_path: Path) -> Path:
        df = lazy.pd.read_parquet(stage_path / "parquet-files")
        output_path = stage_path / "callback-outputs" / "kept"
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True)
        df[df["keep"]].to_parquet(output_path / "data.parquet", index=False)
        return output_path

    workflow = _real_data_designer(tmp_path / "artifacts", stub_model_providers).compose_workflow(name="repeat-partial")
    workflow.add_stage(
        "candidates",
        stage_1,
        num_records=2,
        on_success=keep_rows,
        on_success_version="kept-v1",
        repeat_until=RepeatUntil(output_records=3, max_iterations=2, on_exhausted="return_partial"),
    )

    results = workflow.run()
    metadata = _load_workflow_metadata(tmp_path / "artifacts", "repeat-partial")

    assert results.load_dataset()["name"].tolist() == ["Grace"]
    assert results.count_records() == 1
    assert metadata["stages"][0]["repeat_iterations"] == 2
    assert metadata["stages"][0]["repeat_generated_records"] == 4
    assert metadata["stages"][0]["repeat_satisfied"] is False


def test_composite_workflow_repeat_until_returns_empty_partial_when_exhausted(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage = _seeded_builder(
        stub_model_configs,
        [{"name": "Ada", "keep": False}, {"name": "Grace", "keep": False}],
    )
    stage.add_column(ExpressionColumnConfig(name="candidate", expr="{{ name }} candidate"))

    def keep_rows(stage_path: Path) -> Path:
        df = lazy.pd.read_parquet(stage_path / "parquet-files")
        output_path = stage_path / "callback-outputs" / "kept"
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True)
        df[df["keep"]].to_parquet(output_path / "data.parquet", index=False)
        return output_path

    workflow = _real_data_designer(tmp_path / "artifacts", stub_model_providers).compose_workflow(
        name="repeat-empty-partial"
    )
    workflow.add_stage(
        "candidates",
        stage,
        num_records=1,
        on_success=keep_rows,
        on_success_version="kept-v1",
        repeat_until=RepeatUntil(output_records=1, max_iterations=2, on_exhausted="return_partial"),
    )

    results = workflow.run()
    metadata = _load_workflow_metadata(tmp_path / "artifacts", "repeat-empty-partial")

    assert results.count_records() == 0
    assert results.load_dataset().empty
    assert metadata["stages"][0]["status"] == "completed_empty"
    assert metadata["stages"][0]["repeat_satisfied"] is False


def test_composite_workflow_repeat_until_raises_when_exhausted(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage = _seeded_builder(stub_model_configs, [{"name": "Ada", "keep": False}])
    stage.add_column(ExpressionColumnConfig(name="candidate", expr="{{ name }} candidate"))

    def keep_rows(stage_path: Path) -> Path:
        df = lazy.pd.read_parquet(stage_path / "parquet-files")
        output_path = stage_path / "callback-outputs" / "kept"
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True)
        df[df["keep"]].to_parquet(output_path / "data.parquet", index=False)
        return output_path

    workflow = _real_data_designer(tmp_path / "artifacts", stub_model_providers).compose_workflow(name="repeat-raises")
    workflow.add_stage(
        "candidates",
        stage,
        num_records=1,
        on_success=keep_rows,
        repeat_until=RepeatUntil(output_records=1, max_iterations=2),
    )

    with pytest.raises(DataDesignerWorkflowError, match="repeat_until exhausted"):
        workflow.run()


def test_composite_workflow_repeat_until_discard_keeps_latest_attempt(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage = _seeded_builder(
        stub_model_configs,
        [{"name": "Ada"}, {"name": "Grace"}, {"name": "Linus"}],
    )
    stage.add_column(ExpressionColumnConfig(name="candidate", expr="{{ name }} candidate"))
    callback_calls = 0

    def keep_more_each_time(stage_path: Path) -> Path:
        nonlocal callback_calls
        callback_calls += 1
        df = lazy.pd.read_parquet(stage_path / "parquet-files").head(callback_calls)
        output_path = stage_path / "callback-outputs" / "latest"
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True)
        df.to_parquet(output_path / "data.parquet", index=False)
        return output_path

    workflow = _real_data_designer(tmp_path / "artifacts", stub_model_providers).compose_workflow(name="repeat-discard")
    workflow.add_stage(
        "candidates",
        stage,
        num_records=3,
        on_success=keep_more_each_time,
        repeat_until=RepeatUntil(output_records=2, max_iterations=3, mode="discard"),
    )

    results = workflow.run()
    metadata = _load_workflow_metadata(tmp_path / "artifacts", "repeat-discard")

    assert callback_calls == 2
    assert results.load_dataset()["name"].tolist() == ["Ada", "Grace"]
    assert results["candidates"].count_records() == 3
    assert metadata["stages"][0]["repeat_iterations"] == 2
    assert metadata["stages"][0]["repeat_generated_records"] == 6


def test_composite_workflow_repeat_until_discard_warns_when_resuming(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
    caplog: pytest.LogCaptureFixture,
) -> None:
    data_designer = _data_designer(stub_artifact_path, stub_model_providers)
    _patch_create(data_designer, stub_dataset_profiler_results)
    workflow = data_designer.compose_workflow(name="repeat-discard-resume")
    workflow.add_stage(
        "candidates",
        _category_builder(stub_model_configs),
        num_records=2,
        repeat_until=RepeatUntil(output_records=1, max_iterations=2, mode="discard"),
    )
    workflow.run()

    metadata_path = stub_artifact_path / "repeat-discard-resume" / "workflow-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    _mark_stage_resumable(metadata, 0, "failed")
    metadata_path.write_text(json.dumps(metadata))

    resumed = data_designer.compose_workflow(name="repeat-discard-resume")
    resumed.add_stage(
        "candidates",
        _category_builder(stub_model_configs),
        num_records=2,
        repeat_until=RepeatUntil(output_records=1, max_iterations=2, mode="discard"),
    )

    with caplog.at_level("WARNING", logger="data_designer.interface.composite_workflow"):
        resumed.run(resume=ResumeMode.IF_POSSIBLE)

    assert "previous attempts cannot be resumed" in caplog.text


def test_composite_workflow_repeat_until_uses_processor_output(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage_1 = _seeded_builder(
        stub_model_configs,
        [{"name": "Ada"}, {"name": "Linus"}, {"name": "Grace"}],
    )
    stage_1.add_column(ExpressionColumnConfig(name="persona", expr="{{ name }}"))
    stage_2 = _expression_builder(stub_model_configs, "final", "{{ compact_name }} final")

    workflow = _real_data_designer(tmp_path / "artifacts", stub_model_providers).compose_workflow(
        name="repeat-processor-output"
    )
    workflow.add_stage(
        "compact",
        stage_1,
        num_records=1,
        output_processors=[SchemaTransformProcessorConfig(name="compact", template={"compact_name": "{{ persona }}"})],
        output="processor:compact",
        repeat_until=RepeatUntil(output_records=2, max_iterations=3),
    )
    workflow.add_stage("final", stage_2)

    results = workflow.run()
    final = results.load_dataset().sort_values("compact_name").reset_index(drop=True)
    stage_output = results.load_stage_output("compact").sort_values("compact_name").reset_index(drop=True)
    metadata = _load_workflow_metadata(tmp_path / "artifacts", "repeat-processor-output")

    assert stage_output.to_dict(orient="records") == [{"compact_name": "Ada"}, {"compact_name": "Linus"}]
    assert final.to_dict(orient="records") == [
        {"compact_name": "Ada", "final": "Ada final"},
        {"compact_name": "Linus", "final": "Linus final"},
    ]
    assert metadata["stages"][0]["num_records_requested"] == 2
    assert metadata["stages"][0]["repeat_iterations"] == 2
    assert metadata["stages"][0]["output_seed_path"].endswith(
        "stage-0-compact/output-processors/processors-files/compact"
    )


def test_composite_workflow_does_not_forward_dropped_processor_columns(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage_1 = _seeded_builder(stub_model_configs, [{"name": "Ada", "secret": "hidden"}])
    stage_1.add_column(ExpressionColumnConfig(name="public_name", expr="{{ name }}"))
    stage_1.add_processor(DropColumnsProcessorConfig(name="drop_secret", column_names=["secret"]))

    stage_2 = _expression_builder(stub_model_configs, "copied_name", "{{ public_name }}")

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

    stage_2 = _expression_builder(stub_model_configs, "final", "{{ compact_name }} final")

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


def test_composite_workflow_missing_callback_output_raises_workflow_error(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage = _seeded_builder(stub_model_configs, [{"name": "Ada"}])
    stage.add_column(ExpressionColumnConfig(name="persona", expr="{{ name }}"))

    def missing_output(stage_path: Path) -> Path:
        return stage_path / "callback-outputs" / "missing"

    workflow = _real_data_designer(tmp_path / "artifacts", stub_model_providers).compose_workflow(
        name="missing-callback"
    )
    workflow.add_stage("base", stage, num_records=1, on_success=missing_output)

    with pytest.raises(DataDesignerWorkflowError, match="No parquet files found"):
        workflow.run()


def test_composite_workflow_callback_replaces_selected_output_before_counting(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage = _seeded_builder(stub_model_configs, [{"name": "Ada", "secret": "hidden"}])
    stage.add_column(ExpressionColumnConfig(name="persona", expr="{{ name }}"))
    stage.add_processor(DropColumnsProcessorConfig(name="drop_secret", column_names=["secret"]))

    def use_main_output(stage_path: Path) -> Path:
        return stage_path / "parquet-files"

    workflow = _real_data_designer(tmp_path / "artifacts", stub_model_providers).compose_workflow(
        name="callback-replaces-output"
    )
    workflow.add_stage(
        "base",
        stage,
        num_records=1,
        output="processor:drop_secret",
        on_success=use_main_output,
    )

    assert workflow.run().load_dataset().to_dict(orient="records") == [{"name": "Ada", "persona": "Ada"}]


def test_composite_workflow_runs_seeded_processor_only_stage(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage_1 = _seeded_builder(stub_model_configs, [{"name": "Ada", "secret": "hidden"}])
    stage_1.add_column(ExpressionColumnConfig(name="public_name", expr="{{ name }}"))

    stage_2 = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    stage_2.add_processor(DropColumnsProcessorConfig(name="drop_secret", column_names=["secret"]))

    stage_3 = _expression_builder(stub_model_configs, "final", "{{ public_name }} final")

    workflow = _real_data_designer(tmp_path / "artifacts", stub_model_providers).compose_workflow(
        name="processor-only-stage"
    )
    workflow.add_stage("base", stage_1, num_records=1)
    workflow.add_stage("redacted", stage_2)
    workflow.add_stage("final", stage_3)

    results = workflow.run()
    redacted = results["redacted"].load_dataset()
    final = results.load_dataset()

    assert "secret" not in redacted.columns
    assert final.to_dict(orient="records") == [{"name": "Ada", "public_name": "Ada", "final": "Ada final"}]


def test_composite_workflow_output_processors_transform_stage_output(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage_1 = _seeded_builder(stub_model_configs, [{"name": "Ada", "secret": "hidden"}])
    stage_1.add_column(ExpressionColumnConfig(name="public_name", expr="{{ name }}"))

    stage_2 = _expression_builder(stub_model_configs, "final", "{{ public_name }} final")

    workflow = _real_data_designer(tmp_path / "artifacts", stub_model_providers).compose_workflow(
        name="output_processor-stage"
    )
    workflow.add_stage(
        "base",
        stage_1,
        num_records=1,
        output_processors=[DropColumnsProcessorConfig(name="drop_secret", column_names=["secret"])],
    )
    workflow.add_stage("final", stage_2)

    results = workflow.run()
    base = results["base"].load_dataset()
    final = results.load_dataset()
    metadata = _load_workflow_metadata(tmp_path / "artifacts", "output_processor-stage")

    assert "secret" not in base.columns
    assert final.to_dict(orient="records") == [{"name": "Ada", "public_name": "Ada", "final": "Ada final"}]
    assert metadata["stages"][0]["output_processor_output_path"].endswith("stage-0-base/output-processors")


def test_composite_workflow_output_can_select_processor_artifact(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage_1 = _seeded_builder(stub_model_configs, [{"name": "Ada"}, {"name": "Linus"}])
    stage_1.add_column(ExpressionColumnConfig(name="persona", expr="{{ name }}"))
    stage_1.add_processor(SchemaTransformProcessorConfig(name="compact", template={"compact_name": "{{ persona }}"}))

    stage_2 = _expression_builder(stub_model_configs, "final", "{{ compact_name }} final")

    workflow = _real_data_designer(tmp_path / "artifacts", stub_model_providers).compose_workflow(
        name="processor-output"
    )
    workflow.add_stage("compact", stage_1, num_records=2, output="processor:compact")
    workflow.add_stage("final", stage_2)

    results = workflow.run()
    df = results.load_dataset().sort_values("compact_name").reset_index(drop=True)
    stage_output = results.load_stage_output("compact").sort_values("compact_name").reset_index(drop=True)
    stage_final = results["compact"].load_dataset().sort_values("name").reset_index(drop=True)
    metadata = _load_workflow_metadata(tmp_path / "artifacts", "processor-output")

    assert df.to_dict(orient="records") == [
        {"compact_name": "Ada", "final": "Ada final"},
        {"compact_name": "Linus", "final": "Linus final"},
    ]
    assert stage_output.to_dict(orient="records") == [
        {"compact_name": "Ada"},
        {"compact_name": "Linus"},
    ]
    assert stage_final[["name", "persona"]].to_dict(orient="records") == [
        {"name": "Ada", "persona": "Ada"},
        {"name": "Linus", "persona": "Linus"},
    ]
    assert results.count_stage_output_records("compact") == 2
    assert metadata["stages"][0]["output"] == "processor:compact"
    assert metadata["stages"][0]["output_seed_path"].endswith("stage-0-compact/processors-files/compact")


def test_composite_workflow_output_processors_can_feed_from_processor_artifact(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage_1 = _seeded_builder(stub_model_configs, [{"name": "Ada"}, {"name": "Linus"}])
    stage_1.add_column(ExpressionColumnConfig(name="persona", expr="{{ name }}"))

    stage_2 = _expression_builder(stub_model_configs, "final", "{{ compact_name }} final")

    workflow = _real_data_designer(tmp_path / "artifacts", stub_model_providers).compose_workflow(
        name="output_processor-output"
    )
    workflow.add_stage(
        "compact",
        stage_1,
        num_records=2,
        output_processors=[SchemaTransformProcessorConfig(name="compact", template={"compact_name": "{{ persona }}"})],
        output="processor:compact",
    )
    workflow.add_stage("final", stage_2)

    results = workflow.run()
    df = results.load_dataset().sort_values("compact_name").reset_index(drop=True)
    stage_output = results.load_stage_output("compact").sort_values("compact_name").reset_index(drop=True)
    stage_final = results["compact"].load_dataset().sort_values("name").reset_index(drop=True)

    assert df.to_dict(orient="records") == [
        {"compact_name": "Ada", "final": "Ada final"},
        {"compact_name": "Linus", "final": "Linus final"},
    ]
    assert stage_output.to_dict(orient="records") == [
        {"compact_name": "Ada"},
        {"compact_name": "Linus"},
    ]
    assert stage_final[["name", "persona"]].to_dict(orient="records") == [
        {"name": "Ada", "persona": "Ada"},
        {"name": "Linus", "persona": "Linus"},
    ]


def test_composite_workflow_output_can_select_main_processor_with_output_processors(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage = _seeded_builder(stub_model_configs, [{"name": "Ada", "scratch": "drop me"}])
    stage.add_column(ExpressionColumnConfig(name="persona", expr="{{ name }}"))
    stage.add_processor(SchemaTransformProcessorConfig(name="compact", template={"compact_name": "{{ persona }}"}))

    workflow = _real_data_designer(tmp_path / "artifacts", stub_model_providers).compose_workflow(
        name="main-processor-output"
    )
    workflow.add_stage(
        "compact",
        stage,
        num_records=1,
        output_processors=[DropColumnsProcessorConfig(name="drop_scratch", column_names=["scratch"])],
        output="processor:compact",
    )

    assert workflow.run().load_dataset().to_dict(orient="records") == [{"compact_name": "Ada"}]


def test_composite_workflow_callback_receives_main_stage_artifact_with_output_processors(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage = _seeded_builder(stub_model_configs, [{"name": "Ada", "scratch": "drop me"}])
    stage.add_column(ExpressionColumnConfig(name="persona", expr="{{ name }}"))
    callback_paths = []

    def keep_output_processed(stage_path: Path) -> Path:
        callback_paths.append(stage_path)
        assert (stage_path / "output-processors" / "parquet-files").is_dir()
        return stage_path / "output-processors" / "parquet-files"

    workflow = _real_data_designer(tmp_path / "artifacts", stub_model_providers).compose_workflow(
        name="output_processor-callback"
    )
    workflow.add_stage(
        "base",
        stage,
        num_records=1,
        output_processors=[DropColumnsProcessorConfig(name="drop_scratch", column_names=["scratch"])],
        on_success=keep_output_processed,
    )

    result = workflow.run()

    assert callback_paths == [tmp_path / "artifacts" / "output_processor-callback" / "stage-0-base"]
    assert "scratch" not in result.load_dataset().columns


def test_composite_workflow_export_uses_selected_final_output(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage = _seeded_builder(stub_model_configs, [{"name": "Ada"}, {"name": "Linus"}])
    stage.add_column(ExpressionColumnConfig(name="persona", expr="{{ name }}"))
    stage.add_processor(SchemaTransformProcessorConfig(name="compact", template={"compact_name": "{{ persona }}"}))

    workflow = _real_data_designer(tmp_path / "artifacts", stub_model_providers).compose_workflow(
        name="selected-export"
    )
    workflow.add_stage("compact", stage, num_records=2, output="processor:compact")

    output = workflow.run().export(tmp_path / "selected.jsonl")

    assert [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] == [
        {"compact_name": "Ada"},
        {"compact_name": "Linus"},
    ]


def test_composite_workflow_export_stage_uses_selected_stage_output(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage = _seeded_builder(stub_model_configs, [{"name": "Ada"}, {"name": "Linus"}])
    stage.add_column(ExpressionColumnConfig(name="persona", expr="{{ name }}"))
    stage.add_processor(SchemaTransformProcessorConfig(name="compact", template={"compact_name": "{{ persona }}"}))

    workflow = _real_data_designer(tmp_path / "artifacts", stub_model_providers).compose_workflow(
        name="selected-stage-export"
    )
    workflow.add_stage("compact", stage, num_records=2, output="processor:compact")
    workflow.add_stage("final", _expression_builder(stub_model_configs, "final", "{{ compact_name }} final"))

    output = workflow.run().export_stage("compact", tmp_path / "compact.jsonl")

    assert [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] == [
        {"compact_name": "Ada"},
        {"compact_name": "Linus"},
    ]


def test_composite_workflow_export_matches_selected_output_files(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage = _seeded_builder(stub_model_configs, [{"name": "Ada"}])
    stage.add_column(ExpressionColumnConfig(name="persona", expr="{{ name }}"))
    stage.add_processor(SchemaTransformProcessorConfig(name="compact", template={"compact_name": "{{ persona }}"}))

    workflow = _real_data_designer(tmp_path / "artifacts", stub_model_providers).compose_workflow(name="mixed-export")
    workflow.add_stage("compact", stage, num_records=1, output="processor:compact")
    results = workflow.run()

    lazy.pd.DataFrame({"compact_name": ["Linus"]}).to_parquet(
        results.get_stage_output_path("compact") / "extra.parquet",
        index=False,
    )

    output = results.export(tmp_path / "mixed.jsonl")

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert sorted(rows, key=lambda row: row["compact_name"]) == [
        {"compact_name": "Ada"},
        {"compact_name": "Linus"},
    ]
    assert results.count_records() == 2


def test_composite_workflow_push_to_hub_rejects_selected_processor_output(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    stage = _seeded_builder(stub_model_configs, [{"name": "Ada"}])
    stage.add_column(ExpressionColumnConfig(name="persona", expr="{{ name }}"))
    stage.add_processor(SchemaTransformProcessorConfig(name="compact", template={"compact_name": "{{ persona }}"}))

    workflow = _real_data_designer(tmp_path / "artifacts", stub_model_providers).compose_workflow(name="selected-push")
    workflow.add_stage("compact", stage, num_records=1, output="processor:compact")

    with pytest.raises(DataDesignerWorkflowError, match="selected workflow outputs"):
        workflow.run().push_to_hub("user/selected", "description")


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
        artifact_path = Path(kwargs.pop("artifact_path", data_designer.artifact_path))
        del num_records, kwargs
        value = "first" if dataset_name == "stage-0-first" else "final"
        return _result_from_df(
            artifact_path,
            dataset_name,
            lazy.pd.DataFrame({"stage": [value]}),
            config_builder,
            stub_dataset_profiler_results,
        )

    data_designer.create = MagicMock(side_effect=fake_create)
    first = _seeded_builder(stub_model_configs, [{"stage": "first"}])
    first.add_column(ExpressionColumnConfig(name="stage_copy", expr="{{ stage }}"))
    last = _expression_builder(stub_model_configs, "stage_final", "{{ stage }}")

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
        artifact_path = Path(kwargs.pop("artifact_path", data_designer.artifact_path))
        del num_records, kwargs
        return _result_from_df(
            artifact_path,
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
    last = _expression_builder(stub_model_configs, "stage_final", "{{ stage }}")

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
