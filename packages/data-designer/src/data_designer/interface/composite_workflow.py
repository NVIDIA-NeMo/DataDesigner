# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from collections.abc import Callable, ItemsView, Iterator, KeysView
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.base import ProcessorConfig
from data_designer.config.config_builder import BuilderConfig, DataDesignerConfigBuilder
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.errors import InvalidFileFormatError
from data_designer.config.seed import IndexRange, PartitionBlock, SamplingStrategy
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.config.utils.constants import DEFAULT_NUM_RECORDS
from data_designer.config.version import get_library_version
from data_designer.engine.dataset_builders.errors import ArtifactStorageError
from data_designer.interface.errors import DataDesignerWorkflowError
from data_designer.interface.results import (
    SUPPORTED_EXPORT_FORMATS,
    DatasetCreationResults,
    ExportFormat,
    _export_csv,
    _export_jsonl,
    _export_parquet,
)

if TYPE_CHECKING:
    import pandas as pd

    from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults
    from data_designer.interface.data_designer import DataDesigner


logger = logging.getLogger(__name__)

OnSuccessCallback = Callable[[Path], Path | str]


@dataclass(frozen=True)
class _WorkflowStage:
    name: str
    config_builder: DataDesignerConfigBuilder
    depends_on: tuple[str, ...]
    num_records: int | None
    on_success: OnSuccessCallback | None
    on_success_version: str | None
    postprocessors: tuple[ProcessorConfig, ...]
    output: str
    allow_empty: bool
    sampling_strategy: SamplingStrategy
    selection_strategy: IndexRange | PartitionBlock | None


@dataclass(frozen=True)
class SkippedStageResult:
    status: str
    upstream_stage: str


class CompositeWorkflowResults:
    """Results for a composite workflow run.

    Per-stage entries are the effective ``DataDesigner.create()`` results. If a
    stage uses ``on_success``, metadata and downstream seeding use the callback
    output path while the stage result still points at the stage's dataset.
    Use ``load_stage_output()`` to load the selected output handed downstream.
    """

    def __init__(
        self,
        *,
        name: str,
        stage_results: dict[str, DatasetCreationResults | SkippedStageResult],
        final_stage_name: str,
        stage_output_paths: dict[str, Path] | None = None,
    ) -> None:
        self.name = name
        self.stage_results = stage_results
        self.final_stage_name = final_stage_name
        self._stage_output_paths = stage_output_paths or {}

    def __getitem__(self, stage_name: str) -> DatasetCreationResults | SkippedStageResult:
        return self.stage_results[stage_name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.stage_results)

    def keys(self) -> KeysView[str]:
        return self.stage_results.keys()

    def items(self) -> ItemsView[str, DatasetCreationResults | SkippedStageResult]:
        return self.stage_results.items()

    @property
    def final_result(self) -> DatasetCreationResults:
        result = self.stage_results[self.final_stage_name]
        if isinstance(result, SkippedStageResult):
            raise DataDesignerWorkflowError(f"Final stage {self.final_stage_name!r} was skipped: {result.status}.")
        return result

    def load_dataset(self) -> pd.DataFrame:
        self.final_result
        return self.load_stage_output(self.final_stage_name)

    def load_analysis(self) -> DatasetProfilerResults:
        return self.final_result.load_analysis()

    def count_records(self) -> int:
        self.final_result
        return self.count_stage_output_records(self.final_stage_name)

    def get_stage_output_path(self, stage_name: str) -> Path:
        result = self.stage_results[stage_name]
        if isinstance(result, SkippedStageResult):
            raise DataDesignerWorkflowError(f"Stage {stage_name!r} was skipped: {result.status}.")
        return self._stage_output_paths.get(stage_name, result.artifact_storage.final_dataset_path)

    def load_stage_output(self, stage_name: str) -> pd.DataFrame:
        return _load_parquet_dataset(self.get_stage_output_path(stage_name))

    def count_stage_output_records(self, stage_name: str) -> int:
        return _count_parquet_records(self.get_stage_output_path(stage_name))

    def export(self, path: Path | str, *, format: ExportFormat | None = None) -> Path:
        self.final_result
        return _export_parquet_dataset(self.get_stage_output_path(self.final_stage_name), Path(path), format=format)

    def push_to_hub(self, *args: Any, **kwargs: Any) -> str:
        final_result = self.final_result
        if self.get_stage_output_path(self.final_stage_name) != final_result.artifact_storage.final_dataset_path:
            raise DataDesignerWorkflowError(
                "push_to_hub() does not support selected workflow outputs yet. "
                "Use export() for the selected output, or push the stage result directly."
            )
        return final_result.push_to_hub(*args, **kwargs)


class CompositeWorkflow:
    def __init__(self, *, name: str, data_designer: DataDesigner) -> None:
        _validate_dir_name(name, "workflow name")
        self.name = name
        self._data_designer = data_designer
        self._stages: list[_WorkflowStage] = []

    def add_stage(
        self,
        name: str,
        config_builder: DataDesignerConfigBuilder,
        *,
        num_records: int | None = None,
        on_success: OnSuccessCallback | None = None,
        on_success_version: str | None = None,
        postprocessors: list[ProcessorConfig] | None = None,
        output: str = "final",
        allow_empty: bool = False,
        sampling_strategy: SamplingStrategy = SamplingStrategy.ORDERED,
        selection_strategy: IndexRange | PartitionBlock | None = None,
    ) -> CompositeWorkflow:
        _validate_dir_name(name, "stage name")
        if any(stage.name == name for stage in self._stages):
            raise DataDesignerWorkflowError(f"Stage name {name!r} is already used in workflow {self.name!r}.")
        if num_records is not None and num_records < 1:
            raise DataDesignerWorkflowError("Stage num_records must be at least 1.")
        _validate_stage_output(output)
        _validate_stage_output_processor(output, config_builder, postprocessors or [])
        self._stages.append(
            _WorkflowStage(
                name=name,
                config_builder=_clone_config_builder(config_builder),
                depends_on=(self._stages[-1].name,) if self._stages else (),
                num_records=num_records,
                on_success=on_success,
                on_success_version=on_success_version,
                postprocessors=_clone_processors(postprocessors or []),
                output=output,
                allow_empty=allow_empty,
                sampling_strategy=sampling_strategy,
                selection_strategy=selection_strategy,
            )
        )
        return self

    def run(self) -> CompositeWorkflowResults:
        """Run all stages from scratch, replacing deterministic stage directories."""
        if not self._stages:
            raise DataDesignerWorkflowError(f"Workflow {self.name!r} has no stages.")

        workflow_path = self._data_designer._artifact_path / self.name
        workflow_path.mkdir(parents=True, exist_ok=True)
        metadata: dict[str, Any] = {
            "name": self.name,
            "library_version": get_library_version(),
            "stages": [],
        }
        stage_results: dict[str, DatasetCreationResults | SkippedStageResult] = {}
        stage_output_paths: dict[str, Path] = {}
        previous_seed_path: Path | None = None
        previous_output_records: int | None = None
        previous_stage_name: str | None = None
        previous_stage_fingerprint: str | None = None
        skipped_upstream_stage: str | None = None

        for index, stage in enumerate(self._stages):
            stage_dir_name = _stage_dir_name(index, stage.name)
            stage_metadata = _base_stage_metadata(index, stage, stage_dir_name)
            metadata["stages"].append(stage_metadata)

            if skipped_upstream_stage is not None:
                stage_metadata.update(
                    {
                        "status": "skipped_empty_upstream",
                        "upstream_stage": skipped_upstream_stage,
                    }
                )
                stage_results[stage.name] = SkippedStageResult(
                    status="skipped_empty_upstream",
                    upstream_stage=skipped_upstream_stage,
                )
                _write_workflow_metadata(workflow_path, metadata)
                continue

            stage_builder = _clone_config_builder(stage.config_builder)
            if previous_seed_path is not None:
                if stage_builder.get_seed_config() is not None:
                    logger.warning(
                        "Stage %r has a seed dataset; workflow will seed it from upstream stage %r.",
                        stage.name,
                        previous_stage_name,
                    )
                stage_builder.with_seed_dataset(
                    _local_seed_source_from_path(previous_seed_path),
                    sampling_strategy=stage.sampling_strategy,
                    selection_strategy=stage.selection_strategy,
                )

            num_records = stage.num_records or previous_output_records or DEFAULT_NUM_RECORDS
            stage_config = stage_builder.build()
            stage_fingerprint = _stage_fingerprint(
                stage_config=stage_config,
                stage=stage,
                num_records=num_records,
                upstream_fingerprint=previous_stage_fingerprint,
            )
            stage_path = workflow_path / stage_dir_name
            if stage_path.exists():
                shutil.rmtree(stage_path)

            stage_metadata.update(
                {
                    "status": "running",
                    "fingerprint": stage_fingerprint,
                    "num_records_requested": num_records,
                    "seeded_from_stage": previous_stage_name,
                    "seed_path": str(previous_seed_path) if previous_seed_path is not None else None,
                    "config": stage_config.model_dump(mode="json"),
                }
            )
            _write_workflow_metadata(workflow_path, metadata)

            start_time = time.monotonic()
            try:
                result = self._data_designer.create(
                    stage_builder,
                    num_records=num_records,
                    dataset_name=stage_dir_name,
                    artifact_path=workflow_path,
                )
                actual_records = result.count_records()
                output_result = result
                output_source_result = result
                if stage.postprocessors:
                    postprocessor_builder = _postprocessor_config_builder(
                        stage_builder=stage_builder,
                        seed_path=result.artifact_storage.final_dataset_path,
                        postprocessors=stage.postprocessors,
                    )
                    output_result = self._data_designer.create(
                        postprocessor_builder,
                        num_records=actual_records,
                        dataset_name="postprocessors",
                        artifact_path=workflow_path / stage_dir_name,
                    )
                    output_source_result = _select_output_result(stage, result, output_result)

                output_seed_path = _resolve_stage_output_path(output_source_result, stage.output)
                callback_output_path = None
                output_records = _count_parquet_records(output_seed_path)

                if stage.on_success is not None:
                    callback_output_path = Path(stage.on_success(result.artifact_storage.base_dataset_path))
                    output_seed_path = callback_output_path
                    output_records = _count_parquet_records(callback_output_path)

                if output_records == 0:
                    if not stage.allow_empty:
                        raise DataDesignerWorkflowError(f"Stage {stage.name!r} produced an empty output.")
                    status = "completed_empty"
                    skipped_upstream_stage = stage.name
                else:
                    status = "completed"

                stage_metadata.update(
                    {
                        "status": status,
                        "num_records_actual": actual_records,
                        "output_records": output_records,
                        "output_seed_path": str(output_seed_path),
                        "callback_output_path": str(callback_output_path) if callback_output_path else None,
                        "postprocessor_output_path": (
                            str(output_result.artifact_storage.base_dataset_path) if stage.postprocessors else None
                        ),
                        "duration_sec": time.monotonic() - start_time,
                    }
                )
            except Exception:
                stage_metadata.update({"status": "failed", "duration_sec": time.monotonic() - start_time})
                _write_workflow_metadata(workflow_path, metadata)
                raise

            stage_results[stage.name] = output_result
            stage_output_paths[stage.name] = output_seed_path
            previous_seed_path = output_seed_path
            previous_output_records = output_records
            previous_stage_name = stage.name
            previous_stage_fingerprint = stage_fingerprint
            _write_workflow_metadata(workflow_path, metadata)

        return CompositeWorkflowResults(
            name=self.name,
            stage_results=stage_results,
            final_stage_name=self._stages[-1].name,
            stage_output_paths=stage_output_paths,
        )


def _clone_config_builder(config_builder: DataDesignerConfigBuilder) -> DataDesignerConfigBuilder:
    return DataDesignerConfigBuilder.from_config(BuilderConfig(data_designer=config_builder.build()))


def _clone_processors(processors: list[ProcessorConfig]) -> tuple[ProcessorConfig, ...]:
    return tuple(processor.model_copy(deep=True) for processor in processors)


def _postprocessor_config_builder(
    *,
    stage_builder: DataDesignerConfigBuilder,
    seed_path: Path,
    postprocessors: tuple[ProcessorConfig, ...],
) -> DataDesignerConfigBuilder:
    builder = DataDesignerConfigBuilder(
        model_configs=stage_builder.model_configs,
        tool_configs=stage_builder.tool_configs,
    ).with_seed_dataset(_local_seed_source_from_path(seed_path))
    for processor in postprocessors:
        builder.add_processor(processor.model_copy(deep=True))
    return builder


def _stage_dir_name(index: int, name: str) -> str:
    return f"stage-{index}-{name}"


def _base_stage_metadata(index: int, stage: _WorkflowStage, stage_dir_name: str) -> dict[str, Any]:
    return {
        "index": index,
        "name": stage.name,
        "stage_dir": stage_dir_name,
        "depends_on": list(stage.depends_on),
        "allow_empty": stage.allow_empty,
        "on_success_version": stage.on_success_version,
        "postprocessors": [processor.model_dump(mode="json") for processor in stage.postprocessors],
        "output": stage.output,
        "sampling_strategy": stage.sampling_strategy.value,
        "selection_strategy": _selection_strategy_payload(stage.selection_strategy),
    }


def _stage_fingerprint(
    *,
    stage_config: DataDesignerConfig,
    stage: _WorkflowStage,
    num_records: int,
    upstream_fingerprint: str | None,
) -> str:
    payload = {
        "config_fingerprint": stage_config.fingerprint(),
        "num_records": num_records,
        "sampling_strategy": stage.sampling_strategy.value,
        "selection_strategy": _selection_strategy_payload(stage.selection_strategy),
        "allow_empty": stage.allow_empty,
        "on_success_version": stage.on_success_version,
        "postprocessors": [processor.model_dump(mode="json") for processor in stage.postprocessors],
        "output": stage.output,
        "library_version": get_library_version(),
        "upstream_fingerprint": upstream_fingerprint,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _selection_strategy_payload(selection_strategy: IndexRange | PartitionBlock | None) -> dict[str, Any] | None:
    if selection_strategy is None:
        return None
    return selection_strategy.model_dump(mode="json")


def _local_seed_source_from_path(path: Path) -> LocalFileSeedSource:
    if path.is_dir():
        return LocalFileSeedSource(path=str(path / "*.parquet"))
    return LocalFileSeedSource(path=str(path))


def _resolve_stage_output_path(result: DatasetCreationResults, output: str) -> Path:
    if output == "final":
        return result.artifact_storage.final_dataset_path
    processor_name = output.removeprefix("processor:")
    processor_path = result.artifact_storage.processors_outputs_path / processor_name
    if processor_path.exists():
        return processor_path
    processor_file_path = result.artifact_storage.processors_outputs_path / f"{processor_name}.parquet"
    if processor_file_path.exists():
        return processor_file_path
    raise DataDesignerWorkflowError(f"Stage output processor {processor_name!r} did not produce artifacts.")


def _select_output_result(
    stage: _WorkflowStage,
    result: DatasetCreationResults,
    output_result: DatasetCreationResults,
) -> DatasetCreationResults:
    if stage.output == "final":
        return output_result
    processor_name = stage.output.removeprefix("processor:")
    if processor_name in {processor.name for processor in stage.postprocessors}:
        return output_result
    return result


def _count_parquet_records(path: Path) -> int:
    parquet_files = _parquet_files(path)
    if not parquet_files:
        raise DataDesignerWorkflowError(f"No parquet files found at {str(path)!r}.")
    return sum(lazy.pq.read_metadata(file_path).num_rows for file_path in parquet_files)


def _load_parquet_dataset(path: Path) -> pd.DataFrame:
    parquet_files = _parquet_files(path)
    if not parquet_files:
        raise DataDesignerWorkflowError(f"No parquet files found at {str(path)!r}.")
    return lazy.pd.concat([lazy.pd.read_parquet(file_path) for file_path in parquet_files], ignore_index=True)


def _export_parquet_dataset(source_path: Path, output_path: Path, *, format: ExportFormat | None = None) -> Path:
    resolved_format: str = format if format is not None else output_path.suffix.lstrip(".").lower()
    if resolved_format not in SUPPORTED_EXPORT_FORMATS:
        raise InvalidFileFormatError(
            f"Unsupported export format: {resolved_format!r}. Choose one of: {', '.join(SUPPORTED_EXPORT_FORMATS)}."
        )
    parquet_files = _parquet_files(source_path)
    if not parquet_files:
        raise ArtifactStorageError("No parquet files found to export.")
    if resolved_format == "jsonl":
        _export_jsonl(parquet_files, output_path)
    elif resolved_format == "csv":
        _export_csv(parquet_files, output_path)
    elif resolved_format == "parquet":
        _export_parquet(parquet_files, output_path)
    return output_path


def _parquet_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return [path]
    return sorted(path.glob("*.parquet"))


def _write_workflow_metadata(workflow_path: Path, metadata: dict[str, Any]) -> None:
    path = workflow_path / "workflow-metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _validate_stage_output(output: str) -> None:
    if output == "final":
        return
    if not output.startswith("processor:"):
        raise DataDesignerWorkflowError("Stage output must be 'final' or 'processor:<name>'.")
    processor_name = output.removeprefix("processor:")
    _validate_dir_name(processor_name, "processor output name")


def _validate_stage_output_processor(
    output: str,
    config_builder: DataDesignerConfigBuilder,
    postprocessors: list[ProcessorConfig],
) -> None:
    if not output.startswith("processor:"):
        return
    processor_name = output.removeprefix("processor:")
    processor_names = {processor.name for processor in config_builder.get_processor_configs()}
    processor_names.update(processor.name for processor in postprocessors)
    if processor_name not in processor_names:
        raise DataDesignerWorkflowError(f"Stage output processor {processor_name!r} is not configured on this stage.")


def _validate_dir_name(name: str, label: str) -> None:
    if not name:
        raise DataDesignerWorkflowError(f"{label} must be a non-empty string.")
    if name in {".", ".."}:
        raise DataDesignerWorkflowError(f"{label} {name!r} is not allowed.")
    invalid_chars = {"<", ">", ":", '"', "/", "\\", "|", "?", "*"}
    if any(char in name for char in invalid_chars):
        raise DataDesignerWorkflowError(f"{label} {name!r} contains invalid path characters.")
