# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.config_builder import BuilderConfig, DataDesignerConfigBuilder
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.seed import IndexRange, PartitionBlock, SamplingStrategy
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.config.utils.constants import DEFAULT_NUM_RECORDS
from data_designer.config.version import get_library_version
from data_designer.interface.errors import DataDesignerWorkflowError
from data_designer.interface.results import DatasetCreationResults

if TYPE_CHECKING:
    from data_designer.interface.data_designer import DataDesigner


OnSuccessCallback = Callable[[Path], Path | str]


@dataclass(frozen=True)
class _WorkflowStage:
    name: str
    config_builder: DataDesignerConfigBuilder
    depends_on: tuple[str, ...]
    num_records: int | None
    on_success: OnSuccessCallback | None
    on_success_version: str | None
    allow_empty: bool
    sampling_strategy: SamplingStrategy
    selection_strategy: IndexRange | PartitionBlock | None


@dataclass(frozen=True)
class SkippedStageResult:
    status: str
    upstream_stage: str


class CompositeWorkflowResults:
    def __init__(
        self,
        *,
        name: str,
        stage_results: dict[str, DatasetCreationResults | SkippedStageResult],
        final_stage_name: str,
    ):
        self.name = name
        self.stage_results = stage_results
        self.final_stage_name = final_stage_name

    def __getitem__(self, stage_name: str) -> DatasetCreationResults | SkippedStageResult:
        return self.stage_results[stage_name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.stage_results)

    def keys(self):
        return self.stage_results.keys()

    def items(self):
        return self.stage_results.items()

    @property
    def final_result(self) -> DatasetCreationResults:
        result = self.stage_results[self.final_stage_name]
        if isinstance(result, SkippedStageResult):
            raise DataDesignerWorkflowError(f"Final stage {self.final_stage_name!r} was skipped: {result.status}.")
        return result

    def load_dataset(self):
        return self.final_result.load_dataset()

    def load_analysis(self):
        return self.final_result.load_analysis()

    def count_records(self) -> int:
        return self.final_result.count_records()

    def export(self, *args, **kwargs):
        return self.final_result.export(*args, **kwargs)

    def push_to_hub(self, *args, **kwargs):
        return self.final_result.push_to_hub(*args, **kwargs)


class CompositeWorkflow:
    def __init__(self, *, name: str, data_designer: DataDesigner):
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
        allow_empty: bool = False,
        sampling_strategy: SamplingStrategy = SamplingStrategy.ORDERED,
        selection_strategy: IndexRange | PartitionBlock | None = None,
    ) -> CompositeWorkflow:
        _validate_dir_name(name, "stage name")
        if any(stage.name == name for stage in self._stages):
            raise DataDesignerWorkflowError(f"Stage name {name!r} is already used in workflow {self.name!r}.")
        if num_records is not None and num_records < 1:
            raise DataDesignerWorkflowError("Stage num_records must be at least 1.")
        self._stages.append(
            _WorkflowStage(
                name=name,
                config_builder=config_builder,
                depends_on=(self._stages[-1].name,) if self._stages else (),
                num_records=num_records,
                on_success=on_success,
                on_success_version=on_success_version,
                allow_empty=allow_empty,
                sampling_strategy=sampling_strategy,
                selection_strategy=selection_strategy,
            )
        )
        return self

    def run(self) -> CompositeWorkflowResults:
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
                with _temporary_artifact_path(self._data_designer, workflow_path):
                    result = self._data_designer.create(
                        stage_builder,
                        num_records=num_records,
                        dataset_name=stage_dir_name,
                    )
                actual_records = result.count_records()
                output_seed_path = result.artifact_storage.final_dataset_path
                callback_output_path = None
                output_records = actual_records

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
                        "duration_sec": time.monotonic() - start_time,
                    }
                )
            except Exception:
                stage_metadata.update({"status": "failed", "duration_sec": time.monotonic() - start_time})
                _write_workflow_metadata(workflow_path, metadata)
                raise

            stage_results[stage.name] = result
            previous_seed_path = output_seed_path
            previous_output_records = output_records
            previous_stage_name = stage.name
            previous_stage_fingerprint = stage_fingerprint
            _write_workflow_metadata(workflow_path, metadata)

        return CompositeWorkflowResults(
            name=self.name,
            stage_results=stage_results,
            final_stage_name=self._stages[-1].name,
        )


def _clone_config_builder(config_builder: DataDesignerConfigBuilder) -> DataDesignerConfigBuilder:
    return DataDesignerConfigBuilder.from_config(BuilderConfig(data_designer=config_builder.build()))


@contextmanager
def _temporary_artifact_path(data_designer: DataDesigner, artifact_path: Path):
    original_artifact_path = data_designer._artifact_path
    data_designer._artifact_path = artifact_path
    try:
        yield
    finally:
        data_designer._artifact_path = original_artifact_path


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


def _count_parquet_records(path: Path) -> int:
    parquet_files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not parquet_files:
        raise DataDesignerWorkflowError(f"No parquet files found at {str(path)!r}.")
    return sum(lazy.pq.read_metadata(file_path).num_rows for file_path in parquet_files)


def _write_workflow_metadata(workflow_path: Path, metadata: dict[str, Any]) -> None:
    path = workflow_path / "workflow-metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _validate_dir_name(name: str, label: str) -> None:
    if not name:
        raise DataDesignerWorkflowError(f"{label} must be a non-empty string.")
    invalid_chars = {"<", ">", ":", '"', "/", "\\", "|", "?", "*"}
    if any(char in name for char in invalid_chars):
        raise DataDesignerWorkflowError(f"{label} {name!r} contains invalid path characters.")
