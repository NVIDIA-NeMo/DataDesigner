# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from collections.abc import Callable, ItemsView, Iterator, KeysView
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults
from data_designer.config.base import ProcessorConfig
from data_designer.config.config_builder import BuilderConfig, DataDesignerConfigBuilder
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.dataset_metadata import DatasetMetadata
from data_designer.config.errors import InvalidFileFormatError
from data_designer.config.seed import IndexRange, PartitionBlock, SamplingStrategy
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.config.utils.constants import DEFAULT_NUM_RECORDS
from data_designer.config.utils.type_helpers import StrEnum
from data_designer.config.version import get_library_version
from data_designer.engine.dataset_builders.errors import ArtifactStorageError
from data_designer.engine.storage.artifact_storage import ArtifactStorage, ResumeMode
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

    from data_designer.interface.data_designer import DataDesigner


logger = logging.getLogger(__name__)

OnSuccessCallback = Callable[[Path], Path | str]
StageTargets = str | list[str] | tuple[str, ...] | set[str]
WORKFLOW_METADATA_FILENAME = "workflow-metadata.json"
COMPLETED_STAGE_STATUSES = {"completed", "completed_empty"}
RESUMABLE_STAGE_STATUSES = {"running", "failed"}
WORKFLOW_PATH_METADATA_KEYS = (
    "seed_path",
    "output_seed_path",
    "callback_output_path",
    "output_processor_output_path",
    "stage_output_override_path",
    "repeat_until_output_path",
)


class RepeatUntilMode(StrEnum):
    APPEND = "append"
    DISCARD = "discard"


class RepeatUntilExhaustion(StrEnum):
    RAISE = "raise"
    RETURN_PARTIAL = "return_partial"


@dataclass(frozen=True)
class RepeatUntil:
    """Bounded stage-level retry policy for exact selected-output counts."""

    output_records: int
    max_iterations: int
    mode: RepeatUntilMode | str = RepeatUntilMode.APPEND
    max_generated_records: int | None = None
    on_exhausted: RepeatUntilExhaustion | str = RepeatUntilExhaustion.RAISE
    trim: bool = True

    def __post_init__(self) -> None:
        if self.output_records < 1:
            raise DataDesignerWorkflowError("repeat_until.output_records must be at least 1.")
        if self.max_iterations < 1:
            raise DataDesignerWorkflowError("repeat_until.max_iterations must be at least 1.")
        if self.max_generated_records is not None and self.max_generated_records < 1:
            raise DataDesignerWorkflowError("repeat_until.max_generated_records must be at least 1.")
        try:
            mode = RepeatUntilMode(self.mode)
        except ValueError as exc:
            raise DataDesignerWorkflowError(
                f"repeat_until.mode must be one of: {_enum_values(RepeatUntilMode)}."
            ) from exc
        try:
            on_exhausted = RepeatUntilExhaustion(self.on_exhausted)
        except ValueError as exc:
            raise DataDesignerWorkflowError(
                f"repeat_until.on_exhausted must be one of: {_enum_values(RepeatUntilExhaustion)}."
            ) from exc
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "on_exhausted", on_exhausted)


@dataclass(frozen=True)
class _StageRunResult:
    output_result: DatasetCreationResults
    actual_records: int
    output_seed_path: Path
    output_records: int
    callback_output_path: Path | None
    output_processor_output_path: Path | None
    num_records_requested: int
    repeat_iterations: int | None = None
    repeat_generated_records: int | None = None
    repeat_satisfied: bool | None = None
    repeat_until_output_path: Path | None = None


@dataclass(frozen=True)
class _WorkflowStage:
    name: str
    config_builder: DataDesignerConfigBuilder
    depends_on: tuple[str, ...]
    num_records: int | None
    on_success: OnSuccessCallback | None
    on_success_version: str | None
    output_processors: tuple[ProcessorConfig, ...]
    output: str
    allow_empty: bool
    sampling_strategy: SamplingStrategy
    selection_strategy: IndexRange | PartitionBlock | None
    repeat_until: RepeatUntil | None


class SkippedStageStatus(StrEnum):
    SKIPPED_EMPTY_UPSTREAM = "skipped_empty_upstream"


@dataclass(frozen=True)
class SkippedStageResult:
    status: SkippedStageStatus
    upstream_stage: str


class CompositeWorkflowResults:
    """Results for a composite workflow run.

    Per-stage entries are the effective ``DataDesigner.create()`` results. For
    stages with ``output_processors``, this is the output-processor create
    result. Use ``load_stage_output()`` to load the selected output handed
    downstream; it follows ``output="processor:<name>"`` and ``on_success``.
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
        """Return the final stage result, or raise if it was skipped."""
        return self._require_final_result()

    def _require_final_result(self) -> DatasetCreationResults:
        result = self.stage_results[self.final_stage_name]
        if isinstance(result, SkippedStageResult):
            raise DataDesignerWorkflowError(
                f"Final stage {self.final_stage_name!r} was skipped: {result.status.value}."
            )
        return result

    def load_dataset(self) -> pd.DataFrame:
        """Load the selected output from the final workflow stage."""
        self._require_final_result()
        return self.load_stage_output(self.final_stage_name)

    def load_analysis(self) -> DatasetProfilerResults:
        """Load analysis from the final stage result."""
        return self.final_result.load_analysis()

    def count_records(self) -> int:
        """Count records in the selected output from the final workflow stage."""
        self._require_final_result()
        return self.count_stage_output_records(self.final_stage_name)

    def get_stage_output_path(self, stage_name: str) -> Path:
        """Return the selected output path handed downstream for a stage."""
        result = self.stage_results[stage_name]
        if isinstance(result, SkippedStageResult):
            raise DataDesignerWorkflowError(f"Stage {stage_name!r} was skipped: {result.status.value}.")
        return self._stage_output_paths.get(stage_name, result.artifact_storage.final_dataset_path)

    def load_stage_output(self, stage_name: str) -> pd.DataFrame:
        """Load the selected output handed downstream for a stage."""
        return _load_parquet_dataset(self.get_stage_output_path(stage_name))

    def count_stage_output_records(self, stage_name: str) -> int:
        """Count records in the selected output handed downstream for a stage."""
        return _count_parquet_records(self.get_stage_output_path(stage_name))

    def export(self, path: Path | str, *, format: ExportFormat | None = None) -> Path:
        """Export the selected output from the final workflow stage."""
        self._require_final_result()
        return _export_parquet_dataset(self.get_stage_output_path(self.final_stage_name), Path(path), format=format)

    def export_stage(self, stage_name: str, path: Path | str, *, format: ExportFormat | None = None) -> Path:
        """Export the selected output from a workflow stage."""
        return _export_parquet_dataset(self.get_stage_output_path(stage_name), Path(path), format=format)

    def push_to_hub(self, *args: Any, **kwargs: Any) -> str:
        """Push the final stage result to Hugging Face Hub when no output override is selected."""
        final_result = self.final_result
        if self.get_stage_output_path(self.final_stage_name) != final_result.artifact_storage.final_dataset_path:
            raise DataDesignerWorkflowError(
                "push_to_hub() does not support selected workflow outputs yet. "
                "Use export() for the selected output, or push the stage result directly."
            )
        return final_result.push_to_hub(*args, **kwargs)


class CompositeWorkflow:
    """Experimental linear workflow for chaining Data Designer stages."""

    def __init__(self, *, name: str, data_designer: DataDesigner) -> None:
        """Create a workflow bound to a parent Data Designer instance."""
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
        output_processors: list[ProcessorConfig] | None = None,
        output: str = "final",
        allow_empty: bool = False,
        sampling_strategy: SamplingStrategy = SamplingStrategy.ORDERED,
        selection_strategy: IndexRange | PartitionBlock | None = None,
        repeat_until: RepeatUntil | None = None,
    ) -> CompositeWorkflow:
        """Add a stage to the workflow.

        Processors on ``config_builder`` run inside the main stage. Use
        ``output_processors`` for stage-boundary transforms whose output should
        feed downstream stages by default. ``output="processor:<name>"`` selects
        a named processor artifact, and ``on_success`` can override the selected
        output by returning a parquet file or directory. Use ``repeat_until`` to
        rerun the stage until the selected output reaches an exact row count.
        """
        _validate_dir_name(name, "stage name")
        if any(stage.name == name for stage in self._stages):
            raise DataDesignerWorkflowError(f"Stage name {name!r} is already used in workflow {self.name!r}.")
        if num_records is not None and num_records < 1:
            raise DataDesignerWorkflowError("Stage num_records must be at least 1.")
        _validate_stage_output(output)
        output_processors = output_processors or []
        _validate_distinct_output_processors(config_builder, output_processors)
        _validate_stage_output_processor(output, config_builder, output_processors)
        self._stages.append(
            _WorkflowStage(
                name=name,
                config_builder=_clone_config_builder(config_builder),
                depends_on=(self._stages[-1].name,) if self._stages else (),
                num_records=num_records,
                on_success=on_success,
                on_success_version=on_success_version,
                output_processors=_clone_processors(output_processors),
                output=output,
                allow_empty=allow_empty,
                sampling_strategy=sampling_strategy,
                selection_strategy=selection_strategy,
                repeat_until=repeat_until,
            )
        )
        return self

    def run(
        self,
        *,
        resume: ResumeMode = ResumeMode.NEVER,
        targets: StageTargets | None = None,
        rerun_from: str | None = None,
        stage_output_overrides: dict[str, Path | str] | None = None,
    ) -> CompositeWorkflowResults:
        """Run all stages, optionally reusing compatible completed stage outputs.

        Each stage writes a deterministic artifact directory under the parent
        Data Designer artifact path. Downstream stages are seeded from the
        selected output of the previous stage.
        """
        if not self._stages:
            raise DataDesignerWorkflowError(f"Workflow {self.name!r} has no stages.")
        if rerun_from is not None and resume == ResumeMode.NEVER:
            raise DataDesignerWorkflowError(
                "rerun_from requires resume to be ResumeMode.IF_POSSIBLE or ResumeMode.ALWAYS."
            )

        stage_indices = _stage_indices_by_name(self._stages)
        target_stage_names = _normalize_stage_names(targets, stage_indices, "target")
        target_stage_index = max(stage_indices[name] for name in target_stage_names)
        rerun_from_index = _stage_index_or_none(rerun_from, stage_indices, "rerun_from")
        if rerun_from_index is not None and rerun_from_index > target_stage_index:
            raise DataDesignerWorkflowError("rerun_from must be an ancestor of at least one target stage.")
        stage_output_overrides = stage_output_overrides or {}
        _validate_stage_output_overrides(stage_output_overrides, stage_indices, target_stage_index)

        workflow_path = self._data_designer.artifact_path / self.name
        workflow_path.mkdir(parents=True, exist_ok=True)
        prior_metadata = _read_prior_workflow_metadata(workflow_path, self.name, resume)
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
        # A stage that runs or resumes may produce new data, so descendants rebuild from its current output.
        force_rerun_downstream = False

        for index, stage in enumerate(self._stages):
            if index > target_stage_index:
                break

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
                    status=SkippedStageStatus.SKIPPED_EMPTY_UPSTREAM,
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
            prior_stage_metadata = _get_prior_stage_metadata(prior_metadata, index, stage, stage_dir_name)
            stage_resume = ResumeMode.NEVER
            force_rerun_current = rerun_from_index is not None and index >= rerun_from_index
            prior_matches = (
                not force_rerun_downstream
                and not force_rerun_current
                and prior_stage_metadata is not None
                and prior_stage_metadata.get("fingerprint") == stage_fingerprint
            )

            if prior_matches and _can_skip_prior_stage(stage, prior_stage_metadata, workflow_path):
                stage_metadata.update(prior_stage_metadata)
                output_seed_path = _resolve_metadata_path(workflow_path, stage_metadata["output_seed_path"])
                _normalize_stage_path_metadata(workflow_path, stage_metadata)
                override_path = _stage_output_override(stage.name, stage_output_overrides)
                if override_path is not None:
                    output_seed_path = override_path
                    stage_metadata["output_seed_path"] = _metadata_path_value(workflow_path, output_seed_path)
                    stage_metadata["stage_output_override_path"] = _metadata_path_value(workflow_path, output_seed_path)
                    force_rerun_downstream = True
                output_records = _count_parquet_records(output_seed_path)
                stage_metadata["output_records"] = output_records
                if override_path is not None:
                    if output_records == 0:
                        if not stage.allow_empty:
                            raise DataDesignerWorkflowError(f"Stage {stage.name!r} produced an empty output.")
                        stage_metadata["status"] = "completed_empty"
                    else:
                        stage_metadata["status"] = "completed"
                output_result = _stage_result_from_metadata(
                    workflow_path=workflow_path,
                    stage=stage,
                    stage_dir_name=stage_dir_name,
                    stage_builder=stage_builder,
                )
                stage_results[stage.name] = output_result
                stage_output_paths[stage.name] = output_seed_path
                previous_seed_path = output_seed_path
                previous_output_records = None if stage_metadata["status"] == "completed_empty" else output_records
                previous_stage_name = stage.name
                previous_stage_fingerprint = stage_fingerprint
                if stage_metadata["status"] == "completed_empty":
                    skipped_upstream_stage = stage.name
                _write_workflow_metadata(workflow_path, metadata)
                continue

            if prior_matches and prior_stage_metadata.get("status") in RESUMABLE_STAGE_STATUSES and stage_path.exists():
                stage_resume = ResumeMode.ALWAYS
            elif resume == ResumeMode.ALWAYS and not force_rerun_downstream and not force_rerun_current:
                raise DataDesignerWorkflowError(
                    f"Cannot resume workflow {self.name!r}: stage {stage.name!r} is not reusable."
                )

            if stage_resume == ResumeMode.NEVER and stage_path.exists():
                shutil.rmtree(stage_path)

            stage_metadata.update(
                {
                    "status": "running",
                    "fingerprint": stage_fingerprint,
                    "num_records_requested": num_records,
                    "seeded_from_stage": previous_stage_name,
                    "seed_path": (
                        _metadata_path_value(workflow_path, previous_seed_path)
                        if previous_seed_path is not None
                        else None
                    ),
                    "config": stage_config.model_dump(mode="json"),
                }
            )
            _write_workflow_metadata(workflow_path, metadata)

            start_time = time.monotonic()
            try:
                run_result = self._run_stage(
                    stage=stage,
                    stage_builder=stage_builder,
                    workflow_path=workflow_path,
                    stage_dir_name=stage_dir_name,
                    stage_path=stage_path,
                    num_records=num_records,
                    resume=stage_resume,
                )
                output_seed_path = run_result.output_seed_path
                override_path = _stage_output_override(stage.name, stage_output_overrides)
                if override_path is not None:
                    output_seed_path = override_path
                output_records = _count_parquet_records(output_seed_path)

                if output_records == 0:
                    if not _allows_empty_stage_output(stage, run_result):
                        raise DataDesignerWorkflowError(f"Stage {stage.name!r} produced an empty output.")
                    status = "completed_empty"
                    skipped_upstream_stage = stage.name
                else:
                    status = "completed"

                stage_metadata.update(
                    {
                        "status": status,
                        "num_records_requested": run_result.num_records_requested,
                        "num_records_actual": run_result.actual_records,
                        "output_records": output_records,
                        "output_seed_path": _metadata_path_value(workflow_path, output_seed_path),
                        "callback_output_path": (
                            _metadata_path_value(workflow_path, run_result.callback_output_path)
                            if run_result.callback_output_path
                            else None
                        ),
                        "stage_output_override_path": (
                            _metadata_path_value(workflow_path, override_path) if override_path else None
                        ),
                        "output_processor_output_path": (
                            _metadata_path_value(workflow_path, run_result.output_processor_output_path)
                            if run_result.output_processor_output_path
                            else None
                        ),
                        "duration_sec": time.monotonic() - start_time,
                    }
                )
                if run_result.repeat_iterations is not None:
                    stage_metadata.update(
                        {
                            "repeat_iterations": run_result.repeat_iterations,
                            "repeat_generated_records": run_result.repeat_generated_records,
                            "repeat_satisfied": run_result.repeat_satisfied,
                            "repeat_until_output_path": (
                                _metadata_path_value(workflow_path, run_result.repeat_until_output_path)
                                if run_result.repeat_until_output_path
                                else None
                            ),
                        }
                    )
            except Exception:
                stage_metadata.update({"status": "failed", "duration_sec": time.monotonic() - start_time})
                _write_workflow_metadata(workflow_path, metadata)
                raise

            stage_results[stage.name] = run_result.output_result
            stage_output_paths[stage.name] = output_seed_path
            previous_seed_path = output_seed_path
            previous_output_records = None if status == "completed_empty" else output_records
            previous_stage_name = stage.name
            previous_stage_fingerprint = stage_fingerprint
            force_rerun_downstream = True
            _write_workflow_metadata(workflow_path, metadata)

        return CompositeWorkflowResults(
            name=self.name,
            stage_results=stage_results,
            final_stage_name=self._stages[target_stage_index].name,
            stage_output_paths=stage_output_paths,
        )

    def _run_stage(
        self,
        *,
        stage: _WorkflowStage,
        stage_builder: DataDesignerConfigBuilder,
        workflow_path: Path,
        stage_dir_name: str,
        stage_path: Path,
        num_records: int,
        resume: ResumeMode,
    ) -> _StageRunResult:
        if stage.repeat_until is None:
            return self._run_stage_attempt(
                stage=stage,
                stage_builder=stage_builder,
                artifact_path=workflow_path,
                dataset_name=stage_dir_name,
                num_records=num_records,
                resume=resume,
            )
        if stage.repeat_until.mode == RepeatUntilMode.DISCARD:
            return self._run_stage_until_discard(
                stage=stage,
                stage_builder=stage_builder,
                workflow_path=workflow_path,
                stage_dir_name=stage_dir_name,
                stage_path=stage_path,
                num_records=num_records,
                resume=resume,
            )
        return self._run_stage_until_append(
            stage=stage,
            stage_builder=stage_builder,
            workflow_path=workflow_path,
            stage_dir_name=stage_dir_name,
            stage_path=stage_path,
            num_records=num_records,
            resume=resume,
        )

    def _run_stage_attempt(
        self,
        *,
        stage: _WorkflowStage,
        stage_builder: DataDesignerConfigBuilder,
        artifact_path: Path,
        dataset_name: str,
        num_records: int,
        resume: ResumeMode,
    ) -> _StageRunResult:
        result = self._data_designer.create(
            stage_builder,
            num_records=num_records,
            dataset_name=dataset_name,
            artifact_path=artifact_path,
            resume=resume,
        )
        actual_records = result.count_records()
        output_result = result
        output_source_result = result
        stage_path = artifact_path / dataset_name
        output_processor_output_path = None
        if stage.output_processors:
            output_processor_path = stage_path / "output-processors"
            if output_processor_path.exists():
                shutil.rmtree(output_processor_path)
            output_processor_builder = _output_processor_config_builder(
                stage_builder=stage_builder,
                seed_path=result.artifact_storage.final_dataset_path,
                output_processors=stage.output_processors,
            )
            output_result = self._data_designer.create(
                output_processor_builder,
                num_records=actual_records,
                dataset_name="output-processors",
                artifact_path=stage_path,
            )
            output_source_result = _select_output_result(stage, result, output_result)
            output_processor_output_path = output_result.artifact_storage.base_dataset_path

        callback_output_path = None
        if stage.on_success is not None:
            callback_output_path = Path(stage.on_success(result.artifact_storage.base_dataset_path))
            output_seed_path = callback_output_path
        else:
            output_seed_path = _resolve_stage_output_path(output_source_result, stage.output)
        output_records = _count_parquet_records(output_seed_path)
        return _StageRunResult(
            output_result=output_result,
            actual_records=actual_records,
            output_seed_path=output_seed_path,
            output_records=output_records,
            callback_output_path=callback_output_path,
            output_processor_output_path=output_processor_output_path,
            num_records_requested=num_records,
        )

    def _run_stage_until_append(
        self,
        *,
        stage: _WorkflowStage,
        stage_builder: DataDesignerConfigBuilder,
        workflow_path: Path,
        stage_dir_name: str,
        stage_path: Path,
        num_records: int,
        resume: ResumeMode,
    ) -> _StageRunResult:
        repeat_until = _require_repeat_until(stage)
        last_result = None
        for iteration in range(1, repeat_until.max_iterations + 1):
            requested_records = num_records * iteration
            if _exceeds_max_generated_records(repeat_until, requested_records):
                break
            attempt_resume = resume if iteration == 1 else ResumeMode.ALWAYS
            last_result = self._run_stage_attempt(
                stage=stage,
                stage_builder=stage_builder,
                artifact_path=workflow_path,
                dataset_name=stage_dir_name,
                num_records=requested_records,
                resume=attempt_resume,
            )
            if last_result.output_records >= repeat_until.output_records:
                return _with_repeat_result(
                    last_result,
                    stage_path=stage_path,
                    repeat_until=repeat_until,
                    iterations=iteration,
                    generated_records=last_result.actual_records,
                    satisfied=True,
                )

        return _handle_repeat_until_exhausted(
            stage=stage,
            repeat_until=repeat_until,
            last_result=last_result,
            stage_path=stage_path,
            iterations=(last_result.num_records_requested // num_records if last_result else 0),
            generated_records=(last_result.actual_records if last_result else 0),
        )

    def _run_stage_until_discard(
        self,
        *,
        stage: _WorkflowStage,
        stage_builder: DataDesignerConfigBuilder,
        workflow_path: Path,
        stage_dir_name: str,
        stage_path: Path,
        num_records: int,
        resume: ResumeMode,
    ) -> _StageRunResult:
        repeat_until = _require_repeat_until(stage)
        if resume == ResumeMode.ALWAYS:
            logger.warning(
                "Stage %r uses repeat_until mode='discard'; previous attempts cannot be resumed and will be replaced.",
                stage.name,
            )
        last_result = None
        generated_records = 0
        iterations_run = 0
        for iteration in range(1, repeat_until.max_iterations + 1):
            if _exceeds_max_generated_records(repeat_until, generated_records + num_records):
                break
            if stage_path.exists():
                shutil.rmtree(stage_path)
            last_result = self._run_stage_attempt(
                stage=stage,
                stage_builder=stage_builder,
                artifact_path=workflow_path,
                dataset_name=stage_dir_name,
                num_records=num_records,
                resume=ResumeMode.NEVER,
            )
            iterations_run = iteration
            generated_records += last_result.actual_records
            if last_result.output_records >= repeat_until.output_records:
                return _with_repeat_result(
                    last_result,
                    stage_path=stage_path,
                    repeat_until=repeat_until,
                    iterations=iteration,
                    generated_records=generated_records,
                    satisfied=True,
                )

        return _handle_repeat_until_exhausted(
            stage=stage,
            repeat_until=repeat_until,
            last_result=last_result,
            stage_path=stage_path,
            iterations=iterations_run,
            generated_records=generated_records,
        )


def _stage_indices_by_name(stages: list[_WorkflowStage]) -> dict[str, int]:
    return {stage.name: index for index, stage in enumerate(stages)}


def _require_repeat_until(stage: _WorkflowStage) -> RepeatUntil:
    if stage.repeat_until is None:
        raise DataDesignerWorkflowError(f"Stage {stage.name!r} has no repeat_until policy.")
    return stage.repeat_until


def _exceeds_max_generated_records(repeat_until: RepeatUntil, generated_records: int) -> bool:
    return repeat_until.max_generated_records is not None and generated_records > repeat_until.max_generated_records


def _with_repeat_result(
    result: _StageRunResult,
    *,
    stage_path: Path,
    repeat_until: RepeatUntil,
    iterations: int,
    generated_records: int,
    satisfied: bool,
) -> _StageRunResult:
    output_seed_path = result.output_seed_path
    output_records = result.output_records
    repeat_until_output_path = None
    if output_records > repeat_until.output_records and repeat_until.trim:
        repeat_until_output_path = stage_path / "repeat-until" / "selected-output"
        _write_parquet_head(output_seed_path, repeat_until_output_path, repeat_until.output_records)
        output_seed_path = repeat_until_output_path
        output_records = repeat_until.output_records
    return _StageRunResult(
        output_result=result.output_result,
        actual_records=result.actual_records,
        output_seed_path=output_seed_path,
        output_records=output_records,
        callback_output_path=result.callback_output_path,
        output_processor_output_path=result.output_processor_output_path,
        num_records_requested=result.num_records_requested,
        repeat_iterations=iterations,
        repeat_generated_records=generated_records,
        repeat_satisfied=satisfied,
        repeat_until_output_path=repeat_until_output_path,
    )


def _handle_repeat_until_exhausted(
    *,
    stage: _WorkflowStage,
    repeat_until: RepeatUntil,
    last_result: _StageRunResult | None,
    stage_path: Path,
    iterations: int,
    generated_records: int,
) -> _StageRunResult:
    selected_records = last_result.output_records if last_result is not None else 0
    if repeat_until.on_exhausted == RepeatUntilExhaustion.RAISE:
        raise DataDesignerWorkflowError(
            f"Stage {stage.name!r} repeat_until exhausted after {iterations} iteration(s): "
            f"selected {selected_records} of {repeat_until.output_records} requested records."
        )
    if last_result is None:
        raise DataDesignerWorkflowError(
            f"Stage {stage.name!r} repeat_until did not run because no iteration fit within the configured limits."
        )
    return _with_repeat_result(
        last_result,
        stage_path=stage_path,
        repeat_until=repeat_until,
        iterations=iterations,
        generated_records=generated_records,
        satisfied=False,
    )


def _allows_empty_stage_output(stage: _WorkflowStage, run_result: _StageRunResult) -> bool:
    if stage.allow_empty:
        return True
    return (
        stage.repeat_until is not None
        and stage.repeat_until.on_exhausted == RepeatUntilExhaustion.RETURN_PARTIAL
        and run_result.repeat_satisfied is False
    )


def _repeat_until_payload(repeat_until: RepeatUntil | None) -> dict[str, Any] | None:
    if repeat_until is None:
        return None
    return {
        "output_records": repeat_until.output_records,
        "max_iterations": repeat_until.max_iterations,
        "mode": repeat_until.mode.value,
        "max_generated_records": repeat_until.max_generated_records,
        "on_exhausted": repeat_until.on_exhausted.value,
        "trim": repeat_until.trim,
    }


def _enum_values(enum_type: type[StrEnum]) -> str:
    return ", ".join(repr(item.value) for item in enum_type)


def _normalize_stage_names(
    stage_names: StageTargets | None,
    stage_indices: dict[str, int],
    label: str,
) -> set[str]:
    if stage_names is None:
        return {next(reversed(stage_indices))}
    if isinstance(stage_names, str):
        names = {stage_names}
    else:
        names = set(stage_names)
    if not names:
        raise DataDesignerWorkflowError(f"{label} must include at least one stage.")
    unknown = sorted(names.difference(stage_indices))
    if unknown:
        raise DataDesignerWorkflowError(f"Unknown {label} stage(s): {', '.join(unknown)}.")
    return names


def _stage_index_or_none(stage_name: str | None, stage_indices: dict[str, int], label: str) -> int | None:
    if stage_name is None:
        return None
    if stage_name not in stage_indices:
        raise DataDesignerWorkflowError(f"Unknown {label} stage: {stage_name!r}.")
    return stage_indices[stage_name]


def _validate_stage_output_overrides(
    stage_output_overrides: dict[str, Path | str],
    stage_indices: dict[str, int],
    target_stage_index: int,
) -> None:
    unknown = sorted(set(stage_output_overrides).difference(stage_indices))
    if unknown:
        raise DataDesignerWorkflowError(f"Unknown stage output override(s): {', '.join(unknown)}.")
    non_ancestors = sorted(name for name in stage_output_overrides if stage_indices[name] > target_stage_index)
    if non_ancestors:
        raise DataDesignerWorkflowError(
            f"Stage output override(s) must be ancestors of a target stage: {', '.join(non_ancestors)}."
        )
    for name in sorted(stage_output_overrides):
        override_path = _stage_output_override(name, stage_output_overrides)
        if override_path is not None:
            try:
                _count_parquet_records(override_path)
            except DataDesignerWorkflowError as exc:
                raise DataDesignerWorkflowError(f"Invalid stage output override for stage {name!r}: {exc}") from exc


def _stage_output_override(
    stage_name: str,
    stage_output_overrides: dict[str, Path | str],
) -> Path | None:
    override = stage_output_overrides.get(stage_name)
    if override is None:
        return None
    path = Path(override).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    return path


def _read_prior_workflow_metadata(
    workflow_path: Path,
    workflow_name: str,
    resume: ResumeMode,
) -> dict[str, Any] | None:
    if resume == ResumeMode.NEVER:
        return None
    metadata_path = workflow_path / WORKFLOW_METADATA_FILENAME
    if not metadata_path.exists():
        if resume == ResumeMode.ALWAYS:
            raise DataDesignerWorkflowError(f"Cannot resume workflow {workflow_name!r}: no workflow metadata found.")
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        if resume != ResumeMode.ALWAYS:
            logger.warning("Workflow metadata for %r is corrupt; starting fresh.", workflow_name)
            return None
        raise DataDesignerWorkflowError(
            f"Cannot resume workflow {workflow_name!r}: workflow metadata is corrupt."
        ) from exc
    except OSError as exc:
        if resume != ResumeMode.ALWAYS:
            logger.warning("Workflow metadata for %r could not be read; starting fresh.", workflow_name)
            return None
        raise DataDesignerWorkflowError(
            f"Cannot resume workflow {workflow_name!r}: workflow metadata could not be read."
        ) from exc
    if not isinstance(metadata, dict):
        if resume != ResumeMode.ALWAYS:
            logger.warning("Workflow metadata for %r has invalid shape; starting fresh.", workflow_name)
            return None
        raise DataDesignerWorkflowError(
            f"Cannot resume workflow {workflow_name!r}: workflow metadata has invalid shape."
        )
    if metadata.get("name") != workflow_name:
        if resume != ResumeMode.ALWAYS:
            logger.warning("Workflow metadata for %r has a different name; starting fresh.", workflow_name)
            return None
        raise DataDesignerWorkflowError(
            f"Cannot resume workflow {workflow_name!r}: workflow metadata name does not match."
        )
    return metadata


def _get_prior_stage_metadata(
    prior_metadata: dict[str, Any] | None,
    index: int,
    stage: _WorkflowStage,
    stage_dir_name: str,
) -> dict[str, Any] | None:
    if prior_metadata is None:
        return None
    stages = prior_metadata.get("stages")
    if not isinstance(stages, list) or index >= len(stages):
        return None
    prior_stage = stages[index]
    if not isinstance(prior_stage, dict):
        return None
    if prior_stage.get("name") != stage.name or prior_stage.get("stage_dir") != stage_dir_name:
        return None
    return prior_stage


def _can_skip_prior_stage(stage: _WorkflowStage, prior_stage_metadata: dict[str, Any], workflow_path: Path) -> bool:
    if prior_stage_metadata.get("status") not in COMPLETED_STAGE_STATUSES:
        return False
    if stage.on_success is not None and stage.on_success_version is None:
        return False
    output_seed_path = prior_stage_metadata.get("output_seed_path")
    if not isinstance(output_seed_path, str) or not output_seed_path:
        return False
    try:
        _count_parquet_records(_resolve_metadata_path(workflow_path, output_seed_path))
    except DataDesignerWorkflowError:
        return False
    return True


def _metadata_path_value(workflow_path: Path, path: Path) -> str:
    if path.is_absolute():
        try:
            return str(path.relative_to(workflow_path))
        except ValueError:
            return str(path)
    return str(path)


def _resolve_metadata_path(workflow_path: Path, path: str) -> Path:
    metadata_path = Path(path)
    if metadata_path.is_absolute():
        return metadata_path
    return workflow_path / metadata_path


def _normalize_stage_path_metadata(workflow_path: Path, stage_metadata: dict[str, Any]) -> None:
    for key in WORKFLOW_PATH_METADATA_KEYS:
        value = stage_metadata.get(key)
        if isinstance(value, str) and value:
            stage_metadata[key] = _metadata_path_value(workflow_path, _resolve_metadata_path(workflow_path, value))


def _stage_result_from_metadata(
    *,
    workflow_path: Path,
    stage: _WorkflowStage,
    stage_dir_name: str,
    stage_builder: DataDesignerConfigBuilder,
) -> DatasetCreationResults:
    main_storage = ArtifactStorage(artifact_path=workflow_path, dataset_name=stage_dir_name, resume=ResumeMode.ALWAYS)
    result_storage = main_storage
    result_builder = stage_builder
    if stage.output_processors:
        result_storage = ArtifactStorage(
            artifact_path=workflow_path / stage_dir_name,
            dataset_name="output-processors",
            resume=ResumeMode.ALWAYS,
        )
        result_builder = _output_processor_config_builder(
            stage_builder=stage_builder,
            seed_path=main_storage.final_dataset_path,
            output_processors=stage.output_processors,
        )
    return DatasetCreationResults(
        artifact_storage=result_storage,
        analysis=_load_stage_analysis(result_storage),
        config_builder=result_builder,
        dataset_metadata=DatasetMetadata(),
    )


def _load_stage_analysis(artifact_storage: ArtifactStorage) -> Any:
    try:
        metadata = artifact_storage.read_metadata()
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    column_statistics = metadata.get("column_statistics")
    if not column_statistics:
        return None
    num_records = metadata.get("actual_num_records")
    if num_records is None:
        num_records = _count_parquet_records(artifact_storage.final_dataset_path)
    try:
        return DatasetProfilerResults.model_validate(
            {
                "num_records": num_records,
                "target_num_records": metadata.get("target_num_records", num_records),
                "column_statistics": column_statistics,
                "side_effect_column_names": metadata.get("side_effect_column_names"),
                "column_profiles": metadata.get("column_profiles"),
            }
        )
    except ValidationError:
        return None


def _clone_config_builder(config_builder: DataDesignerConfigBuilder) -> DataDesignerConfigBuilder:
    return DataDesignerConfigBuilder.from_config(BuilderConfig(data_designer=config_builder.build()))


def _clone_processors(processors: list[ProcessorConfig]) -> tuple[ProcessorConfig, ...]:
    return tuple(processor.model_copy(deep=True) for processor in processors)


def _output_processor_config_builder(
    *,
    stage_builder: DataDesignerConfigBuilder,
    seed_path: Path,
    output_processors: tuple[ProcessorConfig, ...],
) -> DataDesignerConfigBuilder:
    builder = DataDesignerConfigBuilder(
        model_configs=stage_builder.model_configs,
        tool_configs=stage_builder.tool_configs,
    ).with_seed_dataset(_local_seed_source_from_path(seed_path))
    for processor in output_processors:
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
        "output_processors": [processor.model_dump(mode="json") for processor in stage.output_processors],
        "output": stage.output,
        "sampling_strategy": stage.sampling_strategy.value,
        "selection_strategy": _selection_strategy_payload(stage.selection_strategy),
        "repeat_until": _repeat_until_payload(stage.repeat_until),
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
        "output_processors": [processor.model_dump(mode="json") for processor in stage.output_processors],
        "output": stage.output,
        "repeat_until": _repeat_until_payload(stage.repeat_until),
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
    if processor_name in {processor.name for processor in stage.output_processors}:
        return output_result
    return result


def _count_parquet_records(path: Path) -> int:
    parquet_files = _parquet_files(path)
    if not parquet_files:
        raise DataDesignerWorkflowError(f"No parquet files found at {str(path)!r}.")
    try:
        return sum(lazy.pq.read_metadata(file_path).num_rows for file_path in parquet_files)
    except Exception as e:
        raise DataDesignerWorkflowError(f"Failed to read parquet files at {str(path)!r}: {e}") from e


def _load_parquet_dataset(path: Path) -> pd.DataFrame:
    parquet_files = _parquet_files(path)
    if not parquet_files:
        raise DataDesignerWorkflowError(f"No parquet files found at {str(path)!r}.")
    try:
        return lazy.pd.concat([lazy.pd.read_parquet(file_path) for file_path in parquet_files], ignore_index=True)
    except Exception as e:
        raise DataDesignerWorkflowError(f"Failed to read parquet files at {str(path)!r}: {e}") from e


def _write_parquet_head(source_path: Path, output_path: Path, num_records: int) -> None:
    df = _load_parquet_dataset(source_path).head(num_records)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    df.to_parquet(output_path / "data.parquet", index=False)


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
    if not path.exists():
        return []
    if not path.is_dir():
        return [path]
    return sorted(path.glob("*.parquet"))


def _write_workflow_metadata(workflow_path: Path, metadata: dict[str, Any]) -> None:
    path = workflow_path / WORKFLOW_METADATA_FILENAME
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


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
    output_processors: list[ProcessorConfig],
) -> None:
    if not output.startswith("processor:"):
        return
    processor_name = output.removeprefix("processor:")
    processor_names = {processor.name for processor in config_builder.get_processor_configs()}
    processor_names.update(processor.name for processor in output_processors)
    if processor_name not in processor_names:
        raise DataDesignerWorkflowError(f"Stage output processor {processor_name!r} is not configured on this stage.")


def _validate_distinct_output_processors(
    config_builder: DataDesignerConfigBuilder,
    output_processors: list[ProcessorConfig],
) -> None:
    seen: set[str] = set()
    duplicate_within: set[str] = set()
    for processor in output_processors:
        if processor.name in seen:
            duplicate_within.add(processor.name)
        seen.add(processor.name)
    if duplicate_within:
        names = ", ".join(sorted(duplicate_within))
        raise DataDesignerWorkflowError(f"Output processor names must be distinct within output_processors: {names}.")
    stage_processor_names = {processor.name for processor in config_builder.get_processor_configs()}
    duplicate_names = stage_processor_names.intersection(processor.name for processor in output_processors)
    if duplicate_names:
        names = ", ".join(sorted(duplicate_names))
        raise DataDesignerWorkflowError(f"Output processor names must be distinct from stage processor names: {names}.")


def _validate_dir_name(name: str, label: str) -> None:
    if not name:
        raise DataDesignerWorkflowError(f"{label} must be a non-empty string.")
    if name in {".", ".."}:
        raise DataDesignerWorkflowError(f"{label} {name!r} is not allowed.")
    invalid_chars = {"<", ">", ":", '"', "/", "\\", "|", "?", "*"}
    if any(char in name for char in invalid_chars):
        raise DataDesignerWorkflowError(f"{label} {name!r} contains invalid path characters.")
