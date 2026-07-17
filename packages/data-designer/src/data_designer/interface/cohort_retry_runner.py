# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from fnmatch import fnmatch
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.config_builder import BuilderConfig, DataDesignerConfigBuilder
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.dataset_metadata import DatasetMetadata
from data_designer.config.processors import ProcessorType
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.engine.models.usage import ModelUsageStats
from data_designer.engine.storage.artifact_storage import (
    FINAL_DATASET_FOLDER_NAME,
    METADATA_FILENAME,
    PROCESSORS_OUTPUTS_FOLDER_NAME,
    SDG_CONFIG_FILENAME,
    ArtifactStorage,
    BatchStage,
    ResumeMode,
)
from data_designer.interface.cohort_retry import (
    RetryExhaustion,
    RetryUntil,
    SamplerRetryMode,
)
from data_designer.interface.cohort_retry_builders import CohortRetryBuilderProjection
from data_designer.interface.errors import CohortRetryExhaustedError, DataDesignerWorkflowError
from data_designer.interface.results import DatasetCreationResults, _load_analysis_from_artifact_storage

if TYPE_CHECKING:
    import pandas as pd

    from data_designer.interface.data_designer import DataDesigner


MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1
BASE_COHORT_PATH = Path("base-cohort/cohort.parquet")
ACCEPTED_DIRECTORY = Path("accepted")
PUBLICATION_INPUT_PATH = ACCEPTED_DIRECTORY / "publication-input.parquet"
ATTEMPTS_DIRECTORY = Path("attempts")
ATTEMPT_RUN_DIRECTORY = "run"
ATTEMPT_COMPLETION_FILENAME = "attempt-completion.json"
FINAL_COMPLETION_FILENAME = "final-completion.json"
INTERNAL_SLOT_COLUMN_BASENAME = "_data_designer_cohort_retry_slot"
INTERNAL_ATTEMPT_COLUMN_BASENAME = "_data_designer_cohort_retry_attempt"
DROPPED_COLUMNS_FOLDER_NAME = "dropped-columns-parquet-files"
PARTIAL_RESULTS_FOLDER_NAME = "tmp-partial-parquet-files"


class _AttemptManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    index: int
    input_path: str
    artifact_path: str
    accepted_path: str
    input_records: int
    output_records: int
    accepted_records: int
    false_records: int
    null_records: int
    missing_records: int
    samplers_executed: bool
    model_usage: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_accounting(self) -> _AttemptManifest:
        if self.index < 0:
            raise ValueError("attempt index must be non-negative")
        counts = (
            self.input_records,
            self.output_records,
            self.accepted_records,
            self.false_records,
            self.null_records,
            self.missing_records,
        )
        if any(count < 0 for count in counts):
            raise ValueError("attempt counts must be non-negative")
        if self.output_records != self.accepted_records + self.false_records + self.null_records:
            raise ValueError("produced-row outcomes do not match output_records")
        if self.input_records != self.output_records + self.missing_records:
            raise ValueError("attempt outcomes do not match input_records")
        attempt_name = f"attempt-{self.index:03d}"
        expected_paths = {
            "input_path": str(ATTEMPTS_DIRECTORY / attempt_name / "input.parquet"),
            "artifact_path": str(ATTEMPTS_DIRECTORY / attempt_name / ATTEMPT_RUN_DIRECTORY),
            "accepted_path": str(ACCEPTED_DIRECTORY / f"{attempt_name}.parquet"),
        }
        for field_name, expected_path in expected_paths.items():
            if getattr(self, field_name) != expected_path:
                raise ValueError(f"attempt {field_name} must match its deterministic stage-relative path")
        return self


class _AttemptCompletion(BaseModel):
    """Durable evidence that one immutable attempt finished generation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    input_slot_ids: list[int]
    output_records: int
    model_usage: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_counts(self) -> _AttemptCompletion:
        if self.output_records < 0 or self.output_records > len(self.input_slot_ids):
            raise ValueError("attempt completion output count must fit within its input slots")
        if len(self.input_slot_ids) != len(set(self.input_slot_ids)):
            raise ValueError("attempt completion input slot IDs must be unique")
        return self


class _FinalCompletion(BaseModel):
    """Durable evidence that terminal processors and profiling completed."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    accepted_records: int
    model_usage: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_accepted_records(self) -> _FinalCompletion:
        if self.accepted_records < 1:
            raise ValueError("final completion requires at least one accepted record")
        return self


class _RetryManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1] = MANIFEST_SCHEMA_VERSION
    fingerprint: str
    status: Literal["running", "finalizing", "complete", "exhausted"] = "running"
    target_records: int
    policy: dict[str, Any]
    slot_column: str
    attempt_column: str
    base_cohort_path: str = str(BASE_COHORT_PATH)
    base_seed_column_names: list[str] = Field(default_factory=list)
    base_model_usage: dict[str, dict[str, Any]] = Field(default_factory=dict)
    attempts: list[_AttemptManifest] = Field(default_factory=list)
    final_model_usage: dict[str, dict[str, Any]] = Field(default_factory=dict)
    unresolved_slot_ids: list[int] = Field(default_factory=list)
    distribution_warning: str | None = None

    @model_validator(mode="after")
    def _validate_attempt_sequence(self) -> _RetryManifest:
        if self.target_records < 1:
            raise ValueError("target_records must be positive")
        if self.base_cohort_path != str(BASE_COHORT_PATH):
            raise ValueError("base cohort path must match its deterministic stage-relative path")
        if not self.slot_column or not self.attempt_column or self.slot_column == self.attempt_column:
            raise ValueError("internal slot and attempt columns must be distinct non-empty names")
        if [attempt.index for attempt in self.attempts] != list(range(len(self.attempts))):
            raise ValueError("attempt indexes must be contiguous and start at zero")
        pending_records = self.target_records
        for attempt in self.attempts:
            if attempt.input_records != pending_records:
                raise ValueError("attempt input counts must match the complete pending cohort")
            pending_records -= attempt.accepted_records
        if len(self.unresolved_slot_ids) != len(set(self.unresolved_slot_ids)) or any(
            slot_id < 0 or slot_id >= self.target_records for slot_id in self.unresolved_slot_ids
        ):
            raise ValueError("unresolved slot IDs must be unique and within the target cohort")
        if self.unresolved_slot_ids != sorted(self.unresolved_slot_ids):
            raise ValueError("unresolved slot IDs must remain in stable cohort order")
        if bool(self.unresolved_slot_ids) != bool(self.distribution_warning):
            raise ValueError("unresolved cohort slots and the partial-result distribution warning must agree")
        if self.unresolved_slot_ids and len(self.unresolved_slot_ids) != pending_records:
            raise ValueError("unresolved slot IDs must match the pending cohort after committed attempts")
        if self.status in {"finalizing", "complete"} and len(self.unresolved_slot_ids) != pending_records:
            raise ValueError("terminal retry state must account for every unresolved cohort slot")
        if self.status == "exhausted" and not self.unresolved_slot_ids:
            raise ValueError("exhausted retry state must contain unresolved cohort slots")
        return self

    @property
    def candidate_records(self) -> int:
        return sum(attempt.input_records for attempt in self.attempts)

    @property
    def accepted_records(self) -> int:
        return sum(attempt.accepted_records for attempt in self.attempts)


@dataclass(frozen=True)
class CohortRetryRun:
    """Canonical result and serializable retry summary for one workflow stage."""

    result: DatasetCreationResults
    summary: dict[str, Any]


@dataclass(frozen=True)
class _AttemptClassification:
    accepted: pd.DataFrame
    output_records: int
    false_records: int
    null_records: int
    missing_records: int


class CohortRetryRunner:
    """Retry rejected logical rows as immutable workflow-level attempts."""

    def __init__(self, data_designer: DataDesigner) -> None:
        self._data_designer = data_designer
        self._task_traces: list[Any] = []

    @staticmethod
    def completed_state_is_reusable(
        *,
        stage_path: Path,
        fingerprint: str,
        policy: RetryUntil,
        target_records: int,
    ) -> bool:
        """Return whether a completed workflow stage has a coherent retry terminal state."""
        try:
            manifest = _RetryManifest.model_validate_json((stage_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
            if (
                manifest.status != "complete"
                or manifest.fingerprint != fingerprint
                or manifest.policy != policy.to_dict()
                or manifest.target_records != target_records
            ):
                return False

            storage = ArtifactStorage(
                artifact_path=stage_path.parent,
                dataset_name=stage_path.name,
                resume=ResumeMode.ALWAYS,
            )
            batch_files = sorted(storage.final_dataset_path.glob("batch_*.parquet"))
            if (
                not batch_files
                or sum(lazy.pq.read_metadata(path).num_rows for path in batch_files) != manifest.accepted_records
            ):
                return False

            metadata = storage.read_metadata()
            if not isinstance(metadata, dict):
                return False
            expected_usage = _aggregate_model_usage(manifest)
            if (
                type(metadata.get("target_num_records")) is not int
                or metadata["target_num_records"] != manifest.target_records
                or type(metadata.get("original_target_num_records")) is not int
                or metadata["original_target_num_records"] != manifest.target_records
                or type(metadata.get("actual_num_records")) is not int
                or metadata["actual_num_records"] != manifest.accepted_records
                or metadata.get("cohort_retry") != _metadata_retry_summary(manifest, expected_usage)
            ):
                return False

            config_path = stage_path / SDG_CONFIG_FILENAME
            config_payload = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(config_payload, dict):
                return False

            completion_path = stage_path / FINAL_COMPLETION_FILENAME
            if manifest.accepted_records == 0:
                return not completion_path.exists() and _load_analysis_from_artifact_storage(storage) is None

            completion = _read_final_completion(completion_path)
            analysis = _load_analysis_from_artifact_storage(storage)
            return bool(
                completion is not None
                and completion.accepted_records == manifest.accepted_records
                and completion.model_usage == manifest.final_model_usage
                and analysis is not None
                and analysis.num_records == manifest.accepted_records
                and analysis.target_num_records == manifest.target_records
            )
        except (
            DataDesignerWorkflowError,
            FileNotFoundError,
            UnicodeError,
            json.JSONDecodeError,
            OSError,
            ValidationError,
            lazy.pa.ArrowException,
        ):
            return False

    def run(
        self,
        *,
        config_builder: DataDesignerConfigBuilder,
        num_records: int,
        dataset_name: str,
        artifact_path: Path,
        policy: RetryUntil,
        fingerprint: str,
        resume: ResumeMode,
        workflow_resume: ResumeMode,
    ) -> CohortRetryRun:
        """Run or resume one cohort-retry stage and publish a normal stage artifact."""
        if num_records < 1:
            raise DataDesignerWorkflowError("Cohort retry requires num_records to be at least 1.")
        if policy.max_candidate_records is not None and policy.max_candidate_records < num_records:
            raise DataDesignerWorkflowError(
                "retry_until.max_candidate_records must be at least the stage num_records so the first "
                "complete cohort attempt can run."
            )

        projection = CohortRetryBuilderProjection(config_builder, policy)
        stage_path = artifact_path / dataset_name
        stage_path.mkdir(parents=True, exist_ok=True)
        manifest = self._load_manifest_for_resume(
            stage_path=stage_path,
            fingerprint=fingerprint,
            policy=policy,
            resume=resume,
            workflow_resume=workflow_resume,
        )

        if manifest is None:
            base_df, manifest = self._materialize_base_cohort(
                projection=projection,
                stage_path=stage_path,
                num_records=num_records,
                fingerprint=fingerprint,
            )
        else:
            base_df = self._load_and_validate_base_cohort(stage_path, manifest)

        accepted_ids = self._load_committed_accepted_ids(stage_path, manifest)
        pending_ids = [slot_id for slot_id in range(num_records) if slot_id not in accepted_ids]

        while pending_ids:
            if self._bounds_exhausted(policy, manifest, len(pending_ids)):
                break
            attempt = self._run_attempt(
                projection=projection,
                stage_path=stage_path,
                manifest=manifest,
                base_df=base_df,
                pending_ids=pending_ids,
            )
            manifest.attempts.append(attempt)
            manifest.status = "running"
            self._write_manifest(stage_path, manifest)
            accepted_ids.update(
                self._read_accepted_slot_ids(
                    stage_path / attempt.accepted_path,
                    manifest.slot_column,
                    manifest.target_records,
                )
            )
            pending_ids = [slot_id for slot_id in pending_ids if slot_id not in accepted_ids]

        exhausted = bool(pending_ids)
        manifest.unresolved_slot_ids = list(pending_ids)
        if exhausted:
            manifest.status = "exhausted"
            manifest.distribution_warning = _distribution_warning(policy.sampler_retry_mode)
            self._write_manifest(stage_path, manifest)
            if policy.on_exhausted == RetryExhaustion.RAISE:
                raise CohortRetryExhaustedError(
                    target_records=num_records,
                    accepted_records=manifest.accepted_records,
                    candidate_records=manifest.candidate_records,
                    attempts=len(manifest.attempts),
                    unresolved_slot_ids=manifest.unresolved_slot_ids,
                )

        coalesced = self._coalesce_accepted(stage_path, manifest, base_df)
        result = self._finalize(
            projection=projection,
            stage_path=stage_path,
            artifact_path=artifact_path,
            dataset_name=dataset_name,
            manifest=manifest,
            coalesced=coalesced,
        )
        manifest.status = "complete"
        self._write_manifest(stage_path, manifest)
        return CohortRetryRun(result=result, summary=_manifest_summary(manifest))

    def _load_manifest_for_resume(
        self,
        *,
        stage_path: Path,
        fingerprint: str,
        policy: RetryUntil,
        resume: ResumeMode,
        workflow_resume: ResumeMode,
    ) -> _RetryManifest | None:
        manifest_path = stage_path / MANIFEST_FILENAME
        if resume == ResumeMode.NEVER:
            return None
        if not manifest_path.exists():
            if not (stage_path / ATTEMPTS_DIRECTORY).exists() and not (stage_path / ACCEPTED_DIRECTORY).exists():
                shutil.rmtree(stage_path)
                stage_path.mkdir(parents=True)
                return None
            return self._handle_incompatible_manifest(
                stage_path,
                workflow_resume,
                "retry manifest is missing while durable attempt state exists",
            )
        try:
            manifest = _RetryManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as exc:
            return self._handle_incompatible_manifest(
                stage_path,
                workflow_resume,
                f"retry manifest is corrupt or invalid: {exc}",
            )
        if manifest.fingerprint != fingerprint or manifest.policy != policy.to_dict():
            return self._handle_incompatible_manifest(
                stage_path,
                workflow_resume,
                "retry manifest does not match the current stage fingerprint and policy",
            )
        return manifest

    @staticmethod
    def _handle_incompatible_manifest(
        stage_path: Path,
        workflow_resume: ResumeMode,
        reason: str,
    ) -> None:
        if workflow_resume == ResumeMode.IF_POSSIBLE:
            shutil.rmtree(stage_path)
            stage_path.mkdir(parents=True)
            return None
        raise DataDesignerWorkflowError(f"Cannot resume cohort retry: {reason}.")

    def _materialize_base_cohort(
        self,
        *,
        projection: CohortRetryBuilderProjection,
        stage_path: Path,
        num_records: int,
        fingerprint: str,
    ) -> tuple[pd.DataFrame, _RetryManifest]:
        base_root = stage_path / BASE_COHORT_PATH.parent
        base_root.mkdir(parents=True, exist_ok=True)
        result = self._data_designer._create(
            projection.build_base_builder(),
            num_records=num_records,
            dataset_name=ATTEMPT_RUN_DIRECTORY,
            artifact_path=base_root,
            profile=False,
        )
        self._task_traces.extend(result.task_traces)
        base_df = result.load_dataset()
        if len(base_df) != num_records:
            raise DataDesignerWorkflowError(
                f"Base cohort materialization produced {len(base_df)} of {num_records} required rows."
            )
        if projection.bootstrap_column_name is not None:
            if projection.bootstrap_column_name not in base_df:
                raise DataDesignerWorkflowError("Base cohort bootstrap column was not materialized.")
            base_df = base_df.drop(columns=[projection.bootstrap_column_name])

        base_seed_column_names = list(result.dataset_metadata.seed_column_names)
        _copy_preserved_seed_media(
            attempt_input=base_df,
            seed_column_names=base_seed_column_names,
            source_root=_local_seed_media_root(projection.original_config),
            run_path=base_root,
        )

        used_names = set(base_df.columns)
        used_names.update(
            output_name
            for column in projection.original_config.columns
            for output_name in (column.name, *column.side_effect_columns)
        )
        slot_column = _unique_name(INTERNAL_SLOT_COLUMN_BASENAME, used_names)
        attempt_column = _unique_name(INTERNAL_ATTEMPT_COLUMN_BASENAME, used_names | {slot_column})
        base_df.insert(0, slot_column, range(num_records))
        _write_parquet_atomic(base_df, stage_path / BASE_COHORT_PATH)

        manifest = _RetryManifest(
            fingerprint=fingerprint,
            target_records=num_records,
            policy=projection.retry_until.to_dict(),
            slot_column=slot_column,
            attempt_column=attempt_column,
            base_seed_column_names=base_seed_column_names,
            base_model_usage=result.model_usage or {},
        )
        self._write_manifest(stage_path, manifest)
        return base_df, manifest

    @staticmethod
    def _load_and_validate_base_cohort(stage_path: Path, manifest: _RetryManifest) -> pd.DataFrame:
        path = stage_path / manifest.base_cohort_path
        if not path.is_file():
            raise DataDesignerWorkflowError(f"Cannot resume cohort retry: base cohort is missing at {str(path)!r}.")
        try:
            base_df = lazy.pd.read_parquet(path)
        except Exception as exc:
            raise DataDesignerWorkflowError(f"Cannot read base cohort at {str(path)!r}: {exc}") from exc
        if len(base_df) != manifest.target_records or manifest.slot_column not in base_df:
            raise DataDesignerWorkflowError("Cannot resume cohort retry: base cohort shape is incompatible.")
        slot_ids = _normalized_slot_ids(base_df[manifest.slot_column], manifest.target_records, "base cohort")
        if slot_ids != list(range(manifest.target_records)):
            raise DataDesignerWorkflowError("Cannot resume cohort retry: base cohort slot IDs are not contiguous.")
        return base_df

    def _run_attempt(
        self,
        *,
        projection: CohortRetryBuilderProjection,
        stage_path: Path,
        manifest: _RetryManifest,
        base_df: pd.DataFrame,
        pending_ids: list[int],
    ) -> _AttemptManifest:
        attempt_index = len(manifest.attempts)
        attempt_name = f"attempt-{attempt_index:03d}"
        attempt_dir = stage_path / ATTEMPTS_DIRECTORY / attempt_name
        input_path = attempt_dir / "input.parquet"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        attempt_input = base_df[base_df[manifest.slot_column].isin(pending_ids)].copy()
        attempt_input[manifest.attempt_column] = attempt_index
        if len(attempt_input) != len(pending_ids):
            raise DataDesignerWorkflowError("Pending slot projection did not produce exactly one input row per slot.")
        _write_or_validate_attempt_input(attempt_input, input_path, manifest.slot_column)

        run_path = attempt_dir / ATTEMPT_RUN_DIRECTORY
        _copy_preserved_seed_media(
            attempt_input=attempt_input,
            seed_column_names=manifest.base_seed_column_names,
            source_root=stage_path / BASE_COHORT_PATH.parent,
            run_path=run_path,
        )
        completion_path = attempt_dir / ATTEMPT_COMPLETION_FILENAME
        completion = _read_attempt_completion(completion_path, pending_ids)
        if completion is None:
            attempt_resume = ResumeMode.ALWAYS if run_path.exists() and any(run_path.iterdir()) else ResumeMode.NEVER
            result = self._data_designer._create(
                projection.build_attempt_builder(input_path),
                num_records=len(attempt_input),
                dataset_name=ATTEMPT_RUN_DIRECTORY,
                artifact_path=attempt_dir,
                resume=attempt_resume,
                profile=False,
                allow_empty=True,
            )
            self._task_traces.extend(result.task_traces)
            output_records = result.count_records()
            output = result.load_dataset() if output_records else _empty_attempt_output(attempt_input, projection)
            completion = _AttemptCompletion(
                input_slot_ids=list(pending_ids),
                output_records=output_records,
                model_usage=result.model_usage or {},
            )
            _write_json_atomic(completion.model_dump(mode="json"), completion_path)
        else:
            output = _load_completed_attempt_output(
                run_path=run_path,
                completion=completion,
                attempt_input=attempt_input,
                projection=projection,
            )
        classification = self._classify_attempt(
            output=output,
            attempt_input=attempt_input,
            manifest=manifest,
            predicate_column=projection.predicate_column,
        )
        accepted = _package_accepted_media(
            classification.accepted,
            attempt_root=run_path,
            stage_path=stage_path,
            attempt_name=attempt_name,
        )
        accepted_path = stage_path / ACCEPTED_DIRECTORY / f"{attempt_name}.parquet"
        _write_parquet_atomic(accepted, accepted_path)

        return _AttemptManifest(
            index=attempt_index,
            input_path=str(input_path.relative_to(stage_path)),
            artifact_path=str(run_path.relative_to(stage_path)),
            accepted_path=str(accepted_path.relative_to(stage_path)),
            input_records=len(attempt_input),
            output_records=classification.output_records,
            accepted_records=len(accepted),
            false_records=classification.false_records,
            null_records=classification.null_records,
            missing_records=classification.missing_records,
            samplers_executed=projection.sampler_retry_mode == SamplerRetryMode.RESAMPLE,
            model_usage=completion.model_usage,
        )

    @staticmethod
    def _classify_attempt(
        *,
        output: pd.DataFrame,
        attempt_input: pd.DataFrame,
        manifest: _RetryManifest,
        predicate_column: str,
    ) -> _AttemptClassification:
        slot_column = manifest.slot_column
        if len(output) and slot_column not in output:
            raise DataDesignerWorkflowError(f"Attempt output is missing internal slot column {slot_column!r}.")
        if len(output) and predicate_column not in output:
            raise DataDesignerWorkflowError(f"Attempt output is missing predicate column {predicate_column!r}.")

        expected_ids = set(_normalized_slot_ids(attempt_input[slot_column], manifest.target_records, "attempt input"))
        if len(output):
            output_ids = _normalized_slot_ids(output[slot_column], manifest.target_records, "attempt output")
            if len(output_ids) != len(set(output_ids)):
                raise DataDesignerWorkflowError("Attempt output contains duplicate rows for one or more slot IDs.")
            unknown = set(output_ids).difference(expected_ids)
            if unknown:
                raise DataDesignerWorkflowError(f"Attempt output contains unknown slot IDs: {sorted(unknown)!r}.")
            _validate_stable_attempt_values(
                output=output,
                attempt_input=attempt_input,
                slot_column=slot_column,
            )
        else:
            output_ids = []

        accepted_mask: list[bool] = []
        false_records = 0
        null_records = 0
        for value in output[predicate_column].tolist() if len(output) else []:
            outcome = _strict_predicate_outcome(value)
            accepted_mask.append(outcome is True)
            if outcome is False:
                false_records += 1
            elif outcome is None:
                null_records += 1

        accepted = output.loc[accepted_mask].copy() if len(output) else output.copy()
        missing_records = len(expected_ids.difference(output_ids))
        return _AttemptClassification(
            accepted=accepted,
            output_records=len(output),
            false_records=false_records,
            null_records=null_records,
            missing_records=missing_records,
        )

    @staticmethod
    def _bounds_exhausted(policy: RetryUntil, manifest: _RetryManifest, pending_records: int) -> bool:
        if policy.max_attempts is not None and len(manifest.attempts) >= policy.max_attempts:
            return True
        return (
            policy.max_candidate_records is not None
            and manifest.candidate_records + pending_records > policy.max_candidate_records
        )

    @staticmethod
    def _load_committed_accepted_ids(stage_path: Path, manifest: _RetryManifest) -> set[int]:
        accepted_ids: set[int] = set()
        for attempt in manifest.attempts:
            path = stage_path / attempt.accepted_path
            ids = CohortRetryRunner._read_accepted_slot_ids(path, manifest.slot_column, manifest.target_records)
            if len(ids) != attempt.accepted_records:
                raise DataDesignerWorkflowError(
                    f"Accepted partition {str(path)!r} does not match its manifest record count."
                )
            overlap = accepted_ids.intersection(ids)
            if overlap:
                raise DataDesignerWorkflowError(f"Slots were accepted more than once: {sorted(overlap)!r}.")
            accepted_ids.update(ids)
        return accepted_ids

    @staticmethod
    def _read_accepted_slot_ids(path: Path, slot_column: str, target_records: int | None = None) -> set[int]:
        if not path.is_file():
            raise DataDesignerWorkflowError(f"Accepted partition is missing at {str(path)!r}.")
        df = lazy.pd.read_parquet(path, columns=[slot_column])
        ids = _normalized_slot_ids(df[slot_column], target_records, "accepted partition")
        if len(ids) != len(set(ids)):
            raise DataDesignerWorkflowError(f"Accepted partition {str(path)!r} contains duplicate slot IDs.")
        return set(ids)

    @staticmethod
    def _coalesce_accepted(
        stage_path: Path,
        manifest: _RetryManifest,
        base_df: pd.DataFrame,
    ) -> pd.DataFrame:
        partitions = [lazy.pd.read_parquet(stage_path / attempt.accepted_path) for attempt in manifest.attempts]
        non_empty = [partition for partition in partitions if len(partition)]
        if non_empty:
            coalesced = lazy.pd.concat(non_empty, ignore_index=True)
            coalesced = coalesced.sort_values(manifest.slot_column, kind="stable").reset_index(drop=True)
        else:
            coalesced = partitions[0].head(0).copy() if partitions else base_df.head(0).copy()
            for column in (manifest.attempt_column,):
                if column not in coalesced:
                    coalesced[column] = lazy.pd.Series(dtype="int64")

        ids = _normalized_slot_ids(coalesced[manifest.slot_column], manifest.target_records, "coalesced output")
        if len(ids) != len(set(ids)) or len(ids) != manifest.accepted_records:
            raise DataDesignerWorkflowError("Coalesced accepted output violates the one-row-per-slot invariant.")
        output_path = stage_path / ACCEPTED_DIRECTORY / "coalesced.parquet"
        _write_parquet_atomic(coalesced, output_path)
        return coalesced

    def _finalize(
        self,
        *,
        projection: CohortRetryBuilderProjection,
        stage_path: Path,
        artifact_path: Path,
        dataset_name: str,
        manifest: _RetryManifest,
        coalesced: pd.DataFrame,
    ) -> DatasetCreationResults:
        if len(coalesced) == 0:
            result = self._write_empty_final(
                projection=projection,
                artifact_path=artifact_path,
                dataset_name=dataset_name,
                manifest=manifest,
                coalesced=coalesced,
            )
            manifest.final_model_usage = {}
            return result

        internal_columns = [manifest.slot_column, manifest.attempt_column]
        missing_internal = [column for column in internal_columns if column not in coalesced]
        if missing_internal:
            raise DataDesignerWorkflowError(
                f"Coalesced output is missing internal cohort identity columns: {missing_internal!r}."
            )
        publication_input_path = stage_path / PUBLICATION_INPUT_PATH
        _write_parquet_atomic(coalesced.drop(columns=internal_columns), publication_input_path)

        manifest.status = "finalizing"
        self._write_manifest(stage_path, manifest)
        completion_path = stage_path / FINAL_COMPLETION_FILENAME
        completion = _read_final_completion(completion_path)
        if completion is None:
            self._clear_ambiguous_finalization(stage_path)
            final_builder = projection.build_final_builder(publication_input_path)
            final_resume = ResumeMode.ALWAYS if (stage_path / METADATA_FILENAME).exists() else ResumeMode.IF_POSSIBLE
            result = self._data_designer._create(
                final_builder,
                num_records=len(coalesced),
                dataset_name=dataset_name,
                artifact_path=artifact_path,
                resume=final_resume,
                profiling_config_builder=_builder_from_projection(projection),
                profiling_target_num_records=manifest.target_records,
            )
            self._task_traces.extend(result.task_traces)
            actual_records = result.count_records()
            if actual_records != len(coalesced):
                raise DataDesignerWorkflowError(
                    "Final cohort-retry processors changed the accepted row count from "
                    f"{len(coalesced)} to {actual_records}; exact cohort retry requires row-count-preserving "
                    "terminal processors."
                )
            if result._load_optional_analysis() is None:
                raise DataDesignerWorkflowError("Final cohort-retry publication completed without profiling analysis.")
            completion = _FinalCompletion(
                accepted_records=actual_records,
                model_usage=result.model_usage or {},
            )
            _write_json_atomic(completion.model_dump(mode="json"), completion_path)
            storage = result.artifact_storage
        else:
            if completion.accepted_records != len(coalesced):
                raise DataDesignerWorkflowError(
                    "Final completion marker does not match the coalesced accepted-record count."
                )
            storage = ArtifactStorage(
                artifact_path=artifact_path,
                dataset_name=dataset_name,
                resume=ResumeMode.ALWAYS,
            )
            if _count_storage_records(storage) != completion.accepted_records:
                raise DataDesignerWorkflowError(
                    "Final dataset does not match its durable cohort-retry completion marker."
                )
            if _load_analysis_from_artifact_storage(storage) is None:
                raise DataDesignerWorkflowError(
                    "Final cohort-retry profiling metadata is missing or invalid after durable completion."
                )

        manifest.final_model_usage = completion.model_usage
        aggregate_usage = _aggregate_model_usage(manifest)
        original_builder = _builder_from_projection(projection)
        _write_original_builder_config(stage_path, projection)
        _write_canonical_metadata(
            storage=storage,
            projection=projection,
            manifest=manifest,
            actual_records=completion.accepted_records,
            model_usage=aggregate_usage,
        )
        canonical_analysis = _load_analysis_from_artifact_storage(storage)
        if canonical_analysis is None:
            raise DataDesignerWorkflowError("Canonical cohort-retry profiling metadata is missing or invalid.")
        return DatasetCreationResults(
            artifact_storage=storage,
            analysis=canonical_analysis,
            config_builder=original_builder,
            dataset_metadata=DatasetMetadata(seed_column_names=manifest.base_seed_column_names),
            task_traces=list(self._task_traces),
            model_usage=aggregate_usage,
        )

    def _write_empty_final(
        self,
        *,
        projection: CohortRetryBuilderProjection,
        artifact_path: Path,
        dataset_name: str,
        manifest: _RetryManifest,
        coalesced: pd.DataFrame,
    ) -> DatasetCreationResults:
        storage = ArtifactStorage(
            artifact_path=artifact_path,
            dataset_name=dataset_name,
            resume=ResumeMode.ALWAYS,
        )
        empty = _empty_publication_dataframe(
            coalesced,
            projection=projection,
            internal_columns={manifest.slot_column, manifest.attempt_column},
        )
        _write_parquet_atomic(empty, storage.create_batch_file_path(0, BatchStage.FINAL_RESULT))
        _write_original_builder_config(storage.base_dataset_path, projection)
        storage.write_metadata(
            {
                **projection.original_config.fingerprint(),
                "target_num_records": manifest.target_records,
                "original_target_num_records": manifest.target_records,
                "actual_num_records": 0,
                "cohort_retry": _metadata_retry_summary(manifest, _aggregate_model_usage(manifest)),
            }
        )
        return DatasetCreationResults(
            artifact_storage=storage,
            analysis=None,
            config_builder=_builder_from_projection(projection),
            dataset_metadata=DatasetMetadata(seed_column_names=manifest.base_seed_column_names),
            task_traces=list(self._task_traces),
            model_usage=_aggregate_model_usage(manifest),
        )

    @staticmethod
    def _clear_ambiguous_finalization(stage_path: Path) -> None:
        metadata_path = stage_path / METADATA_FILENAME
        if not metadata_path.exists():
            return
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
        if metadata.get("post_generation_state") != "started":
            return
        for name in (
            FINAL_DATASET_FOLDER_NAME,
            PARTIAL_RESULTS_FOLDER_NAME,
            DROPPED_COLUMNS_FOLDER_NAME,
            PROCESSORS_OUTPUTS_FOLDER_NAME,
            METADATA_FILENAME,
            SDG_CONFIG_FILENAME,
        ):
            path = stage_path / name
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)

    @staticmethod
    def _write_manifest(stage_path: Path, manifest: _RetryManifest) -> None:
        _write_json_atomic(manifest.model_dump(mode="json"), stage_path / MANIFEST_FILENAME)


def _normalized_slot_ids(series: pd.Series, target_records: int | None, label: str) -> list[int]:
    normalized: list[int] = []
    for value in series.tolist():
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise DataDesignerWorkflowError(f"{label} contains a non-integer slot ID: {value!r}.")
        slot_id = int(value)
        if slot_id < 0 or (target_records is not None and slot_id >= target_records):
            raise DataDesignerWorkflowError(f"{label} contains out-of-range slot ID {slot_id}.")
        normalized.append(slot_id)
    return normalized


def _validate_stable_attempt_values(
    *,
    output: pd.DataFrame,
    attempt_input: pd.DataFrame,
    slot_column: str,
) -> None:
    stable_columns = [column for column in attempt_input.columns if column != slot_column]
    missing_columns = [column for column in stable_columns if column not in output]
    if missing_columns:
        raise DataDesignerWorkflowError(f"Attempt output removed stable seed/cohort columns: {missing_columns!r}.")

    expected_by_slot = attempt_input.set_index(slot_column, drop=False)
    for _, row in output.iterrows():
        slot_id = int(row[slot_column])
        expected = expected_by_slot.loc[slot_id]
        for column in stable_columns:
            if not _values_equal(row[column], expected[column]):
                raise DataDesignerWorkflowError(
                    f"Attempt output mutated stable seed/cohort column {column!r} for slot {slot_id}."
                )


def _values_equal(left: Any, right: Any) -> bool:
    left_missing = _is_scalar_missing(left)
    right_missing = _is_scalar_missing(right)
    if left_missing or right_missing:
        return left_missing and right_missing
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict) or left.keys() != right.keys():
            return False
        return all(_values_equal(left[key], right[key]) for key in left)
    sequence_types = (list, tuple, lazy.np.ndarray)
    if isinstance(left, sequence_types) or isinstance(right, sequence_types):
        if not isinstance(left, sequence_types) or not isinstance(right, sequence_types):
            return False
        left_items = list(left)
        right_items = list(right)
        return len(left_items) == len(right_items) and all(
            _values_equal(left_item, right_item) for left_item, right_item in zip(left_items, right_items)
        )
    try:
        equal = left == right
    except (TypeError, ValueError):
        return False
    return isinstance(equal, (bool, lazy.np.bool_)) and bool(equal)


def _is_scalar_missing(value: Any) -> bool:
    if value is None or value is lazy.pd.NA:
        return True
    try:
        missing = lazy.pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, lazy.np.bool_)) and bool(missing)


def _strict_predicate_outcome(value: Any) -> bool | None:
    if isinstance(value, (bool, lazy.np.bool_)):
        return bool(value)
    if value is None or value is lazy.pd.NA:
        return None
    try:
        missing = lazy.pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, lazy.np.bool_)) and bool(missing):
        return None
    raise DataDesignerWorkflowError(
        "retry_until predicate values must be strict scalar booleans or null; "
        f"received {value!r} ({type(value).__name__})."
    )


def _empty_attempt_output(
    attempt_input: pd.DataFrame,
    projection: CohortRetryBuilderProjection | None,
) -> pd.DataFrame:
    output = attempt_input.head(0).copy()
    if projection is not None:
        for column in projection.original_config.columns:
            for name in (column.name, *column.side_effect_columns):
                if name not in output:
                    output[name] = lazy.pd.Series(dtype="object")
    return output


def _package_accepted_media(
    accepted: pd.DataFrame,
    *,
    attempt_root: Path,
    stage_path: Path,
    attempt_name: str,
) -> pd.DataFrame:
    if accepted.empty:
        return accepted
    packaged = accepted.copy()
    for column in packaged.columns:
        packaged[column] = packaged[column].map(
            lambda value: _package_media_value(
                value,
                attempt_root=attempt_root,
                stage_path=stage_path,
                attempt_name=attempt_name,
            )
        )
    return packaged


def _copy_preserved_seed_media(
    *,
    attempt_input: pd.DataFrame,
    seed_column_names: list[str],
    source_root: Path | None,
    run_path: Path,
) -> None:
    media_paths: set[Path] = set()
    for column in seed_column_names:
        if column not in attempt_input:
            continue
        for value in attempt_input[column].tolist():
            media_paths.update(_relative_media_paths(value))
    if not media_paths:
        return

    resolved_run = run_path.resolve()
    missing_destinations: list[tuple[Path, Path]] = []
    for relative in sorted(media_paths):
        destination = (run_path / relative).resolve()
        if resolved_run not in destination.parents:
            raise DataDesignerWorkflowError(f"Preserved seed media destination {str(relative)!r} is unsafe.")
        if not destination.is_file():
            missing_destinations.append((relative, destination))
    if not missing_destinations:
        return

    if source_root is None:
        raise DataDesignerWorkflowError(
            "Cohort retry cannot resolve preserved 'images/...' seed media. Use a LocalFileSeedSource whose "
            "dataset has an adjacent images directory."
        )
    resolved_root = source_root.resolve()
    for relative, destination in missing_destinations:
        source = (source_root / relative).resolve()
        if resolved_root not in source.parents or not source.is_file():
            raise DataDesignerWorkflowError(
                f"Preserved seed media {str(relative)!r} is missing beneath {str(source_root)!r}."
            )
        _copy_file_atomic(source, destination)


def _relative_media_paths(value: Any) -> set[Path]:
    if isinstance(value, str) and value.startswith("images/"):
        relative = Path(value)
        if relative.is_absolute() or relative.parts[:1] != ("images",) or ".." in relative.parts:
            raise DataDesignerWorkflowError(f"Preserved seed media path {value!r} is unsafe.")
        return {relative}
    if isinstance(value, dict):
        paths: set[Path] = set()
        for item in value.values():
            paths.update(_relative_media_paths(item))
        return paths
    if isinstance(value, lazy.np.ndarray):
        return _relative_media_paths(value.tolist())
    if isinstance(value, (list, tuple)):
        paths = set()
        for item in value:
            paths.update(_relative_media_paths(item))
        return paths
    return set()


def _local_seed_media_root(config: DataDesignerConfig) -> Path | None:
    if config.seed_config is None or not isinstance(config.seed_config.source, LocalFileSeedSource):
        return None
    runtime_path = Path(config.seed_config.source.runtime_path)
    start = runtime_path if runtime_path.is_dir() else runtime_path.parent
    for candidate in (start, *start.parents):
        if (candidate / "images").is_dir():
            return candidate
    return None


def _package_media_value(value: Any, *, attempt_root: Path, stage_path: Path, attempt_name: str) -> Any:
    if isinstance(value, str) and value.startswith("images/"):
        relative = Path(value)
        if relative.is_absolute() or relative.parts[:1] != ("images",) or ".." in relative.parts:
            return value
        source = (attempt_root / relative).resolve()
        resolved_root = attempt_root.resolve()
        if source.is_file() and resolved_root in source.parents:
            destination_relative = Path("images") / attempt_name / Path(*relative.parts[1:])
            destination = stage_path / destination_relative
            resolved_stage = stage_path.resolve()
            if resolved_stage not in destination.resolve().parents:
                return value
            _copy_file_atomic(source, destination)
            return str(destination_relative)
        return value
    if isinstance(value, list):
        return [
            _package_media_value(item, attempt_root=attempt_root, stage_path=stage_path, attempt_name=attempt_name)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _package_media_value(item, attempt_root=attempt_root, stage_path=stage_path, attempt_name=attempt_name)
            for item in value
        )
    if isinstance(value, dict):
        return {
            key: _package_media_value(item, attempt_root=attempt_root, stage_path=stage_path, attempt_name=attempt_name)
            for key, item in value.items()
        }
    if isinstance(value, lazy.np.ndarray):
        return [
            _package_media_value(item, attempt_root=attempt_root, stage_path=stage_path, attempt_name=attempt_name)
            for item in value.tolist()
        ]
    return value


def _write_or_validate_attempt_input(df: pd.DataFrame, path: Path, slot_column: str) -> None:
    if not path.exists():
        _write_parquet_atomic(df, path)
        return
    stored = lazy.pd.read_parquet(path)
    expected_ids = _normalized_slot_ids(df[slot_column], None, "attempt input")
    stored_ids = _normalized_slot_ids(stored[slot_column], None, "stored attempt input")
    if expected_ids != stored_ids or list(stored.columns) != list(df.columns):
        raise DataDesignerWorkflowError(f"Stored immutable attempt input at {str(path)!r} is incompatible.")
    for expected_row, stored_row in zip(
        df.itertuples(index=False, name=None), stored.itertuples(index=False, name=None)
    ):
        if len(expected_row) != len(stored_row) or any(
            not _values_equal(expected, persisted) for expected, persisted in zip(expected_row, stored_row)
        ):
            raise DataDesignerWorkflowError(f"Stored immutable attempt input at {str(path)!r} was modified.")


def _read_attempt_completion(path: Path, expected_slot_ids: list[int]) -> _AttemptCompletion | None:
    if not path.exists():
        return None
    try:
        completion = _AttemptCompletion.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise DataDesignerWorkflowError(f"Attempt completion marker at {str(path)!r} is corrupt: {exc}") from exc
    if completion.input_slot_ids != expected_slot_ids:
        raise DataDesignerWorkflowError(
            f"Attempt completion marker at {str(path)!r} does not match the immutable attempt input."
        )
    return completion


def _read_final_completion(path: Path) -> _FinalCompletion | None:
    if not path.exists():
        return None
    try:
        return _FinalCompletion.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise DataDesignerWorkflowError(f"Final completion marker at {str(path)!r} is corrupt: {exc}") from exc


def _load_completed_attempt_output(
    *,
    run_path: Path,
    completion: _AttemptCompletion,
    attempt_input: pd.DataFrame,
    projection: CohortRetryBuilderProjection,
) -> pd.DataFrame:
    if completion.output_records == 0:
        return _empty_attempt_output(attempt_input, projection)
    try:
        storage = ArtifactStorage(
            artifact_path=run_path.parent,
            dataset_name=run_path.name,
            resume=ResumeMode.ALWAYS,
        )
        output = storage.load_dataset()
    except Exception as exc:
        raise DataDesignerWorkflowError(f"Completed attempt output at {str(run_path)!r} cannot be read: {exc}") from exc
    if len(output) != completion.output_records:
        raise DataDesignerWorkflowError(
            f"Completed attempt output at {str(run_path)!r} contains {len(output)} records; "
            f"its completion marker records {completion.output_records}."
        )
    return output


def _empty_publication_dataframe(
    coalesced: pd.DataFrame,
    *,
    projection: CohortRetryBuilderProjection,
    internal_columns: set[str],
) -> pd.DataFrame:
    empty = coalesced.head(0).copy()
    drop_patterns: list[str] = list(projection.original_dropped_names)
    for processor in projection.original_config.processors or []:
        if processor.processor_type == ProcessorType.DROP_COLUMNS:
            drop_patterns.extend(processor.column_names)
    to_drop = {
        column
        for column in empty.columns
        if column in internal_columns or any(fnmatch(column, pattern) for pattern in drop_patterns)
    }
    return empty.drop(columns=sorted(to_drop), errors="ignore")


def _builder_from_projection(projection: CohortRetryBuilderProjection) -> DataDesignerConfigBuilder:
    return DataDesignerConfigBuilder.from_config(
        BuilderConfig(data_designer=projection.original_config.model_copy(deep=True))
    )


def _write_original_builder_config(stage_path: Path, projection: CohortRetryBuilderProjection) -> None:
    BuilderConfig(data_designer=projection.original_config.model_copy(deep=True)).to_json(
        stage_path / SDG_CONFIG_FILENAME
    )


def _count_storage_records(storage: ArtifactStorage) -> int:
    return sum(lazy.pq.read_metadata(path).num_rows for path in storage.final_dataset_path.glob("batch_*.parquet"))


def _write_canonical_metadata(
    *,
    storage: ArtifactStorage,
    projection: CohortRetryBuilderProjection,
    manifest: _RetryManifest,
    actual_records: int,
    model_usage: dict[str, dict[str, Any]],
) -> None:
    try:
        metadata = storage.read_metadata()
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise DataDesignerWorkflowError(
            "Final cohort-retry metadata is missing or invalid after durable completion."
        ) from exc
    if not metadata.get("column_statistics"):
        raise DataDesignerWorkflowError("Final cohort-retry profiling metadata is missing after durable completion.")
    storage.write_metadata(
        {
            **metadata,
            **projection.original_config.fingerprint(),
            "target_num_records": manifest.target_records,
            "original_target_num_records": manifest.target_records,
            "actual_num_records": actual_records,
            "cohort_retry": _metadata_retry_summary(manifest, model_usage),
        }
    )


def _aggregate_model_usage(manifest: _RetryManifest) -> dict[str, dict[str, Any]]:
    aggregate: dict[str, ModelUsageStats] = {}
    snapshots = [
        manifest.base_model_usage,
        *(attempt.model_usage for attempt in manifest.attempts),
        manifest.final_model_usage,
    ]
    for snapshot in snapshots:
        for model_name, payload in snapshot.items():
            try:
                incoming = ModelUsageStats.model_validate(payload)
            except ValidationError as exc:
                raise DataDesignerWorkflowError(
                    f"Cohort-retry model usage for {model_name!r} is invalid: {exc}"
                ) from exc
            current = aggregate.setdefault(model_name, ModelUsageStats())
            current.extend(
                token_usage=incoming.token_usage,
                request_usage=incoming.request_usage,
                tool_usage=incoming.tool_usage,
                image_usage=incoming.image_usage,
            )
    return {name: usage.model_dump(mode="json") for name, usage in aggregate.items()}


def _manifest_summary(manifest: _RetryManifest) -> dict[str, Any]:
    return {
        "target_records": manifest.target_records,
        "accepted_records": manifest.accepted_records,
        "unresolved_records": len(manifest.unresolved_slot_ids),
        "unresolved_slot_ids": manifest.unresolved_slot_ids,
        "candidate_records": manifest.candidate_records,
        "attempts": len(manifest.attempts),
        "sampler_retry_mode": manifest.policy["sampler_retry_mode"],
        "exhausted": bool(manifest.unresolved_slot_ids),
        "distribution_warning": manifest.distribution_warning,
    }


def _metadata_retry_summary(
    manifest: _RetryManifest,
    model_usage: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        **_manifest_summary(manifest),
        "seed_column_names": manifest.base_seed_column_names,
        "model_usage": model_usage,
    }


def _distribution_warning(mode: SamplerRetryMode) -> str:
    if mode == SamplerRetryMode.PRESERVE:
        return (
            "The partial result omits unresolved seed/sampler slots and is biased toward cohort combinations "
            "that passed within the retry bounds."
        )
    return (
        "The partial result omits unresolved seed slots and accepted sampler values are additionally conditioned "
        "on passing the predicate."
    )


def _write_manifest_atomic_payload(payload: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with tmp_path.open("w", encoding="utf-8") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    _write_manifest_atomic_payload(json.dumps(payload, indent=2, sort_keys=True), path)


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.stem}.tmp.{os.getpid()}.{uuid.uuid4().hex}.parquet")
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(f"{destination.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        shutil.copy2(source, tmp_path)
        os.replace(tmp_path, destination)
    finally:
        tmp_path.unlink(missing_ok=True)


def _unique_name(base_name: str, used_names: set[str]) -> str:
    if base_name not in used_names:
        return base_name
    suffix = 1
    while f"{base_name}_{suffix}" in used_names:
        suffix += 1
    return f"{base_name}_{suffix}"
