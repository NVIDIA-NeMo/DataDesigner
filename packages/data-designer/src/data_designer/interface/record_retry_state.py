# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from data_designer.engine.storage.artifact_storage import ResumeMode
from data_designer.interface.errors import DataDesignerWorkflowError
from data_designer.interface.record_retry import RetryUntil

if TYPE_CHECKING:
    import pandas as pd


MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1
BASE_COHORT_PATH = Path("base-cohort/cohort.parquet")
ACCEPTED_DIRECTORY = Path("accepted")
PUBLICATION_INPUT_PATH = ACCEPTED_DIRECTORY / "publication-input.parquet"
COALESCED_ACCEPTED_PATH = ACCEPTED_DIRECTORY / "coalesced.parquet"
ATTEMPTS_DIRECTORY = Path("attempts")
ATTEMPT_INPUT_FILENAME = "input.parquet"
ATTEMPT_RUN_DIRECTORY = "run"
ATTEMPT_COMPLETION_FILENAME = "attempt-completion.json"
FINAL_COMPLETION_FILENAME = "final-completion.json"
INTERNAL_SLOT_COLUMN_BASENAME = "_data_designer_record_retry_slot"
INTERNAL_ATTEMPT_COLUMN_BASENAME = "_data_designer_record_retry_attempt"


class AttemptManifest(BaseModel):
    """Persisted accounting for one committed record-retry attempt."""

    model_config = ConfigDict(extra="forbid", strict=True)

    input_records: int = Field(ge=0)
    output_records: int = Field(ge=0)
    accepted_records: int = Field(ge=0)
    false_records: int = Field(ge=0)
    null_records: int = Field(ge=0)
    missing_records: int = Field(ge=0)
    model_usage: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_accounting(self) -> AttemptManifest:
        if self.output_records != self.accepted_records + self.false_records + self.null_records:
            raise ValueError("produced-row outcomes do not match output_records")
        if self.input_records != self.output_records + self.missing_records:
            raise ValueError("attempt outcomes do not match input_records")
        return self


class AttemptCompletion(BaseModel):
    """Durable evidence that one immutable attempt finished generation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    input_slot_ids: list[int]
    output_records: int = Field(ge=0)
    model_usage: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_counts(self) -> AttemptCompletion:
        if self.output_records > len(self.input_slot_ids):
            raise ValueError("attempt completion output count must fit within its input slots")
        if len(self.input_slot_ids) != len(set(self.input_slot_ids)):
            raise ValueError("attempt completion input slot IDs must be unique")
        return self


class FinalCompletion(BaseModel):
    """Durable evidence that terminal processors and profiling completed."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    accepted_records: int = Field(ge=1)
    model_usage: dict[str, dict[str, Any]] = Field(default_factory=dict)


class RetryManifest(BaseModel):
    """Durable state and accounting for a record-retry stage."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1] = MANIFEST_SCHEMA_VERSION
    fingerprint: str
    status: Literal["running", "finalizing", "complete", "exhausted"] = "running"
    target_records: int = Field(ge=1)
    policy: RetryUntil
    slot_column: str = Field(min_length=1)
    attempt_column: str = Field(min_length=1)
    base_seed_column_names: list[str] = Field(default_factory=list)
    base_model_usage: dict[str, dict[str, Any]] = Field(default_factory=dict)
    attempts: list[AttemptManifest] = Field(default_factory=list)
    final_model_usage: dict[str, dict[str, Any]] = Field(default_factory=dict)
    unresolved_slot_ids: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_attempt_sequence(self) -> RetryManifest:
        if self.slot_column == self.attempt_column:
            raise ValueError("internal slot and attempt columns must be distinct")

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
        if self.unresolved_slot_ids and len(self.unresolved_slot_ids) != pending_records:
            raise ValueError("unresolved slot IDs must match the pending cohort after committed attempts")
        if self.status in {"finalizing", "complete"} and len(self.unresolved_slot_ids) != pending_records:
            raise ValueError("terminal retry state must account for every unresolved cohort slot")
        if self.status == "exhausted" and not self.unresolved_slot_ids:
            raise ValueError("exhausted retry state must contain unresolved cohort slots")
        return self

    @property
    def candidate_records(self) -> int:
        """Return the number of candidate records consumed by committed attempts."""
        return sum(attempt.input_records for attempt in self.attempts)

    @property
    def accepted_records(self) -> int:
        """Return the number of records accepted by committed attempts."""
        return sum(attempt.accepted_records for attempt in self.attempts)


def get_attempt_name(attempt_index: int) -> str:
    """Return the deterministic directory name for an attempt index."""
    if type(attempt_index) is not int or attempt_index < 0:
        raise ValueError("attempt index must be a non-negative integer")
    return f"attempt-{attempt_index:03d}"


def get_attempt_directory(attempt_index: int) -> Path:
    """Return the stage-relative directory for an attempt index."""
    return ATTEMPTS_DIRECTORY / get_attempt_name(attempt_index)


def get_attempt_input_path(attempt_index: int) -> Path:
    """Return the stage-relative immutable input path for an attempt index."""
    return get_attempt_directory(attempt_index) / ATTEMPT_INPUT_FILENAME


def get_attempt_artifact_path(attempt_index: int) -> Path:
    """Return the stage-relative ordinary-run artifact path for an attempt index."""
    return get_attempt_directory(attempt_index) / ATTEMPT_RUN_DIRECTORY


def get_attempt_accepted_path(attempt_index: int) -> Path:
    """Return the stage-relative accepted partition path for an attempt index."""
    return ACCEPTED_DIRECTORY / f"{get_attempt_name(attempt_index)}.parquet"


def read_retry_manifest(stage_path: Path) -> RetryManifest:
    """Read and validate a stage's durable retry manifest."""
    return RetryManifest.model_validate_json((stage_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))


def load_retry_manifest(
    *,
    stage_path: Path,
    fingerprint: str,
    policy: RetryUntil,
    resume: ResumeMode,
    workflow_resume: ResumeMode,
) -> RetryManifest | None:
    """Load compatible retry state or restart it under ``IF_POSSIBLE`` semantics."""
    manifest_path = stage_path / MANIFEST_FILENAME
    if resume == ResumeMode.NEVER:
        return None
    if not manifest_path.exists():
        if not (stage_path / ATTEMPTS_DIRECTORY).exists() and not (stage_path / ACCEPTED_DIRECTORY).exists():
            _restart_stage(stage_path)
            return None
        return _handle_incompatible_manifest(
            stage_path,
            workflow_resume,
            "retry manifest is missing while durable attempt state exists",
        )
    try:
        manifest = read_retry_manifest(stage_path)
    except (OSError, UnicodeError, ValidationError) as exc:
        return _handle_incompatible_manifest(
            stage_path,
            workflow_resume,
            f"retry manifest is corrupt or invalid: {exc}",
        )
    if manifest.fingerprint != fingerprint or manifest.policy != policy:
        return _handle_incompatible_manifest(
            stage_path,
            workflow_resume,
            "retry manifest does not match the current stage fingerprint and policy",
        )
    return manifest


def _handle_incompatible_manifest(
    stage_path: Path,
    workflow_resume: ResumeMode,
    reason: str,
) -> None:
    if workflow_resume == ResumeMode.IF_POSSIBLE:
        _restart_stage(stage_path)
        return None
    raise DataDesignerWorkflowError(f"Cannot resume record retry: {reason}.")


def _restart_stage(stage_path: Path) -> None:
    shutil.rmtree(stage_path)
    stage_path.mkdir(parents=True)


def read_attempt_completion(path: Path, expected_slot_ids: list[int]) -> AttemptCompletion | None:
    """Read and validate an attempt completion marker when one exists."""
    if not path.exists():
        return None
    try:
        completion = AttemptCompletion.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise DataDesignerWorkflowError(f"Attempt completion marker at {str(path)!r} is corrupt: {exc}") from exc
    if completion.input_slot_ids != expected_slot_ids:
        raise DataDesignerWorkflowError(
            f"Attempt completion marker at {str(path)!r} does not match the immutable attempt input."
        )
    return completion


def read_final_completion(path: Path) -> FinalCompletion | None:
    """Read and validate a final completion marker when one exists."""
    if not path.exists():
        return None
    try:
        return FinalCompletion.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise DataDesignerWorkflowError(f"Final completion marker at {str(path)!r} is corrupt: {exc}") from exc


def write_retry_manifest(stage_path: Path, manifest: RetryManifest) -> None:
    """Atomically persist a retry manifest at the stage root."""
    write_json_atomic(manifest.model_dump(mode="json"), stage_path / MANIFEST_FILENAME)


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    """Atomically write a JSON object with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            file.write(json.dumps(payload, indent=2, sort_keys=True))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """Atomically write a dataframe as a parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.stem}.tmp.{os.getpid()}.{uuid.uuid4().hex}.parquet")
    try:
        df.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def copy_file_atomic(source: Path, destination: Path) -> None:
    """Atomically copy one file to its destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f"{destination.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
