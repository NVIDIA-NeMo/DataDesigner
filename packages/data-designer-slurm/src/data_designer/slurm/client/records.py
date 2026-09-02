# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Annotated, Literal

from pydantic import NonNegativeInt, PositiveInt, StringConstraints, field_validator, model_validator

from data_designer.slurm.config.images import InstalledDistribution
from data_designer.slurm.contracts import (
    ArtifactReference,
    AttemptId,
    ContractRecord,
    ContractValue,
    Identifier,
    Sha256Digest,
    ShardId,
    validate_absolute_path,
    validate_plain_text,
)


class ClientErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    DEPENDENCY_ARTIFACT_MISSING = "dependency_artifact_missing"
    DEPENDENCY_DIGEST_MISMATCH = "dependency_digest_mismatch"
    DEPENDENCY_CONFLICT = "dependency_conflict"
    DEPENDENCY_INSTALL_FAILED = "dependency_install_failed"
    PLUGIN_LOAD_FAILED = "plugin_load_failed"
    CONFIG_INVALID = "config_invalid"
    GENERATION_FAILED = "generation_failed"
    INTERRUPTED = "interrupted"
    OUTPUT_INVALID = "output_invalid"


class ClientEnvironmentOutcome(str, Enum):
    READY = "ready"
    FAILED = "failed"


class ClientInstallerOutcome(str, Enum):
    NOT_REQUIRED = "not_required"
    INSTALLED = "installed"
    REUSED = "reused"


class ClientProgressPhase(str, Enum):
    PREPARING_ENVIRONMENT = "preparing_environment"
    VALIDATING_PLUGINS = "validating_plugins"
    VALIDATING_CONFIG = "validating_config"
    GENERATING = "generating"
    FINALIZING = "finalizing"
    COMPLETE = "complete"
    FAILED = "failed"


class ClientPluginEntryPoint(ContractValue):
    entry_point: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    value: Annotated[str, StringConstraints(min_length=1, max_length=512)]
    distribution: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    distribution_version: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    plugin_name: Identifier
    plugin_type: Literal["column-generator", "seed-reader", "processor"]

    @field_validator("entry_point", "value", "distribution", "distribution_version")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return validate_plain_text(value, field_name="plugin entry point")


class ClientEnvironmentManifest(ContractRecord):
    run_id: Identifier
    shard_id: ShardId
    attempt_id: AttemptId
    created_at: datetime
    outcome: ClientEnvironmentOutcome
    dependency_lock: ArtifactReference
    client_image_sha256: Sha256Digest
    python_abi: Identifier
    overlay_path: str
    installer_outcome: ClientInstallerOutcome
    installed_distributions: tuple[InstalledDistribution, ...]
    plugins: tuple[ClientPluginEntryPoint, ...]
    error_code: ClientErrorCode | None = None
    redacted_message: Annotated[str, StringConstraints(max_length=512)] | None = None

    _overlay_path_is_absolute = field_validator("overlay_path")(validate_absolute_path)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("created_at must be timezone-aware UTC")
        return value

    @field_validator("redacted_message")
    @classmethod
    def validate_redacted_message(cls, value: str | None) -> str | None:
        return None if value is None else validate_plain_text(value, field_name="redacted message")

    @model_validator(mode="after")
    def validate_environment(self) -> ClientEnvironmentManifest:
        distribution_names = tuple(distribution.name for distribution in self.installed_distributions)
        if distribution_names != tuple(sorted(distribution_names)) or len(distribution_names) != len(
            set(distribution_names)
        ):
            raise ValueError("installed distributions must be sorted and unique")
        plugin_keys = tuple((plugin.distribution, plugin.entry_point) for plugin in self.plugins)
        if plugin_keys != tuple(sorted(plugin_keys)) or len(plugin_keys) != len(set(plugin_keys)):
            raise ValueError("plugin entry points must be sorted and unique")
        if self.outcome is ClientEnvironmentOutcome.READY:
            if self.error_code is not None or self.redacted_message is not None:
                raise ValueError("ready client environments cannot contain failure details")
        elif self.error_code is None:
            raise ValueError("failed client environments require error_code")
        return self


class ClientProgress(ContractRecord):
    run_id: Identifier
    shard_id: ShardId
    attempt_id: AttemptId
    revision: PositiveInt
    updated_at: datetime
    phase: ClientProgressPhase
    requested_records: PositiveInt
    completed_records: NonNegativeInt | None = None
    error_code: ClientErrorCode | None = None

    @field_validator("updated_at")
    @classmethod
    def validate_updated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("updated_at must be timezone-aware UTC")
        return value

    @model_validator(mode="after")
    def validate_progress(self) -> ClientProgress:
        if self.completed_records is not None and self.completed_records > self.requested_records:
            raise ValueError("completed_records must not exceed requested_records")
        if (self.phase is ClientProgressPhase.FAILED) != (self.error_code is not None):
            raise ValueError("only failed progress requires an error code")
        return self


class ClientOutcome(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class ClientResult(ContractRecord):
    """Semantic Data Designer outcome independent of engine-internal result types."""

    run_id: Identifier
    shard_id: ShardId
    attempt_id: AttemptId
    completed_at: datetime
    requested_records: PositiveInt
    actual_records: NonNegativeInt | None
    outcome: ClientOutcome
    dataset_path: str | None = None
    early_shutdown: bool | None = None
    requested_resume_mode: Literal["never", "always", "if_possible"]
    effective_resume_mode: Literal["never", "always"] | None = None
    candidate_output_manifest: ArtifactReference | None = None
    error_code: Identifier | None = None
    redacted_message: Annotated[str, StringConstraints(max_length=512)] | None = None

    @field_validator("completed_at")
    @classmethod
    def validate_completed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("completed_at must be timezone-aware UTC")
        return value

    @field_validator("dataset_path")
    @classmethod
    def validate_dataset_path(cls, value: str | None) -> str | None:
        return None if value is None else validate_absolute_path(value)

    @field_validator("redacted_message")
    @classmethod
    def validate_message(cls, value: str | None) -> str | None:
        if value is not None and any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError("redacted_message must not contain control characters")
        return value

    @model_validator(mode="after")
    def validate_outcome(self) -> ClientResult:
        if self.actual_records is not None and self.actual_records > self.requested_records:
            raise ValueError("actual_records must not exceed requested_records")
        if self.requested_resume_mode == "never" and self.effective_resume_mode not in {None, "never"}:
            raise ValueError("resume mode never cannot become effective resume")
        if self.outcome is not ClientOutcome.FAILED:
            if self.early_shutdown is None or self.effective_resume_mode is None:
                raise ValueError("non-failed client results require resume and early-shutdown facts")
        if self.outcome is ClientOutcome.COMPLETE:
            if self.actual_records != self.requested_records:
                raise ValueError("complete client results require the requested record count")
            if self.early_shutdown:
                raise ValueError("complete client results cannot report early shutdown")
            self._require_success_artifacts()
        elif self.outcome is ClientOutcome.PARTIAL:
            if self.actual_records is None or self.actual_records >= self.requested_records:
                raise ValueError("partial client results require fewer than the requested record count")
            self._require_success_artifacts()
        else:
            if self.candidate_output_manifest is not None:
                raise ValueError("failed client results cannot reference a candidate output manifest")
            if self.error_code is None:
                raise ValueError("failed client results require error_code")
        return self

    def _require_success_artifacts(self) -> None:
        if self.dataset_path is None or self.candidate_output_manifest is None:
            raise ValueError("successful client results require dataset and candidate manifest paths")
        if self.error_code is not None or self.redacted_message is not None:
            raise ValueError("successful client results cannot contain failure details")
        shard_root = f"/runs/{self.run_id}/shards/{self.shard_id}"
        if self.effective_resume_mode == "never":
            expected_dataset = f"{shard_root}/attempts/{self.attempt_id}/dataset"
        else:
            expected_dataset = f"{shard_root}/dataset"
        if not self.dataset_path.endswith(expected_dataset):
            raise ValueError("dataset path must match the run, shard, attempt, and resume policy")
        expected_manifest = f"{shard_root}/attempts/{self.attempt_id}/output-manifest.json"
        if not self.candidate_output_manifest.path.endswith(expected_manifest):
            raise ValueError("candidate output reference must match the run, shard, and attempt")
