# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable values returned by public Slurm operations."""

from __future__ import annotations

from typing import Literal

from pydantic import PositiveInt, model_validator

from data_designer.slurm.contracts import ContractValue, Identifier, Sha256Digest
from data_designer.slurm.state import (
    AttemptManifest,
    AttemptReadiness,
    RunManifest,
    ShardManifest,
    ShardWinner,
)


class SlurmRunExecution(ContractValue):
    """One rendered dry run or accepted Slurm submission."""

    run_id: Identifier
    state: Literal["dry_run", "submitted"]
    plan_sha256: Sha256Digest
    shard_count: PositiveInt
    job_id: PositiveInt | None = None
    batch_script: str | None = None

    @model_validator(mode="after")
    def validate_state(self) -> SlurmRunExecution:
        if self.state == "dry_run":
            if self.job_id is not None or not self.batch_script:
                raise ValueError("dry-run execution requires only a rendered batch script")
        elif self.job_id is None or self.batch_script is not None:
            raise ValueError("submitted execution requires only a Slurm job ID")
        return self


class SlurmPersistedAttemptStatus(ContractValue):
    """Persisted attempt state returned after scheduler reconciliation."""

    attempt: AttemptManifest
    readiness: AttemptReadiness | None = None

    @model_validator(mode="after")
    def validate_readiness(self) -> SlurmPersistedAttemptStatus:
        if self.readiness is not None and (
            self.readiness.run_id,
            self.readiness.shard_id,
            self.readiness.attempt_id,
        ) != (self.attempt.run_id, self.attempt.shard_id, self.attempt.attempt_id):
            raise ValueError("readiness identity does not match its attempt")
        return self


class SlurmPersistedShardStatus(ContractValue):
    """Persisted attempt and winner state for one shard."""

    shard: ShardManifest
    attempts: tuple[SlurmPersistedAttemptStatus, ...]
    winner: ShardWinner | None = None

    @model_validator(mode="after")
    def validate_shard(self) -> SlurmPersistedShardStatus:
        if any(status.attempt.shard_id != self.shard.shard_id for status in self.attempts):
            raise ValueError("persisted status contains an attempt for another shard")
        if self.winner is not None and (
            self.winner.run_id != self.shard.run_id or self.winner.shard_id != self.shard.shard_id
        ):
            raise ValueError("persisted winner does not match its shard")
        return self


class SlurmPersistedRunStatus(ContractValue):
    """M2 status assembled only from durable run records."""

    run: RunManifest
    shards: tuple[SlurmPersistedShardStatus, ...]

    @model_validator(mode="after")
    def validate_run(self) -> SlurmPersistedRunStatus:
        if len(self.shards) != self.run.shard_count:
            raise ValueError("persisted status must contain every run shard")
        if tuple(status.shard.shard_index for status in self.shards) != tuple(range(self.run.shard_count)):
            raise ValueError("persisted status shards must remain in planned order")
        if any(status.shard.run_id != self.run.run_id for status in self.shards):
            raise ValueError("persisted status contains a shard for another run")
        return self


class SlurmRunCancellation(ContractValue):
    """Managed Slurm array jobs selected for cancellation."""

    run_id: Identifier
    job_ids: tuple[PositiveInt, ...]

    @model_validator(mode="after")
    def validate_jobs(self) -> SlurmRunCancellation:
        if self.job_ids != tuple(sorted(set(self.job_ids))):
            raise ValueError("cancelled job IDs must be sorted and unique")
        return self


__all__ = [
    "SlurmPersistedAttemptStatus",
    "SlurmPersistedRunStatus",
    "SlurmPersistedShardStatus",
    "SlurmRunCancellation",
    "SlurmRunExecution",
]
