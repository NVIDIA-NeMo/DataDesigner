# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical storage for persisted Slurm state records."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import ValidationError

from data_designer.slurm.client import ClientResult
from data_designer.slurm.config import DataDesignerSlurmConfig
from data_designer.slurm.contracts import AttemptId, ContractRecord, Identifier, ShardId
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.state.errors import SlurmStateError, StateCorruptionError, StateNotFoundError
from data_designer.slurm.state.execution import AttemptManifest, RunManifest, ShardManifest
from data_designer.slurm.state.filesystem import (
    acquire_file_lock,
    ensure_private_child_directory,
    is_state_temporary_name,
    open_verified_child_directory,
    open_verified_directory,
    publish_immutable_text,
    read_regular_text,
    replace_text,
    sync_directory,
)
from data_designer.slurm.state.outputs import CandidateOutputManifest, ShardWinner
from data_designer.slurm.state.readiness import AttemptReadiness
from data_designer.slurm.state.scheduler import SchedulerObservation

_RUN_FILENAME = "run.json"
_AUTHORED_CONFIG_FILENAME = "authored-config.json"
_RESOLVED_PLAN_FILENAME = "resolved-plan.json"
_SHARDS_DIRECTORY_NAME = "shards"
_ATTEMPTS_DIRECTORY_NAME = "attempts"
_SHARD_FILENAME = "shard.json"
_SHARD_LOCK_FILENAME = "shard.lock"
_ATTEMPT_FILENAME = "attempt.json"
_READINESS_FILENAME = "readiness.json"
_SCHEDULER_OBSERVATION_FILENAME = "scheduler.json"
_CLIENT_RESULT_FILENAME = "client-result.json"
_CANDIDATE_OUTPUT_FILENAME = "output-manifest.json"
_WINNER_FILENAME = "winner.json"
_DATASET_DIRECTORY_NAME = "dataset"
_RESUME_LOCK_FILENAME = "resume.lock"
_LOCK_DIRECTORY_NAME = ".locks"
_MAXIMUM_RECORD_SIZE = 16 * 1024 * 1024
_ATTEMPT_NAME_PATTERN = re.compile(r"^attempt-[0-9]{4,}$")
_RecordT = TypeVar("_RecordT", bound=ContractRecord)


class StateStorage:
    """Own descriptor-bound paths, locking, and record serialization."""

    def __init__(self, workspace_root: Path, run_id: Identifier) -> None:
        self.workspace_root = workspace_root
        self.run_id = run_id
        self.runs_root = workspace_root / "runs"
        self.locks_root = self.runs_root / _LOCK_DIRECTORY_NAME
        self.run_root = self.runs_root / run_id

    @property
    def authored_config_path(self) -> Path:
        return self.run_root / _AUTHORED_CONFIG_FILENAME

    @property
    def resolved_plan_path(self) -> Path:
        return self.run_root / _RESOLVED_PLAN_FILENAME

    def get_shard_path(self, shard_id: str) -> Path:
        return self.run_root / _SHARDS_DIRECTORY_NAME / shard_id

    def get_attempt_path(self, shard_id: str, attempt_id: str) -> Path:
        return self.get_shard_path(shard_id) / _ATTEMPTS_DIRECTORY_NAME / attempt_id

    def get_readiness_path(self, shard_id: str, attempt_id: str) -> Path:
        return self.get_attempt_path(shard_id, attempt_id) / _READINESS_FILENAME

    def get_scheduler_observation_path(self, shard_id: str, attempt_id: str) -> Path:
        return self.get_attempt_path(shard_id, attempt_id) / _SCHEDULER_OBSERVATION_FILENAME

    def get_winner_path(self, shard_id: str) -> Path:
        return self.get_shard_path(shard_id) / _WINNER_FILENAME

    def ensure_storage(self) -> None:
        with open_verified_directory(self.workspace_root) as workspace_descriptor:
            ensure_private_child_directory(workspace_descriptor, "runs", self.runs_root)
            with open_verified_child_directory(workspace_descriptor, "runs", self.runs_root) as runs_descriptor:
                ensure_private_child_directory(runs_descriptor, _LOCK_DIRECTORY_NAME, self.locks_root)

    @contextmanager
    def acquire_run_lock(self) -> Iterator[None]:
        try:
            with open_verified_directory(self.runs_root, require_private=True) as runs_descriptor:
                with open_verified_child_directory(
                    runs_descriptor,
                    _LOCK_DIRECTORY_NAME,
                    self.locks_root,
                ) as locks_descriptor:
                    lock_name = f"run-{self.run_id}.lock"
                    with acquire_file_lock(locks_descriptor, lock_name, self.locks_root / lock_name):
                        yield
        except FileNotFoundError as error:
            raise StateNotFoundError(f"run {self.run_id!r} is not initialized") from error

    @contextmanager
    def acquire_shard_lock(self, shard_id: ShardId) -> Iterator[None]:
        try:
            with self.open_shard_directory(shard_id) as shard_descriptor:
                with acquire_file_lock(
                    shard_descriptor,
                    _SHARD_LOCK_FILENAME,
                    self.get_shard_path(shard_id) / _SHARD_LOCK_FILENAME,
                ):
                    yield
        except FileNotFoundError as error:
            raise StateNotFoundError(f"shard {shard_id!r} is not persisted") from error

    @contextmanager
    def acquire_resume_lock(self, shard_id: ShardId) -> Iterator[None]:
        with ExitStack() as contexts:
            try:
                shard_descriptor = contexts.enter_context(self.open_shard_directory(shard_id))
                contexts.enter_context(
                    acquire_file_lock(
                        shard_descriptor,
                        _RESUME_LOCK_FILENAME,
                        self.get_shard_path(shard_id) / _RESUME_LOCK_FILENAME,
                    )
                )
            except FileNotFoundError as error:
                raise StateNotFoundError(f"shard {shard_id!r} is not persisted") from error
            except OSError as error:
                raise SlurmStateError(f"cannot lock dataset workspace for shard {shard_id!r}") from error
            yield

    @contextmanager
    def acquire_resume_and_shard_locks(self, shard_id: ShardId) -> Iterator[None]:
        """Acquire dataset-resume then shard state locks in the canonical order."""
        with self.acquire_resume_lock(shard_id):
            with self.acquire_shard_lock(shard_id):
                yield

    def publish_initial_state(
        self,
        authored_config: DataDesignerSlurmConfig,
        resolved_plan: ResolvedSlurmRunPlan,
        run: RunManifest,
        shards: tuple[ShardManifest, ...],
    ) -> None:
        with open_verified_directory(self.runs_root, require_private=True) as runs_descriptor:
            ensure_private_child_directory(runs_descriptor, self.run_id, self.run_root)
            with open_verified_child_directory(runs_descriptor, self.run_id, self.run_root) as run_descriptor:
                ensure_private_child_directory(
                    run_descriptor,
                    _SHARDS_DIRECTORY_NAME,
                    self.run_root / _SHARDS_DIRECTORY_NAME,
                )
                self._publish_immutable_record(run_descriptor, _AUTHORED_CONFIG_FILENAME, authored_config)
                self._publish_immutable_record(run_descriptor, _RESOLVED_PLAN_FILENAME, resolved_plan)
                with open_verified_child_directory(
                    run_descriptor,
                    _SHARDS_DIRECTORY_NAME,
                    self.run_root / _SHARDS_DIRECTORY_NAME,
                ) as shards_descriptor:
                    for shard in shards:
                        self._publish_initial_shard(shards_descriptor, shard)
                self._publish_immutable_record(run_descriptor, _RUN_FILENAME, run)

    def read_run(self) -> RunManifest:
        with self.open_run_directory() as run_descriptor:
            return self.read_record(run_descriptor, _RUN_FILENAME, self.run_root / _RUN_FILENAME, RunManifest)

    def read_authored_config(self) -> DataDesignerSlurmConfig:
        with self.open_run_directory() as run_descriptor:
            return self.read_record(
                run_descriptor,
                _AUTHORED_CONFIG_FILENAME,
                self.authored_config_path,
                DataDesignerSlurmConfig,
            )

    def read_resolved_plan(self) -> ResolvedSlurmRunPlan:
        with self.open_run_directory() as run_descriptor:
            return self.read_record(
                run_descriptor,
                _RESOLVED_PLAN_FILENAME,
                self.resolved_plan_path,
                ResolvedSlurmRunPlan,
            )

    def read_shards(self, shard_count: int) -> tuple[ShardManifest, ...]:
        with self.open_run_directory() as run_descriptor:
            with open_verified_child_directory(
                run_descriptor,
                _SHARDS_DIRECTORY_NAME,
                self.run_root / _SHARDS_DIRECTORY_NAME,
            ) as shards_descriptor:
                expected_names = tuple(f"shard-{index:05d}" for index in range(shard_count))
                if set(os.listdir(shards_descriptor)) != set(expected_names):
                    raise StateCorruptionError(f"run {self.run_id!r} has an incomplete shard directory set")
                return tuple(self._read_shard(shards_descriptor, shard_name) for shard_name in expected_names)

    def read_shard(self, shard_id: ShardId) -> ShardManifest:
        """Read one shard without scanning unrelated shard state."""
        with self.open_run_directory() as run_descriptor:
            with open_verified_child_directory(
                run_descriptor,
                _SHARDS_DIRECTORY_NAME,
                self.run_root / _SHARDS_DIRECTORY_NAME,
            ) as shards_descriptor:
                return self._read_shard(shards_descriptor, shard_id)

    def read_attempts(self, shard_id: ShardId) -> tuple[AttemptManifest, ...]:
        with self.open_shard_directory(shard_id) as shard_descriptor:
            with open_verified_child_directory(
                shard_descriptor,
                _ATTEMPTS_DIRECTORY_NAME,
                self.get_shard_path(shard_id) / _ATTEMPTS_DIRECTORY_NAME,
            ) as attempts_descriptor:
                names = tuple(os.listdir(attempts_descriptor))
                if any(_ATTEMPT_NAME_PATTERN.fullmatch(name) is None for name in names):
                    raise StateCorruptionError(f"shard {shard_id!r} contains an invalid attempt directory")
                ordered_names = tuple(sorted(names, key=lambda name: int(name.rsplit("-", maxsplit=1)[1])))
                attempts, incomplete_names = self._read_attempt_directories(
                    attempts_descriptor,
                    shard_id,
                    ordered_names,
                )
        self._validate_attempt_sequence(shard_id, attempts, incomplete_names)
        return attempts

    def publish_attempt(self, attempt: AttemptManifest) -> None:
        with self.open_shard_directory(attempt.shard_id) as shard_descriptor:
            with open_verified_child_directory(
                shard_descriptor,
                _ATTEMPTS_DIRECTORY_NAME,
                self.get_shard_path(attempt.shard_id) / _ATTEMPTS_DIRECTORY_NAME,
            ) as attempts_descriptor:
                attempt_root = self.get_attempt_path(attempt.shard_id, attempt.attempt_id)
                ensure_private_child_directory(attempts_descriptor, attempt.attempt_id, attempt_root)
                with open_verified_child_directory(
                    attempts_descriptor,
                    attempt.attempt_id,
                    attempt_root,
                ) as attempt_descriptor:
                    self._publish_immutable_record(attempt_descriptor, _ATTEMPT_FILENAME, attempt)

    def replace_attempt(self, attempt: AttemptManifest) -> None:
        with self.open_attempt_directory(attempt.shard_id, attempt.attempt_id) as attempt_descriptor:
            self._replace_record(attempt_descriptor, _ATTEMPT_FILENAME, attempt)

    def read_readiness(self, shard_id: ShardId, attempt_id: AttemptId) -> AttemptReadiness:
        with self.open_attempt_directory(shard_id, attempt_id) as attempt_descriptor:
            return self.read_record(
                attempt_descriptor,
                _READINESS_FILENAME,
                self.get_readiness_path(shard_id, attempt_id),
                AttemptReadiness,
            )

    def publish_readiness(self, readiness: AttemptReadiness) -> None:
        with self.open_attempt_directory(readiness.shard_id, readiness.attempt_id) as attempt_descriptor:
            self._publish_immutable_record(attempt_descriptor, _READINESS_FILENAME, readiness)

    def replace_readiness(self, readiness: AttemptReadiness) -> None:
        with self.open_attempt_directory(readiness.shard_id, readiness.attempt_id) as attempt_descriptor:
            self._replace_record(attempt_descriptor, _READINESS_FILENAME, readiness)

    def read_scheduler_observation(
        self,
        shard_id: ShardId,
        attempt_id: AttemptId,
    ) -> SchedulerObservation:
        """Read one attempt's latest reconciled scheduler evidence."""
        path = self.get_scheduler_observation_path(shard_id, attempt_id)
        with self.open_attempt_directory(shard_id, attempt_id) as attempt_descriptor:
            return self.read_record(
                attempt_descriptor,
                _SCHEDULER_OBSERVATION_FILENAME,
                path,
                SchedulerObservation,
            )

    def publish_scheduler_observation(
        self,
        shard_id: ShardId,
        attempt_id: AttemptId,
        observation: SchedulerObservation,
    ) -> None:
        """Publish the first scheduler observation for one attempt."""
        path = self.get_scheduler_observation_path(shard_id, attempt_id)
        with self.open_attempt_directory(shard_id, attempt_id) as attempt_descriptor:
            publish_immutable_text(
                attempt_descriptor,
                _SCHEDULER_OBSERVATION_FILENAME,
                observation.serialize_json(),
                path,
                maximum_size=_MAXIMUM_RECORD_SIZE,
            )

    def replace_scheduler_observation(
        self,
        shard_id: ShardId,
        attempt_id: AttemptId,
        observation: SchedulerObservation,
    ) -> None:
        """Atomically replace one attempt's scheduler observation."""
        path = self.get_scheduler_observation_path(shard_id, attempt_id)
        with self.open_attempt_directory(shard_id, attempt_id) as attempt_descriptor:
            replace_text(
                attempt_descriptor,
                _SCHEDULER_OBSERVATION_FILENAME,
                observation.serialize_json(),
                path,
                maximum_size=_MAXIMUM_RECORD_SIZE,
            )

    def sync_attempt_directory(self, shard_id: ShardId, attempt_id: AttemptId) -> None:
        with self.open_attempt_directory(shard_id, attempt_id) as attempt_descriptor:
            sync_directory(attempt_descriptor)

    def ensure_dataset_directory(
        self,
        shard_id: ShardId,
        attempt_id: AttemptId,
        effective_resume_mode: Literal["never", "always"],
    ) -> Path:
        if effective_resume_mode == "always":
            dataset_path = self.get_shard_path(shard_id) / _DATASET_DIRECTORY_NAME
            with self.open_shard_directory(shard_id) as descriptor:
                ensure_private_child_directory(descriptor, _DATASET_DIRECTORY_NAME, dataset_path)
            return dataset_path
        dataset_path = self.get_attempt_path(shard_id, attempt_id) / _DATASET_DIRECTORY_NAME
        with self.open_attempt_directory(shard_id, attempt_id) as descriptor:
            ensure_private_child_directory(descriptor, _DATASET_DIRECTORY_NAME, dataset_path)
        return dataset_path

    def read_finalization_records(
        self,
        shard_id: ShardId,
        attempt_id: AttemptId,
    ) -> tuple[ClientResult, CandidateOutputManifest]:
        attempt_path = self.get_attempt_path(shard_id, attempt_id)
        with self.open_attempt_directory(shard_id, attempt_id) as descriptor:
            client_result = self.read_record(
                descriptor,
                _CLIENT_RESULT_FILENAME,
                attempt_path / _CLIENT_RESULT_FILENAME,
                ClientResult,
            )
            candidate = self.read_record(
                descriptor,
                _CANDIDATE_OUTPUT_FILENAME,
                attempt_path / _CANDIDATE_OUTPUT_FILENAME,
                CandidateOutputManifest,
            )
        return client_result, candidate

    def publish_finalization_records(
        self,
        client_result: ClientResult,
        candidate: CandidateOutputManifest,
    ) -> None:
        """Convergently publish one producer-owned attempt result pair."""
        with self.open_attempt_directory(candidate.shard_id, candidate.attempt_id) as descriptor:
            self._publish_immutable_record(descriptor, _CANDIDATE_OUTPUT_FILENAME, candidate)
            self._publish_immutable_record(descriptor, _CLIENT_RESULT_FILENAME, client_result)

    def read_winner(self, shard_id: ShardId) -> ShardWinner:
        with self.open_shard_directory(shard_id) as descriptor:
            return self.read_record(descriptor, _WINNER_FILENAME, self.get_winner_path(shard_id), ShardWinner)

    def publish_winner(self, winner: ShardWinner) -> None:
        with self.open_shard_directory(winner.shard_id) as descriptor:
            self._publish_immutable_record(descriptor, _WINNER_FILENAME, winner)

    def read_record(
        self,
        directory_descriptor: int,
        name: str,
        display_path: Path,
        record_type: type[_RecordT],
    ) -> _RecordT:
        try:
            content = read_regular_text(
                directory_descriptor,
                name,
                display_path,
                maximum_size=_MAXIMUM_RECORD_SIZE,
            )
            _validate_supported_record_version(content, display_path)
            record = record_type.model_validate_json(content)
            if record.serialize_json() != content:
                raise StateCorruptionError(
                    f"persisted state record {display_path.name!r} is not canonical for schema_version 1; "
                    "persisted schema migrations are not supported"
                )
            return record
        except StateCorruptionError:
            raise
        except (RecursionError, UnicodeError, ValueError, ValidationError) as error:
            raise StateCorruptionError(f"persisted state record {display_path.name!r} is invalid") from error

    @contextmanager
    def open_run_directory(self) -> Iterator[int]:
        with open_verified_directory(self.runs_root, require_private=True) as runs_descriptor:
            with open_verified_child_directory(runs_descriptor, self.run_id, self.run_root) as run_descriptor:
                yield run_descriptor

    @contextmanager
    def open_shard_directory(self, shard_id: ShardId) -> Iterator[int]:
        with self.open_run_directory() as run_descriptor:
            with open_verified_child_directory(
                run_descriptor,
                _SHARDS_DIRECTORY_NAME,
                self.run_root / _SHARDS_DIRECTORY_NAME,
            ) as shards_descriptor:
                with open_verified_child_directory(
                    shards_descriptor,
                    shard_id,
                    self.get_shard_path(shard_id),
                ) as shard_descriptor:
                    yield shard_descriptor

    @contextmanager
    def open_attempt_directory(self, shard_id: ShardId, attempt_id: AttemptId) -> Iterator[int]:
        with self.open_shard_directory(shard_id) as shard_descriptor:
            with open_verified_child_directory(
                shard_descriptor,
                _ATTEMPTS_DIRECTORY_NAME,
                self.get_shard_path(shard_id) / _ATTEMPTS_DIRECTORY_NAME,
            ) as attempts_descriptor:
                with open_verified_child_directory(
                    attempts_descriptor,
                    attempt_id,
                    self.get_attempt_path(shard_id, attempt_id),
                ) as attempt_descriptor:
                    yield attempt_descriptor

    def _publish_initial_shard(self, shards_descriptor: int, shard: ShardManifest) -> None:
        shard_root = self.get_shard_path(shard.shard_id)
        ensure_private_child_directory(shards_descriptor, shard.shard_id, shard_root)
        with open_verified_child_directory(shards_descriptor, shard.shard_id, shard_root) as shard_descriptor:
            ensure_private_child_directory(
                shard_descriptor,
                _ATTEMPTS_DIRECTORY_NAME,
                shard_root / _ATTEMPTS_DIRECTORY_NAME,
            )
            self._publish_immutable_record(shard_descriptor, _SHARD_FILENAME, shard)

    def _read_shard(self, shards_descriptor: int, shard_id: str) -> ShardManifest:
        shard_root = self.get_shard_path(shard_id)
        with open_verified_child_directory(shards_descriptor, shard_id, shard_root) as shard_descriptor:
            return self.read_record(shard_descriptor, _SHARD_FILENAME, shard_root / _SHARD_FILENAME, ShardManifest)

    def _read_attempt_directories(
        self,
        attempts_descriptor: int,
        shard_id: ShardId,
        names: tuple[str, ...],
    ) -> tuple[tuple[AttemptManifest, ...], tuple[str, ...]]:
        published: list[AttemptManifest] = []
        incomplete: list[str] = []
        for name in names:
            attempt = self._read_attempt_if_published(attempts_descriptor, shard_id, name)
            if attempt is None:
                incomplete.append(name)
            else:
                published.append(attempt)
        return tuple(published), tuple(incomplete)

    def _read_attempt_if_published(
        self,
        attempts_descriptor: int,
        shard_id: ShardId,
        attempt_id: str,
    ) -> AttemptManifest | None:
        try:
            return self._read_attempt(attempts_descriptor, shard_id, attempt_id)
        except FileNotFoundError:
            pass

        attempt_root = self.get_attempt_path(shard_id, attempt_id)
        with open_verified_child_directory(attempts_descriptor, attempt_id, attempt_root) as attempt_descriptor:
            unexpected = tuple(name for name in os.listdir(attempt_descriptor) if not is_state_temporary_name(name))
        if unexpected == ():
            return None
        if unexpected == (_ATTEMPT_FILENAME,):
            return self._read_attempt(attempts_descriptor, shard_id, attempt_id)
        raise StateCorruptionError(f"attempt {attempt_id!r} contains unpublished state records")

    def _read_attempt(self, attempts_descriptor: int, shard_id: ShardId, attempt_id: str) -> AttemptManifest:
        attempt_root = self.get_attempt_path(shard_id, attempt_id)
        with open_verified_child_directory(attempts_descriptor, attempt_id, attempt_root) as attempt_descriptor:
            return self.read_record(
                attempt_descriptor,
                _ATTEMPT_FILENAME,
                attempt_root / _ATTEMPT_FILENAME,
                AttemptManifest,
            )

    @staticmethod
    def _validate_attempt_sequence(
        shard_id: ShardId,
        attempts: tuple[AttemptManifest, ...],
        incomplete_names: tuple[str, ...],
    ) -> None:
        ordinals = tuple(attempt.attempt_ordinal for attempt in attempts)
        if ordinals != tuple(range(1, len(attempts) + 1)):
            raise StateCorruptionError(f"shard {shard_id!r} attempts are not a complete monotonic sequence")
        if tuple(attempt.attempt_id for attempt in attempts) != tuple(f"attempt-{ordinal:04d}" for ordinal in ordinals):
            raise StateCorruptionError(f"shard {shard_id!r} attempt IDs do not match their ordinals")
        expected_incomplete = f"attempt-{len(attempts) + 1:04d}"
        if incomplete_names not in ((), (expected_incomplete,)):
            raise StateCorruptionError(f"shard {shard_id!r} contains an invalid incomplete attempt")

    def _publish_immutable_record(self, directory_descriptor: int, name: str, record: ContractRecord) -> None:
        publish_immutable_text(
            directory_descriptor,
            name,
            record.serialize_json(),
            self._get_record_path(name, record),
            maximum_size=_MAXIMUM_RECORD_SIZE,
        )

    def _replace_record(self, directory_descriptor: int, name: str, record: ContractRecord) -> None:
        replace_text(
            directory_descriptor,
            name,
            record.serialize_json(),
            self._get_record_path(name, record),
            maximum_size=_MAXIMUM_RECORD_SIZE,
        )

    def _get_record_path(self, name: str, record: ContractRecord) -> Path:
        if name == _RUN_FILENAME:
            return self.run_root / name
        if name in (_AUTHORED_CONFIG_FILENAME, _RESOLVED_PLAN_FILENAME):
            return self.run_root / name
        if isinstance(record, ShardManifest):
            return self.get_shard_path(record.shard_id) / name
        if isinstance(record, ShardWinner):
            return self.get_winner_path(record.shard_id)
        if isinstance(record, (AttemptManifest, AttemptReadiness, ClientResult, CandidateOutputManifest)):
            return self.get_attempt_path(record.shard_id, record.attempt_id) / name
        raise TypeError(f"unsupported persisted record type: {type(record).__name__}")


def _validate_supported_record_version(content: str, display_path: Path) -> None:
    payload = json.loads(content)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise StateCorruptionError(f"persisted state record {display_path.name!r} uses an unsupported schema_version")
