# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic manifest-backed persistence for one Slurm run."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

from pydantic import TypeAdapter, ValidationError

from data_designer.slurm.config import DataDesignerSlurmConfig
from data_designer.slurm.contracts import AttemptId, ContractRecord, Identifier, ShardId, validate_absolute_path
from data_designer.slurm.integration import IntegrationContractError, PlanStateValidator
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.state.errors import (
    SlurmStateError,
    StateConflictError,
    StateCorruptionError,
    StateNotFoundError,
)
from data_designer.slurm.state.execution import AttemptLifecycleState, AttemptManifest, RunManifest, ShardManifest
from data_designer.slurm.state.filesystem import (
    acquire_file_lock,
    ensure_private_child_directory,
    open_verified_child_directory,
    open_verified_directory,
    publish_immutable_text,
    read_regular_text,
    replace_text,
)
from data_designer.slurm.state.readiness import AttemptReadiness
from data_designer.slurm.state.reconciliation import validate_readiness_transition
from data_designer.slurm.state.validation import (
    StateContractError,
    validate_attempt_manifest,
    validate_attempt_set,
    validate_attempt_transition,
)

_RUN_FILENAME = "run.json"
_AUTHORED_CONFIG_FILENAME = "authored-config.json"
_RESOLVED_PLAN_FILENAME = "resolved-plan.json"
_SHARDS_DIRECTORY_NAME = "shards"
_ATTEMPTS_DIRECTORY_NAME = "attempts"
_SHARD_FILENAME = "shard.json"
_SHARD_LOCK_FILENAME = "shard.lock"
_ATTEMPT_FILENAME = "attempt.json"
_READINESS_FILENAME = "readiness.json"
_LOCK_DIRECTORY_NAME = ".locks"
_MAXIMUM_RECORD_SIZE = 16 * 1024 * 1024
_SHARD_NAME_PATTERN = re.compile(r"^shard-[0-9]{5,}$")
_ATTEMPT_NAME_PATTERN = re.compile(r"^attempt-[0-9]{4,}$")
_IDENTIFIER_ADAPTER = TypeAdapter(Identifier)
_SHARD_ID_ADAPTER = TypeAdapter(ShardId)
_ATTEMPT_ID_ADAPTER = TypeAdapter(AttemptId)
_RecordT = TypeVar("_RecordT", bound=ContractRecord)


class SlurmStateWriter:
    """Persist and reload one run's immutable and revisioned state.

    The writer derives every mutable path from an explicit workspace and
    validated run ID. Run initialization publishes ``run.json`` last, so
    readers never accept a partially initialized run. Attempt writes acquire
    the run lock before the shard lock; readiness writes require only the
    shard lock.

    Args:
        workspace_root: Selected compute-visible workspace root.
        run_id: Stable application-owned run identity.
    """

    def __init__(self, workspace_root: str | Path, run_id: Identifier) -> None:
        try:
            normalized_root = validate_absolute_path(Path(workspace_root).as_posix())
            normalized_run_id = _IDENTIFIER_ADAPTER.validate_python(run_id, strict=True)
        except (ValidationError, ValueError) as error:
            raise SlurmStateError("invalid persisted run location") from error
        self._workspace_root = Path(normalized_root)
        self._runs_root = self._workspace_root / "runs"
        self._locks_root = self._runs_root / _LOCK_DIRECTORY_NAME
        self._run_id = normalized_run_id
        self._run_root = self._runs_root / normalized_run_id

    @property
    def run_root(self) -> Path:
        """Return the workspace-derived root for this run."""
        return self._run_root

    def initialize_run(
        self,
        authored_config: DataDesignerSlurmConfig,
        resolved_plan: ResolvedSlurmRunPlan,
        run: RunManifest,
        shards: tuple[ShardManifest, ...],
    ) -> RunManifest:
        """Convergently publish immutable run intent, plan, and shards.

        The authored config, resolved plan, and shards publish before
        ``run.json``. A retry after process interruption may reuse identical
        records, but it cannot replace different bytes.

        Raises:
            StateConflictError: If inputs disagree or immutable state differs.
            SlurmStateError: If package-owned state cannot be persisted.
        """
        self._validate_initial_state(authored_config, resolved_plan, run, shards)
        try:
            self._ensure_storage()
            with self._run_lock():
                self._initialize_run_locked(authored_config, resolved_plan, run, shards)
            return run
        except StateConflictError:
            raise
        except FileExistsError as error:
            raise StateConflictError(f"run {self._run_id!r} already contains different immutable state") from error
        except OSError as error:
            raise SlurmStateError(f"cannot initialize persisted run {self._run_id!r}") from error

    def load_run(self) -> RunManifest:
        """Load the committed immutable run manifest."""
        try:
            with self._open_run_directory() as run_descriptor:
                return self._read_record(run_descriptor, _RUN_FILENAME, self._run_root / _RUN_FILENAME, RunManifest)
        except FileNotFoundError as error:
            raise StateNotFoundError(f"run {self._run_id!r} is not initialized") from error
        except StateCorruptionError:
            raise
        except OSError as error:
            raise StateCorruptionError(f"cannot load persisted run {self._run_id!r}") from error

    def load_authored_config(self) -> DataDesignerSlurmConfig:
        """Load and digest-verify the run's immutable authored config."""
        run = self.load_run()
        try:
            with self._open_run_directory() as run_descriptor:
                authored_config = self._read_record(
                    run_descriptor,
                    _AUTHORED_CONFIG_FILENAME,
                    self._run_root / _AUTHORED_CONFIG_FILENAME,
                    DataDesignerSlurmConfig,
                )
        except (FileNotFoundError, OSError) as error:
            raise StateCorruptionError(f"run {self._run_id!r} has no valid authored config") from error
        expected_path = (self._run_root / _AUTHORED_CONFIG_FILENAME).as_posix()
        if run.authored_config.path != expected_path or run.authored_config.sha256 != authored_config.compute_sha256():
            raise StateCorruptionError(f"run {self._run_id!r} authored config does not match its manifest")
        return authored_config

    def load_resolved_plan(self) -> ResolvedSlurmRunPlan:
        """Load and digest-verify the run's immutable resolved plan."""
        run = self.load_run()
        try:
            with self._open_run_directory() as run_descriptor:
                plan = self._read_record(
                    run_descriptor,
                    _RESOLVED_PLAN_FILENAME,
                    self._run_root / _RESOLVED_PLAN_FILENAME,
                    ResolvedSlurmRunPlan,
                )
        except (FileNotFoundError, OSError) as error:
            raise StateCorruptionError(f"run {self._run_id!r} has no valid resolved plan") from error
        expected_path = (self._run_root / _RESOLVED_PLAN_FILENAME).as_posix()
        if run.resolved_plan.path != expected_path or run.resolved_plan.sha256 != plan.compute_sha256():
            raise StateCorruptionError(f"run {self._run_id!r} resolved plan does not match its manifest")
        if run.authored_config != plan.authored_config:
            raise StateCorruptionError(f"run {self._run_id!r} resolved plan does not bind its authored config")
        self.load_authored_config()
        return plan

    def load_shards(self) -> tuple[ShardManifest, ...]:
        """Load and validate the complete ordered shard set."""
        run = self.load_run()
        try:
            with self._open_run_directory() as run_descriptor:
                with open_verified_child_directory(
                    run_descriptor,
                    _SHARDS_DIRECTORY_NAME,
                    self._run_root / _SHARDS_DIRECTORY_NAME,
                ) as shards_descriptor:
                    expected_names = tuple(f"shard-{index:05d}" for index in range(run.shard_count))
                    actual_names = tuple(
                        sorted(name for name in os.listdir(shards_descriptor) if not name.startswith("."))
                    )
                    if actual_names != expected_names:
                        raise StateCorruptionError(f"run {self._run_id!r} has an incomplete shard directory set")
                    shards = tuple(self._read_shard(shards_descriptor, shard_name) for shard_name in expected_names)
            PlanStateValidator(self.load_resolved_plan()).validate_plan_shards(run, shards)
            return shards
        except (IntegrationContractError, StateContractError) as error:
            raise StateCorruptionError(f"run {self._run_id!r} has invalid persisted shards") from error
        except StateCorruptionError:
            raise
        except (FileNotFoundError, OSError) as error:
            raise StateCorruptionError(f"run {self._run_id!r} has unreadable persisted shards") from error

    def create_attempt(self, attempt: AttemptManifest) -> AttemptManifest:
        """Publish the next monotonically numbered attempt for one shard."""
        self._validate_attempt_location(attempt)
        try:
            with self._run_lock(), self._shard_lock(attempt.shard_id):
                run, plan, shards = self._load_context()
                shard = self._get_shard(shards, attempt.shard_id)
                attempts_by_shard = self._load_validated_attempts(run, plan, shards)
                shard_attempts = attempts_by_shard[attempt.shard_id]
                existing = next((item for item in shard_attempts if item.attempt_id == attempt.attempt_id), None)
                if existing is not None:
                    if existing == attempt:
                        return existing
                    raise StateConflictError(f"attempt {attempt.attempt_id!r} already contains different state")
                expected_ordinal = len(shard_attempts) + 1
                if (
                    attempt.attempt_ordinal != expected_ordinal
                    or attempt.attempt_id != f"attempt-{expected_ordinal:04d}"
                ):
                    raise StateConflictError("attempt identity does not match the next shard ordinal")
                self._validate_attempt_against_plan(run, plan, shard, attempt)
                all_attempts = tuple(
                    item for current_shard in shards for item in attempts_by_shard[current_shard.shard_id]
                )
                all_attempts += (attempt,)
                validate_attempt_set(run, shards, all_attempts)
                self._publish_attempt(attempt)
                return attempt
        except (StateConflictError, StateCorruptionError, StateNotFoundError):
            raise
        except (IntegrationContractError, StateContractError) as error:
            raise StateConflictError("attempt does not match persisted run intent") from error
        except (FileNotFoundError, OSError) as error:
            raise SlurmStateError(f"cannot create attempt {attempt.attempt_id!r}") from error

    def update_attempt(self, attempt: AttemptManifest) -> AttemptManifest:
        """Atomically replace one attempt after validating its monotonic transition."""
        self._validate_attempt_location(attempt)
        try:
            with self._run_lock(), self._shard_lock(attempt.shard_id):
                run, plan, shards = self._load_context()
                shard = self._get_shard(shards, attempt.shard_id)
                attempts_by_shard = self._load_validated_attempts(run, plan, shards)
                previous = self._get_attempt(attempts_by_shard[attempt.shard_id], attempt.attempt_id)
                if previous == attempt:
                    return previous
                validate_attempt_transition(previous, attempt)
                self._validate_attempt_against_plan(run, plan, shard, attempt)
                all_attempts = tuple(
                    attempt if item.shard_id == attempt.shard_id and item.attempt_id == attempt.attempt_id else item
                    for current_shard in shards
                    for item in attempts_by_shard[current_shard.shard_id]
                )
                validate_attempt_set(run, shards, all_attempts)
                self._replace_attempt(attempt)
                return attempt
        except (StateConflictError, StateCorruptionError, StateNotFoundError):
            raise
        except (IntegrationContractError, StateContractError) as error:
            raise StateConflictError("attempt update is not a valid monotonic transition") from error
        except (FileNotFoundError, OSError) as error:
            raise SlurmStateError(f"cannot update attempt {attempt.attempt_id!r}") from error

    def load_attempts(self, shard_id: ShardId) -> tuple[AttemptManifest, ...]:
        """Load one shard's attempts in ordinal order."""
        normalized_shard_id = self._validate_shard_id(shard_id)
        self.load_run()
        try:
            run, plan, shards = self._load_context()
            self._get_shard(shards, normalized_shard_id)
            return self._load_validated_attempts(run, plan, shards)[normalized_shard_id]
        except (IntegrationContractError, StateContractError) as error:
            raise StateCorruptionError(f"shard {normalized_shard_id!r} has invalid attempts") from error
        except (StateCorruptionError, StateNotFoundError):
            raise
        except (FileNotFoundError, OSError) as error:
            raise StateCorruptionError(f"cannot load attempts for shard {normalized_shard_id!r}") from error

    def load_attempt(self, shard_id: ShardId, attempt_id: AttemptId) -> AttemptManifest:
        """Load one persisted attempt."""
        normalized_shard_id = self._validate_shard_id(shard_id)
        normalized_attempt_id = self._validate_attempt_id(attempt_id)
        for attempt in self.load_attempts(normalized_shard_id):
            if attempt.attempt_id == normalized_attempt_id:
                return attempt
        raise StateNotFoundError(f"attempt {normalized_attempt_id!r} is not persisted")

    def write_readiness(self, readiness: AttemptReadiness) -> AttemptReadiness:
        """Create or atomically replace one validated readiness snapshot."""
        self._validate_readiness_location(readiness)
        try:
            with self._shard_lock(readiness.shard_id):
                run, plan, shards = self._load_context()
                attempts_by_shard = self._load_validated_attempts(run, plan, shards)
                attempt = self._get_attempt(attempts_by_shard[readiness.shard_id], readiness.attempt_id)
                with self._open_attempt_directory(readiness.shard_id, readiness.attempt_id) as attempt_descriptor:
                    try:
                        previous = self._read_record(
                            attempt_descriptor,
                            _READINESS_FILENAME,
                            self._readiness_path(readiness.shard_id, readiness.attempt_id),
                            AttemptReadiness,
                        )
                    except FileNotFoundError:
                        PlanStateValidator(plan).validate_initial_readiness(attempt, readiness)
                        publish_immutable_text(
                            attempt_descriptor,
                            _READINESS_FILENAME,
                            readiness.serialize_json(),
                            self._readiness_path(readiness.shard_id, readiness.attempt_id),
                            maximum_size=_MAXIMUM_RECORD_SIZE,
                        )
                        return readiness
                    if previous == readiness:
                        return previous
                    validate_readiness_transition(previous, readiness)
                    replace_text(
                        attempt_descriptor,
                        _READINESS_FILENAME,
                        readiness.serialize_json(),
                        self._readiness_path(readiness.shard_id, readiness.attempt_id),
                        maximum_size=_MAXIMUM_RECORD_SIZE,
                    )
                    return readiness
        except (StateConflictError, StateCorruptionError, StateNotFoundError):
            raise
        except (IntegrationContractError, StateContractError) as error:
            raise StateConflictError("readiness update is not a valid monotonic transition") from error
        except (FileNotFoundError, OSError) as error:
            raise SlurmStateError(f"cannot persist readiness for attempt {readiness.attempt_id!r}") from error

    def load_readiness(self, shard_id: ShardId, attempt_id: AttemptId) -> AttemptReadiness:
        """Load one attempt's latest readiness snapshot."""
        normalized_shard_id = self._validate_shard_id(shard_id)
        normalized_attempt_id = self._validate_attempt_id(attempt_id)
        self.load_attempt(normalized_shard_id, normalized_attempt_id)
        try:
            with self._open_attempt_directory(normalized_shard_id, normalized_attempt_id) as attempt_descriptor:
                return self._read_record(
                    attempt_descriptor,
                    _READINESS_FILENAME,
                    self._readiness_path(normalized_shard_id, normalized_attempt_id),
                    AttemptReadiness,
                )
        except FileNotFoundError as error:
            raise StateNotFoundError(f"attempt {normalized_attempt_id!r} has no readiness snapshot") from error
        except StateCorruptionError:
            raise
        except OSError as error:
            raise StateCorruptionError(f"attempt {normalized_attempt_id!r} has unreadable readiness") from error

    def _validate_initial_state(
        self,
        authored_config: DataDesignerSlurmConfig,
        resolved_plan: ResolvedSlurmRunPlan,
        run: RunManifest,
        shards: tuple[ShardManifest, ...],
    ) -> None:
        try:
            if not isinstance(authored_config, DataDesignerSlurmConfig):
                raise StateContractError("authored config has an invalid type")
            if not isinstance(resolved_plan, ResolvedSlurmRunPlan):
                raise StateContractError("resolved plan has an invalid type")
            if not isinstance(run, RunManifest):
                raise StateContractError("run manifest has an invalid type")
            if not isinstance(shards, tuple) or any(not isinstance(shard, ShardManifest) for shard in shards):
                raise StateContractError("shard manifests have an invalid type")
            if run.run_id != self._run_id or resolved_plan.run_id != self._run_id:
                raise StateContractError("run identity does not match the state writer")
            if resolved_plan.selected_profile.profile.workspace_root != self._workspace_root.as_posix():
                raise StateContractError("resolved plan workspace does not match the state writer")
            if resolved_plan.authored_config.sha256 != authored_config.compute_sha256():
                raise StateContractError("authored config digest does not match the resolved plan")
            if run.authored_config != resolved_plan.authored_config:
                raise StateContractError("run authored config does not match the resolved plan")
            expected_authored_path = (self._run_root / _AUTHORED_CONFIG_FILENAME).as_posix()
            if run.authored_config.path != expected_authored_path:
                raise StateContractError("run authored config reference does not match its persisted location")
            expected_plan_path = (self._run_root / _RESOLVED_PLAN_FILENAME).as_posix()
            if (
                run.resolved_plan.path != expected_plan_path
                or run.resolved_plan.sha256 != resolved_plan.compute_sha256()
            ):
                raise StateContractError("run resolved plan reference does not match persisted plan bytes")
            PlanStateValidator(resolved_plan).validate_plan_shards(run, shards)
        except (IntegrationContractError, StateContractError) as error:
            raise StateConflictError("run initialization does not match resolved plan intent") from error

    def _initialize_run_locked(
        self,
        authored_config: DataDesignerSlurmConfig,
        resolved_plan: ResolvedSlurmRunPlan,
        run: RunManifest,
        shards: tuple[ShardManifest, ...],
    ) -> None:
        with open_verified_directory(self._runs_root, require_private=True) as runs_descriptor:
            ensure_private_child_directory(runs_descriptor, self._run_id, self._run_root)
            with open_verified_child_directory(runs_descriptor, self._run_id, self._run_root) as run_descriptor:
                ensure_private_child_directory(
                    run_descriptor,
                    _SHARDS_DIRECTORY_NAME,
                    self._run_root / _SHARDS_DIRECTORY_NAME,
                )
                publish_immutable_text(
                    run_descriptor,
                    _AUTHORED_CONFIG_FILENAME,
                    authored_config.serialize_json(),
                    self._run_root / _AUTHORED_CONFIG_FILENAME,
                    maximum_size=_MAXIMUM_RECORD_SIZE,
                )
                publish_immutable_text(
                    run_descriptor,
                    _RESOLVED_PLAN_FILENAME,
                    resolved_plan.serialize_json(),
                    self._run_root / _RESOLVED_PLAN_FILENAME,
                    maximum_size=_MAXIMUM_RECORD_SIZE,
                )
                with open_verified_child_directory(
                    run_descriptor,
                    _SHARDS_DIRECTORY_NAME,
                    self._run_root / _SHARDS_DIRECTORY_NAME,
                ) as shards_descriptor:
                    for shard in shards:
                        self._initialize_shard(shards_descriptor, shard)
                publish_immutable_text(
                    run_descriptor,
                    _RUN_FILENAME,
                    run.serialize_json(),
                    self._run_root / _RUN_FILENAME,
                    maximum_size=_MAXIMUM_RECORD_SIZE,
                )

    def _initialize_shard(self, shards_descriptor: int, shard: ShardManifest) -> None:
        shard_root = self._shard_path(shard.shard_id)
        ensure_private_child_directory(shards_descriptor, shard.shard_id, shard_root)
        with open_verified_child_directory(shards_descriptor, shard.shard_id, shard_root) as shard_descriptor:
            ensure_private_child_directory(
                shard_descriptor,
                _ATTEMPTS_DIRECTORY_NAME,
                shard_root / _ATTEMPTS_DIRECTORY_NAME,
            )
            publish_immutable_text(
                shard_descriptor,
                _SHARD_FILENAME,
                shard.serialize_json(),
                shard_root / _SHARD_FILENAME,
                maximum_size=_MAXIMUM_RECORD_SIZE,
            )

    def _ensure_storage(self) -> None:
        with open_verified_directory(self._workspace_root) as workspace_descriptor:
            ensure_private_child_directory(workspace_descriptor, "runs", self._runs_root)
            with open_verified_child_directory(workspace_descriptor, "runs", self._runs_root) as runs_descriptor:
                ensure_private_child_directory(runs_descriptor, _LOCK_DIRECTORY_NAME, self._locks_root)

    @contextmanager
    def _run_lock(self) -> Iterator[None]:
        try:
            with open_verified_directory(self._runs_root, require_private=True) as runs_descriptor:
                with open_verified_child_directory(
                    runs_descriptor,
                    _LOCK_DIRECTORY_NAME,
                    self._locks_root,
                ) as locks_descriptor:
                    lock_name = f"run-{self._run_id}.lock"
                    with acquire_file_lock(locks_descriptor, lock_name, self._locks_root / lock_name):
                        yield
        except FileNotFoundError as error:
            raise StateNotFoundError(f"run {self._run_id!r} is not initialized") from error

    @contextmanager
    def _shard_lock(self, shard_id: ShardId) -> Iterator[None]:
        normalized_shard_id = self._validate_shard_id(shard_id)
        try:
            with self._open_shard_directory(normalized_shard_id) as shard_descriptor:
                with acquire_file_lock(
                    shard_descriptor,
                    _SHARD_LOCK_FILENAME,
                    self._shard_path(normalized_shard_id) / _SHARD_LOCK_FILENAME,
                ):
                    yield
        except FileNotFoundError as error:
            raise StateNotFoundError(f"shard {normalized_shard_id!r} is not persisted") from error

    @contextmanager
    def _open_run_directory(self) -> Iterator[int]:
        with open_verified_directory(self._runs_root, require_private=True) as runs_descriptor:
            with open_verified_child_directory(runs_descriptor, self._run_id, self._run_root) as run_descriptor:
                yield run_descriptor

    @contextmanager
    def _open_shard_directory(self, shard_id: ShardId) -> Iterator[int]:
        with self._open_run_directory() as run_descriptor:
            with open_verified_child_directory(
                run_descriptor,
                _SHARDS_DIRECTORY_NAME,
                self._run_root / _SHARDS_DIRECTORY_NAME,
            ) as shards_descriptor:
                with open_verified_child_directory(
                    shards_descriptor,
                    shard_id,
                    self._shard_path(shard_id),
                ) as shard_descriptor:
                    yield shard_descriptor

    @contextmanager
    def _open_attempt_directory(self, shard_id: ShardId, attempt_id: AttemptId) -> Iterator[int]:
        with self._open_shard_directory(shard_id) as shard_descriptor:
            with open_verified_child_directory(
                shard_descriptor,
                _ATTEMPTS_DIRECTORY_NAME,
                self._shard_path(shard_id) / _ATTEMPTS_DIRECTORY_NAME,
            ) as attempts_descriptor:
                with open_verified_child_directory(
                    attempts_descriptor,
                    attempt_id,
                    self._attempt_path(shard_id, attempt_id),
                ) as attempt_descriptor:
                    yield attempt_descriptor

    def _load_context(self) -> tuple[RunManifest, ResolvedSlurmRunPlan, tuple[ShardManifest, ...]]:
        run = self.load_run()
        plan = self.load_resolved_plan()
        shards = self.load_shards()
        return run, plan, shards

    def _read_shard(self, shards_descriptor: int, shard_id: str) -> ShardManifest:
        if _SHARD_NAME_PATTERN.fullmatch(shard_id) is None:
            raise StateCorruptionError(f"run {self._run_id!r} contains an invalid shard directory")
        shard_root = self._shard_path(shard_id)
        with open_verified_child_directory(shards_descriptor, shard_id, shard_root) as shard_descriptor:
            return self._read_record(
                shard_descriptor,
                _SHARD_FILENAME,
                shard_root / _SHARD_FILENAME,
                ShardManifest,
            )

    def _load_attempts_locked(self, shard_id: ShardId) -> tuple[AttemptManifest, ...]:
        with self._open_shard_directory(shard_id) as shard_descriptor:
            with open_verified_child_directory(
                shard_descriptor,
                _ATTEMPTS_DIRECTORY_NAME,
                self._shard_path(shard_id) / _ATTEMPTS_DIRECTORY_NAME,
            ) as attempts_descriptor:
                names = tuple(sorted(name for name in os.listdir(attempts_descriptor) if not name.startswith(".")))
                if any(_ATTEMPT_NAME_PATTERN.fullmatch(name) is None for name in names):
                    raise StateCorruptionError(f"shard {shard_id!r} contains an invalid attempt directory")
                published_attempts: list[AttemptManifest] = []
                incomplete_names: list[str] = []
                for name in names:
                    try:
                        published_attempts.append(self._read_attempt(attempts_descriptor, shard_id, name))
                    except FileNotFoundError:
                        incomplete_names.append(name)
                attempts = tuple(published_attempts)
        ordinals = tuple(attempt.attempt_ordinal for attempt in attempts)
        if ordinals != tuple(range(1, len(attempts) + 1)):
            raise StateCorruptionError(f"shard {shard_id!r} attempts are not a complete monotonic sequence")
        if tuple(attempt.attempt_id for attempt in attempts) != tuple(f"attempt-{ordinal:04d}" for ordinal in ordinals):
            raise StateCorruptionError(f"shard {shard_id!r} attempt IDs do not match their ordinals")
        expected_incomplete = f"attempt-{len(attempts) + 1:04d}"
        if incomplete_names not in ([], [expected_incomplete]):
            raise StateCorruptionError(f"shard {shard_id!r} contains an invalid incomplete attempt")
        return attempts

    def _read_attempt(self, attempts_descriptor: int, shard_id: ShardId, attempt_id: str) -> AttemptManifest:
        attempt_root = self._attempt_path(shard_id, attempt_id)
        with open_verified_child_directory(attempts_descriptor, attempt_id, attempt_root) as attempt_descriptor:
            return self._read_record(
                attempt_descriptor,
                _ATTEMPT_FILENAME,
                attempt_root / _ATTEMPT_FILENAME,
                AttemptManifest,
            )

    def _load_validated_attempts(
        self,
        run: RunManifest,
        plan: ResolvedSlurmRunPlan,
        shards: tuple[ShardManifest, ...],
    ) -> dict[ShardId, tuple[AttemptManifest, ...]]:
        try:
            attempts_by_shard = {shard.shard_id: self._load_attempts_locked(shard.shard_id) for shard in shards}
            all_attempts = tuple(attempt for shard in shards for attempt in attempts_by_shard[shard.shard_id])
            for shard in shards:
                for attempt in attempts_by_shard[shard.shard_id]:
                    self._validate_attempt_against_plan(run, plan, shard, attempt)
            validate_attempt_set(run, shards, all_attempts)
            return attempts_by_shard
        except (IntegrationContractError, StateContractError) as error:
            raise StateCorruptionError(f"run {self._run_id!r} has invalid persisted attempts") from error

    def _publish_attempt(self, attempt: AttemptManifest) -> None:
        with self._open_shard_directory(attempt.shard_id) as shard_descriptor:
            with open_verified_child_directory(
                shard_descriptor,
                _ATTEMPTS_DIRECTORY_NAME,
                self._shard_path(attempt.shard_id) / _ATTEMPTS_DIRECTORY_NAME,
            ) as attempts_descriptor:
                attempt_root = self._attempt_path(attempt.shard_id, attempt.attempt_id)
                ensure_private_child_directory(attempts_descriptor, attempt.attempt_id, attempt_root)
                with open_verified_child_directory(
                    attempts_descriptor,
                    attempt.attempt_id,
                    attempt_root,
                ) as attempt_descriptor:
                    publish_immutable_text(
                        attempt_descriptor,
                        _ATTEMPT_FILENAME,
                        attempt.serialize_json(),
                        attempt_root / _ATTEMPT_FILENAME,
                        maximum_size=_MAXIMUM_RECORD_SIZE,
                    )

    def _replace_attempt(self, attempt: AttemptManifest) -> None:
        with self._open_attempt_directory(attempt.shard_id, attempt.attempt_id) as attempt_descriptor:
            replace_text(
                attempt_descriptor,
                _ATTEMPT_FILENAME,
                attempt.serialize_json(),
                self._attempt_path(attempt.shard_id, attempt.attempt_id) / _ATTEMPT_FILENAME,
                maximum_size=_MAXIMUM_RECORD_SIZE,
            )

    def _validate_attempt_against_plan(
        self,
        run: RunManifest,
        plan: ResolvedSlurmRunPlan,
        shard: ShardManifest,
        attempt: AttemptManifest,
    ) -> None:
        validate_attempt_manifest(run, shard, attempt)
        planned_shard = plan.shards[shard.shard_index]
        if planned_shard.shard_id != attempt.shard_id:
            raise StateContractError("attempt shard does not match resolved plan order")
        if attempt.attempt_id != f"attempt-{attempt.attempt_ordinal:04d}":
            raise StateContractError("attempt ID does not match its ordinal")
        if attempt.scheduler is not None and attempt.scheduler.array_task_id != planned_shard.array_task_index:
            raise StateContractError("attempt scheduler task does not match the resolved plan shard")
        if attempt.state is not AttemptLifecycleState.CREATED:
            PlanStateValidator(plan).validate_planned_attempt(planned_shard, attempt)

    @staticmethod
    def _get_shard(shards: tuple[ShardManifest, ...], shard_id: ShardId) -> ShardManifest:
        try:
            return next(shard for shard in shards if shard.shard_id == shard_id)
        except StopIteration:
            raise StateNotFoundError(f"shard {shard_id!r} is not persisted") from None

    @staticmethod
    def _get_attempt(attempts: tuple[AttemptManifest, ...], attempt_id: AttemptId) -> AttemptManifest:
        try:
            return next(attempt for attempt in attempts if attempt.attempt_id == attempt_id)
        except StopIteration:
            raise StateNotFoundError(f"attempt {attempt_id!r} is not persisted") from None

    def _validate_attempt_location(self, attempt: AttemptManifest) -> None:
        if not isinstance(attempt, AttemptManifest) or attempt.run_id != self._run_id:
            raise StateConflictError("attempt identity does not match the state writer")

    def _validate_readiness_location(self, readiness: AttemptReadiness) -> None:
        if not isinstance(readiness, AttemptReadiness) or readiness.run_id != self._run_id:
            raise StateConflictError("readiness identity does not match the state writer")

    @staticmethod
    def _validate_shard_id(shard_id: ShardId) -> ShardId:
        try:
            return _SHARD_ID_ADAPTER.validate_python(shard_id, strict=True)
        except ValidationError as error:
            raise SlurmStateError("invalid shard identity") from error

    @staticmethod
    def _validate_attempt_id(attempt_id: AttemptId) -> AttemptId:
        try:
            return _ATTEMPT_ID_ADAPTER.validate_python(attempt_id, strict=True)
        except ValidationError as error:
            raise SlurmStateError("invalid attempt identity") from error

    def _read_record(
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
            record = record_type.model_validate_json(content)
            if record.serialize_json() != content:
                raise ValueError("record is not deterministically serialized")
            return record
        except (RecursionError, UnicodeError, ValueError, ValidationError) as error:
            raise StateCorruptionError(f"persisted state record {display_path.name!r} is invalid") from error

    def _shard_path(self, shard_id: str) -> Path:
        return self._run_root / _SHARDS_DIRECTORY_NAME / shard_id

    def _attempt_path(self, shard_id: str, attempt_id: str) -> Path:
        return self._shard_path(shard_id) / _ATTEMPTS_DIRECTORY_NAME / attempt_id

    def _readiness_path(self, shard_id: str, attempt_id: str) -> Path:
        return self._attempt_path(shard_id, attempt_id) / _READINESS_FILENAME
