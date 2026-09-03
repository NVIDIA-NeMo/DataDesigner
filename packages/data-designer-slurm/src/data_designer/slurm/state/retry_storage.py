# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Descriptor-bound persistence for retry plans and submission progress."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from data_designer.slurm.contracts import ArtifactReference, Identifier
from data_designer.slurm.state.filesystem import (
    acquire_file_lock,
    ensure_private_child_directory,
    is_state_temporary_name,
    open_verified_child_directory,
    publish_immutable_text,
    replace_text,
)
from data_designer.slurm.state.outputs import RetryPlan
from data_designer.slurm.state.retry_records import RetryStatus
from data_designer.slurm.state.storage import StateStorage

_RETRIES_DIRECTORY = "retries"
_RETRY_LOCK = "retry.lock"
_PLAN_FILENAME = "plan.json"
_STATUS_FILENAME = "status.json"
_RETRY_PATTERN = re.compile(r"^retry-[0-9]{4,}$")
_MAXIMUM_RECORD_SIZE = 16 * 1024 * 1024


class RetryStorage:
    """Persist retry journals separately from run and attempt storage."""

    def __init__(self, state_storage: StateStorage) -> None:
        self._state = state_storage
        self.retries_root = state_storage.run_root / _RETRIES_DIRECTORY

    @contextmanager
    def acquire_lock(self) -> Iterator[None]:
        """Serialize retry selection and submission for one run."""
        with self._state.open_run_directory() as run_descriptor:
            with acquire_file_lock(run_descriptor, _RETRY_LOCK, self._state.run_root / _RETRY_LOCK):
                yield

    def get_next_retry_id(self) -> Identifier:
        """Return the next monotonic retry identity."""
        return f"retry-{len(self.list_retry_ids()) + 1:04d}"

    def discard_incomplete_tail(self) -> None:
        """Discard one trailing journal that cannot have reached submission."""
        try:
            with self._open_retries_directory() as retries_descriptor:
                retry_ids = _validated_retry_ids(tuple(os.listdir(retries_descriptor)))
                if not retry_ids:
                    return
                retry_id = retry_ids[-1]
                with open_verified_child_directory(
                    retries_descriptor,
                    retry_id,
                    self.get_retry_root(retry_id),
                ) as retry_descriptor:
                    if _record_exists(retry_descriptor, _STATUS_FILENAME):
                        return
                    _discard_prepared_files(retry_descriptor, self.get_retry_root(retry_id))
                os.rmdir(retry_id, dir_fd=retries_descriptor)
                os.fsync(retries_descriptor)
        except FileNotFoundError:
            return

    def list_retry_ids(self) -> tuple[Identifier, ...]:
        """List a complete monotonic set of managed retry directories."""
        try:
            with self._open_retries_directory() as descriptor:
                names = tuple(os.listdir(descriptor))
        except FileNotFoundError:
            return ()
        return _validated_retry_ids(names)

    def ensure_retry(self, retry_id: Identifier) -> None:
        """Create one private retry journal directory."""
        with self._state.open_run_directory() as run_descriptor:
            ensure_private_child_directory(run_descriptor, _RETRIES_DIRECTORY, self.retries_root)
            with open_verified_child_directory(run_descriptor, _RETRIES_DIRECTORY, self.retries_root) as descriptor:
                ensure_private_child_directory(descriptor, retry_id, self.get_retry_root(retry_id))

    def get_retry_root(self, retry_id: Identifier) -> Path:
        return self.retries_root / retry_id

    def get_plan_path(self, retry_id: Identifier) -> Path:
        """Return the canonical immutable retry-plan path."""
        return self.get_retry_root(retry_id) / _PLAN_FILENAME

    def get_plan_reference(self, plan: RetryPlan) -> ArtifactReference:
        """Return the exact path and digest bound by retry status."""
        return ArtifactReference(path=self.get_plan_path(plan.retry_id).as_posix(), sha256=plan.compute_sha256())

    def publish_plan(self, plan: RetryPlan) -> None:
        self._require_run_id(plan.run_id)
        with self._open_retry_directory(plan.retry_id) as descriptor:
            publish_immutable_text(
                descriptor,
                _PLAN_FILENAME,
                plan.serialize_json(),
                self.get_plan_path(plan.retry_id),
                maximum_size=_MAXIMUM_RECORD_SIZE,
            )

    def read_plan(self, retry_id: Identifier) -> RetryPlan:
        with self._open_retry_directory(retry_id) as descriptor:
            plan = self._state.read_record(
                descriptor,
                _PLAN_FILENAME,
                self.get_plan_path(retry_id),
                RetryPlan,
            )
        if plan.retry_id != retry_id or plan.run_id != self._state.run_id:
            raise OSError("retry plan identity does not match its persisted location")
        return plan

    def publish_status(self, status: RetryStatus) -> None:
        self._require_run_id(status.run_id)
        with self._open_retry_directory(status.retry_id) as descriptor:
            publish_immutable_text(
                descriptor,
                _STATUS_FILENAME,
                status.serialize_json(),
                self.get_retry_root(status.retry_id) / _STATUS_FILENAME,
                maximum_size=_MAXIMUM_RECORD_SIZE,
            )

    def replace_status(self, status: RetryStatus) -> None:
        self._require_run_id(status.run_id)
        with self._open_retry_directory(status.retry_id) as descriptor:
            replace_text(
                descriptor,
                _STATUS_FILENAME,
                status.serialize_json(),
                self.get_retry_root(status.retry_id) / _STATUS_FILENAME,
                maximum_size=_MAXIMUM_RECORD_SIZE,
            )

    def read_status(self, retry_id: Identifier) -> RetryStatus:
        with self._open_retry_directory(retry_id) as descriptor:
            status = self._state.read_record(
                descriptor,
                _STATUS_FILENAME,
                self.get_retry_root(retry_id) / _STATUS_FILENAME,
                RetryStatus,
            )
        if status.retry_id != retry_id or status.run_id != self._state.run_id:
            raise OSError("retry status identity does not match its persisted location")
        return status

    @contextmanager
    def _open_retries_directory(self) -> Iterator[int]:
        with self._state.open_run_directory() as run_descriptor:
            with open_verified_child_directory(run_descriptor, _RETRIES_DIRECTORY, self.retries_root) as descriptor:
                yield descriptor

    @contextmanager
    def _open_retry_directory(self, retry_id: Identifier) -> Iterator[int]:
        with self._open_retries_directory() as retries_descriptor:
            with open_verified_child_directory(
                retries_descriptor,
                retry_id,
                self.get_retry_root(retry_id),
            ) as descriptor:
                yield descriptor

    def _require_run_id(self, run_id: Identifier) -> None:
        if run_id != self._state.run_id:
            raise OSError("retry record run identity does not match storage")


def _validated_retry_ids(names: tuple[str, ...]) -> tuple[Identifier, ...]:
    if any(_RETRY_PATTERN.fullmatch(name) is None for name in names):
        raise OSError("retry state contains an unowned directory")
    ordered = tuple(sorted(names, key=lambda name: int(name.rsplit("-", maxsplit=1)[1])))
    expected = tuple(f"retry-{index:04d}" for index in range(1, len(ordered) + 1))
    if ordered != expected:
        raise OSError("retry identities are not a complete monotonic sequence")
    return ordered


def _record_exists(directory_descriptor: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _discard_prepared_files(directory_descriptor: int, display_path: Path) -> None:
    names = tuple(os.listdir(directory_descriptor))
    if any(name != _PLAN_FILENAME and not is_state_temporary_name(name) for name in names):
        raise OSError(f"incomplete retry journal {display_path} contains an unowned entry")
    for name in names:
        status = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if not stat.S_ISREG(status.st_mode) or status.st_mode & 0o077:
            raise OSError(f"incomplete retry journal entry {display_path / name} is unsafe")
        os.unlink(name, dir_fd=directory_descriptor)
    os.fsync(directory_descriptor)


__all__ = ["RetryStorage"]
