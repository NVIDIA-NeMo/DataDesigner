# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Descriptor-bound persistence for collection plans and lifecycle."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from data_designer.slurm.contracts import ArtifactReference, Identifier
from data_designer.slurm.filesystem import get_file_facts
from data_designer.slurm.state.collection_records import CollectedOutputFile, CollectionResult, CollectionStatus
from data_designer.slurm.state.filesystem import (
    acquire_file_lock,
    ensure_private_child_directory,
    is_state_temporary_name,
    open_verified_child_directory,
    open_verified_directory,
    open_verified_regular_file,
    publish_immutable_text,
    replace_text,
)
from data_designer.slurm.state.outputs import CollectionPlan
from data_designer.slurm.state.storage import StateStorage

_COLLECTIONS_DIRECTORY = "collections"
_COLLECTION_LOCK = "collection.lock"
_PLAN_FILENAME = "plan.json"
_STATUS_FILENAME = "status.json"
_RESULT_FILENAME = "collection-result.json"
_COLLECTION_PATTERN = re.compile(r"^collection-[0-9]{4,}$")
_MAXIMUM_RECORD_SIZE = 16 * 1024 * 1024


class CollectionStorage:
    """Persist collection state without adding collection concerns to run storage."""

    def __init__(self, state_storage: StateStorage) -> None:
        self._state = state_storage
        self.collections_root = state_storage.run_root / _COLLECTIONS_DIRECTORY

    @contextmanager
    def acquire_lock(self) -> Iterator[None]:
        """Serialize collection preparation, refresh, and worker publication."""
        with self._state.open_run_directory() as run_descriptor:
            with acquire_file_lock(
                run_descriptor,
                _COLLECTION_LOCK,
                self._state.run_root / _COLLECTION_LOCK,
            ):
                yield

    def get_next_collection_id(self) -> Identifier:
        """Return the next monotonic identity after validating existing directories."""
        names = self.list_collection_ids()
        return f"collection-{len(names) + 1:04d}"

    def discard_incomplete_tail(self) -> None:
        """Discard one trailing collection journal that predates submission."""
        try:
            with self._open_collections_directory() as collections_descriptor:
                collection_ids = _validated_collection_ids(tuple(os.listdir(collections_descriptor)))
                if not collection_ids:
                    return
                collection_id = collection_ids[-1]
                with open_verified_child_directory(
                    collections_descriptor,
                    collection_id,
                    self.get_collection_root(collection_id),
                ) as collection_descriptor:
                    if _record_exists(collection_descriptor, _STATUS_FILENAME):
                        return
                    _discard_prepared_files(
                        collection_descriptor,
                        self.get_collection_root(collection_id),
                    )
                os.rmdir(collection_id, dir_fd=collections_descriptor)
                os.fsync(collections_descriptor)
        except FileNotFoundError:
            return

    def list_collection_ids(self) -> tuple[Identifier, ...]:
        """List only a complete monotonic set of managed collection directories."""
        try:
            with self._open_collections_directory() as descriptor:
                names = tuple(os.listdir(descriptor))
        except FileNotFoundError:
            return ()
        return _validated_collection_ids(names)

    def ensure_collection(self, collection_id: Identifier) -> None:
        """Create one private collection state directory."""
        with self._state.open_run_directory() as run_descriptor:
            ensure_private_child_directory(run_descriptor, _COLLECTIONS_DIRECTORY, self.collections_root)
            with open_verified_child_directory(
                run_descriptor,
                _COLLECTIONS_DIRECTORY,
                self.collections_root,
            ) as collections_descriptor:
                ensure_private_child_directory(
                    collections_descriptor,
                    collection_id,
                    self.get_collection_root(collection_id),
                )

    def get_collection_root(self, collection_id: Identifier) -> Path:
        return self.collections_root / collection_id

    def get_plan_path(self, collection_id: Identifier) -> Path:
        return self.get_collection_root(collection_id) / _PLAN_FILENAME

    def get_status_path(self, collection_id: Identifier) -> Path:
        return self.get_collection_root(collection_id) / _STATUS_FILENAME

    def get_result_path(self, plan: CollectionPlan) -> Path:
        return Path(plan.host_destination) / _RESULT_FILENAME

    def get_result_reference(self, plan: CollectionPlan, result: CollectionResult) -> ArtifactReference:
        """Return the canonical host-view reference for a published result."""
        return ArtifactReference(path=self.get_result_path(plan).as_posix(), sha256=result.compute_sha256())

    def publish_plan(self, plan: CollectionPlan) -> None:
        self._require_run_id(plan.run_id)
        with self._open_collection_directory(plan.collection_id) as descriptor:
            publish_immutable_text(
                descriptor,
                _PLAN_FILENAME,
                plan.serialize_json(),
                self.get_plan_path(plan.collection_id),
                maximum_size=_MAXIMUM_RECORD_SIZE,
            )

    def read_plan(self, collection_id: Identifier) -> CollectionPlan:
        with self._open_collection_directory(collection_id) as descriptor:
            plan = self._state.read_record(
                descriptor,
                _PLAN_FILENAME,
                self.get_plan_path(collection_id),
                CollectionPlan,
            )
        if plan.collection_id != collection_id or plan.run_id != self._state.run_id:
            raise OSError("collection plan identity does not match its persisted location")
        return plan

    def get_plan_reference(self, plan: CollectionPlan) -> ArtifactReference:
        """Return the canonical persisted reference for an immutable collection plan."""
        return ArtifactReference(path=self.get_plan_path(plan.collection_id).as_posix(), sha256=plan.compute_sha256())

    def publish_status(self, status: CollectionStatus) -> None:
        self._require_run_id(status.run_id)
        with self._open_collection_directory(status.collection_id) as descriptor:
            publish_immutable_text(
                descriptor,
                _STATUS_FILENAME,
                status.serialize_json(),
                self.get_status_path(status.collection_id),
                maximum_size=_MAXIMUM_RECORD_SIZE,
            )

    def replace_status(self, status: CollectionStatus) -> None:
        self._require_run_id(status.run_id)
        with self._open_collection_directory(status.collection_id) as descriptor:
            replace_text(
                descriptor,
                _STATUS_FILENAME,
                status.serialize_json(),
                self.get_status_path(status.collection_id),
                maximum_size=_MAXIMUM_RECORD_SIZE,
            )

    def read_status(self, collection_id: Identifier) -> CollectionStatus:
        with self._open_collection_directory(collection_id) as descriptor:
            status = self._state.read_record(
                descriptor,
                _STATUS_FILENAME,
                self.get_status_path(collection_id),
                CollectionStatus,
            )
        if status.collection_id != collection_id or status.run_id != self._state.run_id:
            raise OSError("collection status identity does not match its persisted location")
        return status

    def read_result(self, plan: CollectionPlan) -> CollectionResult:
        return self.read_result_from(plan, Path(plan.host_destination))

    def read_result_from(self, plan: CollectionPlan, destination: Path) -> CollectionResult:
        """Read a result through an explicitly selected host or container view."""
        if destination.as_posix() not in {plan.host_destination, plan.container_destination}:
            raise OSError("collection result destination does not match its immutable plan")
        with open_verified_directory(destination, require_private=True) as descriptor:
            return self._state.read_record(
                descriptor,
                _RESULT_FILENAME,
                destination / _RESULT_FILENAME,
                CollectionResult,
            )

    def verify_result_files(
        self,
        plan: CollectionPlan,
        result: CollectionResult,
        destination: Path,
        *,
        verify_digests: bool = True,
    ) -> None:
        """Verify exact inventory with bounded metadata or full output digests."""
        if destination.as_posix() not in {plan.host_destination, plan.container_destination}:
            raise OSError("collection result destination does not match its immutable plan")
        expected_names = tuple(output.relative_path for output in result.files) + (_RESULT_FILENAME,)
        if any("/" in name for name in expected_names):
            raise OSError("collection result inventory must contain only direct child files")
        with open_verified_directory(destination, require_private=True) as descriptor:
            if set(os.listdir(descriptor)) != set(expected_names):
                raise OSError("published collection inventory does not match its result manifest")
            for output in result.files:
                if verify_digests:
                    with open_verified_regular_file(
                        descriptor,
                        output.relative_path,
                        destination / output.relative_path,
                        expected_size=output.byte_size,
                        expected_sha256=output.sha256,
                        require_private=False,
                    ):
                        pass
                else:
                    _verify_output_metadata(descriptor, output, destination)
            result_bytes = result.serialize_json().encode("utf-8")
            with open_verified_regular_file(
                descriptor,
                _RESULT_FILENAME,
                destination / _RESULT_FILENAME,
                expected_size=len(result_bytes),
                expected_sha256=result.compute_sha256(),
            ):
                pass

    @contextmanager
    def _open_collections_directory(self) -> Iterator[int]:
        with self._state.open_run_directory() as run_descriptor:
            with open_verified_child_directory(
                run_descriptor,
                _COLLECTIONS_DIRECTORY,
                self.collections_root,
            ) as collections_descriptor:
                yield collections_descriptor

    @contextmanager
    def _open_collection_directory(self, collection_id: Identifier) -> Iterator[int]:
        with self._open_collections_directory() as collections_descriptor:
            with open_verified_child_directory(
                collections_descriptor,
                collection_id,
                self.get_collection_root(collection_id),
            ) as descriptor:
                yield descriptor

    def _require_run_id(self, run_id: Identifier) -> None:
        if run_id != self._state.run_id:
            raise OSError("collection record run identity does not match storage")


def _verify_output_metadata(
    directory_descriptor: int,
    output: CollectedOutputFile,
    destination: Path,
) -> None:
    before = os.stat(output.relative_path, dir_fd=directory_descriptor, follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_mode & 0o022
        or before.st_size != output.byte_size
        or before.st_mtime_ns != output.modified_at_ns
        or before.st_ctime_ns != output.changed_at_ns
    ):
        raise OSError(f"collected output {destination / output.relative_path} is not a safe regular file")
    descriptor = os.open(
        output.relative_path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        dir_fd=directory_descriptor,
    )
    try:
        after = os.fstat(descriptor)
        rebound = os.stat(output.relative_path, dir_fd=directory_descriptor, follow_symlinks=False)
    finally:
        os.close(descriptor)
    if get_file_facts(before) != get_file_facts(after) or get_file_facts(after) != get_file_facts(rebound):
        raise OSError(f"collected output {destination / output.relative_path} changed during validation")


def _validated_collection_ids(names: tuple[str, ...]) -> tuple[Identifier, ...]:
    if any(_COLLECTION_PATTERN.fullmatch(name) is None for name in names):
        raise OSError("collection state contains an unowned directory")
    ordered = tuple(sorted(names, key=lambda name: int(name.rsplit("-", maxsplit=1)[1])))
    expected = tuple(f"collection-{index:04d}" for index in range(1, len(ordered) + 1))
    if ordered != expected:
        raise OSError("collection state identities are not a complete monotonic sequence")
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
        raise OSError(f"incomplete collection journal {display_path} contains an unowned entry")
    for name in names:
        status = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if not stat.S_ISREG(status.st_mode) or status.st_mode & 0o077:
            raise OSError(f"incomplete collection journal entry {display_path / name} is unsafe")
        os.unlink(name, dir_fd=directory_descriptor)
    os.fsync(directory_descriptor)


__all__ = ["CollectionStorage"]
