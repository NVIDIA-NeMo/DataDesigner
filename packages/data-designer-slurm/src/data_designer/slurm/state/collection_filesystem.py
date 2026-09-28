# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Same-parent staging and atomic no-overwrite collection publication."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import stat
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from data_designer.slurm.contracts import is_path_below
from data_designer.slurm.filesystem import PRIVATE_DIRECTORY_MODE, open_verified_directory
from data_designer.slurm.state.errors import StateConflictError
from data_designer.slurm.state.filesystem import open_verified_child_directory, open_verified_regular_file
from data_designer.slurm.state.outputs import CollectionPlan

_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x00000004
_UNSUPPORTED_NO_REPLACE_ERRORS = frozenset(
    {
        errno.EINVAL,
        errno.ENOSYS,
        errno.ENOTSUP,
        errno.EOPNOTSUPP,
    }
)


@dataclass(frozen=True, slots=True)
class StagedFile:
    """Expected immutable bytes in a collection staging directory."""

    name: str
    sha256: str
    byte_size: int


@dataclass(slots=True)
class StagedCollection:
    """Private sibling directory that becomes visible only after publication."""

    path: Path
    destination: Path
    _parent_descriptor: int
    _parent_identity: tuple[int, int]
    _stage_identity: tuple[int, int]
    _published: bool = False

    def publish(self, expected_files: tuple[StagedFile, ...]) -> None:
        """Atomically rename the complete stage while refusing any collision."""
        if self._published:
            return
        self._rebind()
        _verify_stage_files(self, expected_files)
        _rename_without_overwrite(
            self._parent_descriptor,
            self.path.name,
            self._parent_descriptor,
            self.destination.name,
        )
        self._published = True
        try:
            self._rebind_published_destination()
            os.fsync(self._parent_descriptor)
            self._rebind_published_destination()
        except OSError:
            self._restore_stage()
            raise

    def _rebind(self) -> None:
        self._rebind_parent()
        stage = os.stat(self.path.name, dir_fd=self._parent_descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(stage.st_mode) or _identity(stage) != self._stage_identity:
            raise OSError(f"collection stage {self.path} changed")

    def _rebind_parent(self) -> None:
        opened = os.fstat(self._parent_descriptor)
        current = self.destination.parent.lstat()
        if (
            not stat.S_ISDIR(current.st_mode)
            or _identity(opened) != self._parent_identity
            or _identity(current) != self._parent_identity
        ):
            raise OSError(f"collection destination parent {self.destination.parent} changed")

    def _rebind_published_destination(self) -> None:
        self._rebind_parent()
        opened_view = os.stat(self.destination.name, dir_fd=self._parent_descriptor, follow_symlinks=False)
        path_view = self.destination.lstat()
        if (
            not stat.S_ISDIR(opened_view.st_mode)
            or not stat.S_ISDIR(path_view.st_mode)
            or _identity(opened_view) != self._stage_identity
            or _identity(path_view) != self._stage_identity
        ):
            raise OSError(f"published collection destination {self.destination} changed")

    def _restore_stage(self) -> None:
        published = os.stat(self.destination.name, dir_fd=self._parent_descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(published.st_mode) or _identity(published) != self._stage_identity:
            raise OSError(f"published collection destination {self.destination} changed before rollback")
        _rename_without_overwrite(
            self._parent_descriptor,
            self.destination.name,
            self._parent_descriptor,
            self.path.name,
        )
        self._published = False
        os.fsync(self._parent_descriptor)


def derive_collection_staging_directory(plan: CollectionPlan) -> str:
    """Derive one collision-resistant, persisted stage identity from the plan."""
    return f".dd-collection-{plan.compute_sha256()[:32]}.tmp"


def remove_collection_stage(destination: Path, staging_directory: str, authorized_root: Path) -> None:
    """Remove only the exact persisted stage for one terminal collection."""
    try:
        with _open_authorized_parent(destination, authorized_root) as parent_descriptor:
            _require_restrictive_parent(parent_descriptor)
            _recover_private_reservation(parent_descriptor, staging_directory, destination.name)
            _remove_existing_stage(parent_descriptor, destination.parent, staging_directory)
    except _MissingDestinationParent:
        return


def prepare_collection_destination(destination: Path, authorized_root: Path) -> None:
    """Create and validate the destination parent without creating the dataset."""
    with _open_authorized_parent(destination, authorized_root, create_missing=True) as parent_descriptor:
        _require_restrictive_parent(parent_descriptor)
        _require_absent(parent_descriptor, destination.name, destination)


@contextmanager
def stage_collection(
    destination: Path,
    staging_directory: str,
    authorized_root: Path,
) -> Iterator[StagedCollection]:
    """Yield a private sibling stage and remove it unless atomically published."""
    if destination.parent == destination:
        raise StateConflictError("collection destination cannot be the filesystem root")
    with _open_authorized_parent(destination, authorized_root) as parent_descriptor:
        parent_status = _require_restrictive_parent(parent_descriptor)
        _recover_private_reservation(parent_descriptor, staging_directory, destination.name)
        _require_absent(parent_descriptor, destination.name, destination)
        _remove_existing_stage(parent_descriptor, destination.parent, staging_directory)
        stage_name = _create_stage_directory(parent_descriptor, staging_directory)
        stage_path = destination.parent / stage_name
        staged = StagedCollection(
            path=stage_path,
            destination=destination,
            _parent_descriptor=parent_descriptor,
            _parent_identity=_identity(parent_status),
            _stage_identity=_identity(os.stat(stage_name, dir_fd=parent_descriptor, follow_symlinks=False)),
        )
        try:
            yield staged
        finally:
            if not staged._published:
                _remove_stage(staged)


class _MissingDestinationParent(FileNotFoundError):
    """A child beneath an existing authorized root has not been created yet."""


@contextmanager
def _open_authorized_parent(
    destination: Path,
    authorized_root: Path,
    *,
    create_missing: bool = False,
) -> Iterator[int]:
    destination_text = destination.as_posix()
    root_text = authorized_root.as_posix()
    if destination_text == root_text or not is_path_below(destination_text, root_text):
        raise StateConflictError("collection destination must be below its authorized mount root")
    relative_parent = PurePosixPath(destination.parent.relative_to(authorized_root)).parts
    with ExitStack() as resources:
        descriptor = resources.enter_context(open_verified_directory(authorized_root, resource_name="collection"))
        current_path = authorized_root
        for part in relative_parent:
            current_path /= part
            if create_missing:
                _ensure_private_child_directory(descriptor, part)
            try:
                descriptor = resources.enter_context(
                    open_verified_child_directory(
                        descriptor,
                        part,
                        current_path,
                        require_private=False,
                    )
                )
            except FileNotFoundError as error:
                raise _MissingDestinationParent(current_path) from error
        yield descriptor


def _ensure_private_child_directory(parent_descriptor: int, name: str) -> None:
    try:
        os.mkdir(name, PRIVATE_DIRECTORY_MODE, dir_fd=parent_descriptor)
    except FileExistsError:
        return
    os.fsync(parent_descriptor)


def _require_restrictive_parent(parent_descriptor: int) -> os.stat_result:
    status = os.fstat(parent_descriptor)
    if status.st_mode & 0o022:
        raise StateConflictError("collection destination parent must not be group- or world-writable")
    return status


def _require_absent(parent_descriptor: int, name: str, display_path: Path) -> None:
    try:
        os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise StateConflictError(f"collection destination {display_path} already exists")


def _create_stage_directory(parent_descriptor: int, name: str) -> str:
    os.mkdir(name, PRIVATE_DIRECTORY_MODE, dir_fd=parent_descriptor)
    os.fsync(parent_descriptor)
    return name


def _remove_existing_stage(parent_descriptor: int, parent_path: Path, name: str) -> None:
    try:
        status = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(status.st_mode) or status.st_mode & 0o077:
        raise OSError(f"persisted collection stage {parent_path / name} is not a private directory")
    _delete_stage_directory(parent_descriptor, name, parent_path / name, _identity(status))


def _remove_stage(staged: StagedCollection) -> None:
    try:
        current = os.stat(staged.path.name, dir_fd=staged._parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(current.st_mode) or _identity(current) != staged._stage_identity:
        raise OSError(f"collection stage {staged.path} changed before cleanup")
    _delete_stage_directory(
        staged._parent_descriptor,
        staged.path.name,
        staged.path,
        staged._stage_identity,
    )


def _delete_stage_directory(
    parent_descriptor: int,
    name: str,
    display_path: Path,
    expected_identity: tuple[int, int],
) -> None:
    with open_verified_child_directory(parent_descriptor, name, display_path) as descriptor:
        if _identity(os.fstat(descriptor)) != expected_identity:
            raise OSError(f"collection stage {display_path} changed before cleanup")
        _clear_directory(descriptor, display_path)
    current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if _identity(current) != expected_identity:
        raise OSError(f"collection stage {display_path} changed during cleanup")
    os.rmdir(name, dir_fd=parent_descriptor)
    os.fsync(parent_descriptor)


def _clear_directory(directory_descriptor: int, display_path: Path) -> None:
    for name in os.listdir(directory_descriptor):
        status = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        child_path = display_path / name
        if stat.S_ISDIR(status.st_mode):
            with open_verified_child_directory(directory_descriptor, name, child_path) as child_descriptor:
                if _identity(os.fstat(child_descriptor)) != _identity(status):
                    raise OSError(f"collection staging directory {child_path} changed before cleanup")
                _clear_directory(child_descriptor, child_path)
            current = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            if _identity(current) != _identity(status):
                raise OSError(f"collection staging directory {child_path} changed during cleanup")
            os.rmdir(name, dir_fd=directory_descriptor)
        else:
            os.unlink(name, dir_fd=directory_descriptor)
    os.fsync(directory_descriptor)


def _verify_stage_files(staged: StagedCollection, expected_files: tuple[StagedFile, ...]) -> None:
    expected_names = tuple(file.name for file in expected_files)
    if len(expected_names) != len(set(expected_names)):
        raise OSError("collection stage expectation contains duplicate paths")
    with open_verified_child_directory(
        staged._parent_descriptor,
        staged.path.name,
        staged.path,
    ) as stage_descriptor:
        if set(os.listdir(stage_descriptor)) != set(expected_names):
            raise OSError("collection stage inventory changed before publication")
        for expected in expected_files:
            with open_verified_regular_file(
                stage_descriptor,
                expected.name,
                staged.path / expected.name,
                expected_size=expected.byte_size,
                expected_sha256=expected.sha256,
                require_private=False,
            ):
                pass


def _rename_without_overwrite(
    source_directory: int,
    source_name: str,
    destination_directory: int,
    destination_name: str,
) -> None:
    try:
        _atomic_rename_without_overwrite(
            source_directory,
            source_name,
            destination_directory,
            destination_name,
        )
    except OSError as error:
        if error.errno not in _UNSUPPORTED_NO_REPLACE_ERRORS:
            raise
        _rename_over_private_reservation(
            source_directory,
            source_name,
            destination_directory,
            destination_name,
        )


def _atomic_rename_without_overwrite(
    source_directory: int,
    source_name: str,
    destination_directory: int,
    destination_name: str,
) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
    if sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        result = library.renameat2(
            source_directory,
            ctypes.c_char_p(source),
            destination_directory,
            ctypes.c_char_p(destination),
            _RENAME_NOREPLACE,
        )
    elif sys.platform == "darwin" and hasattr(library, "renameatx_np"):
        result = library.renameatx_np(
            source_directory,
            ctypes.c_char_p(source),
            destination_directory,
            ctypes.c_char_p(destination),
            _RENAME_EXCL,
        )
    else:
        raise OSError(errno.ENOTSUP, "atomic no-overwrite directory rename is unavailable")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise StateConflictError(f"collection destination {destination_name!r} already exists")
    raise OSError(error_number, os.strerror(error_number), destination_name)


def _rename_over_private_reservation(
    source_directory: int,
    source_name: str,
    destination_directory: int,
    destination_name: str,
) -> None:
    """Publish through an exclusive empty directory on filesystems without no-replace rename."""
    marker_name = _reservation_marker_name(source_name, destination_name)
    reservation: os.stat_result | None = None
    remove_marker = False
    try:
        try:
            os.mkdir(destination_name, PRIVATE_DIRECTORY_MODE, dir_fd=destination_directory)
        except FileExistsError:
            raise StateConflictError(f"collection destination {destination_name!r} already exists") from None
        reservation = os.stat(destination_name, dir_fd=destination_directory, follow_symlinks=False)
        if not stat.S_ISDIR(reservation.st_mode) or reservation.st_mode & 0o077:
            raise OSError(f"collection destination reservation {destination_name!r} is not private")
        os.fsync(destination_directory)
        # An unmarked directory is ambiguous after a crash and must never be removed by recovery.
        marker_bytes = _reservation_marker_bytes(source_name, destination_name, _reservation_identity(reservation))
        _create_reservation_marker(destination_directory, marker_name, marker_bytes)
        remove_marker = True
        current = os.stat(destination_name, dir_fd=destination_directory, follow_symlinks=False)
        if _identity(current) != _identity(reservation):
            raise OSError(f"collection destination reservation {destination_name!r} changed")
        os.rename(
            source_name,
            destination_name,
            src_dir_fd=source_directory,
            dst_dir_fd=destination_directory,
        )
    except BaseException:
        if reservation is not None:
            _remove_unchanged_reservation(destination_directory, destination_name, reservation)
        raise
    finally:
        if remove_marker:
            os.unlink(marker_name, dir_fd=destination_directory)
            os.fsync(destination_directory)


def _reservation_marker_name(source_name: str, destination_name: str) -> str:
    identity = hashlib.sha256(f"{source_name}\n{destination_name}\n".encode("utf-8")).hexdigest()[:32]
    return f".dd-collection-reservation-{identity}"


def _reservation_marker_bytes(source_name: str, destination_name: str, identity: tuple[int, int, int]) -> bytes:
    return f"{source_name}\n{destination_name}\n{identity[0]}\n{identity[1]}\n{identity[2]}\n".encode("utf-8")


def _create_reservation_marker(directory: int, name: str, content: bytes) -> None:
    descriptor = os.open(
        name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=directory
    )
    try:
        if os.write(descriptor, content) != len(content):
            raise OSError("collection reservation marker write was incomplete")
        os.fsync(descriptor)
    except BaseException:
        os.unlink(name, dir_fd=directory)
        raise
    finally:
        os.close(descriptor)
    os.fsync(directory)


def _recover_private_reservation(directory: int, source_name: str, destination_name: str) -> None:
    marker_name = _reservation_marker_name(source_name, destination_name)
    try:
        marker = os.stat(marker_name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return
    if (
        not stat.S_ISREG(marker.st_mode)
        or marker.st_mode & 0o077
        or marker.st_nlink != 1
        or marker.st_uid != os.getuid()
    ):
        raise OSError(f"collection reservation marker {marker_name!r} is unsafe")
    descriptor = os.open(marker_name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory)
    try:
        if _identity(os.fstat(descriptor)) != _identity(marker):
            raise OSError(f"collection reservation marker {marker_name!r} changed")
        marker_content = os.read(descriptor, 1024)
    finally:
        os.close(descriptor)
    marker_prefix = f"{source_name}\n{destination_name}\n".encode("utf-8")
    if not marker_content.startswith(marker_prefix):
        raise OSError(f"collection reservation marker {marker_name!r} is invalid")
    identity_parts = marker_content[len(marker_prefix) :].split(b"\n")
    if len(identity_parts) != 4 or identity_parts[3] or not all(part.isdigit() for part in identity_parts[:3]):
        raise OSError(f"collection reservation marker {marker_name!r} is invalid")
    marker_identity = int(identity_parts[0]), int(identity_parts[1]), int(identity_parts[2])
    try:
        reservation = os.stat(destination_name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        reservation = None
    if (
        reservation is not None
        and stat.S_ISDIR(reservation.st_mode)
        and not reservation.st_mode & 0o077
        and _reservation_identity(reservation) == marker_identity
    ):
        opened = os.open(
            destination_name, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory
        )
        try:
            if _identity(os.fstat(opened)) == _identity(reservation) and not os.listdir(opened):
                _remove_unchanged_reservation(directory, destination_name, reservation)
        finally:
            os.close(opened)
    os.unlink(marker_name, dir_fd=directory)
    os.fsync(directory)


def _remove_unchanged_reservation(
    destination_directory: int,
    destination_name: str,
    reservation: os.stat_result,
) -> None:
    try:
        current = os.stat(destination_name, dir_fd=destination_directory, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(current.st_mode) or _identity(current) != _identity(reservation):
        return
    os.rmdir(destination_name, dir_fd=destination_directory)
    os.fsync(destination_directory)


def _identity(status: os.stat_result) -> tuple[int, int]:
    return status.st_dev, status.st_ino


def _reservation_identity(status: os.stat_result) -> tuple[int, int, int]:
    return status.st_dev, status.st_ino, status.st_ctime_ns


__all__ = [
    "StagedCollection",
    "StagedFile",
    "derive_collection_staging_directory",
    "prepare_collection_destination",
    "remove_collection_stage",
    "stage_collection",
]
