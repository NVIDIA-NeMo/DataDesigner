# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Descriptor-bound filesystem operations for persisted Slurm run state."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from data_designer.slurm.filesystem import (
    PRIVATE_DIRECTORY_MODE,
    acquire_restrictive_file_lock,
    create_restrictive_temporary_file,
    get_file_facts,
    is_managed_temporary_name,
)
from data_designer.slurm.filesystem import (
    open_verified_child_directory as open_shared_child_directory,
)
from data_designer.slurm.filesystem import (
    open_verified_directory as open_shared_directory,
)

_READ_RETRY_ATTEMPTS = 10
_TEMPORARY_PREFIX = ".state."
_TEMPORARY_SUFFIX = ".tmp"


@contextmanager
def acquire_file_lock(directory_descriptor: int, name: str, display_path: Path) -> Iterator[None]:
    """Acquire an exclusive package-owned lock relative to a verified directory."""
    with acquire_restrictive_file_lock(
        directory_descriptor,
        name,
        display_path,
        resource_name="state",
    ):
        yield


def ensure_private_child_directory(parent_descriptor: int, name: str, display_path: Path) -> None:
    """Create or restrict one child of an already verified directory."""
    created = False
    try:
        os.mkdir(name, PRIVATE_DIRECTORY_MODE, dir_fd=parent_descriptor)
        created = True
    except FileExistsError:
        pass
    with open_verified_child_directory(
        parent_descriptor,
        name,
        display_path,
        require_private=False,
    ) as descriptor:
        os.fchmod(descriptor, PRIVATE_DIRECTORY_MODE)
        os.fsync(descriptor)
    if created:
        os.fsync(parent_descriptor)


@contextmanager
def open_verified_directory(path: Path, *, require_private: bool = False) -> Iterator[int]:
    """Open one real directory and reject path replacement during the open."""
    with open_shared_directory(path, resource_name="state", require_private=require_private) as descriptor:
        yield descriptor


@contextmanager
def open_verified_child_directory(
    parent_descriptor: int,
    name: str,
    display_path: Path,
    *,
    require_private: bool = True,
) -> Iterator[int]:
    """Open one real child directory relative to a verified parent."""
    with open_shared_child_directory(
        parent_descriptor,
        name,
        display_path,
        resource_name="state",
        require_private=require_private,
    ) as descriptor:
        yield descriptor


def read_regular_text(
    directory_descriptor: int,
    name: str,
    display_path: Path,
    *,
    maximum_size: int,
) -> str:
    """Read one restrictive regular UTF-8 file without following a symlink."""
    for _ in range(_READ_RETRY_ATTEMPTS):
        descriptor: int | None = None
        try:
            before_open = _repair_interrupted_publication(
                directory_descriptor,
                name,
                display_path,
                os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False),
            )
            if not _is_restrictive_regular(before_open):
                raise OSError(f"state record {display_path} is not a restrictive regular file")
            if before_open.st_size > maximum_size:
                raise OSError(f"state record {display_path} exceeds its size limit")
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
                dir_fd=directory_descriptor,
            )
            after_open = os.fstat(descriptor)
            if not _is_restrictive_regular(after_open):
                raise OSError(f"state record {display_path} is not a restrictive regular file")
            if get_file_facts(before_open) != get_file_facts(after_open):
                continue
            record_file = os.fdopen(descriptor, "r", encoding="utf-8", newline="")
            descriptor = None
            with record_file:
                content = record_file.read(maximum_size + 1)
                if len(content.encode("utf-8")) > maximum_size:
                    raise OSError(f"state record {display_path} exceeds its size limit")
                after_read = os.fstat(record_file.fileno())
            if not _is_restrictive_regular(after_read):
                raise OSError(f"state record {display_path} is not a restrictive regular file")
            if get_file_facts(after_open) != get_file_facts(after_read):
                continue
            return content
        finally:
            if descriptor is not None:
                os.close(descriptor)
    raise OSError(f"state record {display_path} changed too often to read consistently")


def verify_regular_file(
    directory_descriptor: int,
    name: str,
    display_path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> None:
    """Digest-verify one restrictive regular file without following a symlink."""
    descriptor: int | None = None
    try:
        before_open = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if not _is_restrictive_regular(before_open) or before_open.st_size != expected_size:
            raise OSError(f"candidate file {display_path} is not the expected restrictive regular file")
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_descriptor,
        )
        after_open = os.fstat(descriptor)
        if not _is_restrictive_regular(after_open) or get_file_facts(before_open) != get_file_facts(after_open):
            raise OSError(f"candidate file {display_path} changed while it was being opened")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after_read = os.fstat(descriptor)
        if not _is_restrictive_regular(after_read) or get_file_facts(after_open) != get_file_facts(after_read):
            raise OSError(f"candidate file {display_path} changed while it was being read")
        if digest.hexdigest() != expected_sha256:
            raise OSError(f"candidate file {display_path} digest does not match its manifest")
    finally:
        if descriptor is not None:
            os.close(descriptor)


def publish_immutable_text(
    directory_descriptor: int,
    name: str,
    content: str,
    display_path: Path,
    *,
    maximum_size: int,
) -> bool:
    """Publish one immutable record without replacing an existing name."""
    temporary_name = _write_temporary_text(directory_descriptor, content, maximum_size=maximum_size)
    try:
        try:
            os.link(
                temporary_name,
                name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            existing = read_regular_text(
                directory_descriptor,
                name,
                display_path,
                maximum_size=maximum_size,
            )
            if existing == content:
                sync_directory(directory_descriptor)
                return False
            raise FileExistsError(f"state record {display_path} already contains different bytes") from None
        try:
            os.unlink(temporary_name, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass
        temporary_name = None
        sync_directory(directory_descriptor)
        return True
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
                sync_directory(directory_descriptor)
            except OSError:
                pass


def replace_text(
    directory_descriptor: int,
    name: str,
    content: str,
    display_path: Path,
    *,
    maximum_size: int,
) -> None:
    """Atomically replace one existing restrictive regular record."""
    status = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    if not stat.S_ISREG(status.st_mode) or status.st_mode & 0o077:
        raise OSError(f"state record {display_path} is not a restrictive regular file")
    temporary_name = _write_temporary_text(directory_descriptor, content, maximum_size=maximum_size)
    try:
        os.replace(
            temporary_name,
            name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        temporary_name = None
        sync_directory(directory_descriptor)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
            except OSError:
                pass


def sync_directory(directory_descriptor: int) -> None:
    """Flush prior changes to an already verified directory."""
    os.fsync(directory_descriptor)


def create_state_temporary_file(directory_descriptor: int) -> tuple[int, str]:
    """Create one restrictive file using the persisted-state temporary convention."""
    return create_restrictive_temporary_file(
        directory_descriptor,
        prefix=_TEMPORARY_PREFIX,
        suffix=_TEMPORARY_SUFFIX,
    )


def is_state_temporary_name(name: str) -> bool:
    """Return whether a name belongs to persisted-state temporary publication."""
    return is_managed_temporary_name(
        name,
        prefix=_TEMPORARY_PREFIX,
        suffix=_TEMPORARY_SUFFIX,
    )


def _write_temporary_text(directory_descriptor: int, content: str, *, maximum_size: int) -> str:
    encoded = content.encode("utf-8")
    if len(encoded) > maximum_size:
        raise OSError("state record exceeds its size limit")
    descriptor: int | None = None
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = create_state_temporary_file(directory_descriptor)
        output = os.fdopen(descriptor, "wb")
        descriptor = None
        with output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        return temporary_name
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
            except OSError:
                pass
        raise


def _repair_interrupted_publication(
    directory_descriptor: int,
    name: str,
    display_path: Path,
    status: os.stat_result,
) -> os.stat_result:
    if _is_restrictive_regular(status):
        return status
    if not stat.S_ISREG(status.st_mode) or status.st_mode & 0o077 or status.st_nlink != 2:
        return status

    temporary_names: list[str] = []
    for candidate_name in os.listdir(directory_descriptor):
        if not is_state_temporary_name(candidate_name):
            continue
        try:
            candidate_status = os.stat(candidate_name, dir_fd=directory_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if (candidate_status.st_dev, candidate_status.st_ino) == (status.st_dev, status.st_ino):
            temporary_names.append(candidate_name)
    if len(temporary_names) != 1:
        return os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)

    try:
        os.unlink(temporary_names[0], dir_fd=directory_descriptor)
        os.fsync(directory_descriptor)
    except FileNotFoundError:
        pass
    repaired = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    if not _is_restrictive_regular(repaired):
        raise OSError(f"state record {display_path} changed while recovering interrupted publication")
    return repaired


def _is_restrictive_regular(status: os.stat_result) -> bool:
    return stat.S_ISREG(status.st_mode) and status.st_nlink == 1 and not status.st_mode & 0o077
