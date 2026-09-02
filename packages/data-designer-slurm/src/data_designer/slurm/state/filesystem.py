# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Descriptor-bound filesystem operations for persisted Slurm run state."""

from __future__ import annotations

import fcntl
import os
import re
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_DIRECTORY_MODE = 0o700
_FILE_MODE = 0o600
_TEMPORARY_NAME_PATTERN = re.compile(r"^\.state\.[0-9a-f]{16}\.tmp$")


@contextmanager
def acquire_file_lock(directory_descriptor: int, name: str, display_path: Path) -> Iterator[None]:
    """Acquire an exclusive package-owned lock relative to a verified directory."""
    descriptor: int | None = None
    try:
        for _ in range(10):
            try:
                descriptor = os.open(
                    name,
                    os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
                    _FILE_MODE,
                    dir_fd=directory_descriptor,
                )
            except FileNotFoundError:
                continue
            break
        if descriptor is None:
            raise OSError(f"state lock {display_path} could not be opened")
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
            raise OSError(f"state lock {display_path} is not a regular file")
        os.fchmod(descriptor, _FILE_MODE)
        if os.fstat(descriptor).st_mode & 0o077:
            raise OSError(f"state lock {display_path} does not have restrictive permissions")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked_status = os.fstat(descriptor)
        current_status = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            (locked_status.st_dev, locked_status.st_ino) != (current_status.st_dev, current_status.st_ino)
            or not stat.S_ISREG(current_status.st_mode)
            or current_status.st_nlink != 1
        ):
            raise OSError(f"state lock {display_path} changed while it was being acquired")
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise

    try:
        yield
    finally:
        assert descriptor is not None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def ensure_private_child_directory(parent_descriptor: int, name: str, display_path: Path) -> None:
    """Create or restrict one child of an already verified directory."""
    created = False
    try:
        os.mkdir(name, _DIRECTORY_MODE, dir_fd=parent_descriptor)
        created = True
    except FileExistsError:
        pass
    with open_verified_child_directory(
        parent_descriptor,
        name,
        display_path,
        require_private=False,
    ) as descriptor:
        os.fchmod(descriptor, _DIRECTORY_MODE)
        os.fsync(descriptor)
    if created:
        os.fsync(parent_descriptor)


@contextmanager
def open_verified_directory(path: Path, *, require_private: bool = False) -> Iterator[int]:
    """Open one real directory and reject path replacement during the open."""
    descriptor: int | None = None
    try:
        before_open = path.lstat()
        if not stat.S_ISDIR(before_open.st_mode):
            raise OSError(f"state directory {path} is not a directory")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        after_open = os.fstat(descriptor)
        if (before_open.st_dev, before_open.st_ino) != (after_open.st_dev, after_open.st_ino):
            raise OSError(f"state directory {path} changed while it was being opened")
        if not stat.S_ISDIR(after_open.st_mode):
            raise OSError(f"state directory {path} is not a directory")
        if require_private and after_open.st_mode & 0o077:
            raise OSError(f"state directory {path} does not have restrictive permissions")
        yield descriptor
    finally:
        if descriptor is not None:
            os.close(descriptor)


@contextmanager
def open_verified_child_directory(
    parent_descriptor: int,
    name: str,
    display_path: Path,
    *,
    require_private: bool = True,
) -> Iterator[int]:
    """Open one real child directory relative to a verified parent."""
    descriptor: int | None = None
    try:
        before_open = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(before_open.st_mode):
            raise OSError(f"state directory {display_path} is not a directory")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        after_open = os.fstat(descriptor)
        if (before_open.st_dev, before_open.st_ino) != (after_open.st_dev, after_open.st_ino):
            raise OSError(f"state directory {display_path} changed while it was being opened")
        if not stat.S_ISDIR(after_open.st_mode):
            raise OSError(f"state directory {display_path} is not a directory")
        if require_private and after_open.st_mode & 0o077:
            raise OSError(f"state directory {display_path} does not have restrictive permissions")
        yield descriptor
    finally:
        if descriptor is not None:
            os.close(descriptor)


def read_regular_text(
    directory_descriptor: int,
    name: str,
    display_path: Path,
    *,
    maximum_size: int,
) -> str:
    """Read one restrictive regular UTF-8 file without following a symlink."""
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
        if not _is_restrictive_regular(after_open) or _get_file_facts(before_open) != _get_file_facts(after_open):
            raise OSError(f"state record {display_path} changed while it was being opened")
        record_file = os.fdopen(descriptor, "r", encoding="utf-8")
        descriptor = None
        with record_file:
            content = record_file.read(maximum_size + 1)
            if len(content.encode("utf-8")) > maximum_size:
                raise OSError(f"state record {display_path} exceeds its size limit")
            after_read = os.fstat(record_file.fileno())
        if not _is_restrictive_regular(after_read) or _get_file_facts(after_open) != _get_file_facts(after_read):
            raise OSError(f"state record {display_path} changed while it was being read")
        return content
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
                return False
            raise FileExistsError(f"state record {display_path} already contains different bytes") from None
        try:
            os.unlink(temporary_name, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass
        temporary_name = None
        os.fsync(directory_descriptor)
        return True
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
                os.fsync(directory_descriptor)
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
        os.fsync(directory_descriptor)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
            except OSError:
                pass


def _write_temporary_text(directory_descriptor: int, content: str, *, maximum_size: int) -> str:
    encoded = content.encode("utf-8")
    if len(encoded) > maximum_size:
        raise OSError("state record exceeds its size limit")
    descriptor: int | None = None
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = _create_temporary_file(directory_descriptor)
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


def _create_temporary_file(directory_descriptor: int) -> tuple[int, str]:
    for _ in range(100):
        name = f".state.{secrets.token_hex(8)}.tmp"
        try:
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                _FILE_MODE,
                dir_fd=directory_descriptor,
            )
        except FileExistsError:
            continue
        try:
            os.fchmod(descriptor, _FILE_MODE)
        except OSError:
            os.close(descriptor)
            os.unlink(name, dir_fd=directory_descriptor)
            raise
        return descriptor, name
    raise OSError("cannot allocate a unique state record temporary name")


def _get_file_facts(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)


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
        if _TEMPORARY_NAME_PATTERN.fullmatch(candidate_name) is None:
            continue
        try:
            candidate_status = os.stat(candidate_name, dir_fd=directory_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if (candidate_status.st_dev, candidate_status.st_ino) == (status.st_dev, status.st_ino):
            temporary_names.append(candidate_name)
    if len(temporary_names) != 1:
        return status

    try:
        os.unlink(temporary_names[0], dir_fd=directory_descriptor)
        os.fsync(directory_descriptor)
    except FileNotFoundError:
        pass
    repaired = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    if (repaired.st_dev, repaired.st_ino) != (status.st_dev, status.st_ino) or not _is_restrictive_regular(repaired):
        raise OSError(f"state record {display_path} changed while recovering interrupted publication")
    return repaired


def _is_restrictive_regular(status: os.stat_result) -> bool:
    return stat.S_ISREG(status.st_mode) and status.st_nlink == 1 and not status.st_mode & 0o077
