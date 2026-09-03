# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared descriptor-bound filesystem primitives for package-owned state."""

from __future__ import annotations

import fcntl
import os
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


@contextmanager
def acquire_restrictive_file_lock(
    directory_descriptor: int,
    name: str,
    display_path: Path,
    *,
    resource_name: str,
) -> Iterator[None]:
    """Acquire one private single-link lock beneath a verified directory."""
    descriptor: int | None = None
    try:
        descriptor = open_lock_file(directory_descriptor, name, display_path, resource_name=resource_name)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked_status = os.fstat(descriptor)
        current_status = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            get_file_facts(locked_status)[:2] != get_file_facts(current_status)[:2]
            or not stat.S_ISREG(current_status.st_mode)
            or current_status.st_nlink != 1
        ):
            raise OSError(f"{resource_name} lock {display_path} changed while it was being acquired")
        yield
    finally:
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def open_lock_file(
    directory_descriptor: int,
    name: str,
    display_path: Path,
    *,
    resource_name: str,
) -> int:
    """Open and restrict one package-owned lock file."""
    for _ in range(10):
        try:
            descriptor = os.open(
                name,
                os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
                PRIVATE_FILE_MODE,
                dir_fd=directory_descriptor,
            )
        except FileNotFoundError:
            continue
        try:
            status = os.fstat(descriptor)
            if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
                raise OSError(f"{resource_name} lock {display_path} is not a regular file")
            os.fchmod(descriptor, PRIVATE_FILE_MODE)
            if os.fstat(descriptor).st_mode & 0o077:
                raise OSError(f"{resource_name} lock {display_path} does not have restrictive permissions")
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor
    raise OSError(f"{resource_name} lock {display_path} could not be opened")


@contextmanager
def open_verified_directory(
    path: Path,
    *,
    resource_name: str,
    require_private: bool = False,
) -> Iterator[int]:
    """Open one real directory and reject replacement during the open."""
    descriptor: int | None = None
    try:
        before_open = path.lstat()
        if not stat.S_ISDIR(before_open.st_mode):
            raise OSError(f"{resource_name} directory {path} is not a directory")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        after_open = os.fstat(descriptor)
        validate_open_directory(before_open, after_open, path, resource_name=resource_name)
        if require_private and after_open.st_mode & 0o077:
            raise OSError(f"{resource_name} directory {path} does not have restrictive permissions")
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
    resource_name: str,
    require_private: bool = False,
) -> Iterator[int]:
    """Open one real child directory relative to a verified parent."""
    descriptor: int | None = None
    try:
        before_open = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(before_open.st_mode):
            raise OSError(f"{resource_name} directory {display_path} is not a directory")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        after_open = os.fstat(descriptor)
        validate_open_directory(before_open, after_open, display_path, resource_name=resource_name)
        if require_private and after_open.st_mode & 0o077:
            raise OSError(f"{resource_name} directory {display_path} does not have restrictive permissions")
        yield descriptor
    finally:
        if descriptor is not None:
            os.close(descriptor)


def validate_open_directory(
    before_open: os.stat_result,
    after_open: os.stat_result,
    display_path: Path,
    *,
    resource_name: str,
) -> None:
    """Require an opened descriptor to identify the directory that was inspected."""
    if get_file_facts(before_open)[:2] != get_file_facts(after_open)[:2]:
        raise OSError(f"{resource_name} directory {display_path} changed while it was being opened")
    if not stat.S_ISDIR(after_open.st_mode):
        raise OSError(f"{resource_name} directory {display_path} is not a directory")


def create_restrictive_temporary_file(
    directory_descriptor: int,
    *,
    prefix: str,
    suffix: str,
) -> tuple[int, str]:
    """Create one private temporary file beneath an open directory."""
    for _ in range(100):
        name = f"{prefix}{secrets.token_hex(8)}{suffix}"
        try:
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                PRIVATE_FILE_MODE,
                dir_fd=directory_descriptor,
            )
        except FileExistsError:
            continue
        try:
            os.fchmod(descriptor, PRIVATE_FILE_MODE)
        except OSError:
            os.close(descriptor)
            try:
                os.unlink(name, dir_fd=directory_descriptor)
            except OSError:
                pass
            raise
        return descriptor, name
    raise OSError("cannot allocate a unique temporary file name")


def is_managed_temporary_name(name: str, *, prefix: str, suffix: str) -> bool:
    """Return whether a name has the package's random temporary-file shape."""
    if not name.startswith(prefix) or not name.endswith(suffix):
        return False
    token = name[len(prefix) : len(name) - len(suffix)]
    return len(token) == 16 and all(character in "0123456789abcdef" for character in token)


def get_file_facts(status: os.stat_result) -> tuple[int, int, int, int, int]:
    """Return stable identity and content facts used for replacement checks."""
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)


__all__ = [
    "PRIVATE_DIRECTORY_MODE",
    "PRIVATE_FILE_MODE",
    "acquire_restrictive_file_lock",
    "create_restrictive_temporary_file",
    "get_file_facts",
    "is_managed_temporary_name",
    "open_lock_file",
    "open_verified_child_directory",
    "open_verified_directory",
    "validate_open_directory",
]
