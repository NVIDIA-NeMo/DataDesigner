# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Descriptor-bound filesystem operations for package-owned Slurm image state."""

from __future__ import annotations

import fcntl
import os
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from data_designer.slurm.images.errors import ImageRegistryError

_DIRECTORY_MODE = 0o700
_FILE_MODE = 0o600


@contextmanager
def acquire_file_lock(directory_descriptor: int, name: str, display_path: Path) -> Iterator[None]:
    """Acquire one exclusive advisory lock without reopening its parent path."""
    descriptor: int | None = None
    try:
        # macOS can report ENOENT to the losing openat(O_CREAT) caller when two
        # processes create the same lock file concurrently. Retrying then opens
        # the winner's regular file with the same no-follow boundary.
        for _ in range(10):
            try:
                descriptor = os.open(
                    name,
                    os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
                    _FILE_MODE,
                    dir_fd=directory_descriptor,
                )
            except FileNotFoundError:
                continue
            break
        if descriptor is None:
            raise OSError(f"lock path {display_path} could not be opened")
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(f"lock path {display_path} is not a regular file")
        os.fchmod(descriptor, _FILE_MODE)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise ImageRegistryError(f"cannot lock image registry target {display_path}") from error

    try:
        yield
    finally:
        assert descriptor is not None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def ensure_private_directory(path: Path, *, parents: bool) -> None:
    """Create or restrict one real directory to its owning user."""
    path.mkdir(mode=_DIRECTORY_MODE, parents=parents, exist_ok=True)
    with open_verified_directory(path) as descriptor:
        os.fchmod(descriptor, _DIRECTORY_MODE)


@contextmanager
def open_verified_directory(path: Path) -> Iterator[int]:
    """Open one real directory and reject path replacement during the open."""
    descriptor: int | None = None
    try:
        before_open = path.lstat()
        if not stat.S_ISDIR(before_open.st_mode):
            raise OSError(f"registry directory {path} is not a directory")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        after_open = os.fstat(descriptor)
        if (before_open.st_dev, before_open.st_ino) != (after_open.st_dev, after_open.st_ino):
            raise OSError(f"registry directory {path} changed while it was being opened")
        if not stat.S_ISDIR(after_open.st_mode):
            raise OSError(f"registry directory {path} is not a directory")
        yield descriptor
    finally:
        if descriptor is not None:
            os.close(descriptor)


@contextmanager
def open_verified_child_directory(
    parent_descriptor: int,
    name: str,
    display_path: Path,
) -> Iterator[int]:
    """Open one real child directory relative to an already verified parent."""
    descriptor: int | None = None
    try:
        before_open = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(before_open.st_mode):
            raise OSError(f"registry directory {display_path} is not a directory")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        after_open = os.fstat(descriptor)
        if (before_open.st_dev, before_open.st_ino) != (after_open.st_dev, after_open.st_ino):
            raise OSError(f"registry directory {display_path} changed while it was being opened")
        if not stat.S_ISDIR(after_open.st_mode):
            raise OSError(f"registry directory {display_path} is not a directory")
        yield descriptor
    finally:
        if descriptor is not None:
            os.close(descriptor)


def create_temporary_file(
    directory_descriptor: int,
    *,
    prefix: str,
    suffix: str,
) -> tuple[int, str]:
    """Create one restrictive temporary file beneath an open directory."""
    for _ in range(100):
        name = f"{prefix}{secrets.token_hex(8)}{suffix}"
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
            raise
        return descriptor, name
    raise OSError("cannot allocate a unique temporary file name")


def read_regular_text(directory_descriptor: int, name: str, display_path: Path) -> str:
    """Read one non-symlink regular text file beneath an open directory."""
    descriptor: int | None = None
    try:
        before_open = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if not stat.S_ISREG(before_open.st_mode):
            raise OSError(f"registry path {display_path} is not a regular file")
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_descriptor,
        )
        after_open = os.fstat(descriptor)
        if (before_open.st_dev, before_open.st_ino) != (after_open.st_dev, after_open.st_ino):
            raise OSError(f"registry path {display_path} changed while it was being opened")
        registry_file = os.fdopen(descriptor, "r", encoding="utf-8")
        descriptor = None
        with registry_file:
            content = registry_file.read()
            after_read = os.fstat(registry_file.fileno())
            after_path = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(after_read.st_mode)
            or not stat.S_ISREG(after_path.st_mode)
            or _get_file_facts(after_open) != _get_file_facts(after_read)
            or _get_file_facts(after_read) != _get_file_facts(after_path)
        ):
            raise OSError(f"registry path {display_path} changed while it was being read")
        return content
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _get_file_facts(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)
