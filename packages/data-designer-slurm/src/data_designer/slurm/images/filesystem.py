# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Descriptor-bound filesystem operations for package-owned Slurm image state."""

from __future__ import annotations

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
)
from data_designer.slurm.filesystem import (
    open_verified_child_directory as open_shared_child_directory,
)
from data_designer.slurm.filesystem import (
    open_verified_directory as open_shared_directory,
)
from data_designer.slurm.images.errors import ImageRegistryError


@contextmanager
def acquire_file_lock(directory_descriptor: int, name: str, display_path: Path) -> Iterator[None]:
    """Acquire one exclusive advisory lock without reopening its parent path."""
    try:
        with acquire_restrictive_file_lock(
            directory_descriptor,
            name,
            display_path,
            resource_name="registry",
        ):
            yield
    except OSError as error:
        raise ImageRegistryError(f"cannot lock image registry target {display_path}") from error


def ensure_private_directory(path: Path, *, parents: bool) -> None:
    """Create or restrict one real directory to its owning user."""
    path.mkdir(mode=PRIVATE_DIRECTORY_MODE, parents=parents, exist_ok=True)
    with open_verified_directory(path) as descriptor:
        os.fchmod(descriptor, PRIVATE_DIRECTORY_MODE)


@contextmanager
def open_verified_directory(path: Path) -> Iterator[int]:
    """Open one real directory and reject path replacement during the open."""
    with open_shared_directory(path, resource_name="registry") as descriptor:
        yield descriptor


@contextmanager
def open_verified_child_directory(
    parent_descriptor: int,
    name: str,
    display_path: Path,
) -> Iterator[int]:
    """Open one real child directory relative to an already verified parent."""
    with open_shared_child_directory(
        parent_descriptor,
        name,
        display_path,
        resource_name="registry",
    ) as descriptor:
        yield descriptor


def create_temporary_file(
    directory_descriptor: int,
    *,
    prefix: str,
    suffix: str,
) -> tuple[int, str]:
    """Create one restrictive temporary file beneath an open directory."""
    return create_restrictive_temporary_file(directory_descriptor, prefix=prefix, suffix=suffix)


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
            or get_file_facts(after_open) != get_file_facts(after_read)
            or get_file_facts(after_read) != get_file_facts(after_path)
        ):
            raise OSError(f"registry path {display_path} changed while it was being read")
        return content
    finally:
        if descriptor is not None:
            os.close(descriptor)
