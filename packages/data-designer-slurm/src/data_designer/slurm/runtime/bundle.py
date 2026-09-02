# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic content-addressed staging for the allocation runtime bundle."""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import re
import secrets
import stat
import tarfile
from contextlib import suppress
from pathlib import Path

from data_designer.slurm.contracts import ArtifactReference, validate_absolute_path
from data_designer.slurm.runtime.errors import SlurmRuntimeError, SlurmRuntimeErrorCode

_DIRECTORY_MODE = 0o700
_FILE_MODE = 0o600
_ENTRYPOINT_MODE = 0o500
_SOURCE_MODE = 0o400
_MAXIMUM_SOURCE_SIZE = 16 * 1024 * 1024
_TEMPORARY_NAME_PATTERN = re.compile(r"^\.runtime\.[0-9a-f]{16}\.tmp$")
_ENTRYPOINT_NAME = "entrypoint.sh"
_SLURM_PACKAGE_NAME = "data_designer/slurm/__init__.py"
_SLURM_PACKAGE_SHIM = (
    b"# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
    b"# SPDX-License-Identifier: Apache-2.0\n\n"
    b'"""Runtime-bundle package shim that extends into the installed Slurm package."""\n\n'
    b"from __future__ import annotations\n\n"
    b"from pkgutil import extend_path\n\n"
    b"__path__ = extend_path(__path__, __name__)\n"
)
_ENTRYPOINT = b"""#!/usr/bin/env bash
set -Eeuo pipefail

dd_slurm_run_allocation() {
    if [[ $# -ne 2 ]]; then
        printf '%s\\n' 'allocation runtime requires plan and attempt directory arguments' >&2
        return 64
    fi
    local runtime_root
    runtime_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
    PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH="${runtime_root}" \
        python3 -m data_designer.slurm.runtime.entrypoint --plan "$1" --attempt-dir "$2"
}
"""


def stage_runtime_bundle(workspace_root: str | Path) -> ArtifactReference:
    """Convergently stage the versioned allocation entrypoint by content digest.

    Raises:
        SlurmRuntimeError: If the workspace or runtime artifact is unsafe or cannot be staged.
    """
    try:
        normalized_workspace = Path(validate_absolute_path(Path(workspace_root).as_posix()))
    except ValueError as error:
        raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "runtime workspace is invalid") from error
    try:
        archive = _build_runtime_archive()
        digest = hashlib.sha256(archive).hexdigest()
        runtime_root = normalized_workspace / "runtime"
        archive_path = runtime_root / f"{digest}.tar.gz"
        _ensure_private_runtime_directory(normalized_workspace, runtime_root)
        _publish_archive(runtime_root, archive_path.name, archive)
    except OSError as error:
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.PREFLIGHT_FAILED, "cannot stage allocation runtime bundle"
        ) from error
    return ArtifactReference(path=archive_path.as_posix(), sha256=digest)


def _build_runtime_archive() -> bytes:
    output = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        _add_archive_file(archive, _ENTRYPOINT_NAME, _ENTRYPOINT, mode=_ENTRYPOINT_MODE)
        _add_archive_file(archive, _SLURM_PACKAGE_NAME, _SLURM_PACKAGE_SHIM, mode=_SOURCE_MODE)
        source_root = Path(__file__).parent
        for source_path in sorted(source_root.glob("*.py")):
            archive_name = f"data_designer/slurm/runtime/{source_path.name}"
            _add_archive_file(archive, archive_name, _read_runtime_source(source_path), mode=_SOURCE_MODE)
    return output.getvalue()


def _add_archive_file(archive: tarfile.TarFile, name: str, content: bytes, *, mode: int) -> None:
    entry = tarfile.TarInfo(name)
    entry.size = len(content)
    entry.mode = mode
    entry.uid = 0
    entry.gid = 0
    entry.uname = ""
    entry.gname = ""
    entry.mtime = 0
    archive.addfile(entry, io.BytesIO(content))


def _read_runtime_source(path: Path) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > _MAXIMUM_SOURCE_SIZE:
        raise OSError(f"allocation runtime source {path} is not a bounded regular file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        opened = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(opened):
            raise OSError(f"allocation runtime source {path} changed while it was opened")
        chunks: list[bytes] = []
        content_size = 0
        while chunk := os.read(descriptor, min(1024 * 1024, _MAXIMUM_SOURCE_SIZE + 1 - content_size)):
            chunks.append(chunk)
            content_size += len(chunk)
            if content_size > _MAXIMUM_SOURCE_SIZE:
                raise OSError(f"allocation runtime source {path} exceeds its size limit")
        after = os.fstat(descriptor)
        if _file_identity(opened) != _file_identity(after):
            raise OSError(f"allocation runtime source {path} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _ensure_private_runtime_directory(workspace_root: Path, runtime_root: Path) -> None:
    workspace_status = workspace_root.lstat()
    if not stat.S_ISDIR(workspace_status.st_mode) or workspace_status.st_mode & 0o022:
        raise OSError(f"runtime workspace {workspace_root} is not a private directory")
    runtime_root.mkdir(mode=_DIRECTORY_MODE, exist_ok=True)
    runtime_status = runtime_root.lstat()
    if not stat.S_ISDIR(runtime_status.st_mode):
        raise OSError(f"runtime root {runtime_root} is not a directory")
    descriptor = os.open(
        runtime_root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened_status = os.fstat(descriptor)
        if (runtime_status.st_dev, runtime_status.st_ino) != (opened_status.st_dev, opened_status.st_ino):
            raise OSError(f"runtime root {runtime_root} changed while it was opened")
        os.fchmod(descriptor, _DIRECTORY_MODE)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_archive(runtime_root: Path, name: str, content: bytes) -> None:
    directory_descriptor = os.open(
        runtime_root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    temporary_name: str | None = None
    descriptor: int | None = None
    try:
        expected_size = len(content)
        existing = _read_existing_archive(directory_descriptor, name, expected_size=expected_size)
        if existing is not None:
            if existing != content:
                raise OSError("content-addressed runtime bundle contains different bytes")
            return
        for _ in range(100):
            temporary_name = f".runtime.{secrets.token_hex(8)}.tmp"
            try:
                descriptor = os.open(
                    temporary_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    _FILE_MODE,
                    dir_fd=directory_descriptor,
                )
            except FileExistsError:
                continue
            break
        if descriptor is None:
            raise OSError("cannot allocate a unique runtime bundle temporary name")
        os.fchmod(descriptor, _FILE_MODE)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = None
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(
                temporary_name,
                name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            if _read_existing_archive(directory_descriptor, name, expected_size=expected_size) != content:
                raise OSError("content-addressed runtime bundle contains different bytes") from None
        try:
            os.unlink(temporary_name, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass
        temporary_name = None
        os.fsync(directory_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_name is not None:
            with suppress(OSError):
                os.unlink(temporary_name, dir_fd=directory_descriptor)
        os.close(directory_descriptor)


def _read_existing_archive(directory_descriptor: int, name: str, *, expected_size: int) -> bytes | None:
    try:
        before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None
    before = _repair_interrupted_archive_publication(directory_descriptor, name, before)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_mode & 0o077
        or before.st_size != expected_size
    ):
        raise OSError("runtime bundle is not a restrictive single-link regular file")
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        dir_fd=directory_descriptor,
    )
    try:
        after_open = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after_open):
            raise OSError("runtime bundle changed while it was opened")
        with os.fdopen(descriptor, "rb") as existing:
            descriptor = -1
            content = existing.read(expected_size + 1)
            after_read = os.fstat(existing.fileno())
        if _file_identity(after_open) != _file_identity(after_read):
            raise OSError("runtime bundle changed while it was read")
        return content
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _file_identity(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns


def _repair_interrupted_archive_publication(
    directory_descriptor: int,
    name: str,
    status: os.stat_result,
) -> os.stat_result:
    if status.st_nlink == 1:
        return status
    if not stat.S_ISREG(status.st_mode) or status.st_mode & 0o077 or status.st_nlink != 2:
        return status
    matching_names: list[str] = []
    for candidate in os.listdir(directory_descriptor):
        if _TEMPORARY_NAME_PATTERN.fullmatch(candidate) is None:
            continue
        try:
            candidate_status = os.stat(candidate, dir_fd=directory_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if (candidate_status.st_dev, candidate_status.st_ino) == (status.st_dev, status.st_ino):
            matching_names.append(candidate)
    if len(matching_names) != 1:
        return status
    try:
        os.unlink(matching_names[0], dir_fd=directory_descriptor)
        os.fsync(directory_descriptor)
    except FileNotFoundError:
        pass
    repaired = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    if (
        (repaired.st_dev, repaired.st_ino) != (status.st_dev, status.st_ino)
        or not stat.S_ISREG(repaired.st_mode)
        or repaired.st_nlink != 1
        or repaired.st_mode & 0o077
    ):
        raise OSError("runtime bundle changed while recovering interrupted publication")
    return repaired
