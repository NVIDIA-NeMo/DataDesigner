# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
import stat
import uuid
from pathlib import Path

from data_designer.slurm.client.errors import ClientWorkerError
from data_designer.slurm.client.records import ClientErrorCode

_MAXIMUM_RECORD_SIZE = 16 * 1024 * 1024


def ensure_private_directory(path: Path) -> None:
    """Create one private directory tree and reject symlink components."""
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    current = path
    while current != current.parent:
        info = current.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "client workspace is not a regular directory")
        if current.parent == path.anchor:
            break
        current = current.parent
    path.chmod(0o700)


def read_regular_bytes(path: Path, *, missing_code: ClientErrorCode) -> bytes:
    """Read one bounded regular file without following a final symlink."""
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "client input is not a regular file")
        if info.st_size > _MAXIMUM_RECORD_SIZE:
            raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "client input exceeds the size limit")
        return path.read_bytes()
    except FileNotFoundError as error:
        raise ClientWorkerError(missing_code, "required client artifact is missing") from error
    except OSError as error:
        raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "required client artifact is unreadable") from error


def compute_file_sha256(path: Path, *, missing_code: ClientErrorCode) -> str:
    """Hash one regular file without retaining its contents."""
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "client artifact is not a regular file")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except FileNotFoundError as error:
        raise ClientWorkerError(missing_code, "required client artifact is missing") from error
    except OSError as error:
        raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "required client artifact is unreadable") from error


def replace_private_text(path: Path, text: str) -> None:
    """Atomically replace one private attempt-local record."""
    payload = text.encode("utf-8")
    if len(payload) > _MAXIMUM_RECORD_SIZE:
        raise ClientWorkerError(ClientErrorCode.OUTPUT_INVALID, "client record exceeds the size limit")
    ensure_private_directory(path.parent)
    temporary = path.parent / f".client.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def publish_private_text(path: Path, text: str) -> None:
    """Convergently publish one immutable private attempt-local record."""
    payload = text.encode("utf-8")
    if len(payload) > _MAXIMUM_RECORD_SIZE:
        raise ClientWorkerError(ClientErrorCode.OUTPUT_INVALID, "client record exceeds the size limit")
    ensure_private_directory(path.parent)
    temporary = path.parent / f".client.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if read_regular_bytes(path, missing_code=ClientErrorCode.OUTPUT_INVALID) != payload:
                raise ClientWorkerError(ClientErrorCode.OUTPUT_INVALID, "immutable client record already differs")
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)
