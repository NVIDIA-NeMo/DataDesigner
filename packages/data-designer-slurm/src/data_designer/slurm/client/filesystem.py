# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
import stat
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from data_designer.slurm.client.errors import ClientWorkerError
from data_designer.slurm.client.records import ClientErrorCode

_MAXIMUM_RECORD_SIZE = 16 * 1024 * 1024
_DIRECTORY_MODE = 0o700
_FILE_MODE = 0o600
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_FILE_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)


def ensure_private_directory(path: Path) -> None:
    """Create one private directory tree and reject symlink components."""
    try:
        with _open_directory(path, create=True) as descriptor:
            os.fchmod(descriptor, _DIRECTORY_MODE)
    except OSError as error:
        raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "client workspace is not a regular directory") from error


def read_regular_bytes(path: Path, *, missing_code: ClientErrorCode) -> bytes:
    """Read one bounded regular file without following a final symlink."""
    try:
        with _open_directory(path.parent, create=False) as directory_descriptor:
            with _open_regular_file(directory_descriptor, path.name) as (descriptor, info):
                if not stat.S_ISREG(info.st_mode):
                    raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "client input is not a regular file")
                return _read_bounded_bytes(descriptor, info)
    except ClientWorkerError:
        raise
    except FileNotFoundError as error:
        raise ClientWorkerError(missing_code, "required client artifact is missing") from error
    except OSError as error:
        raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "required client artifact is unreadable") from error


def compute_file_sha256(path: Path, *, missing_code: ClientErrorCode) -> str:
    """Hash one regular file without retaining its contents."""
    try:
        with _open_directory(path.parent, create=False) as directory_descriptor:
            with _open_regular_file(directory_descriptor, path.name) as (descriptor, info):
                if not stat.S_ISREG(info.st_mode):
                    raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "client artifact is not a regular file")
                digest = hashlib.sha256()
                while chunk := os.read(descriptor, 1024 * 1024):
                    digest.update(chunk)
                return digest.hexdigest()
    except ClientWorkerError:
        raise
    except FileNotFoundError as error:
        raise ClientWorkerError(missing_code, "required client artifact is missing") from error
    except OSError as error:
        raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "required client artifact is unreadable") from error


def replace_private_text(path: Path, text: str) -> None:
    """Atomically replace one private attempt-local record."""
    with _write_private_temporary(path, text) as (_, directory_descriptor, temporary_name):
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )


def publish_private_text(path: Path, text: str) -> None:
    """Convergently publish one immutable private attempt-local record."""
    with _write_private_temporary(path, text) as (payload, directory_descriptor, temporary_name):
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            try:
                with _open_regular_file(directory_descriptor, path.name) as (descriptor, info):
                    if not stat.S_ISREG(info.st_mode):
                        raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "client input is not a regular file")
                    if _read_bounded_bytes(descriptor, info) != payload:
                        raise ClientWorkerError(
                            ClientErrorCode.OUTPUT_INVALID, "immutable client record already differs"
                        )
                    os.fchmod(descriptor, _FILE_MODE)
            except ClientWorkerError:
                raise
            except FileNotFoundError as error:
                raise ClientWorkerError(
                    ClientErrorCode.OUTPUT_INVALID, "required client artifact is missing"
                ) from error
            except OSError as error:
                raise ClientWorkerError(
                    ClientErrorCode.INVALID_INPUT, "required client artifact is unreadable"
                ) from error


@contextmanager
def _write_private_temporary(path: Path, text: str) -> Iterator[tuple[bytes, int, str]]:
    payload = text.encode("utf-8")
    if len(payload) > _MAXIMUM_RECORD_SIZE:
        raise ClientWorkerError(ClientErrorCode.OUTPUT_INVALID, "client record exceeds the size limit")
    with _open_directory(path.parent, create=True) as directory_descriptor:
        os.fchmod(directory_descriptor, _DIRECTORY_MODE)
        temporary_name = f".client.{uuid.uuid4().hex}.tmp"
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            _FILE_MODE,
            dir_fd=directory_descriptor,
        )
        try:
            try:
                os.fchmod(descriptor, _FILE_MODE)
                remaining = memoryview(payload)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written == 0:
                        raise OSError("client temporary file write made no progress")
                    remaining = remaining[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            yield payload, directory_descriptor, temporary_name
        finally:
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass


@contextmanager
def _open_directory(path: Path, *, create: bool) -> Iterator[int]:
    start = path.anchor or "."
    parts = path.parts[1:] if path.anchor else path.parts
    descriptor = os.open(start, _DIRECTORY_FLAGS)
    try:
        for part in parts:
            if part in ("", ".", ".."):
                raise OSError("client directory path is not canonical")
            try:
                child_descriptor = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, _DIRECTORY_MODE, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child_descriptor = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            try:
                if not stat.S_ISDIR(os.fstat(child_descriptor).st_mode):
                    raise OSError("client workspace component is not a directory")
            except BaseException:
                os.close(child_descriptor)
                raise
            previous_descriptor = descriptor
            descriptor = child_descriptor
            os.close(previous_descriptor)
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def _open_regular_file(directory_descriptor: int, name: str) -> Iterator[tuple[int, os.stat_result]]:
    descriptor = os.open(name, _FILE_READ_FLAGS, dir_fd=directory_descriptor)
    try:
        yield descriptor, os.fstat(descriptor)
    finally:
        os.close(descriptor)


def _read_bounded_bytes(descriptor: int, info: os.stat_result) -> bytes:
    if info.st_size > _MAXIMUM_RECORD_SIZE:
        raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "client input exceeds the size limit")
    payload = bytearray()
    while len(payload) <= _MAXIMUM_RECORD_SIZE:
        chunk = os.read(descriptor, min(1024 * 1024, _MAXIMUM_RECORD_SIZE + 1 - len(payload)))
        if not chunk:
            return bytes(payload)
        payload.extend(chunk)
    raise ClientWorkerError(ClientErrorCode.INVALID_INPUT, "client input exceeds the size limit")
