# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Descriptor-bound validation of candidate dataset artifacts."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

import data_designer.lazy_heavy_imports as lazy
from data_designer.slurm.state.filesystem import (
    open_verified_child_directory,
    open_verified_directory,
    open_verified_regular_file,
)
from data_designer.slurm.state.outputs import CandidateOutputFile, CandidateOutputManifest


@dataclass(frozen=True, slots=True)
class _DirectoryBinding:
    parent_descriptor: int | None
    name: str | None
    descriptor: int
    display_path: Path

    def rebind(self) -> None:
        opened = os.fstat(self.descriptor)
        current = (
            self.display_path.lstat()
            if self.parent_descriptor is None or self.name is None
            else os.stat(self.name, dir_fd=self.parent_descriptor, follow_symlinks=False)
        )
        if not stat.S_ISDIR(current.st_mode) or current.st_mode & 0o022 or _identity(current) != _identity(opened):
            raise OSError(f"candidate directory {self.display_path} changed during finalization")


@dataclass(frozen=True, slots=True)
class _FileBinding:
    parent_descriptor: int
    name: str
    descriptor: int
    display_path: Path

    def rebind(self) -> None:
        opened = os.fstat(self.descriptor)
        current = os.stat(self.name, dir_fd=self.parent_descriptor, follow_symlinks=False)
        if not _is_safe_file(current) or _file_facts(current) != _file_facts(opened):
            raise OSError(f"candidate file {self.display_path} changed during finalization")


@dataclass(frozen=True, slots=True)
class VerifiedCandidateArtifacts:
    """Held candidate descriptors plus metadata derived from their bytes."""

    record_counts: tuple[int, ...]
    dataset_schema_digest: str
    _directories: tuple[_DirectoryBinding, ...]
    _files: tuple[_FileBinding, ...]

    def rebind(self) -> None:
        """Verify every held directory and file still occupies its original path."""
        for directory in self._directories:
            directory.rebind()
        for output_file in self._files:
            output_file.rebind()


class CandidateArtifactVerifier:
    """Open, verify, and retain one candidate's complete artifact chain."""

    @contextmanager
    def verify(
        self,
        candidate: CandidateOutputManifest,
        output_format: Literal["parquet", "jsonl", "csv"],
    ) -> Iterator[VerifiedCandidateArtifacts]:
        if output_format != "parquet":
            raise OSError("winner metadata verification currently requires Parquet candidate output")
        dataset_path = Path(candidate.dataset_path)
        with ExitStack() as resources:
            dataset_descriptor = resources.enter_context(open_verified_directory(dataset_path, require_private=True))
            directories = [_DirectoryBinding(None, None, dataset_descriptor, dataset_path)]
            opened_files = tuple(
                self._open_output_file(resources, dataset_path, dataset_descriptor, output_file, directories)
                for output_file in candidate.files
            )
            schema_payloads = tuple(schema_payload for _, _, schema_payload in opened_files)
            if not schema_payloads or any(payload != schema_payloads[0] for payload in schema_payloads[1:]):
                raise OSError("candidate Parquet files do not share one dataset schema")
            verified = VerifiedCandidateArtifacts(
                record_counts=tuple(record_count for _, record_count, _ in opened_files),
                dataset_schema_digest=hashlib.sha256(schema_payloads[0]).hexdigest(),
                _directories=tuple(directories),
                _files=tuple(binding for binding, _, _ in opened_files),
            )
            yield verified

    def _open_output_file(
        self,
        resources: ExitStack,
        dataset_path: Path,
        dataset_descriptor: int,
        output_file: CandidateOutputFile,
        directories: list[_DirectoryBinding],
    ) -> tuple[_FileBinding, int, bytes]:
        parent_descriptor, parent_path = self._open_parent_directories(
            resources,
            dataset_path,
            dataset_descriptor,
            PurePosixPath(output_file.relative_path).parts[:-1],
            directories,
        )
        name = PurePosixPath(output_file.relative_path).parts[-1]
        display_path = parent_path / name
        descriptor = resources.enter_context(
            open_verified_regular_file(
                parent_descriptor,
                name,
                display_path,
                expected_size=output_file.byte_size,
                expected_sha256=output_file.sha256,
                require_private=False,
            )
        )
        record_count, schema_payload = _read_parquet_metadata(descriptor, display_path)
        return _FileBinding(parent_descriptor, name, descriptor, display_path), record_count, schema_payload

    @staticmethod
    def _open_parent_directories(
        resources: ExitStack,
        dataset_path: Path,
        dataset_descriptor: int,
        parts: tuple[str, ...],
        directories: list[_DirectoryBinding],
    ) -> tuple[int, Path]:
        parent_descriptor = dataset_descriptor
        parent_path = dataset_path
        for part in parts:
            child_path = parent_path / part
            child_descriptor = resources.enter_context(
                open_verified_child_directory(
                    parent_descriptor,
                    part,
                    child_path,
                    require_private=False,
                )
            )
            binding = _DirectoryBinding(parent_descriptor, part, child_descriptor, child_path)
            binding.rebind()
            directories.append(binding)
            parent_descriptor, parent_path = child_descriptor, child_path
        return parent_descriptor, parent_path


def _read_parquet_metadata(descriptor: int, display_path: Path) -> tuple[int, bytes]:
    try:
        with os.fdopen(os.dup(descriptor), "rb") as source:
            parquet_file = lazy.pq.ParquetFile(source)
            schema_payload = parquet_file.schema_arrow.remove_metadata().serialize().to_pybytes()
            return parquet_file.metadata.num_rows, schema_payload
    except (OSError, lazy.pa.ArrowException) as error:
        raise OSError(f"candidate Parquet metadata {display_path} is invalid") from error


def _identity(status: os.stat_result) -> tuple[int, int]:
    return status.st_dev, status.st_ino


def _file_facts(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns


def _is_safe_file(status: os.stat_result) -> bool:
    return stat.S_ISREG(status.st_mode) and status.st_nlink == 1 and not status.st_mode & 0o022
