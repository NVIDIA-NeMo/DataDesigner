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
from typing import BinaryIO, Protocol

import data_designer.lazy_heavy_imports as lazy
from data_designer.slurm.state.filesystem import (
    open_verified_child_directory,
    open_verified_directory,
    open_verified_regular_file,
)
from data_designer.slurm.state.outputs import CandidateOutputFile, CandidateOutputManifest


class SerializedCandidateSchema(Protocol):
    """Serialized schema bytes returned by an Arrow-compatible schema."""

    def to_pybytes(self) -> bytes:
        """Return the canonical serialized schema bytes."""
        ...


class CandidateSchema(Protocol):
    """Structural schema interface used by candidate producers and readers."""

    def remove_metadata(self) -> CandidateSchema:
        """Return the schema without producer-specific metadata."""
        ...

    def serialize(self) -> SerializedCandidateSchema:
        """Serialize the normalized schema."""
        ...


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
    relative_path: str
    descriptor: int
    file_facts: tuple[int, int, int, int, int]

    def rebind(self, dataset_descriptor: int, dataset_path: Path) -> None:
        parts = PurePosixPath(self.relative_path).parts
        with _open_parent_directory(dataset_descriptor, dataset_path, parts[:-1]) as (
            parent_descriptor,
            parent_path,
        ):
            current = os.stat(parts[-1], dir_fd=parent_descriptor, follow_symlinks=False)
            if not _is_safe_file(current) or _file_facts(current) != self.file_facts:
                raise OSError(f"candidate file {parent_path / parts[-1]} changed during finalization")

    def validate_lease(self) -> None:
        current = os.fstat(self.descriptor)
        if not _is_safe_file(current) or _file_facts(current) != self.file_facts:
            raise OSError(f"candidate file {self.relative_path!r} changed during finalization")


@dataclass(frozen=True, slots=True)
class VerifiedCandidateArtifacts:
    """Bounded live candidate leases plus metadata derived from artifact bytes."""

    record_counts: tuple[int, ...]
    dataset_schema_digest: str
    _dataset: _DirectoryBinding
    _files: tuple[_FileBinding, ...]

    def rebind(self) -> None:
        """Verify every leased file still occupies its original candidate path."""
        self._dataset.rebind()
        for output_file in self._files:
            output_file.rebind(self._dataset.descriptor, self._dataset.display_path)
        for output_file in self._files:
            output_file.validate_lease()


@dataclass(frozen=True, slots=True)
class CandidateArtifactSnapshot:
    """Descriptor-free identity snapshot retained during bounded collection."""

    record_counts: tuple[int, ...]
    dataset_schema_digest: str
    _dataset_identity: tuple[int, int]
    _bindings: tuple[_ArtifactBinding, ...]


class CandidateArtifactVerifier:
    """Verify one manifest-bounded candidate and lease its files through publication."""

    @contextmanager
    def verify(self, candidate: CandidateOutputManifest) -> Iterator[VerifiedCandidateArtifacts]:
        dataset_path = Path(candidate.dataset_path)
        with ExitStack() as resources:
            dataset_descriptor = resources.enter_context(open_verified_directory(dataset_path, require_private=True))
            dataset = _DirectoryBinding(None, None, dataset_descriptor, dataset_path)
            metadata = tuple(
                _open_output_file(resources, dataset_descriptor, dataset_path, output_file)
                for output_file in candidate.files
            )
            schema_digests = tuple(schema_digest for _, schema_digest, _ in metadata)
            if not schema_digests or any(digest != schema_digests[0] for digest in schema_digests[1:]):
                raise OSError("candidate Parquet files do not share one dataset schema")
            yield VerifiedCandidateArtifacts(
                record_counts=tuple(record_count for record_count, _, _ in metadata),
                dataset_schema_digest=schema_digests[0],
                _dataset=dataset,
                _files=tuple(binding for _, _, binding in metadata),
            )

    def inspect(self, candidate: CandidateOutputManifest) -> CandidateArtifactSnapshot:
        """Inspect one candidate and return identities without retaining descriptors."""
        with self.verify(candidate) as verified:
            return CandidateArtifactSnapshot(
                record_counts=verified.record_counts,
                dataset_schema_digest=verified.dataset_schema_digest,
                _dataset_identity=_identity(os.fstat(verified._dataset.descriptor)),
                _bindings=verified._bindings,
            )

    def rebind(self, candidate: CandidateOutputManifest, expected: CandidateArtifactSnapshot) -> None:
        """Reopen a candidate and require the identities captured before collection."""
        with self.verify(candidate) as current:
            actual = CandidateArtifactSnapshot(
                record_counts=current.record_counts,
                dataset_schema_digest=current.dataset_schema_digest,
                _dataset_identity=_identity(os.fstat(current._dataset.descriptor)),
                _bindings=current._bindings,
            )
        if actual != expected:
            raise OSError("candidate paths or metadata changed during collection")

    @contextmanager
    def open_output(
        self,
        candidate: CandidateOutputManifest,
        output_file: CandidateOutputFile,
    ) -> Iterator[BinaryIO]:
        """Yield one digest-verified candidate file through a bounded descriptor."""
        if output_file not in candidate.files:
            raise ValueError("candidate output file is not declared by the manifest")
        dataset_path = Path(candidate.dataset_path)
        parts = PurePosixPath(output_file.relative_path).parts
        with open_verified_directory(dataset_path, require_private=True) as dataset_descriptor:
            with _open_parent_directory(dataset_descriptor, dataset_path, parts[:-1]) as (
                parent_descriptor,
                parent_path,
                _,
            ):
                display_path = parent_path / parts[-1]
                with open_verified_regular_file(
                    parent_descriptor,
                    parts[-1],
                    display_path,
                    expected_size=output_file.byte_size,
                    expected_sha256=output_file.sha256,
                    require_private=False,
                ) as descriptor:
                    with os.fdopen(os.dup(descriptor), "rb") as source:
                        yield source


def compute_candidate_schema_digest(schema: CandidateSchema) -> str:
    """Compute the version-1 digest for an attempt-local candidate schema."""
    payload = schema.remove_metadata().serialize().to_pybytes()
    return hashlib.sha256(payload).hexdigest()


def _open_output_file(
    resources: ExitStack,
    dataset_descriptor: int,
    dataset_path: Path,
    output_file: CandidateOutputFile,
) -> tuple[int, str, _FileBinding]:
    parts = PurePosixPath(output_file.relative_path).parts
    parent_descriptor, parent_path = resources.enter_context(
        _open_parent_directory(dataset_descriptor, dataset_path, parts[:-1])
    )
    name = parts[-1]
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
    record_count, schema_digest = _read_parquet_metadata(descriptor, display_path)
    binding = _FileBinding(
        relative_path=output_file.relative_path,
        descriptor=descriptor,
        file_facts=_file_facts(os.fstat(descriptor)),
    )
    return record_count, schema_digest, binding


@contextmanager
def _open_parent_directory(
    dataset_descriptor: int,
    dataset_path: Path,
    parts: tuple[str, ...],
) -> Iterator[tuple[int, Path]]:
    parent_descriptor = os.dup(dataset_descriptor)
    parent_path = dataset_path
    try:
        for part in parts:
            child_path = parent_path / part
            with open_verified_child_directory(
                parent_descriptor,
                part,
                child_path,
                require_private=False,
            ) as child_descriptor:
                next_descriptor = os.dup(child_descriptor)
            os.close(parent_descriptor)
            parent_descriptor = next_descriptor
            parent_path = child_path
        yield parent_descriptor, parent_path
    finally:
        os.close(parent_descriptor)


def _read_parquet_metadata(descriptor: int, display_path: Path) -> tuple[int, str]:
    try:
        with os.fdopen(os.dup(descriptor), "rb") as source:
            parquet_file = lazy.pq.ParquetFile(source)
            return parquet_file.metadata.num_rows, compute_candidate_schema_digest(parquet_file.schema_arrow)
    except (OSError, lazy.pa.ArrowException) as error:
        raise OSError(f"candidate Parquet metadata {display_path} is invalid") from error


def _identity(status: os.stat_result) -> tuple[int, int]:
    return status.st_dev, status.st_ino


def _file_facts(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns


def _is_safe_file(status: os.stat_result) -> bool:
    return stat.S_ISREG(status.st_mode) and status.st_nlink == 1 and not status.st_mode & 0o022


__all__ = [
    "CandidateArtifactSnapshot",
    "CandidateArtifactVerifier",
    "CandidateSchema",
    "SerializedCandidateSchema",
    "VerifiedCandidateArtifacts",
    "compute_candidate_schema_digest",
]
