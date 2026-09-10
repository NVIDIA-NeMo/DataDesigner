# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded-memory merging of validated Parquet shard winners."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Protocol, TextIO

import data_designer.lazy_heavy_imports as lazy
from data_designer.slurm.filesystem import get_file_facts
from data_designer.slurm.state.artifacts import CandidateArtifactSnapshot, CandidateArtifactVerifier
from data_designer.slurm.state.collection_filesystem import StagedCollection, StagedFile
from data_designer.slurm.state.collection_records import CollectedOutputFile, CollectionResult
from data_designer.slurm.state.filesystem import open_verified_directory, publish_immutable_text, sync_directory
from data_designer.slurm.state.outputs import CandidateOutputManifest, CollectionPlan

_BATCH_SIZE = 65_536
_RESULT_FILENAME = "collection-result.json"
_MAXIMUM_RECORD_SIZE = 16 * 1024 * 1024


class CollectionPartitionWriter(Protocol):
    """Bounded writer for one deterministic output partition."""

    def write(self, batch: object) -> None:
        """Append one Arrow record batch."""
        ...

    def finish(self, relative_path: str, record_count: int) -> CollectedOutputFile:
        """Seal and describe the exact file descriptor that received the records."""
        ...


class _ArrowBatchWriter(Protocol):
    def write_batch(self, batch: object) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _PartitionLayout:
    stage_path: Path
    output_format: str
    schema: object
    record_counts: tuple[int, ...]


class CollectionMerger:
    """Merge ordered winner files while retaining only bounded data and descriptors."""

    def __init__(
        self,
        output_format: str,
        *,
        verifier: CandidateArtifactVerifier | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        self._output_format = output_format
        self._verifier = verifier if verifier is not None else CandidateArtifactVerifier()
        self._completed_at = completed_at

    def merge(
        self,
        collection_plan: CollectionPlan,
        candidates: tuple[CandidateOutputManifest, ...],
        staged: StagedCollection,
    ) -> CollectionResult:
        """Write deterministic partitions, rebind inputs, and publish the stage."""
        snapshots = tuple(self._inspect_candidate(candidate) for candidate in candidates)
        batches = self._iter_batches(candidates)
        try:
            first_batch = next(batches, None)
            if first_batch is None:
                raise OSError("collection inputs contain no records")
            layout = _PartitionLayout(
                stage_path=staged.path,
                output_format=self._output_format,
                schema=first_batch.schema.remove_metadata(),
                record_counts=_partition_record_counts(
                    sum(candidate.actual_records for candidate in candidates),
                    collection_plan.num_partitions,
                ),
            )
            files = self._write_partitions(layout, first_batch, batches)
        finally:
            batches.close()
        completion_time = self._completed_at if self._completed_at is not None else _utc_now()
        result = CollectionResult(
            schema_version=1,
            collection_id=collection_plan.collection_id,
            run_id=collection_plan.run_id,
            completed_at=completion_time,
            collection_plan_sha256=collection_plan.compute_sha256(),
            actual_records=sum(output.record_count for output in files),
            files=files,
        )
        result_bytes = self._write_result(staged.path, result)
        for candidate, snapshot in zip(candidates, snapshots, strict=True):
            self._verifier.rebind(candidate, snapshot)
        staged.publish(_stage_expectations(files, result, result_bytes))
        return result

    def _inspect_candidate(self, candidate: CandidateOutputManifest) -> CandidateArtifactSnapshot:
        snapshot = self._verifier.inspect(candidate)
        if snapshot.record_counts != tuple(output.record_count for output in candidate.files):
            raise OSError("candidate Parquet row counts changed before collection")
        if snapshot.dataset_schema_digest != candidate.dataset_schema_digest:
            raise OSError("candidate Parquet schema changed before collection")
        return snapshot

    def _iter_batches(self, candidates: tuple[CandidateOutputManifest, ...]) -> Generator[object, None, None]:
        for candidate in candidates:
            for output_file in candidate.files:
                with self._verifier.open_output(candidate, output_file) as source:
                    parquet_file = lazy.pq.ParquetFile(source)
                    yield from parquet_file.iter_batches(batch_size=_BATCH_SIZE)

    def _write_partitions(
        self,
        layout: _PartitionLayout,
        first_batch: object,
        remaining_batches: Generator[object, None, None],
    ) -> tuple[CollectedOutputFile, ...]:
        cursor = _BatchCursor(first_batch, remaining_batches)
        outputs: list[CollectedOutputFile] = []
        for partition_index, record_count in enumerate(layout.record_counts):
            suffix = "jsonl" if layout.output_format == "jsonl" else layout.output_format
            relative_path = f"part-{partition_index:05d}.{suffix}"
            output_path = layout.stage_path / relative_path
            with _open_partition_writer(output_path, layout.output_format, layout.schema) as writer:
                cursor.write_records(writer, record_count)
                outputs.append(writer.finish(relative_path, record_count))
        if cursor.has_remaining_records():
            raise OSError("collection inputs contain more rows than declared")
        return tuple(outputs)

    @staticmethod
    def _write_result(stage_path: Path, result: CollectionResult) -> bytes:
        serialized = result.serialize_json()
        with open_verified_directory(stage_path, require_private=True) as descriptor:
            publish_immutable_text(
                descriptor,
                _RESULT_FILENAME,
                serialized,
                stage_path / _RESULT_FILENAME,
                maximum_size=_MAXIMUM_RECORD_SIZE,
            )
            sync_directory(descriptor)
        return serialized.encode("utf-8")


class _BatchCursor:
    def __init__(self, first_batch: object, remaining: Generator[object, None, None]) -> None:
        self._batch = first_batch
        self._remaining = remaining
        self._offset = 0

    def write_records(self, writer: CollectionPartitionWriter, record_count: int) -> None:
        remaining = record_count
        while remaining:
            available = self._batch.num_rows - self._offset
            if available == 0:
                self._advance()
                continue
            size = min(remaining, available)
            writer.write(self._batch.slice(self._offset, size))
            self._offset += size
            remaining -= size

    def has_remaining_records(self) -> bool:
        if self._offset < self._batch.num_rows:
            return True
        return next(self._remaining, None) is not None

    def _advance(self) -> None:
        next_batch = next(self._remaining, None)
        if next_batch is None:
            raise OSError("collection inputs contain fewer rows than declared")
        self._batch = next_batch
        self._offset = 0


class _BoundOutput:
    def __init__(self, path: Path, output: BinaryIO | TextIO) -> None:
        self.path = path
        self._output = output

    def describe(self, relative_path: str, record_count: int) -> CollectedOutputFile:
        _sync_output(self._output)
        return _describe_open_output(self._output.fileno(), self.path, relative_path, record_count)


class _ParquetWriter:
    def __init__(self, writer: _ArrowBatchWriter, output: _BoundOutput) -> None:
        self._writer = writer
        self._output = output
        self._closed = False

    def write(self, batch: object) -> None:
        self._writer.write_batch(batch.replace_schema_metadata(None))

    def finish(self, relative_path: str, record_count: int) -> CollectedOutputFile:
        self.close()
        return self._output.describe(relative_path, record_count)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._writer.close()


class _TextWriter:
    def __init__(self, output: TextIO, output_format: str, bound_output: _BoundOutput) -> None:
        self._output = output
        self._bound_output = bound_output
        self._format = output_format
        self._is_first = True

    def write(self, batch: object) -> None:
        frame = batch.to_pandas()
        if self._format == "csv":
            frame.to_csv(self._output, header=self._is_first, index=False)
        else:
            content = frame.to_json(orient="records", lines=True, force_ascii=False, date_format="iso")
            if content:
                self._output.write(content)
        self._is_first = False

    def finish(self, relative_path: str, record_count: int) -> CollectedOutputFile:
        return self._bound_output.describe(relative_path, record_count)


def _open_partition_writer(
    output_path: Path,
    output_format: str,
    schema: object,
) -> AbstractContextManager[CollectionPartitionWriter]:
    if output_format == "parquet":
        return _open_parquet_writer(output_path, schema)
    if output_format in {"csv", "jsonl"}:
        return _open_text_writer(output_path, output_format)
    raise ValueError(f"unsupported collection output format {output_format!r}")


@contextmanager
def _open_parquet_writer(output_path: Path, schema: object) -> Generator[CollectionPartitionWriter, None, None]:
    with output_path.open("x+b") as output:
        os.fchmod(output.fileno(), 0o600)
        partition_writer = _ParquetWriter(lazy.pq.ParquetWriter(output, schema), _BoundOutput(output_path, output))
        try:
            yield partition_writer
        finally:
            partition_writer.close()
            _sync_output(output)


@contextmanager
def _open_text_writer(output_path: Path, output_format: str) -> Generator[CollectionPartitionWriter, None, None]:
    with output_path.open("x+", encoding="utf-8", newline="") as output:
        os.fchmod(output.fileno(), 0o600)
        try:
            yield _TextWriter(output, output_format, _BoundOutput(output_path, output))
        finally:
            _sync_output(output)


def _sync_output(output: BinaryIO | TextIO) -> None:
    output.flush()
    os.fsync(output.fileno())


def _partition_record_counts(record_count: int, partition_count: int) -> tuple[int, ...]:
    floor_count = record_count // partition_count
    return tuple(
        record_count - floor_count * (partition_count - 1) if index == partition_count - 1 else floor_count
        for index in range(partition_count)
    )


def _describe_open_output(
    descriptor: int,
    output_path: Path,
    relative_path: str,
    record_count: int,
) -> CollectedOutputFile:
    before = os.fstat(descriptor)
    _require_safe_output(before, output_path)
    digest = hashlib.sha256()
    offset = 0
    while block := os.pread(descriptor, 1024 * 1024, offset):
        digest.update(block)
        offset += len(block)
    after = os.fstat(descriptor)
    path_status = output_path.lstat()
    if get_file_facts(before) != get_file_facts(after) or get_file_facts(after) != get_file_facts(path_status):
        raise OSError(f"collection output {output_path} changed while it was being described")
    _require_safe_output(after, output_path)
    return CollectedOutputFile(
        relative_path=relative_path,
        sha256=digest.hexdigest(),
        byte_size=after.st_size,
        record_count=record_count,
        modified_at_ns=after.st_mtime_ns,
        changed_at_ns=after.st_ctime_ns,
    )


def _require_safe_output(status: os.stat_result, output_path: Path) -> None:
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1 or status.st_mode & 0o077:
        raise OSError(f"collection output {output_path} is not a private single-link regular file")


def _stage_expectations(
    files: tuple[CollectedOutputFile, ...],
    result: CollectionResult,
    result_bytes: bytes,
) -> tuple[StagedFile, ...]:
    return (
        *(StagedFile(file.relative_path, file.sha256, file.byte_size) for file in files),
        StagedFile(_RESULT_FILENAME, result.compute_sha256(), len(result_bytes)),
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = ["CollectionMerger", "CollectionPartitionWriter"]
