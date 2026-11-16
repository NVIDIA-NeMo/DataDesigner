# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable benchmark metadata beneath the selected Slurm workspace."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic import TypeAdapter, ValidationError

from data_designer.slurm.benchmark.records import BenchmarkChildRun, BenchmarkManifest, BenchmarkReport
from data_designer.slurm.config import DataDesignerSlurmBenchmarkConfig, DataDesignerSlurmConfig
from data_designer.slurm.contracts import ArtifactReference, ContractRecord, Identifier, validate_absolute_path
from data_designer.slurm.images.filesystem import ensure_private_directory
from data_designer.slurm.state.filesystem import open_verified_directory, publish_immutable_text, read_regular_text

_CONFIG_FILENAME = "config.json"
_BASE_RUN_FILENAME = "base-run.json"
_MANIFEST_FILENAME = "benchmark.json"
_REPORTS_DIRECTORY = "reports"
_MAXIMUM_RECORD_SIZE = 16 * 1024 * 1024
_IDENTIFIER_ADAPTER = TypeAdapter(Identifier)
_RecordT = TypeVar("_RecordT", bound=ContractRecord)


class BenchmarkStoreError(Exception):
    """Base error for benchmark metadata persistence."""


class BenchmarkNotFoundError(BenchmarkStoreError):
    """A benchmark metadata record is missing."""


class BenchmarkConflictError(BenchmarkStoreError):
    """Benchmark metadata is unsafe, corrupt, or conflicts with immutable state."""


class BenchmarkStore:
    """Persist one benchmark config, manifest, and point-in-time reports."""

    def __init__(self, workspace_root: str | Path, benchmark_id: Identifier) -> None:
        try:
            root = Path(validate_absolute_path(Path(workspace_root).as_posix()))
            normalized_id = _IDENTIFIER_ADAPTER.validate_python(benchmark_id, strict=True)
        except (ValidationError, ValueError) as error:
            raise BenchmarkStoreError("invalid benchmark location") from error
        self.workspace_root = root
        self.benchmark_id = normalized_id
        self.benchmarks_root = root / "benchmarks"
        self.benchmark_root = self.benchmarks_root / normalized_id
        self.config_path = self.benchmark_root / _CONFIG_FILENAME
        self.base_run_path = self.benchmark_root / _BASE_RUN_FILENAME
        self.manifest_path = self.benchmark_root / _MANIFEST_FILENAME
        self.reports_root = self.benchmark_root / _REPORTS_DIRECTORY

    def build_manifest(
        self,
        config: DataDesignerSlurmBenchmarkConfig,
        children: tuple[BenchmarkChildRun, ...],
    ) -> BenchmarkManifest:
        """Build a manifest bound to this store's immutable locations."""
        return BenchmarkManifest(
            schema_version=1,
            benchmark_id=self.benchmark_id,
            benchmark_config=ArtifactReference(path=self.config_path.as_posix(), sha256=config.compute_sha256()),
            children=children,
        )

    def publish(
        self,
        config: DataDesignerSlurmBenchmarkConfig,
        base_run: DataDesignerSlurmConfig,
        manifest: BenchmarkManifest,
    ) -> BenchmarkManifest:
        """Convergently publish resolved inputs then the manifest commit record."""
        self._validate_manifest(config, manifest)
        try:
            ensure_private_directory(self.benchmark_root, parents=True)
            with open_verified_directory(self.benchmark_root, require_private=True) as descriptor:
                publish_immutable_text(
                    descriptor,
                    _BASE_RUN_FILENAME,
                    base_run.serialize_json(),
                    self.base_run_path,
                    maximum_size=_MAXIMUM_RECORD_SIZE,
                )
                publish_immutable_text(
                    descriptor,
                    _CONFIG_FILENAME,
                    config.serialize_json(),
                    self.config_path,
                    maximum_size=_MAXIMUM_RECORD_SIZE,
                )
                publish_immutable_text(
                    descriptor,
                    _MANIFEST_FILENAME,
                    manifest.serialize_json(),
                    self.manifest_path,
                    maximum_size=_MAXIMUM_RECORD_SIZE,
                )
            return manifest
        except FileExistsError as error:
            raise BenchmarkConflictError("benchmark already contains different immutable metadata") from error
        except OSError as error:
            raise BenchmarkStoreError("benchmark metadata cannot be persisted") from error

    def load(self) -> tuple[DataDesignerSlurmBenchmarkConfig, DataDesignerSlurmConfig, BenchmarkManifest]:
        """Load and verify the committed benchmark metadata."""
        try:
            with open_verified_directory(self.benchmark_root, require_private=True) as descriptor:
                config = self._read_record(
                    descriptor,
                    _CONFIG_FILENAME,
                    self.config_path,
                    DataDesignerSlurmBenchmarkConfig,
                )
                base_run = self._read_record(
                    descriptor,
                    _BASE_RUN_FILENAME,
                    self.base_run_path,
                    DataDesignerSlurmConfig,
                )
                manifest = self._read_record(
                    descriptor,
                    _MANIFEST_FILENAME,
                    self.manifest_path,
                    BenchmarkManifest,
                )
            self._validate_manifest(config, manifest)
            return config, base_run, manifest
        except FileNotFoundError as error:
            raise BenchmarkNotFoundError(f"benchmark {self.benchmark_id!r} is not initialized") from error
        except BenchmarkConflictError:
            raise
        except (OSError, ValidationError, ValueError) as error:
            raise BenchmarkConflictError(f"benchmark {self.benchmark_id!r} contains invalid metadata") from error

    def publish_report(self, report: BenchmarkReport) -> Path:
        """Persist one immutable point-in-time report."""
        if report.benchmark_id != self.benchmark_id:
            raise BenchmarkConflictError("benchmark report identity does not match its store")
        _, _, manifest = self.load()
        expected_reference = ArtifactReference(path=self.manifest_path.as_posix(), sha256=manifest.compute_sha256())
        if report.benchmark_manifest != expected_reference:
            raise BenchmarkConflictError("benchmark report does not bind the persisted manifest")
        report_root = self.reports_root / report.analysis_id
        report_path = report_root / "report.json"
        try:
            ensure_private_directory(self.reports_root, parents=False)
            ensure_private_directory(report_root, parents=False)
            with open_verified_directory(report_root, require_private=True) as descriptor:
                publish_immutable_text(
                    descriptor,
                    "report.json",
                    report.serialize_json(),
                    report_path,
                    maximum_size=_MAXIMUM_RECORD_SIZE,
                )
            return report_path
        except FileExistsError as error:
            raise BenchmarkConflictError("benchmark report identity already contains different bytes") from error
        except OSError as error:
            raise BenchmarkStoreError("benchmark report cannot be persisted") from error

    def manifest_reference(self, manifest: BenchmarkManifest) -> ArtifactReference:
        """Return the immutable reference for a verified manifest."""
        return ArtifactReference(path=self.manifest_path.as_posix(), sha256=manifest.compute_sha256())

    def _validate_manifest(
        self,
        config: DataDesignerSlurmBenchmarkConfig,
        manifest: BenchmarkManifest,
    ) -> None:
        expected = ArtifactReference(path=self.config_path.as_posix(), sha256=config.compute_sha256())
        if manifest.benchmark_id != self.benchmark_id or manifest.benchmark_config != expected:
            raise BenchmarkConflictError("benchmark manifest does not match its immutable config")
        for child in manifest.children:
            expected_path = self.workspace_root / "runs" / child.child_run_id / "authored-config.json"
            if child.child_authored_config.path != expected_path.as_posix():
                raise BenchmarkConflictError("benchmark child reference is outside its ordinary run state")

    @staticmethod
    def _read_record(
        directory_descriptor: int,
        name: str,
        path: Path,
        record_type: type[_RecordT],
    ) -> _RecordT:
        content = read_regular_text(
            directory_descriptor,
            name,
            path,
            maximum_size=_MAXIMUM_RECORD_SIZE,
        )
        record = record_type.model_validate_json(content)
        if record.serialize_json() != content:
            raise BenchmarkConflictError("benchmark metadata is not canonical")
        return record


__all__ = [
    "BenchmarkConflictError",
    "BenchmarkNotFoundError",
    "BenchmarkStore",
    "BenchmarkStoreError",
]
