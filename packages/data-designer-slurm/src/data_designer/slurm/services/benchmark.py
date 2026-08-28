# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public benchmark facade for Data Designer Slurm."""

from __future__ import annotations

from typing import Protocol

from pydantic import TypeAdapter, ValidationError

from data_designer.slurm.benchmark import BenchmarkManifest, BenchmarkReport
from data_designer.slurm.config import DataDesignerSlurmBenchmarkConfig
from data_designer.slurm.contracts import Identifier
from data_designer.slurm.services.errors import (
    SlurmServiceOperation,
    _invoke_service_backend,
    _make_invalid_request_error,
)

_IDENTIFIER_ADAPTER = TypeAdapter(Identifier)


class _BenchmarkBackend(Protocol):
    """Process benchmarks; any normalized failure must be caller-safe."""

    def run(self, config: DataDesignerSlurmBenchmarkConfig) -> BenchmarkManifest:
        """Persist and start the ordinary child runs for one benchmark."""

    def analyze(self, benchmark_id: Identifier, *, refresh_state: bool = False) -> BenchmarkReport:
        """Return one point-in-time benchmark report."""


class SlurmBenchmarkService:
    """Expose benchmark operations through a package-owned boundary.

    The service borrows its injected dependency and does not manage its lifecycle.

    Args:
        backend: Package-owned benchmark execution and analysis boundary.
    """

    def __init__(self, backend: _BenchmarkBackend) -> None:
        self._backend = backend

    def run(self, config: DataDesignerSlurmBenchmarkConfig) -> BenchmarkManifest:
        """Start all ordinary child runs and return their immutable mapping.

        The returned manifest must reference the exact serialized benchmark config.

        Raises:
            SlurmServiceError: If the request is invalid or benchmark launch fails.
        """
        operation = SlurmServiceOperation.RUN_BENCHMARK
        if not isinstance(config, DataDesignerSlurmBenchmarkConfig):
            raise _make_invalid_request_error(operation, "config must be a DataDesignerSlurmBenchmarkConfig")

        def run_benchmark() -> BenchmarkManifest:
            manifest = self._backend.run(config)
            if not isinstance(manifest, BenchmarkManifest):
                raise TypeError("benchmark backend returned an invalid manifest")
            if manifest.benchmark_config.sha256 != config.compute_sha256():
                raise ValueError("benchmark manifest does not match the requested config")
            return manifest

        return _invoke_service_backend(operation, run_benchmark)

    def analyze(self, benchmark_id: Identifier, *, refresh_state: bool = False) -> BenchmarkReport:
        """Analyze one persisted benchmark without a resident monitor.

        Args:
            benchmark_id: Persisted benchmark identity to analyze.
            refresh_state: Request one point-in-time state refresh before analysis.

        Raises:
            SlurmServiceError: If the request is invalid or analysis fails.
        """
        operation = SlurmServiceOperation.ANALYZE_BENCHMARK
        try:
            validated_id = _IDENTIFIER_ADAPTER.validate_python(benchmark_id, strict=True)
        except ValidationError:
            raise _make_invalid_request_error(operation, "benchmark_id must be a valid identifier") from None
        if type(refresh_state) is not bool:
            raise _make_invalid_request_error(operation, "refresh_state must be a boolean")

        def analyze_benchmark() -> BenchmarkReport:
            report = self._backend.analyze(validated_id, refresh_state=refresh_state)
            if not isinstance(report, BenchmarkReport):
                raise TypeError("benchmark backend returned an invalid report")
            if report.benchmark_id != validated_id:
                raise ValueError("benchmark report does not match the requested benchmark")
            return report

        return _invoke_service_backend(operation, analyze_benchmark)
