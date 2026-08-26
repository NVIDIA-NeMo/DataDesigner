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
    invalid_request,
    invoke_backend,
)

_IDENTIFIER_ADAPTER = TypeAdapter(Identifier)


class BenchmarkBackend(Protocol):
    """Execute and analyze benchmarks without exposing implementation types."""

    def run(self, config: DataDesignerSlurmBenchmarkConfig) -> BenchmarkManifest:
        """Persist and start the ordinary child runs for one benchmark."""

    def analyze(self, benchmark_id: Identifier, *, refresh_state: bool = False) -> BenchmarkReport:
        """Return one point-in-time benchmark report."""


class SlurmBenchmarkService:
    """Expose benchmark operations through an injected benchmark implementation."""

    def __init__(self, backend: BenchmarkBackend) -> None:
        self._backend = backend

    def run(self, config: DataDesignerSlurmBenchmarkConfig) -> BenchmarkManifest:
        """Start all ordinary child runs and return their immutable mapping."""
        operation = SlurmServiceOperation.RUN_BENCHMARK
        if not isinstance(config, DataDesignerSlurmBenchmarkConfig):
            raise invalid_request(operation, "config must be a DataDesignerSlurmBenchmarkConfig")

        def run_benchmark() -> BenchmarkManifest:
            manifest = self._backend.run(config)
            if not isinstance(manifest, BenchmarkManifest):
                raise TypeError("benchmark backend returned an invalid manifest")
            return manifest

        return invoke_backend(operation, run_benchmark)

    def analyze(self, benchmark_id: Identifier, *, refresh_state: bool = False) -> BenchmarkReport:
        """Analyze one persisted benchmark without a resident monitor."""
        operation = SlurmServiceOperation.ANALYZE_BENCHMARK
        try:
            validated_id = _IDENTIFIER_ADAPTER.validate_python(benchmark_id, strict=True)
        except ValidationError:
            raise invalid_request(operation, "benchmark_id must be a valid identifier") from None
        if type(refresh_state) is not bool:
            raise invalid_request(operation, "refresh_state must be a boolean")

        def analyze_benchmark() -> BenchmarkReport:
            report = self._backend.analyze(validated_id, refresh_state=refresh_state)
            if not isinstance(report, BenchmarkReport):
                raise TypeError("benchmark backend returned an invalid report")
            return report

        return invoke_backend(operation, analyze_benchmark)
