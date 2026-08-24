# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable benchmark records for Data Designer Slurm."""

from __future__ import annotations

from data_designer.slurm.benchmark.records import (
    BenchmarkCaseResult,
    BenchmarkChildRun,
    BenchmarkManifest,
    BenchmarkOutcome,
    BenchmarkRecommendation,
    BenchmarkRecommendationKind,
    BenchmarkReport,
)

__all__ = [
    "BenchmarkCaseResult",
    "BenchmarkChildRun",
    "BenchmarkManifest",
    "BenchmarkOutcome",
    "BenchmarkRecommendation",
    "BenchmarkRecommendationKind",
    "BenchmarkReport",
]
