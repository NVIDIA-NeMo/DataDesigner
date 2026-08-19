# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import BaseModel

from data_designer.slurm._contracts import AuthoredConfig, ContractRecord
from data_designer.slurm.benchmark import BenchmarkManifest, BenchmarkReport
from data_designer.slurm.client import ClientResult
from data_designer.slurm.config import (
    DataDesignerSlurmBenchmarkConfig,
    DataDesignerSlurmConfig,
    ImageInspectionRecord,
    SlurmProfileCatalog,
)
from data_designer.slurm.planning import ResolvedDependencyLock, ResolvedSlurmRunPlan

GOLDEN_DIR = Path(__file__).parent / "golden"


@pytest.mark.parametrize(
    ("record_type", "filename"),
    [
        (DataDesignerSlurmConfig, "authored_run.json"),
        (DataDesignerSlurmConfig, "authored_run_single.json"),
        (SlurmProfileCatalog, "profile_catalog.json"),
        (ImageInspectionRecord, "client_image_inspection.json"),
        (ImageInspectionRecord, "serving_image_inspection.json"),
        (ResolvedDependencyLock, "dependency_lock.json"),
        (ResolvedDependencyLock, "dependency_lock_single.json"),
        (ResolvedSlurmRunPlan, "single_node_plan.json"),
        (ResolvedSlurmRunPlan, "multi_node_plan.json"),
        (ClientResult, "client_result.json"),
        (DataDesignerSlurmBenchmarkConfig, "benchmark_config.json"),
        (BenchmarkManifest, "benchmark_manifest.json"),
        (BenchmarkReport, "benchmark_report.json"),
    ],
)
def test_golden_record_round_trip(record_type: type[BaseModel], filename: str) -> None:
    fixture = (GOLDEN_DIR / filename).read_text()
    record = record_type.model_validate_json(fixture)

    assert record_type.model_validate_json(record.model_dump_json()) == record
    if isinstance(record, (AuthoredConfig, ContractRecord)):
        assert record_type.model_validate_json(record.serialize_json()) == record
        assert record.compute_sha256() == hashlib.sha256(record.serialize_json().encode()).hexdigest()
    if isinstance(record, ContractRecord):
        assert record.serialize_json() == fixture


def test_golden_records_are_sanitized() -> None:
    contents = "\n".join(path.read_text().casefold() for path in GOLDEN_DIR.glob("*.json"))

    assert "nvidia" not in contents
    assert "secret-value" not in contents
    assert "lustre" not in contents


def test_canonical_serialization_ignores_mapping_order(authored_run: DataDesignerSlurmConfig) -> None:
    payload = authored_run.model_dump(mode="json")
    payload["invocation"]["model_concurrency"] = {"judge": 32, "generator": 64}
    reordered = DataDesignerSlurmConfig.model_validate(payload)

    assert authored_run.serialize_json() == reordered.serialize_json()
    assert authored_run.compute_sha256() == reordered.compute_sha256()
