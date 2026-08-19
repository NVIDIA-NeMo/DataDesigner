# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from data_designer.slurm.config import DataDesignerSlurmConfig, SlurmProfileCatalog
from data_designer.slurm.planning import ResolvedDependencyLock, ResolvedSlurmRunPlan

GOLDEN_DIR = Path(__file__).parent / "golden"


@pytest.fixture
def authored_run() -> DataDesignerSlurmConfig:
    return DataDesignerSlurmConfig.model_validate_json((GOLDEN_DIR / "authored_run.json").read_text())


@pytest.fixture
def authored_run_single() -> DataDesignerSlurmConfig:
    return DataDesignerSlurmConfig.model_validate_json((GOLDEN_DIR / "authored_run_single.json").read_text())


@pytest.fixture
def profile_catalog() -> SlurmProfileCatalog:
    return SlurmProfileCatalog.model_validate_json((GOLDEN_DIR / "profile_catalog.json").read_text())


@pytest.fixture
def dependency_lock() -> ResolvedDependencyLock:
    return ResolvedDependencyLock.model_validate_json((GOLDEN_DIR / "dependency_lock.json").read_text())


@pytest.fixture
def dependency_lock_single() -> ResolvedDependencyLock:
    return ResolvedDependencyLock.model_validate_json((GOLDEN_DIR / "dependency_lock_single.json").read_text())


@pytest.fixture
def single_node_plan() -> ResolvedSlurmRunPlan:
    return ResolvedSlurmRunPlan.model_validate_json((GOLDEN_DIR / "single_node_plan.json").read_text())


@pytest.fixture
def multi_node_plan() -> ResolvedSlurmRunPlan:
    return ResolvedSlurmRunPlan.model_validate_json((GOLDEN_DIR / "multi_node_plan.json").read_text())
