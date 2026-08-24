# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest
from slurm_test_fakes import (
    FakeClock,
    FakeCommandResponse,
    FakeLogicalEndpoint,
    FakeSlurmArray,
    FakeSlurmRunner,
    FakeSlurmTask,
    FakeVllmBackend,
)

from data_designer.slurm.client import ClientResult
from data_designer.slurm.config import DataDesignerSlurmConfig, SlurmProfileCatalog
from data_designer.slurm.planning import ResolvedDependencyLock, ResolvedSlurmRunPlan
from data_designer.slurm.state import (
    AttemptManifest,
    AttemptReadiness,
    CandidateOutputManifest,
    RunManifest,
    SchedulerIdentity,
    ShardManifest,
    ShardWinner,
)

TEST_DIRECTORY = Path(__file__).parent
CONTRACT_GOLDEN_DIRECTORY = TEST_DIRECTORY / "contracts" / "golden"
INTEGRATION_GOLDEN_PATH = TEST_DIRECTORY / "integration" / "golden" / "finalization_chain.json"
SLURM_GOLDEN_DIRECTORY = TEST_DIRECTORY / "slurm_test_fakes" / "golden" / "slurm"


@pytest.fixture
def authored_run() -> DataDesignerSlurmConfig:
    return DataDesignerSlurmConfig.model_validate_json((CONTRACT_GOLDEN_DIRECTORY / "authored_run.json").read_text())


@pytest.fixture
def authored_run_single() -> DataDesignerSlurmConfig:
    return DataDesignerSlurmConfig.model_validate_json(
        (CONTRACT_GOLDEN_DIRECTORY / "authored_run_single.json").read_text()
    )


@pytest.fixture
def profile_catalog() -> SlurmProfileCatalog:
    return SlurmProfileCatalog.model_validate_json((CONTRACT_GOLDEN_DIRECTORY / "profile_catalog.json").read_text())


@pytest.fixture
def dependency_lock() -> ResolvedDependencyLock:
    return ResolvedDependencyLock.model_validate_json((CONTRACT_GOLDEN_DIRECTORY / "dependency_lock.json").read_text())


@pytest.fixture
def dependency_lock_single() -> ResolvedDependencyLock:
    return ResolvedDependencyLock.model_validate_json(
        (CONTRACT_GOLDEN_DIRECTORY / "dependency_lock_single.json").read_text()
    )


@pytest.fixture
def single_node_plan() -> ResolvedSlurmRunPlan:
    return ResolvedSlurmRunPlan.model_validate_json((CONTRACT_GOLDEN_DIRECTORY / "single_node_plan.json").read_text())


@pytest.fixture
def multi_node_plan() -> ResolvedSlurmRunPlan:
    return ResolvedSlurmRunPlan.model_validate_json((CONTRACT_GOLDEN_DIRECTORY / "multi_node_plan.json").read_text())


@pytest.fixture
def single_node_render_plan(single_node_plan: ResolvedSlurmRunPlan) -> ResolvedSlurmRunPlan:
    return single_node_plan.model_copy(
        update={"submission": single_node_plan.submission.model_copy(update={"account": None, "partition": None})}
    )


@pytest.fixture
def multi_node_render_plan(multi_node_plan: ResolvedSlurmRunPlan) -> ResolvedSlurmRunPlan:
    return multi_node_plan.model_copy(
        update={"submission": multi_node_plan.submission.model_copy(update={"account": None, "partition": None})}
    )


@pytest.fixture
def client_result() -> ClientResult:
    return ClientResult.model_validate_json((CONTRACT_GOLDEN_DIRECTORY / "client_result.json").read_text())


@pytest.fixture
def finalization_chain_payload() -> dict[str, object]:
    return cast(dict[str, object], json.loads(INTEGRATION_GOLDEN_PATH.read_text()))


@pytest.fixture
def run_manifest(finalization_chain_payload: dict[str, object]) -> RunManifest:
    return RunManifest.model_validate_json(json.dumps(finalization_chain_payload["run"]))


@pytest.fixture
def shard_manifests(finalization_chain_payload: dict[str, object]) -> tuple[ShardManifest, ...]:
    return tuple(
        ShardManifest.model_validate_json(json.dumps(payload))
        for payload in cast(list[object], finalization_chain_payload["shards"])
    )


@pytest.fixture
def attempt_manifest(finalization_chain_payload: dict[str, object]) -> AttemptManifest:
    return AttemptManifest.model_validate_json(json.dumps(finalization_chain_payload["attempt"]))


@pytest.fixture
def attempt_readiness(finalization_chain_payload: dict[str, object]) -> AttemptReadiness:
    return AttemptReadiness.model_validate_json(json.dumps(finalization_chain_payload["readiness"]))


@pytest.fixture
def finalization_client_result(finalization_chain_payload: dict[str, object]) -> ClientResult:
    return ClientResult.model_validate_json(json.dumps(finalization_chain_payload["client_result"]))


@pytest.fixture
def candidate_output_manifest(finalization_chain_payload: dict[str, object]) -> CandidateOutputManifest:
    return CandidateOutputManifest.model_validate_json(json.dumps(finalization_chain_payload["candidate"]))


@pytest.fixture
def shard_winner(finalization_chain_payload: dict[str, object]) -> ShardWinner:
    return ShardWinner.model_validate_json(json.dumps(finalization_chain_payload["winner"]))


@pytest.fixture
def fake_clock() -> FakeClock:
    """Return an isolated explicitly controlled clock."""
    return FakeClock(datetime(2026, 8, 18, 12, tzinfo=timezone.utc), monotonic_time=100.0)


@pytest.fixture
def fake_slurm_array() -> FakeSlurmArray:
    """Return a two-task array copied when the fake runner is constructed."""
    return FakeSlurmArray(
        tasks=(
            FakeSlurmTask(SchedulerIdentity(array_job_id=4101, array_task_id=0)),
            FakeSlurmTask(
                SchedulerIdentity(array_job_id=4101, array_task_id=1),
                queue_state="RUNNING",
            ),
        )
    )


@pytest.fixture
def fake_slurm_runner(fake_slurm_array: FakeSlurmArray) -> FakeSlurmRunner:
    """Return an isolated Slurm runner with one array and one bounded sinfo query."""
    return FakeSlurmRunner(
        arrays=(fake_slurm_array,),
        sinfo_responses={
            ("sinfo", "--noheader", "--format=%G"): FakeCommandResponse(
                stdout=(SLURM_GOLDEN_DIRECTORY / "sinfo_gres.txt").read_text()
            )
        },
    )


@pytest.fixture
def fake_plugin_overlay() -> Path:
    """Return the installed-layout fake Data Designer plugin overlay."""
    return TEST_DIRECTORY / "fixtures" / "fake_plugin_overlay"


@pytest.fixture
def fake_logical_endpoint() -> FakeLogicalEndpoint:
    """Return an isolated two-backend logical endpoint."""
    return FakeLogicalEndpoint(
        "http://127.0.0.1:31000",
        (
            FakeVllmBackend("http://127.0.0.1:31001", rank=0),
            FakeVllmBackend("http://127.0.0.1:31002", rank=1),
        ),
    )
