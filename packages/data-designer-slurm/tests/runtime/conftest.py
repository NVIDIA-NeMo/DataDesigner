# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest

from data_designer.slurm.client import ClientResult
from data_designer.slurm.config import SlurmProfile
from data_designer.slurm.contracts import ArtifactReference, compute_canonical_json_sha256
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.runtime.models import AllocationContext, RuntimeEndpoint, RuntimeStep, RuntimeStepRole
from data_designer.slurm.runtime.preflight import AllocationLayout
from data_designer.slurm.state import (
    AttemptLifecycleState,
    AttemptManifest,
    AttemptReadiness,
    CandidateOutputManifest,
    SchedulerIdentity,
    StateNotFoundError,
    validate_attempt_transition,
    validate_readiness_transition,
)


@dataclass(slots=True)
class RuntimeCase:
    workspace: Path
    context: AllocationContext
    created_at: datetime


@dataclass(slots=True)
class FakeStateStore:
    attempt: AttemptManifest
    readiness: list[AttemptReadiness] = field(default_factory=list)

    def update_attempt(self, attempt: AttemptManifest) -> AttemptManifest:
        validate_attempt_transition(self.attempt, attempt)
        self.attempt = attempt
        return attempt

    def publish_attempt_result(
        self,
        client_result: ClientResult,
        candidate: CandidateOutputManifest,
    ) -> tuple[ClientResult, CandidateOutputManifest]:
        reference = client_result.candidate_output_manifest
        assert reference is not None
        assert candidate.compute_sha256() == reference.sha256
        assert self.attempt.state is AttemptLifecycleState.RUNNING
        bound_attempt = self.attempt.model_copy(update={"candidate_output": reference})
        validate_attempt_transition(self.attempt, bound_attempt)
        self.attempt = bound_attempt
        return client_result, candidate

    def write_readiness(self, readiness: AttemptReadiness) -> AttemptReadiness:
        if self.readiness:
            validate_readiness_transition(self.readiness[-1], readiness)
        else:
            assert readiness.revision == 1
        self.readiness.append(readiness)
        return readiness

    def load_readiness(self, shard_id: str, attempt_id: str) -> AttemptReadiness:
        if not self.readiness:
            raise StateNotFoundError(f"attempt {attempt_id!r} has no readiness snapshot")
        readiness = self.readiness[-1]
        assert readiness.shard_id == shard_id
        assert readiness.attempt_id == attempt_id
        return readiness


class FakePreflight:
    def __init__(self, failure: BaseException | None = None) -> None:
        self.failure = failure
        self.calls = 0

    def verify(self, context: AllocationContext, environment: object) -> AllocationLayout:
        del environment
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        node_count = max(index for deployment in context.plan.deployments for index in deployment.node_indices) + 1
        return AllocationLayout(tuple(f"node-{index}" for index in range(node_count)))


class FakeClientStepBuilder:
    def build_preflight_step(
        self,
        plan: ResolvedSlurmRunPlan,
        shard: object,
        attempt: object,
        attempt_directory: Path,
        endpoints: tuple[RuntimeEndpoint, ...],
        source_environment: object,
    ) -> RuntimeStep:
        del plan, shard, attempt, endpoints, source_environment
        return _step("client-preflight", RuntimeStepRole.CLIENT_PREFLIGHT, attempt_directory)

    def build_generation_step(
        self,
        plan: ResolvedSlurmRunPlan,
        shard: object,
        attempt: object,
        attempt_directory: Path,
        endpoints: tuple[RuntimeEndpoint, ...],
        source_environment: object,
    ) -> RuntimeStep:
        del plan, shard, attempt, endpoints, source_environment
        return _step("client-generation", RuntimeStepRole.CLIENT, attempt_directory)


@pytest.fixture
def runtime_case(tmp_path: Path, single_node_plan: ResolvedSlurmRunPlan) -> RuntimeCase:
    return _build_runtime_case(tmp_path, single_node_plan)


@pytest.fixture
def multi_node_runtime_case(tmp_path: Path, multi_node_plan: ResolvedSlurmRunPlan) -> RuntimeCase:
    return _build_runtime_case(tmp_path, multi_node_plan)


def _build_runtime_case(tmp_path: Path, source_plan: ResolvedSlurmRunPlan) -> RuntimeCase:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    plan = relocate_plan(source_plan, workspace)
    created_at = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)
    plan_reference = ArtifactReference(
        path=Path(plan.authored_config.path).with_name("resolved-plan.json").as_posix(),
        sha256=plan.compute_sha256(),
    )
    attempt = AttemptManifest(
        schema_version=1,
        run_id=plan.run_id,
        shard_id=plan.shards[0].shard_id,
        attempt_id="attempt-0001",
        attempt_ordinal=1,
        resolved_plan=plan_reference,
        state=AttemptLifecycleState.SUBMITTED,
        scheduler=SchedulerIdentity(array_job_id=4101, array_task_id=0),
        created_at=created_at,
        updated_at=created_at + timedelta(seconds=1),
    )
    attempt_directory = (
        Path(plan.authored_config.path).parent / "shards" / attempt.shard_id / "attempts" / attempt.attempt_id
    )
    attempt_directory.mkdir(parents=True, mode=0o700)
    return RuntimeCase(
        workspace=workspace,
        context=AllocationContext(
            plan=plan,
            shard=plan.shards[0],
            attempt=attempt,
            attempt_directory=attempt_directory,
        ),
        created_at=created_at,
    )


def relocate_plan(plan: ResolvedSlurmRunPlan, workspace: Path) -> ResolvedSlurmRunPlan:
    previous_workspace = plan.selected_profile.profile.workspace_root
    payload = cast(
        dict[str, object],
        json.loads(plan.serialize_json().replace(previous_workspace, workspace.as_posix())),
    )
    selected_profile = cast(dict[str, object], payload["selected_profile"])
    profile_payload = cast(dict[str, object], selected_profile["profile"])
    relocated_mount = {
        "source": workspace.as_posix(),
        "target": previous_workspace,
        "read_only": False,
    }
    profile_payload["container_mounts"] = [relocated_mount]
    payload["container_mounts"] = [relocated_mount]
    profile = SlurmProfile.model_validate(profile_payload)
    selected_profile["profile_sha256"] = compute_canonical_json_sha256(profile.model_dump(mode="json"))
    return ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))


def _step(step_id: str, role: RuntimeStepRole, attempt_directory: Path) -> RuntimeStep:
    return RuntimeStep(
        step_id=step_id,
        role=role,
        command=("true",),
        environment={"PATH": "/usr/bin", "LC_ALL": "C"},
        stdout_path=attempt_directory / "logs" / f"{step_id}.out",
        stderr_path=attempt_directory / "logs" / f"{step_id}.err",
    )
