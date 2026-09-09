# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from data_designer.slurm.config import DataDesignerSlurmConfig, SlurmProfile
from data_designer.slurm.contracts import ArtifactReference, compute_canonical_json_sha256
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.runtime.context import load_allocation_context
from data_designer.slurm.state import (
    AttemptLifecycleState,
    AttemptManifest,
    RunManifest,
    SchedulerIdentity,
    ShardManifest,
    SlurmStateWriter,
)


def test_allocation_context_reads_and_updates_state_through_remapped_workspace(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    physical_workspace = tmp_path / "workspace"
    physical_workspace.mkdir()
    logical_workspace = single_node_plan.selected_profile.profile.workspace_root
    logical_attempts = (
        Path(logical_workspace)
        / "runs"
        / single_node_plan.run_id
        / "shards"
        / single_node_plan.shards[0].shard_id
        / "attempts"
    )
    fast_attempts = tmp_path / "fast-attempts"
    payload = cast(dict[str, object], json.loads(single_node_plan.serialize_json()))
    selected = cast(dict[str, object], payload["selected_profile"])
    profile_payload = cast(dict[str, object], selected["profile"])
    mounts = [
        {"source": logical_workspace, "target": physical_workspace.as_posix(), "read_only": False},
        {"source": logical_attempts.as_posix(), "target": fast_attempts.as_posix(), "read_only": False},
    ]
    profile_payload["container_mounts"] = mounts
    payload["container_mounts"] = mounts
    profile = SlurmProfile.model_validate(profile_payload)
    selected["profile_sha256"] = compute_canonical_json_sha256(profile.model_dump(mode="json"))
    plan = ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))
    created_at = datetime(2026, 9, 9, tzinfo=timezone.utc)
    plan_reference = ArtifactReference(
        path=f"{logical_workspace}/runs/{plan.run_id}/resolved-plan.json",
        sha256=plan.compute_sha256(),
    )
    run = RunManifest(
        schema_version=1,
        run_id=plan.run_id,
        created_at=created_at,
        authored_config=plan.authored_config,
        resolved_plan=plan_reference,
        shard_count=1,
    )
    shard = ShardManifest(
        schema_version=1,
        run_id=plan.run_id,
        shard_id=plan.shards[0].shard_id,
        shard_index=0,
        record_range=plan.shards[0].record_range,
        input_partition=plan.shards[0].input_partition,
        resume_workspace=plan.shards[0].resume_workspace,
        created_at=created_at,
    )
    scheduler = SchedulerIdentity(array_job_id=4101, array_task_id=0)
    attempt = AttemptManifest(
        schema_version=1,
        run_id=plan.run_id,
        shard_id=shard.shard_id,
        attempt_id="attempt-0001",
        attempt_ordinal=1,
        resolved_plan=plan_reference,
        state=AttemptLifecycleState.SUBMITTED,
        scheduler=scheduler,
        created_at=created_at,
        updated_at=created_at,
    )
    host_writer = SlurmStateWriter(
        physical_workspace,
        plan.run_id,
        logical_workspace_root=logical_workspace,
    )
    host_writer.initialize_run(authored_run_single, plan, run, (shard,))
    host_writer.create_attempt(attempt)
    plan_path = physical_workspace / "runs" / plan.run_id / "resolved-plan.json"
    attempt_directory = fast_attempts / attempt.attempt_id
    attempt_directory.mkdir(parents=True, mode=0o700)

    context, runtime_writer = load_allocation_context(
        plan_path,
        attempt_directory,
        {"SLURM_ARRAY_TASK_ID": "0", "SLURM_ARRAY_JOB_ID": "4101"},
    )
    runtime_writer.update_attempt(context.attempt.model_copy(update={"state": AttemptLifecycleState.RUNNING}))
    with runtime_writer.acquire_dataset_workspace(shard.shard_id, attempt.attempt_id, "never") as dataset_path:
        assert dataset_path == attempt_directory / "dataset"

    assert context.attempt_directory.as_posix().startswith(logical_workspace)
    assert host_writer.load_attempt(shard.shard_id, attempt.attempt_id).state is AttemptLifecycleState.RUNNING
