# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

from conftest import RuntimeCase

from data_designer.slurm.runtime.bootstrap import RuntimeBootstrapManifest, build_runtime_manifest
from data_designer.slurm.runtime.models import RuntimeStepRole
from data_designer.slurm.state import RetryPlan, RetryShard


def test_bootstrap_manifest_builds_typed_one_node_steps_without_secret_values(runtime_case: RuntimeCase) -> None:
    context = runtime_case.context
    runtime_root = context.attempt_directory / "runtime"
    log_directory = context.attempt_directory / "logs/execution-00000002"

    manifest = build_runtime_manifest(
        context,
        {"SLURM_JOB_GPUS": "0"},
        runtime_root=runtime_root,
        log_directory=log_directory,
    )
    reloaded = RuntimeBootstrapManifest.model_validate_json(manifest.serialize_json())

    assert reloaded == manifest
    assert [step.role for step in manifest.steps] == [
        RuntimeStepRole.CLIENT_PREFLIGHT,
        RuntimeStepRole.SERVER,
        RuntimeStepRole.ENDPOINT,
        RuntimeStepRole.CLIENT,
    ]
    assert all(step.command[0] != "srun" for step in manifest.steps)
    assert all(step.stdout_path.startswith(context.attempt_directory.as_posix()) for step in manifest.steps)
    assert manifest.steps[-1].command[:4] == (
        "python3",
        "-m",
        "data_designer.slurm.runtime.entrypoint",
        "client",
    )
    assert "--shard-id" not in manifest.steps[-1].command
    assert "--attempt-id" not in manifest.steps[-1].command
    assert "--plan" in manifest.steps[-1].command
    assert "--attempt-dir" in manifest.steps[-1].command


def test_bootstrap_manifest_binds_retry_plan_to_control_and_client_workers(runtime_case: RuntimeCase) -> None:
    context = runtime_case.context
    retry = RetryPlan(
        schema_version=1,
        retry_id="retry-0001",
        run_id=context.plan.run_id,
        created_at=runtime_case.created_at,
        resolved_plan=context.attempt.resolved_plan,
        planned_shards=(
            RetryShard(
                shard_id=context.shard.shard_id,
                attempt_id=context.attempt.attempt_id,
                attempt_ordinal=context.attempt.attempt_ordinal,
                array_task_index=context.shard.array_task_index,
            ),
        ),
        effective_resume_mode="never",
    )
    retry_context = replace(context, retry_plan=retry)

    manifest = build_runtime_manifest(
        retry_context,
        {"SLURM_JOB_GPUS": "0"},
        runtime_root=context.attempt_directory / "runtime",
        log_directory=context.attempt_directory / "logs/execution-00000002",
    )

    preflight = manifest.steps[0].command
    client = manifest.steps[-1].command
    assert ("--resume-mode", "never") == preflight[preflight.index("--resume-mode") :][:2]
    assert ("--retry-id", retry.retry_id) == client[client.index("--retry-id") :][:2]
    assert ("--retry-plan-sha256", retry.compute_sha256()) == client[client.index("--retry-plan-sha256") :][:2]
    assert ("--effective-resume-mode", "never") == client[client.index("--effective-resume-mode") :][:2]
    assert "--resume-mode" not in client
