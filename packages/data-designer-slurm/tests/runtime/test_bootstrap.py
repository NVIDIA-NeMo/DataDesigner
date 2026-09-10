# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from conftest import RuntimeCase, relocate_plan

from data_designer.slurm.contracts import ArtifactReference
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.runtime.bootstrap import RuntimeBootstrapManifest, build_runtime_manifest
from data_designer.slurm.runtime.models import AllocationContext, RuntimeStepRole
from data_designer.slurm.runtime.node_spec import decode_node_worker_spec
from data_designer.slurm.runtime.preflight import AllocationLayout
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
        layout=AllocationLayout(("compute-001",)),
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
    assert all(step.node_hosts == ("compute-001",) for step in manifest.steps)
    assert all(step.role is not RuntimeStepRole.SERVER_PREFLIGHT for step in manifest.steps)


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


def test_bootstrap_manifest_composes_multi_node_workers_and_remote_endpoints(
    runtime_case: RuntimeCase,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    deployments = tuple(
        deployment.model_copy(
            update={
                "authored": deployment.authored.model_copy(
                    update={
                        "model": f"/workspace/primary/models/model-{deployment_index}",
                        "served_model_name": deployment.served_model_name,
                    }
                ),
                "model": f"/workspace/primary/models/model-{deployment_index}",
            }
        )
        for deployment_index, deployment in enumerate(multi_node_plan.deployments)
    )
    context = _replace_plan(runtime_case, multi_node_plan.model_copy(update={"deployments": deployments}))
    layout = AllocationLayout(("compute-001", "compute-002", "compute-003"))

    manifest = build_runtime_manifest(
        context,
        {"SLURM_JOB_GPUS": "0,1,2,3,4,5,6,7"},
        runtime_root=context.attempt_directory / "runtime",
        log_directory=context.attempt_directory / "logs/execution-00000002",
        layout=layout,
    )

    distributed = next(step for step in manifest.steps if step.step_id == "deployment-00000-serve")
    preflight = next(step for step in manifest.steps if step.step_id == "deployment-00000-preflight")
    worker_spec = decode_node_worker_spec(distributed.command[-1])
    endpoint = next(step for step in manifest.steps if step.step_id == "deployment-00000-endpoint")
    remote_server = next(step for step in manifest.steps if step.step_id == "deployment-00001-replica-00000-rank-00000")
    remote_preflight = next(step for step in manifest.steps if step.step_id == "deployment-00001-preflight")

    assert distributed.node_hosts == ("compute-001", "compute-002")
    assert distributed.kill_on_bad_exit
    assert preflight.role is RuntimeStepRole.SERVER_PREFLIGHT
    assert tuple(node.host for node in worker_spec.nodes) == distributed.node_hosts
    assert "--master-addr" in worker_spec.nodes[0].processes[0].command
    assert "compute-001" in worker_spec.nodes[0].processes[0].command
    assert worker_spec.nodes[0].processes[0].command[2] == "/workspace/primary/models/model-0"
    assert "--headless" in worker_spec.nodes[1].processes[0].command
    assert tuple(probe.host for probe in distributed.readiness) == ("compute-001",)
    assert "http://compute-001:" in " ".join(endpoint.command)
    assert endpoint.node_hosts == ("compute-001",)
    assert remote_preflight.node_hosts == ("compute-003",)
    assert remote_server.node_hosts == ("compute-003",)
    assert remote_server.command[2] == "/workspace/primary/models/model-1"
    assert tuple(probe.host for probe in remote_server.readiness) == ("compute-003",)


def _replace_plan(runtime_case: RuntimeCase, source_plan: ResolvedSlurmRunPlan) -> AllocationContext:
    plan = relocate_plan(source_plan, runtime_case.workspace)
    shard = plan.shards[0]
    plan_path = Path(plan.authored_config.path).with_name("resolved-plan.json")
    attempt = runtime_case.context.attempt.model_copy(
        update={
            "run_id": plan.run_id,
            "shard_id": shard.shard_id,
            "resolved_plan": ArtifactReference(path=plan_path.as_posix(), sha256=plan.compute_sha256()),
            "scheduler": runtime_case.context.attempt.scheduler.model_copy(
                update={"array_task_id": shard.array_task_index}
            ),
        }
    )
    attempt_directory = plan_path.parent / "shards" / shard.shard_id / "attempts" / attempt.attempt_id
    attempt_directory.mkdir(parents=True, mode=0o700)
    return AllocationContext(
        plan=plan,
        shard=shard,
        attempt=attempt,
        attempt_directory=attempt_directory,
    )
