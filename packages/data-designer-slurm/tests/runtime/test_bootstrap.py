# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shlex
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import RuntimeCase, relocate_plan

from data_designer.slurm.contracts import ArtifactReference
from data_designer.slurm.planning import PortClaim, ResolvedSlurmRunPlan, ResolvedTopology
from data_designer.slurm.runtime.bootstrap import RuntimeBootstrapManifest, RuntimeStepSpec, build_runtime_manifest
from data_designer.slurm.runtime.distributed import build_vllm_process_command
from data_designer.slurm.runtime.errors import SlurmRuntimeError
from data_designer.slurm.runtime.models import AllocationContext, RuntimeStepRole
from data_designer.slurm.runtime.node_spec import decode_node_worker_spec
from data_designer.slurm.runtime.ports import resolve_allocation_deployments
from data_designer.slurm.runtime.preflight import AllocationLayout
from data_designer.slurm.serving.vllm import ResolvedVllmProcess
from data_designer.slurm.state import RetryPlan, RetryShard


def test_bootstrap_manifest_builds_typed_one_node_steps_without_secret_values(runtime_case: RuntimeCase) -> None:
    context = runtime_case.context
    runtime_root = Path("/tmp/data-designer-slurm-4101-0/runtime")
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
        context.plan.client.runtime.python_executable,
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
    server = next(step for step in manifest.steps if step.role is RuntimeStepRole.SERVER)
    assert server.command[server.command.index("--host") + 1] == "127.0.0.1"
    endpoint = next(step for step in manifest.steps if step.role is RuntimeStepRole.ENDPOINT)
    assert endpoint.literal_environment["PYTHONPATH"] == runtime_root.as_posix()
    assert endpoint.container_environment == ()
    client_steps = tuple(
        step for step in manifest.steps if step.role in {RuntimeStepRole.CLIENT_PREFLIGHT, RuntimeStepRole.CLIENT}
    )
    assert all(
        step.literal_environment["DATA_DESIGNER_SLURM_SCRATCH_ROOT"] == runtime_root.parent.as_posix()
        for step in client_steps
    )
    assert "SLURM_TMPDIR" not in manifest.serialize_json()


@pytest.mark.parametrize(
    ("nodes", "tensor_parallel", "replicas", "stagger"), ((1, 1, 8, 0), (1, 2, 4, 2), (1, 4, 2, 0), (2, 4, 4, 2))
)
def test_multi_replica_manifest_uses_one_gpu_owned_worker(
    runtime_case: RuntimeCase, nodes: int, tensor_parallel: int, replicas: int, stagger: int
) -> None:
    plan = runtime_case.context.plan
    placement = plan.deployments[0]
    authored = placement.authored.model_copy(
        update={
            "topology": placement.authored.topology.model_copy(update={"tensor_parallel": tensor_parallel}),
            "resources": placement.authored.resources.model_copy(update={"nodes": nodes}),
            "server": placement.authored.server.model_copy(
                update={
                    "startup_timeout": "2s",
                    "distributed_init_timeout": "1s",
                    "lead_boot_standoff": "4s",
                    "rank_launch_stagger": f"{stagger}s",
                }
            ),
        }
    )
    topology = ResolvedTopology.derive(
        node_count=nodes,
        gpus_per_node=plan.resolved_gpus_per_node,
        tensor_parallel=tensor_parallel,
        nodes_per_replica=1,
    )
    ports = tuple(
        PortClaim(
            name=f"{placement.deployment_id}-http-{index:05d}",
            role="http",
            node_index=index // (replicas // nodes),
            port=18000 + index % (replicas // nodes),
        )
        for index in range(replicas)
    )
    placement = placement.model_copy(
        update={"authored": authored, "topology": topology, "ports": ports, "node_indices": tuple(range(nodes))}
    )
    plan = ResolvedSlurmRunPlan.model_validate(plan.model_copy(update={"deployments": (placement,)}).model_dump())
    context = replace(runtime_case.context, plan=plan)

    layout = AllocationLayout(tuple(f"compute-{index + 1:03d}" for index in range(nodes)))
    manifest = build_runtime_manifest(
        context,
        {"SLURM_JOB_GPUS": "0,1,2,3,4,5,6,7"},
        runtime_root=Path("/tmp/data-designer-slurm-4101-0/runtime"),
        log_directory=context.attempt_directory / "logs/execution-00000002",
        layout=layout,
    )

    servers = tuple(step for step in manifest.steps if step.role is RuntimeStepRole.SERVER)
    assert len(servers) == 1
    assert servers[0].gpu_indices == tuple(range(8))
    assert len(servers[0].readiness) == replicas
    assert {probe.host for probe in servers[0].readiness} == ({"127.0.0.1"} if nodes == 1 else set(layout.node_hosts))
    worker = decode_node_worker_spec(servers[0].command[-1])
    processes = tuple(process for node in worker.nodes for process in node.processes)
    assert len(processes) == replicas
    assert [process.gpu_indices for process in processes] == [
        tuple(
            range((index % (replicas // nodes)) * tensor_parallel, (index % (replicas // nodes) + 1) * tensor_parallel)
        )
        for index in range(replicas)
    ]
    assert all(
        ("--host", "127.0.0.1" if nodes == 1 else "0.0.0.0") == process.command[process.command.index("--host") :][:2]
        for process in processes
    )
    endpoint = next(step for step in manifest.steps if step.role is RuntimeStepRole.ENDPOINT)
    assert "0.0.0.0" not in endpoint.command
    assert " ".join(endpoint.command).count("http://") == replicas
    _assert_launch_readiness(manifest, runtime_case.workspace)


@pytest.mark.parametrize("role", (RuntimeStepRole.CLIENT_PREFLIGHT, RuntimeStepRole.SERVER))
def test_runtime_step_rejects_execution_mode_that_conflicts_with_role(
    runtime_case: RuntimeCase, role: RuntimeStepRole
) -> None:
    context = runtime_case.context
    manifest = build_runtime_manifest(
        context,
        {"SLURM_JOB_GPUS": "0"},
        runtime_root=Path("/tmp/data-designer-slurm-4101-0/runtime"),
        log_directory=context.attempt_directory / "logs/execution-00000002",
        layout=AllocationLayout(("compute-001",)),
    )
    step = next(item for item in manifest.steps if item.role is role)
    payload = step.model_dump(mode="python")
    payload["execution"] = "native" if step.execution == "container" else "container"
    payload["image_path"] = None if payload["execution"] == "native" else "/workspace/client.sqsh"
    if payload["execution"] == "native":
        payload["container_environment"] = ()

    with pytest.raises(ValueError, match="execution does not match its role"):
        RuntimeStepSpec.model_validate(payload)


def test_bootstrap_manifest_rejects_runtime_outside_allocation_scratch(runtime_case: RuntimeCase) -> None:
    with pytest.raises(SlurmRuntimeError, match="outside allocation-local scratch"):
        build_runtime_manifest(
            runtime_case.context,
            {"SLURM_JOB_GPUS": "0"},
            runtime_root=Path("/shared/runtime"),
            log_directory=runtime_case.context.attempt_directory / "logs/execution-00000002",
            layout=AllocationLayout(("compute-001",)),
        )


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
        runtime_root=Path("/tmp/data-designer-slurm-4101-0/runtime"),
        log_directory=context.attempt_directory / "logs/execution-00000002",
        layout=AllocationLayout(("compute-001",)),
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
                        "server": deployment.authored.server.model_copy(
                            update={
                                "startup_timeout": "2s",
                                "distributed_init_timeout": "1s",
                                "lead_boot_standoff": "4s",
                                "rank_launch_stagger": "2s",
                            }
                        ),
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
        runtime_root=Path("/tmp/data-designer-slurm-4101-0/runtime"),
        log_directory=context.attempt_directory / "logs/execution-00000002",
        layout=layout,
    )

    distributed = next(step for step in manifest.steps if step.step_id == "deployment-00000-serve")
    preflight = next(step for step in manifest.steps if step.step_id == "deployment-00000-preflight")
    worker_spec = decode_node_worker_spec(distributed.command[-1])
    endpoint = next(step for step in manifest.steps if step.step_id == "deployment-00000-endpoint")
    remote_server = next(step for step in manifest.steps if step.step_id == "deployment-00001-serve")
    remote_preflight = next(step for step in manifest.steps if step.step_id == "deployment-00001-preflight")

    assert distributed.node_hosts == ("compute-001", "compute-002")
    assert distributed.kill_on_bad_exit
    assert preflight.role is RuntimeStepRole.SERVER_PREFLIGHT
    assert tuple(node.host for node in worker_spec.nodes) == distributed.node_hosts
    assert "--master-addr" in worker_spec.nodes[0].processes[0].command
    assert "compute-001" in worker_spec.nodes[0].processes[0].command
    assert worker_spec.nodes[0].processes[0].command[2] == "/workspace/primary/models/model-0"
    assert worker_spec.required_model_path == "/workspace/primary/models/model-0"
    assert "--headless" in worker_spec.nodes[1].processes[0].command
    assert tuple(probe.host for probe in distributed.readiness) == ("compute-001",)
    assert "http://compute-001:" in " ".join(endpoint.command)
    assert endpoint.node_hosts == ("compute-001",)
    assert endpoint.literal_environment["PYTHONPATH"] == "/tmp/data-designer-slurm-4101-0/runtime"
    assert endpoint.container_environment == ()
    assert remote_preflight.node_hosts == ("compute-003",)
    remote_worker_spec = decode_node_worker_spec(remote_preflight.command[-1])
    assert remote_worker_spec.required_model_path == "/workspace/primary/models/model-1"
    assert remote_server.node_hosts == ("compute-003",)
    assert remote_server.gpu_indices == tuple(range(8))
    remote_server_spec = decode_node_worker_spec(remote_server.command[-1])
    assert remote_server_spec.required_model_path == "/workspace/primary/models/model-1"
    assert remote_server_spec.nodes[0].host == "compute-003"
    assert len(remote_server_spec.nodes[0].processes) == 8
    assert all(
        process.command[process.command.index("--distributed-executor-backend") + 1] == "uni"
        for process in remote_server_spec.nodes[0].processes
    )
    assert tuple(probe.host for probe in remote_server.readiness) == ("compute-003",) * 8
    _assert_launch_readiness(manifest, runtime_case.workspace)


def _assert_launch_readiness(manifest: RuntimeBootstrapManifest, workspace: Path) -> None:
    manifest_path = workspace / "readiness-manifest.json"
    manifest_path.write_text(manifest.serialize_json())
    cases: list[str] = []
    budgets: list[tuple[int, int]] = []
    for step in manifest.steps:
        if step.role is not RuntimeStepRole.SERVER:
            continue
        worker = decode_node_worker_spec(step.command[-1])
        for probe in step.readiness:
            process = next(
                process
                for node in worker.nodes
                if node.host == probe.host or probe.host == "127.0.0.1"
                for process in node.processes
                if "--headless" not in process.command
                if str(probe.port) == process.command[process.command.index("--port") + 1]
            )
            budgets.append((probe.deadline_seconds, process.launch_delay_seconds + 2))
            cases.append(f"*://{probe.host}:{probe.port}/*) ((SECONDS >= {process.launch_delay_seconds + 1}));;")
    entrypoint = Path(__file__).parents[2] / "src/data_designer/slurm/runtime/entrypoint.sh"
    command = f"""
source {shlex.quote(str(entrypoint))}
DD_RUNTIME_MANIFEST={shlex.quote(str(manifest_path))}
SECONDS=0
dd_start_step() {{ DD_LAST_PID=1; }}
dd_register_required_pid() {{ :; }}
dd_require_running() {{ :; }}
dd_sleep() {{ if [[ $1 == 0.5 ]]; then SECONDS=$((SECONDS + 1)); else SECONDS=$((SECONDS + $1)); fi; }}
curl() {{ case "${{@: -1}}" in {" ".join(cases)} *) return 99;; esac; }}
dd_start_servers
dd_wait_for_role_readiness server
"""
    completed = subprocess.run(("bash", "-c", command), capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    assert all(actual == expected for actual, expected in budgets)


def test_pipeline_parallel_process_uses_multi_process_executor(
    runtime_case: RuntimeCase,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    context = _replace_plan(runtime_case, multi_node_plan)
    layout = AllocationLayout(("compute-001", "compute-002", "compute-003"))
    deployment = resolve_allocation_deployments(
        context,
        {"SLURM_JOB_GPUS": "0,1,2,3,4,5,6,7"},
    )[0]
    payload = deployment.processes[0].model_dump(mode="python")
    payload.update({"gpu_indices": (0,), "tensor_parallel": 1})
    process = ResolvedVllmProcess.model_validate(payload)

    command = build_vllm_process_command(deployment, process, context.plan, layout)

    executor_index = command.index("--distributed-executor-backend")
    assert process.tensor_parallel == 1
    assert process.pipeline_parallel == 2
    assert command[executor_index + 1] == "mp"


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
