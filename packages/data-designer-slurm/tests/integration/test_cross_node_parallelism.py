# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from data_designer.slurm.config import DataDesignerSlurmConfig, injected_profile
from data_designer.slurm.planning import ResolvedDependencyLock, ResolvedSlurmRunPlan
from data_designer.slurm.planning.compiler import SlurmRunCompiler
from data_designer.slurm.planning.errors import SlurmPlanCompilationError
from data_designer.slurm.planning.resolution import resolve_slurm_config
from data_designer.slurm.runtime.distributed import build_node_worker_spec, build_vllm_process_command
from data_designer.slurm.runtime.preflight import AllocationLayout
from data_designer.slurm.runtime.steps import build_vllm_command
from data_designer.slurm.serving.deployment import ResolvedVllmServerDeployment
from data_designer.slurm.serving.resolver import resolve_vllm_server
from data_designer.slurm.serving.vllm import ResolvedVllmProcess, VllmProcessRole


def _compile(
    authored: DataDesignerSlurmConfig,
    lock: ResolvedDependencyLock,
    template: ResolvedSlurmRunPlan,
    *,
    tp: int,
    pp: int | None,
    nodes: int,
    groups: int = 1,
) -> ResolvedSlurmRunPlan:
    payload = authored.model_dump(mode="json")
    deployment = payload["deployments"][0]
    deployment["resources"]["nodes"] = nodes * groups
    deployment["topology"] = {"tensor_parallel": tp, "nodes_per_replica": nodes}
    if pp is not None:
        deployment["topology"]["pipeline_parallel"] = pp
    deployment["server"]["enable_expert_parallel"] = True
    authored = DataDesignerSlurmConfig.model_validate(payload)
    profile = injected_profile(template.selected_profile.profile.model_copy(update={"gpus_per_node": 4}))
    return SlurmRunCompiler.compile(
        resolve_slurm_config(
            authored,
            selected_profile=profile,
            client_runtime=template.client.runtime,
            deployment_images=(template.deployments[0].image,),
            dependency_lock=lock,
            runtime_bundle=template.runtime_bundle,
            run_id=template.run_id,
            package_version=template.package_version,
            resolved_gpus_per_node=4,
        )
    )


@pytest.mark.parametrize(
    ("tp", "pp", "nodes", "groups", "replicas"),
    [
        (16, 1, 4, 1, 1),
        (8, 1, 2, 1, 1),
        (4, 1, 2, 2, 4),
        (8, 2, 4, 1, 1),
        (4, None, 2, 1, 1),
        (2, 2, 1, 1, 1),
        (1, 4, 1, 1, 1),
    ],
    ids=["tp16-ep", "tp8-ep", "two-lanes-two-groups", "hybrid-tp-pp", "legacy-pp", "local-pp", "local-pp-tp1"],
)
def test_compiled_parallel_topology_drives_node_gpu_ownership_and_commands(
    authored_run_single: DataDesignerSlurmConfig,
    dependency_lock_single: ResolvedDependencyLock,
    single_node_plan: ResolvedSlurmRunPlan,
    tp: int,
    pp: int | None,
    nodes: int,
    groups: int,
    replicas: int,
) -> None:
    plan = _compile(
        authored_run_single, dependency_lock_single, single_node_plan, tp=tp, pp=pp, nodes=nodes, groups=groups
    )
    assert ResolvedSlurmRunPlan.model_validate_json(plan.serialize_json()).compute_sha256() == plan.compute_sha256()
    server = resolve_vllm_server(plan, plan.deployments[0].deployment_id)
    assert ResolvedVllmServerDeployment.model_validate_json(server.model_dump_json()) == server
    effective_pp = nodes if pp is None else pp
    assert server.topology.replica_count == replicas
    assert server.topology.gpus_per_replica == tp * effective_pp
    assert len(server.processes) == nodes * replicas
    assert len(server.backend_endpoints) == replicas
    assert len(server.readiness_probes) == replicas
    layout = AllocationLayout(tuple(f"gpu-{index}" for index in range(nodes * groups)))
    spec = build_node_worker_spec(server, plan, layout)
    assert not spec.allow_gpu_sharing
    assert len(spec.nodes) == nodes * groups
    for node in spec.nodes:
        assigned = [gpu for process in node.processes for gpu in process.gpu_indices]
        assert sorted(assigned) == list(range(4))
    for process in server.processes:
        assert process.node_count == nodes
        assert process.node_rank < nodes
        assert len(process.gpu_indices) * nodes == tp * effective_pp
        assert (process.role is VllmProcessRole.API_SERVER) == (process.node_rank == 0)
        command = build_vllm_process_command(server, process, plan, layout)
        assert command[command.index("--tensor-parallel-size") + 1] == str(tp)
        assert command[command.index("--distributed-executor-backend") + 1] == "mp"
        assert "--enable-expert-parallel" in command
        if nodes > 1:
            assert command[command.index("--pipeline-parallel-size") + 1] == str(effective_pp)
            assert command[command.index("--nnodes") + 1] == str(nodes)
            assert command[command.index("--node-rank") + 1] == str(process.node_rank)
            assert ("--headless" in command) == (process.node_rank != 0)
            assert process.rendezvous is not None
            assert command[command.index("--master-addr") + 1] == layout.get_host(process.rendezvous.master_node_index)
        else:
            local = build_vllm_command(server, process, plan)
            assert local[local.index("--pipeline-parallel-size") + 1] == str(effective_pp)
            assert "--nnodes" not in command


@pytest.mark.parametrize(("tp", "pp", "nodes"), [(3, 1, 2), (32, 1, 4), (12, 1, 2)])
def test_invalid_local_gpu_counts_fail_before_submission(
    authored_run_single: DataDesignerSlurmConfig,
    dependency_lock_single: ResolvedDependencyLock,
    single_node_plan: ResolvedSlurmRunPlan,
    tp: int,
    pp: int,
    nodes: int,
) -> None:
    with pytest.raises(SlurmPlanCompilationError, match="divide"):
        _compile(authored_run_single, dependency_lock_single, single_node_plan, tp=tp, pp=pp, nodes=nodes)


@pytest.mark.parametrize("mutation", ["node_rank", "node_count", "local_gpu_count", "rendezvous"])
def test_cross_node_process_rejects_invalid_world_geometry(
    authored_run_single: DataDesignerSlurmConfig,
    dependency_lock_single: ResolvedDependencyLock,
    single_node_plan: ResolvedSlurmRunPlan,
    mutation: str,
) -> None:
    plan = _compile(authored_run_single, dependency_lock_single, single_node_plan, tp=16, pp=1, nodes=4)
    server = resolve_vllm_server(plan, plan.deployments[0].deployment_id)
    payload = server.processes[-1].model_dump(mode="json")
    assert payload["node_rank"] == 3
    assert "pipeline_rank" not in payload
    if mutation == "node_rank":
        payload["node_rank"] = 4
    elif mutation == "node_count":
        payload["nodes_per_replica"] = 2
    elif mutation == "local_gpu_count":
        payload["gpu_indices"].pop()
    else:
        payload["rendezvous"] = None
    with pytest.raises(ValidationError):
        ResolvedVllmProcess.model_validate_json(json.dumps(payload))
