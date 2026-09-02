# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import data_designer.slurm.serving.resolver as resolver_module
from data_designer.slurm.config import QueueBackpressureConfig
from data_designer.slurm.contracts import compute_serialized_json_sha256, pretty_json
from data_designer.slurm.planning import ResolvedDeployment, ResolvedSlurmRunPlan
from data_designer.slurm.serving.deployment import ResolvedVllmServerDeployment
from data_designer.slurm.serving.resolver import (
    VllmServerResolutionError,
    resolve_vllm_server,
)
from data_designer.slurm.serving.vllm import VllmProcessRole

GOLDEN_DIRECTORY = Path(__file__).parent / "golden"


def test_single_node_resolution_matches_golden(single_node_plan: ResolvedSlurmRunPlan) -> None:
    placement = single_node_plan.deployments[0]

    first = _resolve(single_node_plan, placement.deployment_id)
    second = _resolve(single_node_plan, placement.deployment_id)

    assert first == second
    assert pretty_json(first.model_dump(mode="json")) == (GOLDEN_DIRECTORY / "single_node.json").read_text()
    assert first.processes[0].role is VllmProcessRole.API_SERVER
    assert first.processes[0].launch_delay_seconds == 0
    assert first.launch_policy.lead_boot_standoff_seconds == 60
    assert first.launch_policy.rank_launch_stagger_seconds == 5
    assert first.launch_policy.queue_backpressure == QueueBackpressureConfig()
    assert first.logical_endpoint.load_balancing == "least_connections"
    assert first.failure_policy == "coordinated"


def test_multi_node_replica_resolution_matches_golden(multi_replica_plan: ResolvedSlurmRunPlan) -> None:
    placement = multi_replica_plan.deployments[0]

    resolved = _resolve(multi_replica_plan, placement.deployment_id)

    assert pretty_json(resolved.model_dump(mode="json")) == (GOLDEN_DIRECTORY / "multi_node.json").read_text()
    assert [process.pipeline_rank for process in resolved.processes] == [0, 1, 0, 1]
    assert [process.gpu_indices for process in resolved.processes] == [
        (0, 1, 2, 3),
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (4, 5, 6, 7),
    ]
    assert [process.launch_delay_seconds for process in resolved.processes] == [0, 0, 32, 32]
    assert [process.role for process in resolved.processes] == [
        VllmProcessRole.API_SERVER,
        VllmProcessRole.FOLLOWER,
        VllmProcessRole.API_SERVER,
        VllmProcessRole.FOLLOWER,
    ]
    assert resolved.logical_endpoint.load_balancing == "least_connections"


def test_multiple_two_node_groups_preserve_global_replica_order_and_delays(
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    plan = _multi_group_plan(multi_node_plan)
    placement = plan.deployments[0]

    resolved = _resolve(plan, placement.deployment_id)

    assert resolved.topology.node_group_count == 2
    assert resolved.topology.replica_count == 4
    assert [endpoint.node_index for endpoint in resolved.backend_endpoints] == [0, 0, 2, 2]
    assert [process.node_index for process in resolved.processes] == [0, 1, 0, 1, 2, 3, 2, 3]
    assert [process.launch_delay_seconds for process in resolved.processes] == [0, 0, 32, 32, 34, 34, 36, 36]


def test_two_deployments_keep_images_and_endpoint_identities_isolated(
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    resolved = tuple(_resolve(multi_node_plan, placement.deployment_id) for placement in multi_node_plan.deployments)

    assert resolved[0].image.sha256 != resolved[1].image.sha256
    assert resolved[0].model_alias != resolved[1].model_alias
    assert resolved[0].model != resolved[1].model
    assert resolved[0].served_model_name != resolved[1].served_model_name
    assert resolved[0].image.inspection_facts.runtime_version != resolved[1].image.inspection_facts.runtime_version
    node_indices = [node_index for deployment in resolved for node_index in deployment.node_indices]
    assert len(node_indices) == len(set(node_indices))
    process_ids = [process.process_id for deployment in resolved for process in deployment.processes]
    assert len(process_ids) == len(set(process_ids))
    backend_addresses = [
        (backend.node_index, backend.port) for deployment in resolved for backend in deployment.backend_endpoints
    ]
    assert len(backend_addresses) == len(set(backend_addresses))
    logical_addresses = [
        (deployment.logical_endpoint.node_index, deployment.logical_endpoint.port) for deployment in resolved
    ]
    assert len(logical_addresses) == len(set(logical_addresses))
    rendezvous_addresses = [
        (process.rendezvous.master_node_index, process.rendezvous.port)
        for deployment in resolved
        for process in deployment.processes
        if process.pipeline_rank == 0 and process.rendezvous is not None
    ]
    network_addresses = backend_addresses + logical_addresses + rendezvous_addresses
    assert len(network_addresses) == len(set(network_addresses))


def test_resolution_preserves_inspected_runtime_version_as_provenance(single_node_plan: ResolvedSlurmRunPlan) -> None:
    payload = single_node_plan.deployments[0].model_dump(mode="json")
    payload["image"]["inspection"]["inspection"]["runtime_version"] = "0.22.0+vendor.1"
    placement = ResolvedDeployment.model_validate_json(json.dumps(payload))
    plan = _plan_with_placement(single_node_plan, placement)

    resolved = _resolve(plan, placement.deployment_id)

    assert resolved.image.inspection_facts.runtime_version == "0.22.0+vendor.1"


@pytest.mark.parametrize(
    "runtime_version",
    ("vendor-vllm-build", "0.20.9", "0.22.0rc1", "0.23.0"),
)
def test_resolution_rejects_unsupported_runtime_versions(
    single_node_plan: ResolvedSlurmRunPlan,
    runtime_version: str,
) -> None:
    payload = single_node_plan.deployments[0].model_dump(mode="json")
    payload["image"]["inspection"]["inspection"]["runtime_version"] = runtime_version
    placement = ResolvedDeployment.model_validate_json(json.dumps(payload))
    plan = _plan_with_placement(single_node_plan, placement)

    with pytest.raises(VllmServerResolutionError, match="unsupported vLLM runtime version"):
        _resolve(plan, placement.deployment_id)


def test_resolution_rejects_image_inspection_mismatch(single_node_plan: ResolvedSlurmRunPlan) -> None:
    placement = single_node_plan.deployments[0].model_copy(update={"image": single_node_plan.client.image})
    invalid_plan = single_node_plan.model_copy(update={"deployments": (placement,)})

    with pytest.raises(VllmServerResolutionError, match="serving image"):
        _resolve(invalid_plan, placement.deployment_id)


def test_resolution_rejects_multi_node_expert_parallel(multi_node_plan: ResolvedSlurmRunPlan) -> None:
    placement = multi_node_plan.deployments[0]
    server = placement.authored.server.model_copy(update={"enable_expert_parallel": True})
    authored = placement.authored.model_copy(update={"server": server})
    invalid_placement = placement.model_copy(update={"authored": authored})
    invalid_plan = multi_node_plan.model_copy(
        update={"deployments": (invalid_placement, *multi_node_plan.deployments[1:])}
    )

    with pytest.raises(VllmServerResolutionError, match="multi-node expert parallel"):
        _resolve(invalid_plan, invalid_placement.deployment_id)


def test_resolution_allows_independent_single_node_expert_parallel_replicas(
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    placement = _independent_replica_placement(multi_node_plan)
    plan = _plan_with_placement(multi_node_plan, placement)

    resolved = _resolve(plan, placement.deployment_id)

    assert resolved.launch_policy.enable_expert_parallel
    assert resolved.topology.pipeline_parallel == 1
    assert [process.node_index for process in resolved.processes] == [0, 1]
    assert all(process.rendezvous is None for process in resolved.processes)


def test_serving_package_has_no_scheduler_shell_or_runtime_dependency() -> None:
    imports: set[str] = set()
    for source_path in Path(resolver_module.__file__).parent.glob("*.py"):
        tree = ast.parse(source_path.read_text())
        imports.update(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
        imports.update(
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
        )

    assert "subprocess" not in imports
    assert not any(
        module.startswith(
            (
                "data_designer.slurm.launcher",
                "data_designer.slurm.runtime",
                "data_designer.slurm.state",
            )
        )
        for module in imports
    )


def test_resolution_requires_matching_logical_endpoint(single_node_plan: ResolvedSlurmRunPlan) -> None:
    deployment = single_node_plan.deployments[0]
    unrelated_port = single_node_plan.client.ports[0].model_copy(update={"name": "other-logical-endpoint"})
    client = single_node_plan.client.model_copy(update={"ports": (unrelated_port,)})
    invalid_plan = single_node_plan.model_copy(update={"client": client})

    with pytest.raises(VllmServerResolutionError, match="exactly one logical endpoint"):
        resolve_vllm_server(invalid_plan, deployment.deployment_id)


def test_resolution_rejects_cross_input_network_collision(single_node_plan: ResolvedSlurmRunPlan) -> None:
    deployment = single_node_plan.deployments[0]
    deployment_port = deployment.ports[0]
    colliding_port = single_node_plan.client.ports[0].model_copy(
        update={"node_index": deployment_port.node_index, "port": deployment_port.port}
    )
    client = single_node_plan.client.model_copy(update={"ports": (colliding_port,)})
    invalid_plan = single_node_plan.model_copy(update={"client": client})

    with pytest.raises(VllmServerResolutionError, match="must not collide"):
        resolve_vllm_server(invalid_plan, deployment.deployment_id)


def test_resolution_normalizes_inconsistent_planner_output(single_node_plan: ResolvedSlurmRunPlan) -> None:
    deployment = single_node_plan.deployments[0]
    inconsistent_topology = deployment.topology.model_copy(update={"replica_count": 2})
    inconsistent_deployment = deployment.model_copy(update={"topology": inconsistent_topology})
    invalid_plan = single_node_plan.model_copy(update={"deployments": (inconsistent_deployment,)})

    with pytest.raises(VllmServerResolutionError, match="inconsistent vLLM server specification"):
        resolve_vllm_server(invalid_plan, deployment.deployment_id)


def test_resolution_normalizes_intermediate_record_validation(single_node_plan: ResolvedSlurmRunPlan) -> None:
    deployment = single_node_plan.deployments[0]
    long_deployment_id = "d" * 128
    renamed_deployment = deployment.model_copy(update={"deployment_id": long_deployment_id})
    endpoint = single_node_plan.client.ports[0].model_copy(update={"name": f"{long_deployment_id}-logical-endpoint"})
    client = single_node_plan.client.model_copy(update={"ports": (endpoint,)})
    invalid_plan = single_node_plan.model_copy(update={"client": client, "deployments": (renamed_deployment,)})

    with pytest.raises(VllmServerResolutionError, match="inconsistent vLLM server specification"):
        resolve_vllm_server(invalid_plan, long_deployment_id)


def test_resolution_derives_placement_and_endpoint_from_one_plan(
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    payload = single_node_plan.model_dump(mode="json")
    payload["deployments"][0]["ports"][0]["port"] = 18100
    payload["client"]["ports"][0]["port"] = 17100
    alternate_plan = ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))

    resolved = _resolve(alternate_plan, alternate_plan.deployments[0].deployment_id)

    assert resolved.backend_endpoints[0].port == 18100
    assert resolved.logical_endpoint.port == 17100


def _resolve(
    plan: ResolvedSlurmRunPlan,
    deployment_id: str,
) -> ResolvedVllmServerDeployment:
    return resolve_vllm_server(plan, deployment_id)


def _plan_with_placement(
    plan: ResolvedSlurmRunPlan,
    placement: ResolvedDeployment,
) -> ResolvedSlurmRunPlan:
    payload = plan.model_dump(mode="json")
    matching_indices = [
        index for index, candidate in enumerate(plan.deployments) if candidate.deployment_id == placement.deployment_id
    ]
    assert len(matching_indices) == 1
    payload["deployments"][matching_indices[0]] = placement.model_dump(mode="json")
    return ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))


def _multi_group_plan(plan: ResolvedSlurmRunPlan) -> ResolvedSlurmRunPlan:
    payload = plan.model_dump(mode="json")
    payload["deployments"] = payload["deployments"][:1]
    payload["client"]["ports"] = payload["client"]["ports"][:1]
    payload["builder"]["inline"]["data_designer"]["model_configs"] = payload["builder"]["inline"]["data_designer"][
        "model_configs"
    ][:1]
    payload["builder"]["model_aliases"] = ["generator"]
    payload["builder"]["content_sha256"] = compute_serialized_json_sha256(payload["builder"]["inline"])
    deployment = payload["deployments"][0]
    deployment["authored"]["resources"]["nodes"] = 4
    deployment["authored"]["topology"]["tensor_parallel"] = 4
    deployment["authored"]["server"].update(
        lead_boot_standoff="30s",
        rank_launch_stagger="2s",
    )
    deployment["node_indices"] = [0, 1, 2, 3]
    deployment["topology"].update(
        tensor_parallel=4,
        node_group_count=2,
        replicas_per_node_group=2,
        replica_count=4,
        gpus_per_replica=8,
    )
    deployment["ports"] = [
        {
            "name": f"deployment-00000-http-{deployment_replica_index:05d}",
            "role": "http",
            "node_index": 0 if deployment_replica_index < 2 else 2,
            "port": 18000 + deployment_replica_index % 2,
        }
        for deployment_replica_index in range(4)
    ] + [
        {
            "name": f"deployment-00000-rendezvous-{deployment_replica_index:05d}",
            "role": "rendezvous",
            "node_index": 0 if deployment_replica_index < 2 else 2,
            "port": 19000 + deployment_replica_index % 2,
        }
        for deployment_replica_index in range(4)
    ]
    return ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))


def _independent_replica_placement(plan: ResolvedSlurmRunPlan) -> ResolvedDeployment:
    payload = plan.deployments[0].model_dump(mode="json")
    payload["authored"]["server"]["enable_expert_parallel"] = True
    payload["authored"]["topology"]["nodes_per_replica"] = 1
    payload["topology"].update(
        nodes_per_replica=1,
        pipeline_parallel=1,
        node_group_count=2,
        replica_count=2,
        gpus_per_replica=8,
    )
    payload["ports"] = [
        {
            "name": "deployment-00000-http-00000",
            "role": "http",
            "node_index": 0,
            "port": 18000,
        },
        {
            "name": "deployment-00000-http-00001",
            "role": "http",
            "node_index": 1,
            "port": 18001,
        },
    ]
    return ResolvedDeployment.model_validate_json(json.dumps(payload))
