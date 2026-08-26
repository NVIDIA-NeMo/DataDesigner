# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from data_designer.slurm.config import QueueBackpressureConfig, ServerDeploymentConfig
from data_designer.slurm.contracts import pretty_json
from data_designer.slurm.planning import PortClaim, ResolvedDeployment, ResolvedSlurmRunPlan
from data_designer.slurm.serving import (
    ResolvedLogicalEndpoint,
    ResolvedServerDeployment,
    ServerResolutionContext,
    ServerResolutionError,
    VllmLaunchPolicy,
    VllmProcessRole,
    VllmProcessSpec,
    resolve_server,
)
from data_designer.slurm.serving import resolver as resolver_module

GOLDEN_DIRECTORY = Path(__file__).parent / "golden"


class _UnsupportedServer:
    type: str = "unsupported"


def test_single_node_resolution_matches_golden(single_node_plan: ResolvedSlurmRunPlan) -> None:
    placement = single_node_plan.deployments[0]
    context = ServerResolutionContext.from_plan(single_node_plan, placement.deployment_id)

    first = resolve_server(placement.authored, context)
    second = resolve_server(placement.authored, context)

    assert first == second
    assert pretty_json(first.model_dump(mode="json")) == (GOLDEN_DIRECTORY / "single_node.json").read_text()
    assert first.processes[0].role is VllmProcessRole.API_SERVER
    assert first.processes[0].launch_delay_seconds == 0
    assert first.launch_policy.lead_boot_standoff_seconds == 60
    assert first.launch_policy.rank_launch_stagger_seconds == 5
    assert first.launch_policy.queue_backpressure == QueueBackpressureConfig()
    assert first.logical_endpoint.load_balancing == "least_connections"
    assert first.failure_policy == "coordinated"


def test_multi_node_lane_resolution_matches_golden(multi_node_plan: ResolvedSlurmRunPlan) -> None:
    placement = _multi_lane_placement(multi_node_plan)
    context = _context_for_placement(multi_node_plan, placement)

    resolved = resolve_server(placement.authored, context)

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

    resolved = resolve_server(placement.authored, ServerResolutionContext.from_plan(plan, placement.deployment_id))

    assert resolved.topology.node_group_count == 2
    assert resolved.topology.replica_count == 4
    assert [endpoint.node_index for endpoint in resolved.backend_endpoints] == [0, 0, 2, 2]
    assert [process.node_index for process in resolved.processes] == [0, 1, 0, 1, 2, 3, 2, 3]
    assert [process.launch_delay_seconds for process in resolved.processes] == [0, 0, 32, 32, 34, 34, 36, 36]


def test_two_deployments_keep_images_and_endpoint_identities_isolated(
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    resolved = tuple(
        resolve_server(
            placement.authored,
            ServerResolutionContext.from_plan(multi_node_plan, placement.deployment_id),
        )
        for placement in multi_node_plan.deployments
    )

    assert resolved[0].image.sha256 != resolved[1].image.sha256
    assert resolved[0].model_alias != resolved[1].model_alias
    assert resolved[0].model != resolved[1].model
    assert resolved[0].served_model_name != resolved[1].served_model_name
    assert (
        resolved[0].image.inspection.inspection.runtime_version
        != resolved[1].image.inspection.inspection.runtime_version
    )
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


def test_resolution_context_rejects_wrong_logical_endpoint(single_node_plan: ResolvedSlurmRunPlan) -> None:
    placement = single_node_plan.deployments[0]
    with pytest.raises(ValidationError, match="logical endpoint claim"):
        ServerResolutionContext(
            placement=placement,
            client_host_node_index=single_node_plan.client.host_node_index,
            logical_endpoint=PortClaim(
                name="other-logical-endpoint",
                role="logical_endpoint",
                node_index=0,
                port=17000,
            ),
        )


def test_resolution_rejects_declaration_that_differs_from_placement(single_node_plan: ResolvedSlurmRunPlan) -> None:
    placement = single_node_plan.deployments[0]
    context = ServerResolutionContext.from_plan(single_node_plan, placement.deployment_id)
    different = placement.authored.model_copy(update={"model": "example/different"})

    with pytest.raises(ServerResolutionError, match="does not match"):
        resolve_server(different, context)


def test_resolution_preserves_inspected_runtime_version_without_gating(single_node_plan: ResolvedSlurmRunPlan) -> None:
    payload = single_node_plan.deployments[0].model_dump(mode="json")
    payload["image"]["inspection"]["inspection"]["runtime_version"] = "vendor-vllm-build"
    placement = ResolvedDeployment.model_validate_json(json.dumps(payload))
    context = _context_for_placement(single_node_plan, placement)

    resolved = resolve_server(placement.authored, context)

    assert resolved.image.inspection.inspection.runtime_version == "vendor-vllm-build"


def test_resolution_rejects_image_inspection_mismatch(single_node_plan: ResolvedSlurmRunPlan) -> None:
    placement = single_node_plan.deployments[0].model_copy(update={"image": single_node_plan.client.image})
    context = ServerResolutionContext.model_construct(
        placement=placement,
        logical_endpoint=single_node_plan.client.ports[0],
    )

    with pytest.raises(ServerResolutionError, match="serving image"):
        resolve_server(placement.authored, context)


def test_resolution_rejects_multi_node_expert_parallel(multi_node_plan: ResolvedSlurmRunPlan) -> None:
    placement = multi_node_plan.deployments[0]
    server = placement.authored.server.model_copy(update={"enable_expert_parallel": True})
    authored = placement.authored.model_copy(update={"server": server})
    invalid_placement = placement.model_copy(update={"authored": authored})
    context = ServerResolutionContext(
        placement=invalid_placement,
        client_host_node_index=multi_node_plan.client.host_node_index,
        logical_endpoint=multi_node_plan.client.ports[0],
    )

    with pytest.raises(ServerResolutionError, match="multi-node expert parallel"):
        resolve_server(authored, context)


def test_resolution_allows_independent_single_node_expert_parallel_replicas(
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    placement = _independent_replica_placement(multi_node_plan)
    context = _context_for_placement(multi_node_plan, placement)

    resolved = resolve_server(placement.authored, context)

    assert resolved.launch_policy.enable_expert_parallel
    assert resolved.topology.pipeline_parallel == 1
    assert [process.node_index for process in resolved.processes] == [0, 1]
    assert all(process.rendezvous is None for process in resolved.processes)


def test_dispatch_rejects_unimplemented_server_type(single_node_plan: ResolvedSlurmRunPlan) -> None:
    placement = single_node_plan.deployments[0]
    authored = ServerDeploymentConfig.model_construct(
        model_alias=placement.authored.model_alias,
        served_model_name=placement.authored.served_model_name,
        model=placement.authored.model,
        server=_UnsupportedServer(),
        resources=placement.authored.resources,
        topology=placement.authored.topology,
    )
    invalid_placement = placement.model_copy(update={"authored": authored})
    context = ServerResolutionContext.model_construct(
        placement=invalid_placement,
        logical_endpoint=single_node_plan.client.ports[0],
    )

    with pytest.raises(ServerResolutionError, match="unsupported server type"):
        resolve_server(authored, context)


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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("gpu_order", "sorted and unique"),
        ("gpu_count", "equal tensor parallelism"),
        ("pipeline_rank", "below pipeline parallelism"),
        ("head_role", "rank zero"),
        ("head_port", "require an HTTP port"),
        ("follower_role", "must be followers"),
        ("follower_port", "must not publish"),
        ("missing_rendezvous", "require rendezvous"),
        ("wrong_rendezvous_lane", "rendezvous identity"),
    ],
)
def test_process_specs_reject_one_field_topology_drift(
    multi_node_plan: ResolvedSlurmRunPlan,
    mutation: str,
    message: str,
) -> None:
    placement = _multi_lane_placement(multi_node_plan)
    resolved = resolve_server(placement.authored, _context_for_placement(multi_node_plan, placement))
    process_index = (
        1 if mutation.startswith("follower") or mutation in {"missing_rendezvous", "wrong_rendezvous_lane"} else 0
    )
    payload = resolved.processes[process_index].model_dump(mode="json")
    if mutation == "gpu_order":
        payload["gpu_indices"] = list(reversed(payload["gpu_indices"]))
    elif mutation == "gpu_count":
        payload["gpu_indices"].pop()
    elif mutation == "pipeline_rank":
        payload["pipeline_rank"] = payload["pipeline_parallel"]
    elif mutation == "head_role":
        payload["role"] = "follower"
    elif mutation == "head_port":
        payload["http_port"] = None
    elif mutation == "follower_role":
        payload["role"] = "api_server"
    elif mutation == "follower_port":
        payload["http_port"] = 18000
    elif mutation == "missing_rendezvous":
        payload["rendezvous"] = None
    else:
        payload["rendezvous"]["lane_index"] = 1

    with pytest.raises(ValidationError, match=message):
        VllmProcessSpec.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("image_kind", "server image inspection"),
        ("executable", "executable path"),
        ("nodes", "sorted and unique"),
        ("node_group_divisibility", "divide evenly into replica groups"),
        ("gpu_divisibility", "divide evenly into tensor-parallel lanes"),
        ("topology", "node and GPU resources"),
        ("expert_parallel", "multi-node expert parallel"),
        ("logical_alias", "model alias"),
        ("logical_served_name", "served model name"),
        ("logical_id", "endpoint ID"),
        ("replica_index", "replica identities"),
        ("backend_id", "endpoint IDs"),
        ("logical_backends", "resolved backend order"),
        ("process_ids", "process IDs"),
        ("process_order", "ordered replica and pipeline identities"),
        ("readiness_count", "readiness probes"),
        ("backend_placement", "replica placement"),
        ("readiness_target", "readiness probes"),
        ("pipeline_ranks", "ordered replica and pipeline identities"),
        ("process_topology", "topology and launch policy"),
        ("head_endpoint", "head process"),
        ("rendezvous_consistency", "share one rendezvous"),
        ("rendezvous_master", "group head and distributed timeout"),
        ("rendezvous_timeout", "group head and distributed timeout"),
        ("network_address", "network addresses must be unique"),
        ("launch_policy_owned_arg", "owned by the compiler or runtime"),
        ("launch_policy_literal_secret", "secret-shaped environment names"),
        ("process_count", "process count"),
    ],
)
def test_resolved_server_rejects_one_field_join_drift(
    multi_node_plan: ResolvedSlurmRunPlan,
    mutation: str,
    message: str,
) -> None:
    placement = _multi_lane_placement(multi_node_plan)
    resolved = resolve_server(placement.authored, _context_for_placement(multi_node_plan, placement))
    payload = resolved.model_dump(mode="json")
    if mutation == "image_kind":
        payload["image"] = multi_node_plan.client.image.model_dump(mode="json")
    elif mutation == "executable":
        payload["executable_path"] = "/usr/local/bin/other"
    elif mutation == "nodes":
        payload["node_indices"] = [1, 0]
    elif mutation == "node_group_divisibility":
        payload["node_indices"] = [0, 1, 2]
    elif mutation == "gpu_divisibility":
        payload["gpus_per_node"] = 7
    elif mutation == "topology":
        payload["topology"]["replica_count"] = 3
    elif mutation == "expert_parallel":
        payload["launch_policy"]["enable_expert_parallel"] = True
    elif mutation == "logical_alias":
        payload["logical_endpoint"]["model_alias"] = "other"
    elif mutation == "logical_served_name":
        payload["logical_endpoint"]["served_model_name"] = "other"
    elif mutation == "logical_id":
        payload["logical_endpoint"]["endpoint_id"] = "other-logical-endpoint"
    elif mutation == "replica_index":
        payload["backend_endpoints"][1]["replica_index"] = 2
    elif mutation == "backend_id":
        payload["backend_endpoints"][1]["backend_id"] = "deployment-00000-backend-99999"
        payload["readiness_probes"][1]["backend_id"] = "deployment-00000-backend-99999"
        payload["logical_endpoint"]["backend_ids"][1] = "deployment-00000-backend-99999"
    elif mutation == "logical_backends":
        payload["logical_endpoint"]["backend_ids"] = list(reversed(payload["logical_endpoint"]["backend_ids"]))
    elif mutation == "process_ids":
        payload["processes"][1]["process_id"] = payload["processes"][0]["process_id"]
    elif mutation == "process_order":
        payload["processes"][1], payload["processes"][2] = payload["processes"][2], payload["processes"][1]
    elif mutation == "readiness_count":
        payload["readiness_probes"].append(payload["readiness_probes"][0])
    elif mutation == "backend_placement":
        payload["backend_endpoints"][1]["lane_index"] = 0
    elif mutation == "readiness_target":
        payload["readiness_probes"][1]["path"] = "/other"
    elif mutation == "pipeline_ranks":
        payload["processes"][1]["replica_index"] = 1
    elif mutation == "process_topology":
        payload["processes"][2]["node_index"] = 1
    elif mutation == "head_endpoint":
        payload["processes"][2]["http_port"] = 18000
    elif mutation == "rendezvous_consistency":
        payload["processes"][1]["rendezvous"]["port"] = 19001
    elif mutation == "rendezvous_master":
        payload["processes"][0]["rendezvous"]["master_node_index"] = 1
        payload["processes"][1]["rendezvous"]["master_node_index"] = 1
    elif mutation == "rendezvous_timeout":
        payload["processes"][0]["rendezvous"]["timeout_seconds"] = 1
        payload["processes"][1]["rendezvous"]["timeout_seconds"] = 1
    elif mutation == "network_address":
        payload["logical_endpoint"]["port"] = payload["backend_endpoints"][0]["port"]
    elif mutation == "launch_policy_owned_arg":
        payload["launch_policy"]["extra_args"] = ["--model", "example/other"]
    elif mutation == "launch_policy_literal_secret":
        payload["launch_policy"]["environment"] = {"HF_TOKEN": {"type": "literal", "value": "secret"}}
    else:
        extra = dict(payload["processes"][0])
        extra.update(process_id="deployment-00000-replica-99999-rank-00000", replica_index=99999)
        payload["processes"].append(extra)

    with pytest.raises(ValidationError, match=message):
        ResolvedServerDeployment.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"readiness_path": "health"}, "absolute URL path"),
        ({"readiness_path": "//other-host/health"}, "absolute URL path"),
        ({"readiness_path": "/health check"}, "without whitespace"),
        ({"startup_timeout_seconds": 10, "distributed_init_timeout_seconds": 11}, "must not exceed"),
    ],
)
def test_launch_policy_rejects_invalid_readiness_or_deadlines(update: dict[str, object], message: str) -> None:
    payload = {
        "startup_timeout_seconds": 900,
        "distributed_init_timeout_seconds": 600,
        "lead_boot_standoff_seconds": 60,
        "rank_launch_stagger_seconds": 5,
        "readiness_path": "/health",
        "enable_expert_parallel": False,
        "queue_backpressure": {"max_waiting_requests": 128, "retry_after_seconds": 1},
        **update,
    }
    with pytest.raises(ValidationError, match=message):
        VllmLaunchPolicy.model_validate(payload)


@pytest.mark.parametrize(
    "argument",
    [
        "-dcp=2",
        "-n2",
        "-n+2",
        "-pcp=2",
        "-r1",
        "-r0",
        "--kv-transfer-config={}",
        "--numa-bind",
        "--reasoning-parser-plugin=/tmp/reasoning.py",
        "--worker-cls=custom.Worker",
    ],
)
def test_launch_policy_rejects_runtime_owned_arguments(argument: str) -> None:
    with pytest.raises(ValidationError, match="owned by the compiler or runtime"):
        VllmLaunchPolicy(
            startup_timeout_seconds=900,
            distributed_init_timeout_seconds=600,
            lead_boot_standoff_seconds=60,
            rank_launch_stagger_seconds=5,
            readiness_path="/health",
            enable_expert_parallel=False,
            queue_backpressure=QueueBackpressureConfig(),
            extra_args=(argument,),
        )


@pytest.mark.parametrize(
    "environment_name",
    ["CUDA_VISIBLE_DEVICES", "VLLM_API_KEY", "VLLM_DP_RANK", "VLLM_HOST_IP"],
)
def test_launch_policy_rejects_runtime_owned_environment_names(environment_name: str) -> None:
    with pytest.raises(ValidationError, match="owned by the compiler or runtime"):
        VllmLaunchPolicy(
            startup_timeout_seconds=900,
            distributed_init_timeout_seconds=600,
            lead_boot_standoff_seconds=60,
            rank_launch_stagger_seconds=5,
            readiness_path="/health",
            enable_expert_parallel=False,
            queue_backpressure=QueueBackpressureConfig(),
            environment={environment_name: {"type": "secret", "environment": "EXTERNAL_VALUE"}},
        )


def test_single_node_process_rejects_rendezvous(single_node_plan: ResolvedSlurmRunPlan) -> None:
    placement = single_node_plan.deployments[0]
    resolved = resolve_server(
        placement.authored,
        ServerResolutionContext.from_plan(single_node_plan, placement.deployment_id),
    )
    payload = resolved.processes[0].model_dump(mode="json")
    payload["rendezvous"] = {
        "node_group_index": 0,
        "lane_index": 0,
        "master_node_index": 0,
        "port": 19000,
        "timeout_seconds": 600,
    }

    with pytest.raises(ValidationError, match="must not carry rendezvous"):
        VllmProcessSpec.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_backends", "must be unique"),
        ("missing_retry", "must retry HTTP 429"),
    ],
)
def test_logical_endpoint_rejects_invalid_backend_or_retry_contract(
    single_node_plan: ResolvedSlurmRunPlan,
    mutation: str,
    message: str,
) -> None:
    placement = single_node_plan.deployments[0]
    resolved = resolve_server(
        placement.authored,
        ServerResolutionContext.from_plan(single_node_plan, placement.deployment_id),
    )
    payload = resolved.logical_endpoint.model_dump(mode="json")
    if mutation == "duplicate_backends":
        payload["backend_ids"] *= 2
    else:
        payload["retry_status_codes"] = []

    with pytest.raises(ValidationError, match=message):
        ResolvedLogicalEndpoint.model_validate_json(json.dumps(payload))


def test_resolution_context_requires_matching_client_endpoint(single_node_plan: ResolvedSlurmRunPlan) -> None:
    client_without_ports = single_node_plan.client.model_copy(update={"ports": ()})
    plan_without_ports = single_node_plan.model_copy(update={"client": client_without_ports})

    with pytest.raises(ValueError, match="exactly one logical endpoint"):
        ServerResolutionContext.from_plan(plan_without_ports, single_node_plan.deployments[0].deployment_id)


def test_resolution_context_derives_placement_and_endpoint_from_one_plan(
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    payload = single_node_plan.model_dump(mode="json")
    payload["deployments"][0]["ports"][0]["port"] = 18100
    payload["client"]["ports"][0]["port"] = 17100
    alternate_plan = ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))

    context = ServerResolutionContext.from_plan(alternate_plan, "deployment-00000")

    assert context.placement.ports[0].port == 18100
    assert context.logical_endpoint.port == 17100


def test_resolution_context_rejects_unknown_deployment(single_node_plan: ResolvedSlurmRunPlan) -> None:
    with pytest.raises(ValueError, match="exactly one deployment"):
        ServerResolutionContext.from_plan(single_node_plan, "deployment-99999")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_host", "resolved client host"),
        ("port_collision", "must not collide"),
    ],
)
def test_resolution_context_rejects_invalid_client_binding(
    single_node_plan: ResolvedSlurmRunPlan,
    mutation: str,
    message: str,
) -> None:
    placement = single_node_plan.deployments[0]
    logical_endpoint = single_node_plan.client.ports[0]
    client_host_node_index = single_node_plan.client.host_node_index
    if mutation == "wrong_host":
        client_host_node_index = 1
    else:
        logical_endpoint = logical_endpoint.model_copy(
            update={
                "node_index": placement.ports[0].node_index,
                "port": placement.ports[0].port,
            }
        )

    with pytest.raises(ValidationError, match=message):
        ServerResolutionContext(
            placement=placement,
            client_host_node_index=client_host_node_index,
            logical_endpoint=logical_endpoint,
        )


def _context_for_placement(
    plan: ResolvedSlurmRunPlan,
    placement: ResolvedDeployment,
) -> ServerResolutionContext:
    payload = plan.model_dump(mode="json")
    matching_indices = [
        index for index, candidate in enumerate(plan.deployments) if candidate.deployment_id == placement.deployment_id
    ]
    assert len(matching_indices) == 1
    payload["deployments"][matching_indices[0]] = placement.model_dump(mode="json")
    updated_plan = ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))
    return ServerResolutionContext.from_plan(updated_plan, placement.deployment_id)


def _multi_lane_placement(plan: ResolvedSlurmRunPlan) -> ResolvedDeployment:
    payload = plan.deployments[0].model_dump(mode="json")
    payload["authored"]["server"].update(
        lead_boot_standoff="30s",
        rank_launch_stagger="2s",
    )
    payload["authored"]["topology"]["tensor_parallel"] = 4
    payload["topology"].update(
        tensor_parallel=4,
        replicas_per_node_group=2,
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
            "node_index": 0,
            "port": 18001,
        },
        {
            "name": "deployment-00000-rendezvous-00000",
            "role": "rendezvous",
            "node_index": 0,
            "port": 19000,
        },
        {
            "name": "deployment-00000-rendezvous-00001",
            "role": "rendezvous",
            "node_index": 0,
            "port": 19001,
        },
    ]
    return ResolvedDeployment.model_validate_json(json.dumps(payload))


def _multi_group_plan(plan: ResolvedSlurmRunPlan) -> ResolvedSlurmRunPlan:
    payload = plan.model_dump(mode="json")
    payload["deployments"] = payload["deployments"][:1]
    payload["client"]["ports"] = payload["client"]["ports"][:1]
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
            "name": f"deployment-00000-http-{replica_index:05d}",
            "role": "http",
            "node_index": 0 if replica_index < 2 else 2,
            "port": 18000 + replica_index % 2,
        }
        for replica_index in range(4)
    ] + [
        {
            "name": f"deployment-00000-rendezvous-{replica_index:05d}",
            "role": "rendezvous",
            "node_index": 0 if replica_index < 2 else 2,
            "port": 19000 + replica_index % 2,
        }
        for replica_index in range(4)
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
