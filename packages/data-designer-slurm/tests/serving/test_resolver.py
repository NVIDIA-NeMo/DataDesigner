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
    VllmRuntimeCompatibility,
    resolve_server,
)
from data_designer.slurm.serving import resolver as resolver_module

GOLDEN_DIRECTORY = Path(__file__).parent / "golden"


class _UnsupportedServer:
    type: str = "unsupported"


def test_single_node_resolution_matches_golden(single_node_plan: ResolvedSlurmRunPlan) -> None:
    placement = single_node_plan.deployments[0]
    context = ServerResolutionContext.from_plan(placement, single_node_plan.client)

    first = resolve_server(placement.authored, context)
    second = resolve_server(placement.authored, context)

    assert first == second
    assert pretty_json(first.model_dump(mode="json")) == (GOLDEN_DIRECTORY / "single_node.json").read_text()
    assert first.processes[0].role is VllmProcessRole.API_SERVER
    assert first.processes[0].launch_delay_seconds == 0
    assert first.launch_policy.lead_boot_standoff_seconds == 60
    assert first.launch_policy.rank_launch_stagger_seconds == 5
    assert first.launch_policy.queue_backpressure == QueueBackpressureConfig()
    assert first.failure_policy == "coordinated"


def test_multi_node_lane_resolution_matches_golden(multi_node_plan: ResolvedSlurmRunPlan) -> None:
    placement = _multi_lane_placement(multi_node_plan)
    context = ServerResolutionContext.from_plan(placement, multi_node_plan.client)

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


def test_two_deployments_keep_images_and_endpoint_identities_isolated(
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    resolved = tuple(
        resolve_server(placement.authored, ServerResolutionContext.from_plan(placement, multi_node_plan.client))
        for placement in multi_node_plan.deployments
    )

    assert resolved[0].image.sha256 != resolved[1].image.sha256
    assert resolved[0].model_alias != resolved[1].model_alias
    assert resolved[0].model != resolved[1].model
    assert resolved[0].served_model_name != resolved[1].served_model_name
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
    context = ServerResolutionContext.from_plan(placement, single_node_plan.client)
    different = placement.authored.model_copy(update={"model": "example/different"})

    with pytest.raises(ServerResolutionError, match="does not match"):
        resolve_server(different, context)


def test_resolution_rejects_unsupported_inspected_version(single_node_plan: ResolvedSlurmRunPlan) -> None:
    payload = single_node_plan.deployments[0].model_dump(mode="json")
    payload["image"]["inspection"]["inspection"]["runtime_version"] = "0.20.0"
    placement = ResolvedDeployment.model_validate_json(json.dumps(payload))
    context = ServerResolutionContext.from_plan(placement, single_node_plan.client)

    with pytest.raises(ValueError, match="unsupported inspected vLLM version"):
        resolve_server(placement.authored, context)


def test_resolution_rejects_image_inspection_mismatch(single_node_plan: ResolvedSlurmRunPlan) -> None:
    placement = single_node_plan.deployments[0].model_copy(update={"image": single_node_plan.client.image})
    context = ServerResolutionContext.model_construct(
        placement=placement,
        logical_endpoint=single_node_plan.client.ports[0],
    )

    with pytest.raises(ServerResolutionError, match="serving image"):
        resolve_server(placement.authored, context)


def test_resolution_rejects_missing_multi_node_capability(
    multi_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    placement = multi_node_plan.deployments[0]
    context = ServerResolutionContext.from_plan(placement, multi_node_plan.client)
    compatibility = resolve_server(placement.authored, context).compatibility.model_copy(
        update={"supports_multi_node": False}
    )
    monkeypatch.setattr(resolver_module, "resolve_vllm_compatibility", lambda _version: compatibility)

    with pytest.raises(ServerResolutionError, match="multi-node topology"):
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
    resolved = resolve_server(placement.authored, ServerResolutionContext.from_plan(placement, multi_node_plan.client))
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
        ("runtime_version", "inspected runtime version"),
        ("executable", "executable path"),
        ("nodes", "sorted and unique"),
        ("node_group_divisibility", "divide evenly into replica groups"),
        ("gpu_divisibility", "divide evenly into tensor-parallel lanes"),
        ("topology", "node and GPU resources"),
        ("multi_node_capability", "package-owned runtime mapping"),
        ("required_capability", "package-owned runtime mapping"),
        ("expert_parallel", "multi-node expert parallel"),
        ("logical_alias", "model alias"),
        ("logical_served_name", "served model name"),
        ("logical_id", "endpoint ID"),
        ("replica_index", "replica identities"),
        ("backend_id", "endpoint IDs"),
        ("logical_backends", "resolved backend order"),
        ("process_ids", "process IDs"),
        ("readiness_count", "readiness probes"),
        ("backend_placement", "replica placement"),
        ("readiness_target", "readiness probes"),
        ("pipeline_ranks", "ordered pipeline rank"),
        ("process_topology", "topology and launch policy"),
        ("head_endpoint", "head process"),
        ("rendezvous_consistency", "share one rendezvous"),
        ("rendezvous_master", "group head and distributed timeout"),
        ("rendezvous_timeout", "group head and distributed timeout"),
        ("network_address", "network addresses must be unique"),
        ("process_count", "process count"),
    ],
)
def test_resolved_server_rejects_one_field_join_drift(
    multi_node_plan: ResolvedSlurmRunPlan,
    mutation: str,
    message: str,
) -> None:
    placement = _multi_lane_placement(multi_node_plan)
    resolved = resolve_server(placement.authored, ServerResolutionContext.from_plan(placement, multi_node_plan.client))
    payload = resolved.model_dump(mode="json")
    if mutation == "image_kind":
        payload["image"] = multi_node_plan.client.image.model_dump(mode="json")
    elif mutation == "runtime_version":
        payload["compatibility"]["runtime_version"] = "0.22.0"
        payload["compatibility"]["runtime_series"] = "0.22"
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
    elif mutation == "multi_node_capability":
        payload["compatibility"]["supports_multi_node"] = False
    elif mutation == "required_capability":
        payload["compatibility"]["supports_http_readiness"] = False
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
    else:
        extra = dict(payload["processes"][0])
        extra.update(process_id="deployment-00000-replica-99999-rank-00000", replica_index=99999)
        payload["processes"].append(extra)

    with pytest.raises(ValidationError, match=message):
        ResolvedServerDeployment.model_validate_json(json.dumps(payload))


def test_compatibility_rejects_runtime_series_drift() -> None:
    with pytest.raises(ValidationError, match="runtime series"):
        VllmRuntimeCompatibility(
            runtime_version="0.21.0",
            runtime_series="0.22",
            supports_single_node=True,
            supports_multi_node=True,
            supports_http_readiness=True,
            supports_queue_backpressure=True,
            supports_coordinated_failure=True,
        )


def test_compatibility_rejects_invalid_runtime_version() -> None:
    with pytest.raises(ValidationError, match="invalid inspected vLLM version"):
        VllmRuntimeCompatibility(
            runtime_version="invalid",
            runtime_series="invalid",
            supports_single_node=True,
            supports_multi_node=True,
            supports_http_readiness=True,
            supports_queue_backpressure=True,
            supports_coordinated_failure=True,
        )


def test_compatibility_rejects_capabilities_outside_package_mapping() -> None:
    with pytest.raises(ValidationError, match="package-owned runtime mapping"):
        VllmRuntimeCompatibility(
            runtime_version="0.21.0",
            runtime_series="0.21",
            supports_single_node=True,
            supports_multi_node=False,
            supports_http_readiness=True,
            supports_queue_backpressure=True,
            supports_coordinated_failure=True,
        )


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"readiness_path": "health"}, "absolute URL path"),
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


def test_single_node_process_rejects_rendezvous(single_node_plan: ResolvedSlurmRunPlan) -> None:
    placement = single_node_plan.deployments[0]
    resolved = resolve_server(placement.authored, ServerResolutionContext.from_plan(placement, single_node_plan.client))
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
    resolved = resolve_server(placement.authored, ServerResolutionContext.from_plan(placement, single_node_plan.client))
    payload = resolved.logical_endpoint.model_dump(mode="json")
    if mutation == "duplicate_backends":
        payload["backend_ids"] *= 2
    else:
        payload["retry_status_codes"] = []

    with pytest.raises(ValidationError, match=message):
        ResolvedLogicalEndpoint.model_validate_json(json.dumps(payload))


def test_resolved_server_rejects_single_node_capability_drift(single_node_plan: ResolvedSlurmRunPlan) -> None:
    placement = single_node_plan.deployments[0]
    resolved = resolve_server(placement.authored, ServerResolutionContext.from_plan(placement, single_node_plan.client))
    payload = resolved.model_dump(mode="json")
    payload["compatibility"]["supports_single_node"] = False

    with pytest.raises(ValidationError, match="package-owned runtime mapping"):
        ResolvedServerDeployment.model_validate_json(json.dumps(payload))


def test_resolution_context_requires_matching_client_endpoint(single_node_plan: ResolvedSlurmRunPlan) -> None:
    client_without_ports = single_node_plan.client.model_copy(update={"ports": ()})

    with pytest.raises(ValueError, match="exactly one logical endpoint"):
        ServerResolutionContext.from_plan(single_node_plan.deployments[0], client_without_ports)


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
