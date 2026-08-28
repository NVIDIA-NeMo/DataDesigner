# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.serving.deployment import ResolvedVllmServerDeployment
from data_designer.slurm.serving.endpoints import ResolvedLogicalEndpoint
from data_designer.slurm.serving.vllm import ResolvedVllmProcess


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
    resolved_multi_replica_server: ResolvedVllmServerDeployment,
    mutation: str,
    message: str,
) -> None:
    process_index = (
        1 if mutation.startswith("follower") or mutation in {"missing_rendezvous", "wrong_rendezvous_lane"} else 0
    )
    payload = resolved_multi_replica_server.processes[process_index].model_dump(mode="json")
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
        payload["rendezvous"]["replica_index_in_node_group"] = 1

    with pytest.raises(ValidationError, match=message):
        ResolvedVllmProcess.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("image_kind", "server image inspection"),
        ("executable", "executable path"),
        ("nodes", "sorted and unique"),
        ("node_group_divisibility", "must divide the deployment node count"),
        ("gpu_divisibility", "must divide resolved GPUs per node"),
        ("topology", "node and GPU resources"),
        ("expert_parallel", "multi-node expert parallel"),
        ("logical_alias", "model alias"),
        ("logical_served_name", "served model name"),
        ("logical_id", "endpoint ID"),
        ("deployment_replica_index", "replica identities"),
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
    resolved_multi_replica_server: ResolvedVllmServerDeployment,
    mutation: str,
    message: str,
) -> None:
    payload = resolved_multi_replica_server.model_dump(mode="json")
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
    elif mutation == "deployment_replica_index":
        payload["backend_endpoints"][1]["deployment_replica_index"] = 2
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
        payload["backend_endpoints"][1]["replica_index_in_node_group"] = 0
    elif mutation == "readiness_target":
        payload["readiness_probes"][1]["path"] = "/other"
    elif mutation == "pipeline_ranks":
        payload["processes"][1]["deployment_replica_index"] = 1
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
        extra.update(
            process_id="deployment-00000-replica-99999-rank-00000",
            deployment_replica_index=99999,
        )
        payload["processes"].append(extra)

    with pytest.raises(ValidationError, match=message):
        ResolvedVllmServerDeployment.model_validate_json(json.dumps(payload))


def test_single_node_process_rejects_rendezvous(
    resolved_single_node_server: ResolvedVllmServerDeployment,
) -> None:
    payload = resolved_single_node_server.processes[0].model_dump(mode="json")
    payload["rendezvous"] = {
        "node_group_index": 0,
        "replica_index_in_node_group": 0,
        "master_node_index": 0,
        "port": 19000,
        "timeout_seconds": 600,
    }

    with pytest.raises(ValidationError, match="must not carry rendezvous"):
        ResolvedVllmProcess.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_backends", "must be unique"),
        ("missing_retry", "must retry HTTP 429"),
    ],
)
def test_logical_endpoint_rejects_invalid_backend_or_retry_contract(
    resolved_single_node_server: ResolvedVllmServerDeployment,
    mutation: str,
    message: str,
) -> None:
    payload = resolved_single_node_server.logical_endpoint.model_dump(mode="json")
    if mutation == "duplicate_backends":
        payload["backend_ids"] *= 2
    else:
        payload["retry_status_codes"] = []

    with pytest.raises(ValidationError, match=message):
        ResolvedLogicalEndpoint.model_validate_json(json.dumps(payload))
