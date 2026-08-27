# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.serving.deployment import ResolvedVllmServerDeployment
from data_designer.slurm.serving.resolver import VllmServerResolutionContext, resolve_vllm_server


@pytest.fixture
def multi_replica_plan(multi_node_plan: ResolvedSlurmRunPlan) -> ResolvedSlurmRunPlan:
    payload = multi_node_plan.model_dump(mode="json")
    deployment = payload["deployments"][0]
    deployment["authored"]["server"].update(
        lead_boot_standoff="30s",
        rank_launch_stagger="2s",
    )
    deployment["authored"]["topology"]["tensor_parallel"] = 4
    deployment["topology"].update(
        tensor_parallel=4,
        replicas_per_node_group=2,
        replica_count=2,
        gpus_per_replica=8,
    )
    deployment["ports"] = [
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
    return ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))


@pytest.fixture
def resolved_multi_replica_server(multi_replica_plan: ResolvedSlurmRunPlan) -> ResolvedVllmServerDeployment:
    deployment = multi_replica_plan.deployments[0]
    return _resolve(multi_replica_plan, deployment.deployment_id)


@pytest.fixture
def resolved_single_node_server(single_node_plan: ResolvedSlurmRunPlan) -> ResolvedVllmServerDeployment:
    deployment = single_node_plan.deployments[0]
    return _resolve(single_node_plan, deployment.deployment_id)


def _resolve(plan: ResolvedSlurmRunPlan, deployment_id: str) -> ResolvedVllmServerDeployment:
    deployments = tuple(deployment for deployment in plan.deployments if deployment.deployment_id == deployment_id)
    endpoint_ports = tuple(port for port in plan.client.ports if port.name == f"{deployment_id}-logical-endpoint")
    assert len(deployments) == 1
    assert len(endpoint_ports) == 1
    return resolve_vllm_server(
        VllmServerResolutionContext(
            deployment=deployments[0],
            logical_endpoint_port=endpoint_ports[0],
        )
    )
