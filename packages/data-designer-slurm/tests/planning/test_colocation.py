# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from data_designer.config import DataDesignerConfig
from data_designer.slurm.config import (
    DataDesignerSlurmConfig,
    DataDesignerSlurmConfigBuilder,
    injected_profile,
    load_run_config,
)
from data_designer.slurm.launcher.renderer import render_generation_attempt_script
from data_designer.slurm.planning import ResolvedDependencyLock, ResolvedSlurmRunPlan
from data_designer.slurm.planning.compiler import SlurmRunCompiler
from data_designer.slurm.planning.errors import SlurmConfigResolutionError
from data_designer.slurm.planning.resolution import resolve_slurm_config
from data_designer.slurm.runtime.ports import resolve_allocation_plan
from data_designer.slurm.serving.resolver import resolve_vllm_server


@pytest.mark.parametrize("suffix", ("yaml", "json"))
def test_shared_node_authoring_round_trip(
    shared_node_config: DataDesignerSlurmConfig, tmp_path: Path, suffix: str
) -> None:
    builder = DataDesignerSlurmConfigBuilder(shared_node_config.builder).with_invocation(
        num_records=4, dataset_name="judged", model_concurrency={"writer": 1, "judge": 1}
    )
    for deployment in shared_node_config.deployments:
        builder.with_deployment(deployment)
    path = tmp_path / f"run.{suffix}"
    builder.write_config(path)
    assert load_run_config(path) == builder.build() == shared_node_config


@pytest.mark.parametrize(
    ("resources", "topology", "message"),
    [
        ({"shared_node": "models"}, {}, "provided together"),
        ({"gpus_per_node": 4}, {}, "provided together"),
        ({"shared_node": "models", "gpus_per_node": 0}, {}, "allowed minimum"),
        ({"shared_node": "models", "gpus_per_node": -1}, {}, "allowed minimum"),
        ({"shared_node": "models", "gpus_per_node": 4, "nodes": 2}, {}, "exactly one"),
        ({"shared_node": "models", "gpus_per_node": 4}, {"nodes_per_replica": 2}, "single-node"),
        ({"shared_node": "models", "gpus_per_node": 3}, {"tensor_parallel": 2}, "GPU quota"),
    ],
)
def test_shared_node_builder_rejects_invalid_resources(
    shared_node_config: DataDesignerSlurmConfig, resources: dict[str, object], topology: dict[str, int], message: str
) -> None:
    deployment = shared_node_config.deployments[0].model_dump(mode="python")
    deployment["resources"] = resources
    deployment["topology"].update(topology)
    with pytest.raises(ValueError, match=message):
        DataDesignerSlurmConfigBuilder(shared_node_config.builder).with_deployment(deployment)


@pytest.mark.parametrize("mismatch", ("singleton", "disjoint", "environment", "queue_backpressure"))
def test_shared_node_rejects_incompatible_groups(shared_node_config: DataDesignerSlurmConfig, mismatch: str) -> None:
    payload = shared_node_config.model_dump(mode="python")
    if mismatch == "singleton":
        payload["deployments"][1]["resources"]["shared_node"] = "other"
    elif mismatch == "disjoint":
        payload["deployments"][1]["resources"] = {"nodes": 1}
    elif mismatch == "environment":
        payload["deployments"][1]["server"]["environment"] = {"MODE": {"type": "literal", "value": "other"}}
    else:
        payload["deployments"][1]["server"]["queue_backpressure"]["max_waiting_requests"] = 2
    with pytest.raises(ValidationError, match="shared_node"):
        DataDesignerSlurmConfig.model_validate(payload)


def test_shared_node_plan_assigns_distinct_models_gpus_ports_and_one_node(
    shared_node_plan: ResolvedSlurmRunPlan,
) -> None:
    plan = ResolvedSlurmRunPlan.model_validate_json(shared_node_plan.serialize_json())
    assert plan.compute_sha256() == shared_node_plan.compute_sha256()
    assert [deployment.node_indices for deployment in plan.deployments] == [(0,), (0,)]
    assert [deployment.gpu_start_index for deployment in plan.deployments] == [0, 4]
    servers = tuple(resolve_vllm_server(plan, deployment.deployment_id) for deployment in plan.deployments)
    assert len({server.model for server in servers}) == 2
    assert [process.gpu_indices for server in servers for process in server.processes] == [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
    ]
    assert [backend.port for server in servers for backend in server.backend_endpoints] == [18000, 18001, 18002, 18003]
    assert [server.logical_endpoint.port for server in servers] == [17000, 17001]
    script = render_generation_attempt_script(plan, attempt_ordinal=1)
    assert "#SBATCH --nodes=1\n" in script
    assert "#SBATCH --gres=gpu:8\n" in script
    assert "#SBATCH --exclusive" not in script


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("overlap", "GPU ownership"),
        ("oversubscription", "quotas exceed"),
        ("node", "placement and GPU ownership"),
        ("image", "same resolved serving image"),
        ("visible", "requires GRES"),
        ("environment", "matching server environment"),
        ("queue", "matching server environment"),
    ),
)
def test_shared_node_plan_rejects_invalid_ownership(
    shared_node_plan: ResolvedSlurmRunPlan, mutation: str, message: str
) -> None:
    payload = shared_node_plan.model_dump(mode="python")
    if mutation == "overlap":
        payload["deployments"][1]["gpu_start_index"] = 0
    elif mutation == "oversubscription":
        payload["deployments"][1]["authored"]["resources"]["gpus_per_node"] = 8
    elif mutation == "node":
        payload["deployments"][1]["node_indices"] = (1,)
        for port in payload["deployments"][1]["ports"]:
            port["node_index"] = 1
    elif mutation == "image":
        payload["deployments"][1]["image"]["path"] = "/images/other.sqsh"
    elif mutation == "visible":
        profile = shared_node_plan.selected_profile.profile
        visible_profile = profile.model_copy(
            update={
                "gpu_request_mode": "visible",
                "scheduler": profile.scheduler.model_copy(update={"mem_per_gpu": None}),
            }
        )
        payload["selected_profile"] = injected_profile(visible_profile).model_dump(mode="python")
    elif mutation == "environment":
        payload["deployments"][1]["authored"]["server"]["environment"] = {"MODE": {"type": "literal", "value": "other"}}
    else:
        payload["deployments"][1]["authored"]["server"]["queue_backpressure"]["max_waiting_requests"] = 2
    with pytest.raises(ValidationError, match=message):
        ResolvedSlurmRunPlan.model_validate(payload)


def test_documented_colocated_example_validates_both_public_configs() -> None:
    path = Path(__file__).parents[2] / "examples/colocated-models.yaml"
    config = load_run_config(path)
    builder = DataDesignerConfig.model_validate(config.builder.inline["data_designer"])
    assert [model.alias for model in builder.model_configs] == ["writer", "judge"]
    assert [column.name for column in builder.columns] == ["answer", "review"]
    assert len({model.model for model in builder.model_configs}) == 2


def test_shared_node_runtime_ports_are_distinct_and_isolated(shared_node_plan: ResolvedSlurmRunPlan) -> None:
    first = resolve_allocation_plan(shared_node_plan, {"SLURM_JOB_GPUS": "1,3,5,7"})
    second = resolve_allocation_plan(shared_node_plan, {"SLURM_JOB_GPUS": "0,2,4,6"})
    first_ports = [port.port for port in first.client.ports] + [
        port.port for deployment in first.deployments for port in deployment.ports
    ]
    second_ports = [port.port for port in second.client.ports] + [
        port.port for deployment in second.deployments for port in deployment.ports
    ]
    assert len(set(first_ports)) == 6
    assert not set(first_ports).intersection(second_ports)
    assert [deployment.node_indices for deployment in first.deployments] == [(0,), (0,)]
    assert first.authored_config == shared_node_plan.authored_config


@pytest.mark.parametrize("invalid", ("oversubscription", "visible", "image"))
def test_preparation_rejects_unsupported_shared_placement(
    shared_node_config: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    dependency_lock_single: ResolvedDependencyLock,
    invalid: str,
) -> None:
    payload = shared_node_config.model_dump(mode="python")
    profile = single_node_plan.selected_profile.profile
    images = (single_node_plan.deployments[0].image,) * 2
    if invalid == "oversubscription":
        payload["deployments"][1]["resources"]["gpus_per_node"] = 8
    elif invalid == "visible":
        profile = profile.model_copy(
            update={
                "gpu_request_mode": "visible",
                "scheduler": profile.scheduler.model_copy(update={"mem_per_gpu": None}),
            }
        )
    else:
        images = (images[0], images[1].model_copy(update={"path": "/images/other.sqsh"}))
    with pytest.raises(SlurmConfigResolutionError, match="shared_node"):
        resolve_slurm_config(
            DataDesignerSlurmConfig.model_validate(payload),
            selected_profile=injected_profile(profile),
            client_runtime=single_node_plan.client.runtime,
            deployment_images=images,
            dependency_lock=dependency_lock_single,
            runtime_bundle=single_node_plan.runtime_bundle,
            run_id=single_node_plan.run_id,
            package_version=single_node_plan.package_version,
        )


def test_partial_gpu_allocation_uses_per_model_quota_for_replica_count(
    shared_node_config: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    dependency_lock_single: ResolvedDependencyLock,
) -> None:
    payload = shared_node_config.model_dump(mode="python")
    for deployment, quota in zip(payload["deployments"], (1, 2), strict=True):
        deployment["resources"]["gpus_per_node"] = quota
        deployment["topology"]["tensor_parallel"] = quota
    profile = single_node_plan.selected_profile.profile.model_copy(update={"gpus_per_node": 3})
    plan = SlurmRunCompiler.compile(
        resolve_slurm_config(
            DataDesignerSlurmConfig.model_validate(payload),
            selected_profile=injected_profile(profile),
            client_runtime=single_node_plan.client.runtime,
            deployment_images=(single_node_plan.deployments[0].image,) * 2,
            dependency_lock=dependency_lock_single,
            runtime_bundle=single_node_plan.runtime_bundle,
            run_id=single_node_plan.run_id,
            package_version=single_node_plan.package_version,
        )
    )
    assert [deployment.topology.replica_count for deployment in plan.deployments] == [1, 1]
    assert [deployment.gpu_start_index for deployment in plan.deployments] == [0, 1]
    assert [deployment.gpus_per_node for deployment in plan.deployments] == [1, 2]
    script = render_generation_attempt_script(plan, attempt_ordinal=1)
    assert "#SBATCH --nodes=1\n" in script and "#SBATCH --gres=gpu:3\n" in script
    assert "#SBATCH --exclusive" not in script
