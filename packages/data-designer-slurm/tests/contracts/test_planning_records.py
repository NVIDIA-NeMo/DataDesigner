# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from data_designer.slurm._contracts import compute_sha256
from data_designer.slurm.config import DataDesignerSlurmConfig
from data_designer.slurm.planning import (
    ArtifactReference,
    PlanContractError,
    ResolvedDependencyLock,
    ResolvedSlurmRunPlan,
    ResolvedSubmission,
    validate_resolved_plan,
)


def test_multi_node_plan_matches_authored_inputs(
    authored_run: DataDesignerSlurmConfig,
    dependency_lock: ResolvedDependencyLock,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    assert validate_resolved_plan(authored_run, dependency_lock, multi_node_plan) is multi_node_plan
    assert multi_node_plan.authored_config.sha256 == compute_sha256(authored_run.model_dump(mode="json"))
    assert [deployment.topology.replica_count for deployment in multi_node_plan.deployments] == [1, 8]
    assert multi_node_plan.client.gpu_count == 0


def test_single_node_plan_matches_authored_inputs(
    authored_run_single: DataDesignerSlurmConfig,
    dependency_lock_single: ResolvedDependencyLock,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    assert validate_resolved_plan(authored_run_single, dependency_lock_single, single_node_plan) is single_node_plan
    assert single_node_plan.authored_config.sha256 == compute_sha256(authored_run_single.model_dump(mode="json"))
    assert [deployment.topology.replica_count for deployment in single_node_plan.deployments] == [1]
    assert single_node_plan.client.gpu_count == 0


def test_plan_canonical_json_is_byte_stable(multi_node_plan: ResolvedSlurmRunPlan) -> None:
    payload = json.loads(multi_node_plan.serialize_json())
    payload["invocation"]["authored"]["model_concurrency"] = {"judge": 32, "generator": 64}
    reordered = ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))

    assert reordered.serialize_canonical_json() == multi_node_plan.serialize_canonical_json()


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload.pop("schema_version"),
        lambda payload: payload.update(schema_version=2),
        lambda payload: payload.update(unknown=True),
        lambda payload: payload.update(resolved_gpus_per_node=4),
        lambda payload: payload["client"].update(host_node_index=1),
        lambda payload: payload["deployments"][1].update(node_indices=[1]),
        lambda payload: payload["deployments"][0]["ports"][1].update(port=18000),
        lambda payload: payload["deployments"][1].update(ports=payload["deployments"][1]["ports"][:1]),
        lambda payload: payload.update(shards=payload["shards"][:1]),
        lambda payload: payload["shards"][1].update(start_index=49),
        lambda payload: payload["output"].update(root="/outside/output"),
        lambda payload: payload.update(container_mounts=[]),
        lambda payload: payload["deployments"][0]["topology"].update(replica_count=2),
    ],
)
def test_plan_rejects_invalid_boundaries(multi_node_plan: ResolvedSlurmRunPlan, mutator: object) -> None:
    payload = deepcopy(multi_node_plan.model_dump(mode="json"))
    mutator(payload)

    with pytest.raises(ValidationError):
        ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))


def test_plan_rejects_unmaterialized_run_config(multi_node_plan: ResolvedSlurmRunPlan) -> None:
    payload = multi_node_plan.model_dump(mode="json")
    payload["invocation"]["effective_run_config"] = {}

    with pytest.raises(ValidationError, match="fully materialized"):
        ResolvedSlurmRunPlan.model_validate(payload)


@pytest.mark.parametrize(
    "update",
    [
        {"time_limit": "invalid"},
        {"comment": "bad\ncomment"},
    ],
)
def test_resolved_submission_preserves_authored_validation(update: dict[str, object]) -> None:
    payload = {"job_name": "data-designer", "time_limit": "03:55:00", **update}

    with pytest.raises(ValidationError):
        ResolvedSubmission.model_validate(payload)


def test_resolved_image_rejects_digest_mismatch(multi_node_plan: ResolvedSlurmRunPlan) -> None:
    payload = multi_node_plan.model_dump(mode="json")
    payload["client"]["image"]["inspection"]["sqsh_sha256"] = "a" * 64

    with pytest.raises(ValidationError, match="digest"):
        ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload["overlay_packages"].append(
            {
                "name": "data-designer",
                "version": "0.9.2",
                "artifact": {"path": "/wheels/data_designer.whl", "sha256": "a" * 64},
            }
        ),
        lambda payload: payload.update(image_distributions=list(reversed(payload["image_distributions"]))),
        lambda payload: payload["overlay_packages"][0]["artifact"].update(path="/wheels/plugin.tar.gz"),
    ],
)
def test_dependency_lock_rejects_overlap_order_and_non_wheel(
    dependency_lock: ResolvedDependencyLock,
    mutator: object,
) -> None:
    payload = deepcopy(dependency_lock.model_dump(mode="json"))
    mutator(payload)

    with pytest.raises(ValidationError):
        ResolvedDependencyLock.model_validate_json(json.dumps(payload))


def test_cross_record_validation_rejects_authored_digest(
    authored_run: DataDesignerSlurmConfig,
    dependency_lock: ResolvedDependencyLock,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    invalid = multi_node_plan.model_copy(
        update={
            "authored_config": ArtifactReference(
                path=multi_node_plan.authored_config.path,
                sha256="0" * 64,
            )
        }
    )

    with pytest.raises(PlanContractError, match="authored config digest"):
        validate_resolved_plan(authored_run, dependency_lock, invalid)


def test_cross_record_validation_rejects_dependency_lock_digest(
    authored_run: DataDesignerSlurmConfig,
    dependency_lock: ResolvedDependencyLock,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    client = multi_node_plan.client.model_copy(
        update={
            "dependency_lock": ArtifactReference(
                path=multi_node_plan.client.dependency_lock.path,
                sha256="0" * 64,
            )
        }
    )
    invalid = multi_node_plan.model_copy(update={"client": client})

    with pytest.raises(PlanContractError, match="dependency lock digest"):
        validate_resolved_plan(authored_run, dependency_lock, invalid)


def test_cross_record_validation_rejects_python_abi(
    authored_run: DataDesignerSlurmConfig,
    dependency_lock: ResolvedDependencyLock,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    invalid_lock = dependency_lock.model_copy(update={"python_abi": "cp311"})
    client = multi_node_plan.client.model_copy(
        update={
            "dependency_lock": multi_node_plan.client.dependency_lock.model_copy(
                update={"sha256": invalid_lock.compute_sha256()}
            )
        }
    )
    invalid_plan = multi_node_plan.model_copy(update={"client": client})

    with pytest.raises(PlanContractError, match="Python ABI"):
        validate_resolved_plan(authored_run, invalid_lock, invalid_plan)


def test_cross_record_validation_rejects_image_inventory(
    authored_run: DataDesignerSlurmConfig,
    dependency_lock: ResolvedDependencyLock,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    invalid_lock = dependency_lock.model_copy(update={"image_distributions": ()})
    client = multi_node_plan.client.model_copy(
        update={
            "dependency_lock": multi_node_plan.client.dependency_lock.model_copy(
                update={"sha256": invalid_lock.compute_sha256()}
            )
        }
    )
    invalid_plan = multi_node_plan.model_copy(update={"client": client})

    with pytest.raises(PlanContractError, match="image inventory"):
        validate_resolved_plan(authored_run, invalid_lock, invalid_plan)


def test_cross_record_validation_rejects_changed_invocation(
    authored_run: DataDesignerSlurmConfig,
    dependency_lock: ResolvedDependencyLock,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    invocation = multi_node_plan.invocation.model_copy(
        update={"authored": authored_run.invocation.model_copy(update={"dataset_name": "other"})}
    )
    invalid = multi_node_plan.model_copy(update={"invocation": invocation})

    with pytest.raises(PlanContractError, match="invocation"):
        validate_resolved_plan(authored_run, dependency_lock, invalid)
