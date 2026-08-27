# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from data_designer.slurm.config import BuilderInput, ClientDependencies, DataDesignerSlurmConfig
from data_designer.slurm.contracts import compute_serialized_json_sha256
from data_designer.slurm.planning import (
    ArtifactReference,
    ResolvedDependencyLock,
    ResolvedDeployment,
    ResolvedSlurmRunPlan,
    ResolvedSubmission,
)
from data_designer.slurm.planning.errors import SlurmPlanContractError
from data_designer.slurm.planning.validation import validate_resolved_plan


def test_multi_node_plan_matches_authored_inputs(
    authored_run: DataDesignerSlurmConfig,
    dependency_lock: ResolvedDependencyLock,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    assert validate_resolved_plan(authored_run, dependency_lock, multi_node_plan) is multi_node_plan
    assert multi_node_plan.authored_config.sha256 == authored_run.compute_sha256()
    assert [deployment.topology.replica_count for deployment in multi_node_plan.deployments] == [1, 8]
    assert multi_node_plan.client.gpu_count == 0


def test_single_node_plan_matches_authored_inputs(
    authored_run_single: DataDesignerSlurmConfig,
    dependency_lock_single: ResolvedDependencyLock,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    assert validate_resolved_plan(authored_run_single, dependency_lock_single, single_node_plan) is single_node_plan
    assert single_node_plan.authored_config.sha256 == authored_run_single.compute_sha256()
    assert [deployment.topology.replica_count for deployment in single_node_plan.deployments] == [1]
    assert single_node_plan.client.gpu_count == 0


def test_plan_canonical_json_is_byte_stable(multi_node_plan: ResolvedSlurmRunPlan) -> None:
    payload = json.loads(multi_node_plan.serialize_json())
    payload["invocation"]["authored"]["model_concurrency"] = {"judge": 32, "generator": 64}
    reordered = ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))

    assert reordered.serialize_canonical_json() == multi_node_plan.serialize_canonical_json()


def test_resolved_plan_is_deeply_immutable(multi_node_plan: ResolvedSlurmRunPlan) -> None:
    inline = multi_node_plan.builder.inline
    assert inline is not None
    data_designer = inline["data_designer"]
    assert isinstance(data_designer, dict)
    columns = data_designer["columns"]
    assert isinstance(columns, list)

    with pytest.raises(TypeError, match="frozen dictionary"):
        multi_node_plan.invocation.effective_run_config["buffer_size"] = 1
    with pytest.raises(TypeError, match="frozen list"):
        columns.append({})


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload.pop("schema_version"),
        lambda payload: payload.update(schema_version=2),
        lambda payload: payload.update(unknown=True),
        lambda payload: payload.update(resolved_gpus_per_node=4),
        lambda payload: payload["client"].update(host_node_index=1),
        lambda payload: payload["deployments"][1].update(node_indices=[1]),
        lambda payload: payload["deployments"][0].update(deployment_id="unrelated-runtime-name"),
        lambda payload: payload["deployments"][0]["ports"][1].update(port=18000),
        lambda payload: payload["deployments"][1].update(ports=payload["deployments"][1]["ports"][:1]),
        lambda payload: payload["client"].update(ports=payload["client"]["ports"][:1]),
        lambda payload: payload.update(shards=payload["shards"][:1]),
        lambda payload: payload["shards"][1]["record_range"].update(start_index=49),
        lambda payload: payload["shards"][1].update(shard_id="shard-00002"),
        lambda payload: payload["shards"][1].update(resume_workspace=payload["shards"][0]["resume_workspace"]),
        lambda payload: payload.update(run_id="run-other"),
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


def test_plan_preserves_default_non_inference_worker_count(
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    payload = multi_node_plan.model_dump(mode="json")
    payload["invocation"]["effective_run_config"]["non_inference_max_parallel_workers"] = 32

    with pytest.raises(ValidationError, match="RunConfig default"):
        ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))


def test_plan_preserves_explicit_non_inference_worker_override(
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    payload = multi_node_plan.model_dump(mode="json")
    payload["invocation"]["authored"]["run_config"]["non_inference_max_parallel_workers"] = 32
    payload["invocation"]["effective_run_config"]["non_inference_max_parallel_workers"] = 32

    plan = ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))

    assert plan.invocation.effective_run_config["non_inference_max_parallel_workers"] == 32


def test_plan_rejects_otel_port_collision(multi_node_plan: ResolvedSlurmRunPlan) -> None:
    payload = multi_node_plan.model_dump(mode="json")
    payload["invocation"]["effective_run_config"]["otel_metrics_port"] = 18000

    with pytest.raises(ValidationError, match="OTEL metrics port collides"):
        ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    "output_root",
    [
        "/workspace/primary/runs/run-001/shards/shard-00000/dataset",
        "/workspace/primary/runs/run-001",
        "/workspace/primary/runs",
    ],
)
def test_plan_rejects_output_overlapping_shard_workspace(
    multi_node_plan: ResolvedSlurmRunPlan,
    output_root: str,
) -> None:
    payload = multi_node_plan.model_dump(mode="json")
    payload["output"]["root"] = output_root

    with pytest.raises(ValidationError, match="must not overlap"):
        ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))


def test_plan_rejects_deployment_alias_missing_from_builder(multi_node_plan: ResolvedSlurmRunPlan) -> None:
    payload = multi_node_plan.model_dump(mode="json")
    payload["builder"]["inline"]["data_designer"]["model_configs"] = payload["builder"]["inline"]["data_designer"][
        "model_configs"
    ][:1]
    payload["builder"]["model_aliases"] = ["generator"]
    payload["builder"]["content_sha256"] = compute_serialized_json_sha256(payload["builder"]["inline"])

    with pytest.raises(ValidationError, match="deployment alias"):
        ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))


def test_multi_node_tp4_requires_rendezvous_per_replica_lane(multi_node_plan: ResolvedSlurmRunPlan) -> None:
    payload = multi_node_plan.deployments[0].model_dump(mode="json")
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
    ]

    with pytest.raises(ValidationError, match="rendezvous"):
        ResolvedDeployment.model_validate_json(json.dumps(payload))

    payload["ports"].append(
        {
            "name": "deployment-00000-rendezvous-00001",
            "role": "rendezvous",
            "node_index": 0,
            "port": 19001,
        }
    )
    assert ResolvedDeployment.model_validate_json(json.dumps(payload)).topology.replica_count == 2


def test_sourced_builder_validation_requires_resolved_payload(
    authored_run: DataDesignerSlurmConfig,
    dependency_lock: ResolvedDependencyLock,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    sourced_authored = authored_run.model_copy(update={"builder": BuilderInput(source="builder.json")})
    builder_digest = compute_serialized_json_sha256(authored_run.builder.inline)
    payload = multi_node_plan.model_dump(mode="json")
    payload["authored_config"]["sha256"] = sourced_authored.compute_sha256()
    payload["builder"] = {
        "authored_source": "builder.json",
        "source": {"path": "/workspace/primary/runs/run-001/builder.json", "sha256": builder_digest},
        "inline": None,
        "content_sha256": builder_digest,
        "model_aliases": ["generator", "judge"],
    }
    sourced_plan = ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))

    with pytest.raises(SlurmPlanContractError, match="resolved payload"):
        validate_resolved_plan(sourced_authored, dependency_lock, sourced_plan)

    assert (
        validate_resolved_plan(
            sourced_authored,
            dependency_lock,
            sourced_plan,
            builder_payload=authored_run.builder.inline,
        )
        is sourced_plan
    )


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
        lambda payload: payload.update(authored_source="lock.json"),
        lambda payload: payload.update(
            authored_source="lock.yaml",
            source={"path": "/workspace/lock.yaml", "sha256": "a" * 64},
        ),
    ],
)
def test_dependency_lock_rejects_invalid_boundaries(
    dependency_lock: ResolvedDependencyLock,
    mutator: object,
) -> None:
    payload = deepcopy(dependency_lock.model_dump(mode="json"))
    mutator(payload)

    with pytest.raises(ValidationError):
        ResolvedDependencyLock.model_validate_json(json.dumps(payload))


def test_dependency_lock_requires_every_authored_package(dependency_lock: ResolvedDependencyLock) -> None:
    payload = dependency_lock.model_dump(mode="json")
    payload["overlay_packages"] = []

    with pytest.raises(ValidationError, match="missing from the dependency lock"):
        ResolvedDependencyLock.model_validate_json(json.dumps(payload))


def test_dependency_lock_requires_compatible_versions(dependency_lock: ResolvedDependencyLock) -> None:
    payload = dependency_lock.model_dump(mode="json")
    payload["overlay_packages"][0]["version"] = "0.3.0"

    with pytest.raises(ValidationError, match="does not satisfy"):
        ResolvedDependencyLock.model_validate_json(json.dumps(payload))


def test_dependency_lock_binds_direct_wheel_digest(dependency_lock: ResolvedDependencyLock) -> None:
    payload = dependency_lock.model_dump(mode="json")
    payload["authored_requirements"] = [
        "data-designer-speech @ https://example.test/data_designer_speech.whl#sha256=" + "a" * 64
    ]

    with pytest.raises(ValidationError, match="locked overlay artifact"):
        ResolvedDependencyLock.model_validate_json(json.dumps(payload))

    payload["overlay_packages"][0]["artifact"]["sha256"] = "a" * 64
    assert ResolvedDependencyLock.model_validate_json(json.dumps(payload)).overlay_packages


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

    with pytest.raises(SlurmPlanContractError, match="authored config digest"):
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

    with pytest.raises(SlurmPlanContractError, match="dependency lock digest"):
        validate_resolved_plan(authored_run, dependency_lock, invalid)


def test_cross_record_validation_binds_authored_lock_source(
    authored_run: DataDesignerSlurmConfig,
    dependency_lock: ResolvedDependencyLock,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    dependencies = ClientDependencies(requirements=None, lock_file="locks/user-lock.json")
    authored = authored_run.model_copy(
        update={"client": authored_run.client.model_copy(update={"dependencies": dependencies})}
    )
    plan_payload = multi_node_plan.model_dump(mode="json")
    plan_payload["authored_config"]["sha256"] = authored.compute_sha256()
    plan_payload["client"]["authored"] = authored.client.model_dump(mode="json")
    plan = ResolvedSlurmRunPlan.model_validate_json(json.dumps(plan_payload))

    with pytest.raises(SlurmPlanContractError, match="authored lock file"):
        validate_resolved_plan(authored, dependency_lock, plan)

    lock_payload = dependency_lock.model_dump(mode="json")
    lock_payload.update(
        authored_source="locks/user-lock.json",
        source={
            "path": "/workspace/primary/runs/run-001/inputs/user-lock.json",
            "sha256": "a" * 64,
        },
    )
    matching_lock = ResolvedDependencyLock.model_validate_json(json.dumps(lock_payload))
    unexpected_source_plan = multi_node_plan.model_copy(
        update={
            "client": multi_node_plan.client.model_copy(
                update={
                    "dependency_lock": multi_node_plan.client.dependency_lock.model_copy(
                        update={"sha256": matching_lock.compute_sha256()}
                    )
                }
            )
        }
    )
    with pytest.raises(SlurmPlanContractError, match="present for authored requirements"):
        validate_resolved_plan(authored_run, matching_lock, unexpected_source_plan)

    matching_plan = plan.model_copy(
        update={
            "client": plan.client.model_copy(
                update={
                    "dependency_lock": plan.client.dependency_lock.model_copy(
                        update={"sha256": matching_lock.compute_sha256()}
                    )
                }
            )
        }
    )

    assert validate_resolved_plan(authored, matching_lock, matching_plan) is matching_plan


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

    with pytest.raises(SlurmPlanContractError, match="Python ABI"):
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

    with pytest.raises(SlurmPlanContractError, match="image inventory"):
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

    with pytest.raises(SlurmPlanContractError, match="invocation"):
        validate_resolved_plan(authored_run, dependency_lock, invalid)


def test_cross_record_validation_preserves_explicit_run_config(
    authored_run: DataDesignerSlurmConfig,
    dependency_lock: ResolvedDependencyLock,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    payload = multi_node_plan.model_dump(mode="json")
    payload["invocation"]["effective_run_config"]["buffer_size"] = 16384
    invalid = ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))

    with pytest.raises(SlurmPlanContractError, match="run_config.buffer_size"):
        validate_resolved_plan(authored_run, dependency_lock, invalid)
