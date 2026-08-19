# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from data_designer.slurm.config import (
    ArrayTasksConfig,
    BenchmarkBaseRun,
    ClientDependencies,
    DataDesignerSlurmBenchmarkConfig,
    DataDesignerSlurmConfig,
    ImageBuildRequest,
    ImageInspectionRecord,
    ImageRef,
    LiteralEnvironmentBinding,
    QueueBackpressureConfig,
    SecretRef,
    ServerDeploymentConfig,
    SubmissionConfig,
    VllmServerConfig,
)


@pytest.mark.parametrize("version", [None, 0, 2, "1"])
def test_run_config_requires_supported_version(authored_run: DataDesignerSlurmConfig, version: object) -> None:
    payload = authored_run.model_dump(mode="json")
    if version is None:
        payload.pop("schema_version")
    else:
        payload["schema_version"] = version

    with pytest.raises(ValidationError):
        DataDesignerSlurmConfig.model_validate(payload)


def test_run_config_rejects_unknown_fields(authored_run: DataDesignerSlurmConfig) -> None:
    payload = authored_run.model_dump(mode="json")
    payload["placement"] = {"gpu_ids": [0]}

    with pytest.raises(ValidationError, match="placement"):
        DataDesignerSlurmConfig.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"name": "image", "path": "/images/image.sqsh"},
        {"path": "relative.sqsh"},
        {"path": "/images/image.tar"},
    ],
)
def test_image_ref_requires_one_registered_alias_or_absolute_sqsh(payload: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        ImageRef.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"requirements": None},
        {"requirements": [], "lock_file": "lock.json"},
        {"requirements": ["-e ./plugin"]},
        {"requirements": ["plugin @ git+https://example.test/plugin.git"]},
        {"requirements": ["plugin @ https://example.test/plugin.whl"]},
        {"requirements": ["my_pkg==1", "my-pkg==2"]},
        {"requirements": None, "lock_file": "../lock.json"},
    ],
)
def test_client_dependencies_reject_mutable_or_ambiguous_sources(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ClientDependencies.model_validate(payload)


def test_client_dependencies_accept_digest_bound_wheel() -> None:
    dependencies = ClientDependencies(requirements=["plugin @ https://example.test/plugin.whl#sha256=" + "a" * 64])

    assert dependencies.requirements is not None


def test_secret_reference_serializes_only_external_binding() -> None:
    secret = SecretRef(type="secret", environment="HF_TOKEN")

    assert secret.model_dump(mode="json") == {"type": "secret", "environment": "HF_TOKEN"}
    with pytest.raises(ValidationError):
        SecretRef.model_validate({"type": "secret", "environment": "HF_TOKEN", "value": "secret-value"})


def test_literal_environment_rejects_control_characters() -> None:
    with pytest.raises(ValidationError, match="control"):
        LiteralEnvironmentBinding(type="literal", value="line\nbreak")


def test_vllm_defaults_and_backpressure_override() -> None:
    server = VllmServerConfig(type="vllm", image=ImageRef(name="vllm"))
    overridden = VllmServerConfig(
        type="vllm",
        image=ImageRef(name="vllm"),
        queue_backpressure=QueueBackpressureConfig(max_waiting_requests=0, retry_after_seconds=None),
    )

    assert server.queue_backpressure.model_dump() == {"max_waiting_requests": 128, "retry_after_seconds": 1}
    assert overridden.queue_backpressure.model_dump() == {"max_waiting_requests": 0, "retry_after_seconds": None}


@pytest.mark.parametrize("argument", ["--port", "--host=0.0.0.0", "--tensor-parallel-size"])
def test_vllm_rejects_runtime_owned_arguments(argument: str) -> None:
    with pytest.raises(ValidationError, match="owned"):
        VllmServerConfig(type="vllm", image=ImageRef(name="vllm"), extra_args=[argument])


def test_deployment_rejects_invalid_topology() -> None:
    payload = {
        "model_alias": "generator",
        "model": "example/generator",
        "server": {"type": "vllm", "image": {"name": "vllm"}},
        "resources": {"nodes": 3},
        "topology": {"tensor_parallel": 8, "nodes_per_replica": 2},
    }

    with pytest.raises(ValidationError, match="divide"):
        ServerDeploymentConfig.model_validate(payload)

    payload["resources"]["nodes"] = 2
    payload["server"]["enable_expert_parallel"] = True
    with pytest.raises(ValidationError, match="expert"):
        ServerDeploymentConfig.model_validate(payload)


def test_run_rejects_duplicate_alias_and_unknown_concurrency(authored_run: DataDesignerSlurmConfig) -> None:
    payload = authored_run.model_dump(mode="json")
    payload["deployments"][1]["model_alias"] = "generator"
    with pytest.raises(ValidationError, match="aliases"):
        DataDesignerSlurmConfig.model_validate(payload)

    payload = authored_run.model_dump(mode="json")
    payload["invocation"]["model_concurrency"]["missing"] = 1
    with pytest.raises(ValidationError, match="undeclared"):
        DataDesignerSlurmConfig.model_validate(payload)


def test_run_rejects_retired_builder_fields(authored_run: DataDesignerSlurmConfig) -> None:
    payload = authored_run.model_dump(mode="json")
    payload["builder"]["inline"]["server_configs"] = []

    with pytest.raises(ValidationError, match="retired"):
        DataDesignerSlurmConfig.model_validate(payload)


def test_run_validates_public_run_config_and_shard_count(authored_run: DataDesignerSlurmConfig) -> None:
    payload = authored_run.model_dump(mode="json")
    payload["invocation"]["run_config"]["buffer_size"] = 0
    with pytest.raises(ValidationError, match="buffer_size"):
        DataDesignerSlurmConfig.model_validate(payload)

    payload = authored_run.model_dump(mode="json")
    payload["array_tasks"]["count"] = 101
    payload["array_tasks"]["max_concurrent"] = 1
    with pytest.raises(ValidationError, match="requested records"):
        DataDesignerSlurmConfig.model_validate(payload)


def test_small_config_values_validate_at_boundary() -> None:
    with pytest.raises(ValidationError, match="concurrency"):
        ArrayTasksConfig(count=2, max_concurrent=3)
    with pytest.raises(ValidationError, match="minutes"):
        SubmissionConfig(time_limit="00:60:00")
    with pytest.raises(ValidationError, match="readiness_path"):
        VllmServerConfig(type="vllm", image=ImageRef(name="vllm"), readiness_path="health")


@pytest.mark.parametrize(
    "source",
    ["nvcr.io/example/vllm:latest", "relative.sqsh"],
)
def test_image_build_request_rejects_mutable_or_relative_source(source: str) -> None:
    with pytest.raises(ValidationError):
        ImageBuildRequest(name="vllm", kind="serving", source=source)


def test_image_inspection_rejects_duplicate_distribution_names() -> None:
    payload = {
        "schema_version": 1,
        "inspector_version": "v1",
        "sqsh_sha256": "a" * 64,
        "inspection": {
            "kind": "client",
            "python_implementation": "cpython",
            "python_version": "3.12.1",
            "python_abi": "cp312",
            "distributions": [
                {"name": "plugin", "version": "1"},
                {"name": "plugin", "version": "2"},
            ],
            "installer_path": "/usr/bin/pip",
            "installer_version": "1",
        },
    }

    with pytest.raises(ValidationError, match="unique"):
        ImageInspectionRecord.model_validate_json(json.dumps(payload))


def test_benchmark_config_rejects_duplicate_axes() -> None:
    payload = {
        "schema_version": 1,
        "name": "bench",
        "base_run": "./run.yaml",
        "model_aliases": ["generator", "generator"],
        "concurrency_values": [32, 32],
        "deployment_cases": [{"name": "case", "deployments": {"generator": {"nodes": 1, "nodes_per_replica": 1}}}],
        "record_policy": {"type": "fixed", "records": 100},
        "analysis": {"target_total_records": 1000, "target_runtime": "1h"},
    }

    with pytest.raises(ValidationError, match="aliases"):
        DataDesignerSlurmBenchmarkConfig.model_validate(payload)

    payload["model_aliases"] = ["generator"]
    with pytest.raises(ValidationError, match="concurrency"):
        DataDesignerSlurmBenchmarkConfig.model_validate(payload)


def test_benchmark_base_run_normalizes_local_source() -> None:
    base_run = BenchmarkBaseRun.model_validate("./run.yaml")

    assert base_run.source == "run.yaml"


def test_config_models_do_not_mutate_input(authored_run: DataDesignerSlurmConfig) -> None:
    payload = authored_run.model_dump(mode="json")
    original = deepcopy(payload)

    DataDesignerSlurmConfig.model_validate(payload)

    assert payload == original
