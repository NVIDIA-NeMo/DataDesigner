# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from data_designer.config import DataDesignerConfigBuilder, HuggingFaceSeedSource, ModelConfig
from data_designer.slurm.config import (
    ArrayTasksConfig,
    BenchmarkBaseRun,
    BuilderInput,
    ClientDependencies,
    DataDesignerSlurmBenchmarkConfig,
    DataDesignerSlurmConfig,
    ImageBuildRequest,
    ImageInspectionRecord,
    ImageRef,
    LiteralEnvironmentBinding,
    LocalStdioMCPProviderConfig,
    QueueBackpressureConfig,
    RemoteMCPProviderConfig,
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
        {"requirements": ["plugin @ https://user:secret@example.test/plugin.whl#sha256=" + "a" * 64]},
        {"requirements": ["plugin @ https://example.test/plugin.whl?token=secret#sha256=" + "a" * 64]},
        {"requirements": ["plugin @ https://example.test:invalid/plugin.whl#sha256=" + "a" * 64]},
        {"requirements": ["my_pkg==1", "my-pkg==2"]},
        {"requirements": ["not valid !!!"]},
        {"requirements": ["plugin=="]},
        {"requirements": ["plugin==1; python_version >= '3.12'"]},
        {"requirements": None, "lock_file": "../lock.json"},
    ],
)
def test_client_dependencies_reject_mutable_or_ambiguous_sources(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ClientDependencies.model_validate(payload)


def test_client_dependencies_accept_digest_bound_wheel() -> None:
    dependencies = ClientDependencies(requirements=["plugin @ https://example.test/plugin.whl#sha256=" + "a" * 64])

    assert dependencies.requirements is not None


def test_rejected_dependency_does_not_appear_in_validation_error() -> None:
    sentinel = "VERY_SECRET_VALUE"

    with pytest.raises(ValidationError) as error:
        ClientDependencies(requirements=[f"not valid {sentinel} !!!"])

    assert sentinel not in str(error.value)


def test_secret_reference_serializes_only_external_binding() -> None:
    secret = SecretRef(type="secret", environment="HF_TOKEN")
    sentinel = "VERY_SECRET_VALUE"

    assert secret.model_dump(mode="json") == {"type": "secret", "environment": "HF_TOKEN"}
    with pytest.raises(ValidationError) as error:
        SecretRef.model_validate({"type": "secret", "environment": "HF_TOKEN", "value": sentinel})
    assert sentinel not in str(error.value)


def test_rejected_secret_argument_does_not_appear_in_validation_error() -> None:
    sentinel = "VERY_SECRET_VALUE"

    with pytest.raises(ValidationError) as error:
        VllmServerConfig(
            type="vllm",
            image=ImageRef(name="vllm"),
            extra_args=[f"--api-key={sentinel}"],
        )

    assert sentinel not in str(error.value)


def test_literal_environment_rejects_control_characters() -> None:
    with pytest.raises(ValidationError, match="control"):
        LiteralEnvironmentBinding(type="literal", value="line\nbreak")


def test_secret_shaped_environment_requires_external_reference() -> None:
    literal = LiteralEnvironmentBinding(type="literal", value="plaintext-secret")

    with pytest.raises(ValidationError, match="external secret references"):
        LocalStdioMCPProviderConfig(
            provider_type="stdio",
            name="provider",
            command="provider",
            environment={"API_TOKEN": literal},
        )
    with pytest.raises(ValidationError, match="external secret references"):
        VllmServerConfig(
            type="vllm",
            image=ImageRef(name="vllm"),
            environment={"MODEL_PASSWORD": literal},
        )


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://user:password@example.test/mcp",
        "https://example.test/mcp?token=plaintext-secret",
        "https://example.test/mcp#secret",
    ],
)
def test_remote_mcp_endpoint_rejects_embedded_credentials(endpoint: str) -> None:
    with pytest.raises(ValidationError, match="must not embed"):
        RemoteMCPProviderConfig(provider_type="sse", name="provider", endpoint=endpoint)


@pytest.mark.parametrize("endpoint", ["https:///missing-host", "https://example.test:invalid/mcp"])
def test_remote_mcp_endpoint_requires_valid_host_and_port(endpoint: str) -> None:
    with pytest.raises(ValidationError, match=r"HTTP\(S\)"):
        RemoteMCPProviderConfig(provider_type="sse", name="provider", endpoint=endpoint)


@pytest.mark.parametrize(
    "argument",
    [
        "--api-key",
        "--api-key plaintext-secret",
        " --api-key plaintext-secret",
        "--access-token=plaintext-secret",
        "password",
    ],
)
def test_stdio_mcp_rejects_secret_shaped_arguments(argument: str) -> None:
    with pytest.raises(ValidationError, match="secret-shaped"):
        LocalStdioMCPProviderConfig(
            provider_type="stdio",
            name="provider",
            command="provider",
            args=[argument],
        )


def test_rejected_stdio_mcp_secret_argument_does_not_appear_in_validation_error() -> None:
    sentinel = "VALUE_THAT_MUST_NOT_APPEAR"

    with pytest.raises(ValidationError, match="secret-shaped") as error:
        LocalStdioMCPProviderConfig(
            provider_type="stdio",
            name="provider",
            command="provider",
            args=[f"--api-key {sentinel}"],
        )

    assert sentinel not in str(error.value)


def test_vllm_defaults_and_backpressure_override() -> None:
    server = VllmServerConfig(type="vllm", image=ImageRef(name="vllm"))
    overridden = VllmServerConfig(
        type="vllm",
        image=ImageRef(name="vllm"),
        lead_boot_standoff="0s",
        rank_launch_stagger="12s",
        queue_backpressure=QueueBackpressureConfig(max_waiting_requests=0, retry_after_seconds=None),
    )

    assert server.lead_boot_standoff == "60s"
    assert server.rank_launch_stagger == "5s"
    assert server.queue_backpressure.model_dump() == {"max_waiting_requests": 128, "retry_after_seconds": 1}
    assert overridden.lead_boot_standoff == "0s"
    assert overridden.rank_launch_stagger == "12s"
    assert overridden.queue_backpressure.model_dump() == {"max_waiting_requests": 0, "retry_after_seconds": None}


def test_vllm_rejects_nonpositive_retry_after_seconds() -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        QueueBackpressureConfig(retry_after_seconds=0)


@pytest.mark.parametrize(
    "argument",
    [
        "--api-key=plaintext-secret",
        "--api-key plaintext-secret",
        "-asc=2",
        "-dp",
        "-dpa=127.0.0.1",
        "-dpb",
        "-dcp=2",
        "-dpe",
        "-dph",
        "-dpl",
        "-dpm",
        "-dpn=1",
        "-dpp",
        "-dpr",
        "-ep",
        "-n",
        "-n+2",
        "-n2",
        "-pcp=2",
        "-pp=2",
        "-r",
        "-r0",
        "-r1",
        "-tp",
        "--api-server-count=2",
        "--config=/tmp/vllm.yaml",
        "--cpu-distributed-timeout-seconds=120",
        "--cp-kv-cache-interleave-size=2",
        "--cpunodebind=0",
        "--data-parallel-address",
        "--data-parallel-backend=mp",
        "--data-parallel-external-lb",
        "--data-parallel-hybrid-lb",
        "--data_parallel_master_ip=127.0.0.1",
        "--data-parallel-rank=1",
        "--data-parallel-rpc-port 29502",
        "--data-parallel-size",
        "--data-parallel-size-local",
        "--data-parallel-start-rank=1",
        "--data-parallel-supervisor-port=9256",
        "--dcp-comm-backend=a2a",
        "--dcp-kv-cache-interleave-size=2",
        "--decode-context-parallel-size=2",
        "--default-mm-loras={}",
        "--distributed-executor-backend=mp",
        "--distributed-init-address",
        "--distributed-timeout-seconds",
        "--dp-supervisor-probe-interval-s=1",
        "--ec-transfer-config={}",
        "--enable-elastic-ep",
        "--enable-expert-parallel",
        "--enable-lora",
        "--enable-ssl-refresh",
        "--grpc",
        "--headless",
        "--host=0.0.0.0",
        "--io-processor-plugin=custom",
        "--kv-events-config={}",
        "--kv-transfer-config={}",
        "--logits-processors=custom.LogitsProcessor",
        "--lora-modules=adapter=/models/adapter",
        "--max-cpu-loras=2",
        "--max-loras=2",
        "--master-addr",
        "--master-port=29501",
        "--middleware",
        "--model",
        "--nnodes",
        "--node-rank=1",
        "--no-data-parallel-external-lb",
        "--no-data-parallel-hybrid-lb",
        "--no-enable-elastic-ep",
        "--no-enable-expert-parallel",
        "--no-enable-lora",
        "--no-enable-ssl-refresh",
        "--no-numa-bind",
        "--numa-bind",
        "--numa_bind_nodes=[0,1]",
        "--physcpubind=0-7",
        "--pipeline-parallel-size=2",
        "--port",
        "--port 9000",
        "--prefill-context-parallel-size=2",
        "--reasoning-parser-plugin=/tmp/reasoning.py",
        "--root-path=/v1",
        "--served-model-name=example/model",
        "--spec-model=example/draft-model",
        "--speculative-config={}",
        "--ssl-ca-certs=/tmp/ca.pem",
        "--ssl-cert-reqs=2",
        "--ssl-certfile=/tmp/cert.pem",
        "--ssl-ciphers=ECDHE-RSA-AES256-GCM-SHA384",
        "--ssl-keyfile=/tmp/key.pem",
        "--tensor-parallel-size",
        "--tensor_parallel_size=2",
        "--tensor=2",
        "--tool-parser-plugin=/tmp/tools.py",
        "--uds=/tmp/vllm.sock",
        "--weight-transfer-config={}",
        "--worker-cls=custom.Worker",
        "--worker-extension-cls=custom.Extension",
    ],
)
def test_vllm_rejects_runtime_owned_arguments(argument: str) -> None:
    with pytest.raises(ValidationError, match="owned"):
        VllmServerConfig(type="vllm", image=ImageRef(name="vllm"), extra_args=[argument])


@pytest.mark.parametrize("abbreviated_flag", ["-a", "-as", "-dc", "-e", "-pc", "-t"])
def test_vllm_rejects_runtime_owned_short_flag_abbreviations(abbreviated_flag: str) -> None:
    with pytest.raises(ValidationError, match="owned"):
        VllmServerConfig(type="vllm", image=ImageRef(name="vllm"), extra_args=[abbreviated_flag, "4"])


def test_vllm_allows_builtin_parser_selection() -> None:
    config = VllmServerConfig(
        type="vllm",
        image=ImageRef(name="vllm"),
        extra_args=["--reasoning_parser", "deepseek_r1", "--tool-call-parser", "hermes"],
    )

    assert config.extra_args == ["--reasoning_parser", "deepseek_r1", "--tool-call-parser", "hermes"]


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--structured-outputs-config.reasoning-parser-plugin=/tmp/reasoning.py"],
        ["--structured-outputs-config", '{"reasoning_parser_plugin":"/tmp/reasoning.py"}'],
        ['--structured-outputs-config={"reasoning_parser_plugin":"/tmp/reasoning.py"}'],
        ['--additional-config={"nested":{"worker_cls":"custom.Worker"}}'],
        ['--additional-config={"partition_key":"group"}'],
    ],
)
def test_vllm_does_not_interpret_plugin_hooks_nested_in_json_arguments(extra_args: list[str]) -> None:
    config = VllmServerConfig(type="vllm", image=ImageRef(name="vllm"), extra_args=extra_args)

    assert config.extra_args == extra_args


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--hf-overrides", '{"api_key":"plaintext-secret"}'],
        ['--hf-overrides={"nested":{"access_token":"plaintext-secret"}}'],
    ],
)
def test_vllm_rejects_plaintext_secrets_in_plugin_owned_json_arguments(extra_args: list[str]) -> None:
    with pytest.raises(ValidationError, match="plaintext values under secret-bearing keys"):
        VllmServerConfig(type="vllm", image=ImageRef(name="vllm"), extra_args=extra_args)


def test_vllm_allows_nonsecret_json_arguments() -> None:
    config = VllmServerConfig(
        type="vllm",
        image=ImageRef(name="vllm"),
        extra_args=['--hf-overrides={"model_type":"custom"}'],
    )

    assert config.extra_args == ['--hf-overrides={"model_type":"custom"}']


def test_vllm_preserves_argv_values_with_whitespace_and_templates() -> None:
    extra_args = ["--hf-overrides", '{"model_type": "custom"}', "--chat-template={{messages}}"]

    config = VllmServerConfig(type="vllm", image=ImageRef(name="vllm"), extra_args=extra_args)

    assert config.extra_args == extra_args


@pytest.mark.parametrize("argument", [" --max-model-len", "32768 ", "   "])
def test_vllm_rejects_argv_with_outer_whitespace(argument: str) -> None:
    with pytest.raises(ValidationError, match="leading or trailing whitespace"):
        VllmServerConfig(type="vllm", image=ImageRef(name="vllm"), extra_args=[argument])


@pytest.mark.parametrize("argument", ["=", "=x"])
def test_vllm_rejects_empty_option_names(argument: str) -> None:
    with pytest.raises(ValidationError, match="option name must not be empty"):
        VllmServerConfig(type="vllm", image=ImageRef(name="vllm"), extra_args=[argument])


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--hf-overrides", "{'model_type':'custom'}"],
        ['--hf-overrides={"model_type":"custom"'],
    ],
)
def test_vllm_preserves_structured_values_for_runtime_validation(extra_args: list[str]) -> None:
    config = VllmServerConfig(type="vllm", image=ImageRef(name="vllm"), extra_args=extra_args)

    assert config.extra_args == extra_args


@pytest.mark.parametrize(
    "environment_name",
    [
        "CUDA_VISIBLE_DEVICES",
        "ENABLE_EP",
        "HF_HOME",
        "GROUP_RANK",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NVIDIA_VISIBLE_DEVICES",
        "PYTHON_EXEC",
        "RANK",
        "ROLE_WORLD_SIZE",
        "SLURM_PROCID",
        "TORCHELASTIC_RUN_ID",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING",
        "VLLM_API_KEY",
        "VLLM_CACHE_ROOT",
        "VLLM_LEAD_BOOT_STANDOFF_SECONDS",
        "VLLM_DP_RANK",
        "VLLM_HOST_IP",
        "VLLM_LORA_RESOLVER_HF_REPO_LIST",
        "VLLM_MODEL_REDIRECT_PATH",
        "VLLM_MOONCAKE_BOOTSTRAP_PORT",
        "VLLM_NIXL_SIDE_CHANNEL_PORT",
        "VLLM_PLUGINS",
        "VLLM_PORT",
        "VLLM_RAY_PER_WORKER_GPUS",
        "VLLM_RPC_BASE_PATH",
        "WORLD_SIZE",
    ],
)
def test_vllm_rejects_runtime_owned_environment_names(environment_name: str) -> None:
    with pytest.raises(ValidationError, match="owned by the compiler or runtime"):
        VllmServerConfig(
            type="vllm",
            image=ImageRef(name="vllm"),
            environment={environment_name: SecretRef(type="secret", environment="EXTERNAL_VALUE")},
        )


def test_vllm_allows_explicit_tuning_environment() -> None:
    config = VllmServerConfig(
        type="vllm",
        image=ImageRef(name="vllm"),
        environment={
            "NCCL_DEBUG": LiteralEnvironmentBinding(type="literal", value="INFO"),
            "VLLM_WORKER_MULTIPROC_METHOD": LiteralEnvironmentBinding(type="literal", value="spawn"),
        },
    )

    assert set(config.environment) == {"NCCL_DEBUG", "VLLM_WORKER_MULTIPROC_METHOD"}


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--hf-token=plaintext-secret"],
        ["--hf-token", "plaintext-secret"],
        ["--hf-token plaintext-secret"],
    ],
)
def test_vllm_rejects_secret_shaped_arguments(extra_args: list[str]) -> None:
    with pytest.raises(ValidationError, match="secret-shaped"):
        VllmServerConfig(type="vllm", image=ImageRef(name="vllm"), extra_args=extra_args)


def test_vllm_rejects_non_tokenized_arguments() -> None:
    with pytest.raises(ValidationError, match="one token"):
        VllmServerConfig(
            type="vllm",
            image=ImageRef(name="vllm"),
            extra_args=["--max-model-len 32768"],
        )


def test_vllm_rejects_duplicate_arguments() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        VllmServerConfig(
            type="vllm",
            image=ImageRef(name="vllm"),
            extra_args=["--max-model-len", "32768", "--max-model-len=16384"],
        )

    with pytest.raises(ValidationError, match="duplicate or conflicting"):
        VllmServerConfig(
            type="vllm",
            image=ImageRef(name="vllm"),
            extra_args=["--max_model_len=4096", "--max-model-len=8192"],
        )


def test_vllm_rejects_conflicting_boolean_arguments() -> None:
    with pytest.raises(ValidationError, match="duplicate or conflicting"):
        VllmServerConfig(
            type="vllm",
            image=ImageRef(name="vllm"),
            extra_args=["--enable-prefix-caching", "--no-enable-prefix-caching"],
        )


def test_vllm_rejects_option_terminator() -> None:
    with pytest.raises(ValidationError, match="option terminators"):
        VllmServerConfig(type="vllm", image=ImageRef(name="vllm"), extra_args=["--"])


def test_vllm_rejects_distributed_timeout_beyond_startup_timeout() -> None:
    with pytest.raises(ValidationError, match="must not exceed"):
        VllmServerConfig(
            type="vllm",
            image=ImageRef(name="vllm"),
            startup_timeout="10m",
            distributed_init_timeout="11m",
        )


@pytest.mark.parametrize("field", ["lead_boot_standoff", "rank_launch_stagger"])
def test_vllm_rejects_negative_launch_timing(field: str) -> None:
    with pytest.raises(ValidationError):
        VllmServerConfig.model_validate({"type": "vllm", "image": {"name": "vllm"}, field: "-1s"})


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
    payload["topology"]["nodes_per_replica"] = 1
    payload["server"]["enable_expert_parallel"] = True

    deployment = ServerDeploymentConfig.model_validate(payload)

    assert deployment.topology.nodes_per_replica == 1

    payload["topology"]["nodes_per_replica"] = 2
    with pytest.raises(ValidationError, match="expert"):
        ServerDeploymentConfig.model_validate(payload)


def test_model_alias_preserves_public_data_designer_values() -> None:
    alias = "judge/v2"
    ModelConfig(alias=alias, model="example/judge", provider="openai")

    deployment = ServerDeploymentConfig.model_validate(
        {
            "model_alias": alias,
            "model": "example/judge",
            "server": {"type": "vllm", "image": {"name": "vllm"}},
        }
    )

    assert deployment.model_alias == alias


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


@pytest.mark.parametrize(
    "secret_key",
    [
        "api_key",
        "accessToken",
        "client-secret",
        "consumer_key",
        "license_key",
        "password",
        "private_key",
        "signing_key",
        "ssh_key",
        "subscription_key",
    ],
)
def test_builder_input_rejects_plaintext_secrets_in_plugin_owned_fields(secret_key: str) -> None:
    inline = {"columns": [], "plugin_config": {secret_key: "opaque-plugin-value"}}

    with pytest.raises(ValidationError, match="plaintext values under secret-bearing keys"):
        BuilderInput(inline=inline)


def test_rejected_embedded_secret_does_not_appear_in_validation_error() -> None:
    sentinel = "VERY_SECRET_VALUE"

    with pytest.raises(ValidationError) as error:
        BuilderInput(inline={"columns": [], "plugin_config": {"api_key": sentinel}})

    assert sentinel not in str(error.value)


@pytest.mark.parametrize("plugin_key", ["sort_key", "partition_key", "primary_key", "idempotency_key"])
def test_builder_input_allows_non_secret_plugin_keys(plugin_key: str) -> None:
    inline = {"columns": [{"column_type": "plugin", plugin_key: "value"}]}

    assert BuilderInput(inline=inline).inline == inline


def test_builder_input_accepts_exported_and_shorthand_configs() -> None:
    exported = DataDesignerConfigBuilder(model_configs=[]).get_builder_config().to_dict()

    assert BuilderInput(inline=exported).inline == exported
    assert BuilderInput(inline={"columns": []}).inline == {"columns": []}


def test_builder_input_accepts_null_secret_fields_from_canonical_export() -> None:
    builder = DataDesignerConfigBuilder(model_configs=[]).with_seed_dataset(
        HuggingFaceSeedSource(path="datasets/example/seed/*.parquet")
    )
    exported = builder.get_builder_config().to_dict()

    assert exported["data_designer"]["seed_config"]["source"]["token"] is None
    assert BuilderInput(inline=exported).inline == exported


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_builder_input_rejects_non_finite_json(value: float) -> None:
    with pytest.raises(ValidationError):
        BuilderInput(inline={"columns": [], "value": value})


@pytest.mark.parametrize(
    "inline",
    [
        {"data_designer": {}, "library_version": 1},
        {"data_designer": {}, "unknown": True},
        {"model_configs": []},
    ],
)
def test_builder_input_rejects_invalid_serialized_shapes(inline: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="complete serialized"):
        BuilderInput.model_validate({"inline": inline})


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


@pytest.mark.parametrize("readiness_path", ["health", "//other-host/health", "/health check"])
def test_small_config_values_validate_at_boundary(readiness_path: str) -> None:
    with pytest.raises(ValidationError, match="concurrency"):
        ArrayTasksConfig(count=2, max_concurrent=3)
    with pytest.raises(ValidationError, match="minutes"):
        SubmissionConfig(time_limit="00:60:00")
    with pytest.raises(ValidationError, match="absolute URL path"):
        VllmServerConfig(type="vllm", image=ImageRef(name="vllm"), readiness_path=readiness_path)


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


def test_config_models_are_deeply_immutable(authored_run: DataDesignerSlurmConfig) -> None:
    inline = authored_run.builder.inline
    assert inline is not None
    data_designer = inline["data_designer"]
    assert isinstance(data_designer, dict)
    columns = data_designer["columns"]
    assert isinstance(columns, list)

    with pytest.raises(TypeError, match="frozen list"):
        authored_run.deployments.clear()
    with pytest.raises(TypeError, match="frozen dictionary"):
        authored_run.invocation.run_config["buffer_size"] = 1
    with pytest.raises(TypeError, match="frozen list"):
        columns.append({})
