# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import posixpath
import re
from collections.abc import Mapping
from typing import Annotated, Literal, TypeVar
from urllib.parse import urlsplit

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from pydantic import (
    Field,
    JsonValue,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from data_designer.config import RunConfig
from data_designer.slurm.config.images import ImageRef
from data_designer.slurm.contracts import (
    AuthoredConfig,
    Duration,
    EnvironmentName,
    Identifier,
    ModelAlias,
    NonNegativeDuration,
    SchemaVersion,
    validate_absolute_path,
    validate_local_config_path,
    validate_plain_text,
    validate_url,
)

_OWNED_VLLM_FLAGS = {
    "-asc",
    "-dcp",
    "-dp",
    "-dpa",
    "-dpb",
    "-dpe",
    "-dph",
    "-dpl",
    "-dpm",
    "-dpn",
    "-dpp",
    "-dpr",
    "-ep",
    "-n",
    "-pcp",
    "-pp",
    "-r",
    "-tp",
    "--api-key",
    "--api-server-count",
    "--config",
    "--cpu-distributed-timeout-seconds",
    "--cp-kv-cache-interleave-size",
    "--cpunodebind",
    "--data-parallel-address",
    "--data-parallel-backend",
    "--data-parallel-external-lb",
    "--data-parallel-hybrid-lb",
    "--data-parallel-rank",
    "--data-parallel-rpc-port",
    "--data-parallel-size",
    "--data-parallel-size-local",
    "--data-parallel-start-rank",
    "--data-parallel-supervisor-port",
    "--dcp-comm-backend",
    "--dcp-kv-cache-interleave-size",
    "--decode-context-parallel-size",
    "--default-mm-loras",
    "--distributed-executor-backend",
    "--distributed-init-address",
    "--distributed-timeout-seconds",
    "--dp-supervisor-probe-failure-threshold",
    "--dp-supervisor-probe-interval-s",
    "--dp-supervisor-probe-timeout-s",
    "--ec-transfer-config",
    "--enable-elastic-ep",
    "--enable-expert-parallel",
    "--enable-lora",
    "--enable-ssl-refresh",
    "--grpc",
    "--headless",
    "--host",
    "--io-processor-plugin",
    "--kv-events-config",
    "--kv-transfer-config",
    "--logits-processors",
    "--master-addr",
    "--master-port",
    "--middleware",
    "--model",
    "--nnodes",
    "--node-rank",
    "--no-data-parallel-external-lb",
    "--no-data-parallel-hybrid-lb",
    "--no-enable-elastic-ep",
    "--no-enable-expert-parallel",
    "--no-enable-lora",
    "--no-enable-ssl-refresh",
    "--no-numa-bind",
    "--numa-bind",
    "--numa-bind-cpus",
    "--numa-bind-nodes",
    "--physcpubind",
    "--pipeline-parallel-size",
    "--port",
    "--prefill-context-parallel-size",
    "--reasoning-parser-plugin",
    "--root-path",
    "--served-model-name",
    "--ssl-ca-certs",
    "--ssl-cert-reqs",
    "--ssl-certfile",
    "--ssl-ciphers",
    "--ssl-keyfile",
    "--tensor-parallel-size",
    "--tool-parser-plugin",
    "--uds",
    "--weight-transfer-config",
    "--worker-cls",
    "--worker-extension-cls",
}
_OWNED_VLLM_FLAG_PREFIXES = (
    "--api-server-",
    "--data-parallel-",
    "--decode-context-parallel-",
    "--distributed-",
    "--dp-supervisor-",
    "--master-",
    "--numa-",
    "--pipeline-parallel-",
    "--prefill-context-parallel-",
    "--ssl-",
    "--tensor-parallel-",
)
_VLLM_EXACT_PASSTHROUGH_FLAGS = frozenset({"--reasoning-parser"})
_OWNED_VLLM_MODEL_FEATURE_SEGMENTS = frozenset({"draft", "lora", "loras", "spec", "speculative"})
_OWNED_VLLM_JSON_KEYS = frozenset(
    {
        "io-processor-plugin",
        "logits-processors",
        "reasoning-parser-plugin",
        "tool-parser-plugin",
        "worker-cls",
        "worker-extension-cls",
    }
)
_OWNED_VLLM_ATTACHED_VALUE_FLAGS = ("-n", "-r")
_OWNED_VLLM_ENVIRONMENT_NAMES = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "GROUP_RANK",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NODE_RANK",
        "NVIDIA_VISIBLE_DEVICES",
        "PYTHON_EXEC",
        "RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING",
        "VLLM_API_KEY",
        "VLLM_HOST_IP",
        "VLLM_MODEL_REDIRECT_PATH",
        "VLLM_PLUGINS",
        "VLLM_PORT",
        "VLLM_RPC_BASE_PATH",
        "WORLD_SIZE",
    }
)
_OWNED_VLLM_ENVIRONMENT_PREFIXES = (
    "SLURM_",
    "TORCHELASTIC_",
    "VLLM_DP_",
    "VLLM_LORA_",
    "VLLM_MOONCAKE_",
    "VLLM_NIXL_",
    "VLLM_RAY_",
)
_DURATION_FACTORS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_NON_SECRET_PAYLOAD_KEYS = frozenset({"idempotency_key", "partition_key", "primary_key", "sort_key"})
_SECRET_NAME_PARTS = frozenset({"credential", "credentials", "password", "secret", "token"})

_Arguments = TypeVar("_Arguments", list[str], tuple[str, ...])


def _duration_seconds(value: Duration) -> int:
    return int(value[:-1]) * _DURATION_FACTORS[value[-1]]


def _secret_name_segments(value: str) -> list[str]:
    snake_case = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    normalized = re.sub(r"[^a-z0-9]+", "_", snake_case.casefold()).strip("_")
    return normalized.split("_")


def _is_secret_name(value: str) -> bool:
    segments = _secret_name_segments(value)
    return bool(
        _SECRET_NAME_PARTS.intersection(segments)
        or {"access", "key"}.issubset(segments)
        or segments[-1] in {"auth", "key"}
    )


def _is_secret_payload_name(value: str) -> bool:
    segments = _secret_name_segments(value)
    normalized = "_".join(segments)
    return normalized not in _NON_SECRET_PAYLOAD_KEYS and _is_secret_name(value)


def _option_flag(value: str) -> str:
    flag = re.split(r"[=\s]", value.lstrip(), maxsplit=1)[0]
    return flag.replace("_", "-") if flag.startswith("--") else flag


def _is_owned_vllm_flag(flag: str) -> bool:
    if flag in _VLLM_EXACT_PASSTHROUGH_FLAGS:
        return False
    if flag.startswith("--") and _OWNED_VLLM_MODEL_FEATURE_SEGMENTS.intersection(flag.removeprefix("--").split("-")):
        return True
    if flag.startswith("--") and _OWNED_VLLM_JSON_KEYS.intersection(flag.removeprefix("--").split(".")):
        return True
    if flag in _OWNED_VLLM_FLAGS or flag.startswith(_OWNED_VLLM_FLAG_PREFIXES):
        return True
    if any(
        flag.startswith(owned_flag) and re.fullmatch(r"[+-]?[0-9]+", flag[len(owned_flag) :]) is not None
        for owned_flag in _OWNED_VLLM_ATTACHED_VALUE_FLAGS
    ):
        return True
    return flag.startswith("--") and any(
        owned_flag.startswith(flag) for owned_flag in _OWNED_VLLM_FLAGS if owned_flag.startswith("--")
    )


def _contains_owned_vllm_json_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).replace("_", "-") in _OWNED_VLLM_JSON_KEYS or _contains_owned_vllm_json_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return any(_contains_owned_vllm_json_key(item) for item in value)
    return False


def _parse_vllm_json_argument(value: str) -> object | None:
    candidate = value.partition("=")[2] if value.startswith("-") and "=" in value else value
    if not candidate.startswith(("{", "[")):
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ValueError("vLLM JSON arguments must contain valid JSON") from error


def _contains_secret_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            (_is_secret_payload_name(str(key)) and item is not None) or _contains_secret_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return any(_contains_secret_key(item) for item in value)
    return False


def validate_environment_bindings(
    values: dict[EnvironmentName, EnvironmentBinding],
) -> dict[EnvironmentName, EnvironmentBinding]:
    """Reject literal values for environment variables that look secret-bearing."""
    literal_secrets = [
        name
        for name, binding in values.items()
        if _is_secret_name(name) and isinstance(binding, LiteralEnvironmentBinding)
    ]
    if literal_secrets:
        raise ValueError("secret-shaped environment names require external secret references")
    return values


def validate_vllm_environment_bindings(
    values: dict[EnvironmentName, EnvironmentBinding],
) -> dict[EnvironmentName, EnvironmentBinding]:
    """Reject environment names whose values are derived by the serving runtime."""
    validate_environment_bindings(values)
    conflicts = sorted(
        name
        for name in values
        if name in _OWNED_VLLM_ENVIRONMENT_NAMES or name.startswith(_OWNED_VLLM_ENVIRONMENT_PREFIXES)
    )
    if conflicts:
        raise ValueError(f"vLLM environment names are owned by the compiler or runtime: {', '.join(conflicts)}")
    return values


def validate_vllm_readiness_path(value: str) -> str:
    """Require an HTTP path that cannot be interpreted as another authority."""
    validate_plain_text(value, field_name="readiness path")
    if (
        not value.startswith("/")
        or value.startswith("//")
        or any(character.isspace() for character in value)
        or "?" in value
        or "#" in value
    ):
        raise ValueError("readiness path must be an absolute URL path without whitespace, query, or fragment")
    return value


def validate_vllm_extra_args(values: _Arguments) -> _Arguments:
    """Validate shell-free vLLM arguments without allowing structured-input overrides."""
    seen_flags: set[str] = set()
    for value in values:
        validate_plain_text(value, field_name="vLLM argument")
        flag = _option_flag(value)
        json_payload = _parse_vllm_json_argument(value)
        if flag == "--":
            raise ValueError("vLLM argument option terminators are not supported")
        if _is_owned_vllm_flag(flag):
            raise ValueError(f"vLLM argument {flag!r} is owned by the compiler or runtime")
        if _contains_owned_vllm_json_key(json_payload):
            raise ValueError("vLLM argument contains a field owned by the compiler or runtime")
        if _contains_secret_key(json_payload):
            raise ValueError("secret-shaped vLLM JSON fields must use an environment secret reference")
        if _is_secret_name(flag.lstrip("-")):
            raise ValueError("secret-shaped vLLM arguments must use an environment secret reference")
        if any(character.isspace() for character in value):
            raise ValueError("each vLLM argument must be one token")
        if re.match(r"^--?[A-Za-z]", flag) is not None:
            canonical_flag = f"--{flag.removeprefix('--no-')}" if flag.startswith("--no-") else flag
            if canonical_flag in seen_flags:
                raise ValueError(f"duplicate or conflicting vLLM argument {flag!r}")
            seen_flags.add(canonical_flag)
    return values


class LiteralEnvironmentBinding(AuthoredConfig):
    type: Literal["literal"]
    value: Annotated[str, StringConstraints(max_length=4096)]

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: str) -> str:
        return validate_plain_text(value, field_name="environment value")


class SecretRef(AuthoredConfig):
    type: Literal["secret"]
    environment: EnvironmentName


EnvironmentBinding = Annotated[
    LiteralEnvironmentBinding | SecretRef,
    Field(discriminator="type"),
]


class BuilderInput(AuthoredConfig):
    source: str | None = None
    inline: dict[str, JsonValue] | None = None

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str | None) -> str | None:
        return None if value is None else validate_local_config_path(value)

    @model_validator(mode="after")
    def validate_input(self) -> BuilderInput:
        if (self.source is None) == (self.inline is None):
            raise ValueError("builder requires exactly one of source or inline")
        if self.inline is not None:
            if not self.inline:
                raise ValueError("inline builder input must not be empty")
            retired = {"dependencies", "sandbox_config", "server_configs"}.intersection(self.inline)
            if retired:
                raise ValueError(f"builder input contains retired Big Iron fields: {', '.join(sorted(retired))}")
            if "data_designer" in self.inline:
                unknown = set(self.inline).difference({"data_designer", "library_version"})
                library_version = self.inline.get("library_version")
                valid = not unknown and isinstance(self.inline["data_designer"], dict)
                valid = valid and (library_version is None or isinstance(library_version, str))
            else:
                valid = isinstance(self.inline.get("columns"), list)
            if not valid:
                raise ValueError("inline builder input must be one complete serialized Data Designer config")
            if _contains_secret_key(self.inline):
                raise ValueError("inline builder input must not contain secret values")
        return self


class InputBindings(AuthoredConfig):
    seed_path: str | None = None
    managed_assets_path: str | None = None

    @field_validator("seed_path", "managed_assets_path")
    @classmethod
    def validate_paths(cls, value: str | None) -> str | None:
        return None if value is None else validate_absolute_path(value)


class RemoteMCPProviderConfig(AuthoredConfig):
    provider_type: Literal["sse", "streamable_http"]
    name: Identifier
    endpoint: str
    api_key: SecretRef | None = None

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        validate_url(value, field_name="MCP endpoint")
        parsed = urlsplit(value)
        if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
            raise ValueError("MCP endpoint must not embed credentials, query parameters, or fragments")
        return value


class LocalStdioMCPProviderConfig(AuthoredConfig):
    provider_type: Literal["stdio"]
    name: Identifier
    command: str
    args: list[str] = Field(default_factory=list)
    environment: dict[EnvironmentName, EnvironmentBinding] = Field(default_factory=dict)

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: str) -> str:
        validate_plain_text(value, field_name="MCP command")
        if any(character.isspace() for character in value):
            raise ValueError("MCP command must be one executable token")
        return value

    @field_validator("args")
    @classmethod
    def validate_args(cls, values: list[str]) -> list[str]:
        for value in values:
            validate_plain_text(value, field_name="MCP argument")
            option = _option_flag(value).lstrip("-")
            if _is_secret_name(option):
                raise ValueError("secret-shaped MCP arguments must use an environment secret reference")
        return values

    _environment_uses_secret_references = field_validator("environment")(validate_environment_bindings)


MCPProviderConfig = Annotated[
    RemoteMCPProviderConfig | LocalStdioMCPProviderConfig,
    Field(discriminator="provider_type"),
]


class InvocationDiagnostics(AuthoredConfig):
    log_requests: bool = False


class InvocationConfig(AuthoredConfig):
    num_records: PositiveInt
    dataset_name: Identifier
    resume: Literal["never", "always", "if_possible"] = "never"
    run_config: dict[str, JsonValue] = Field(default_factory=dict)
    input_bindings: InputBindings = Field(default_factory=InputBindings)
    mcp_providers: list[MCPProviderConfig] = Field(default_factory=list)
    model_concurrency: dict[ModelAlias, PositiveInt] = Field(default_factory=dict)
    diagnostics: InvocationDiagnostics = Field(default_factory=InvocationDiagnostics)

    @field_validator("run_config")
    @classmethod
    def validate_run_config_keys(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        unknown = set(value).difference(RunConfig.model_fields)
        if unknown:
            raise ValueError(f"unknown Data Designer RunConfig fields: {', '.join(sorted(unknown))}")
        RunConfig.model_validate(value)
        return value

    @model_validator(mode="after")
    def validate_mcp_providers(self) -> InvocationConfig:
        names = [provider.name for provider in self.mcp_providers]
        if len(names) != len(set(names)):
            raise ValueError("MCP provider names must be unique")
        return self


class ClientDependencies(AuthoredConfig):
    requirements: list[str] | None = Field(default_factory=list)
    lock_file: str | None = None
    index_credentials: dict[str, SecretRef] = Field(default_factory=dict)

    @field_validator("requirements")
    @classmethod
    def validate_requirements(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        names: list[str] = []
        for value in values:
            validate_plain_text(value, field_name="dependency requirement")
            if value != value.strip() or value.startswith(("-e ", "/", "./", "../")) or "git+" in value:
                raise ValueError(f"dependency requirement must identify a package or immutable wheel: {value!r}")
            try:
                requirement = Requirement(value)
            except InvalidRequirement as error:
                raise ValueError(f"invalid dependency requirement: {value!r}") from error
            if requirement.marker is not None:
                raise ValueError("dependency requirement environment markers are not supported")
            if requirement.url is not None:
                validate_url(requirement.url, field_name="direct dependency URL")
                parsed = urlsplit(requirement.url)
                valid_wheel = (
                    parsed.scheme == "https"
                    and parsed.hostname is not None
                    and parsed.username is None
                    and parsed.password is None
                    and not parsed.query
                    and parsed.path.endswith(".whl")
                    and re.fullmatch(r"sha256=[0-9a-f]{64}", parsed.fragment) is not None
                )
                if not valid_wheel:
                    raise ValueError("direct dependency URLs must be HTTPS wheels with a SHA-256 fragment")
            names.append(canonicalize_name(requirement.name))
        if len(names) != len(set(names)):
            raise ValueError("dependency requirements must have unique normalized names")
        return values

    @field_validator("lock_file")
    @classmethod
    def validate_lock_file(cls, value: str | None) -> str | None:
        if value is None:
            return None
        validate_plain_text(value, field_name="dependency lock path")
        if "://" in value or ".." in value.split("/"):
            raise ValueError("dependency lock must be a normalized local path")
        normalized = posixpath.normpath(value)
        if not normalized.endswith(".json"):
            raise ValueError("dependency lock path must end in .json")
        return normalized

    @model_validator(mode="after")
    def validate_source(self) -> ClientDependencies:
        if self.lock_file is None and self.requirements is None:
            raise ValueError("client dependencies require requirements or lock_file")
        if self.lock_file is not None and self.requirements is not None:
            raise ValueError("client dependencies cannot contain both requirements and lock_file")
        return self


class ClientConfig(AuthoredConfig):
    cpus: PositiveInt = 32
    image: ImageRef
    dependencies: ClientDependencies = Field(default_factory=ClientDependencies)


class QueueBackpressureConfig(AuthoredConfig):
    max_waiting_requests: NonNegativeInt = 128
    retry_after_seconds: NonNegativeInt | None = 1


class VllmServerConfig(AuthoredConfig):
    type: Literal["vllm"]
    image: ImageRef
    startup_timeout: Duration = "15m"
    distributed_init_timeout: Duration = "10m"
    lead_boot_standoff: NonNegativeDuration = "60s"
    rank_launch_stagger: NonNegativeDuration = "5s"
    readiness_path: str = "/health"
    enable_expert_parallel: bool = False
    queue_backpressure: QueueBackpressureConfig = Field(default_factory=QueueBackpressureConfig)
    extra_args: list[str] = Field(default_factory=list)
    environment: dict[EnvironmentName, EnvironmentBinding] = Field(default_factory=dict)

    _readiness_path_is_safe = field_validator("readiness_path")(validate_vllm_readiness_path)

    @field_validator("extra_args")
    @classmethod
    def validate_extra_args(cls, values: list[str]) -> list[str]:
        return validate_vllm_extra_args(values)

    _environment_is_safe = field_validator("environment")(validate_vllm_environment_bindings)

    @model_validator(mode="after")
    def validate_timeouts(self) -> VllmServerConfig:
        if _duration_seconds(self.distributed_init_timeout) > _duration_seconds(self.startup_timeout):
            raise ValueError("distributed_init_timeout must not exceed startup_timeout")
        return self


class DeploymentResources(AuthoredConfig):
    nodes: PositiveInt = 1


class DeploymentTopology(AuthoredConfig):
    tensor_parallel: PositiveInt = 1
    nodes_per_replica: PositiveInt = 1


class ServerDeploymentConfig(AuthoredConfig):
    model_alias: ModelAlias
    served_model_name: str | None = None
    model: str
    server: VllmServerConfig
    resources: DeploymentResources = Field(default_factory=DeploymentResources)
    topology: DeploymentTopology = Field(default_factory=DeploymentTopology)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        validate_plain_text(value, field_name="model")
        if value.startswith("/"):
            return validate_absolute_path(value)
        if any(character.isspace() for character in value):
            raise ValueError("Hugging Face model identifiers must not contain whitespace")
        return value

    @field_validator("served_model_name")
    @classmethod
    def validate_served_model_name(cls, value: str | None) -> str | None:
        return None if value is None else validate_plain_text(value, field_name="served model name")

    @model_validator(mode="after")
    def validate_topology(self) -> ServerDeploymentConfig:
        if self.resources.nodes % self.topology.nodes_per_replica:
            raise ValueError("nodes_per_replica must divide deployment nodes")
        if self.server.enable_expert_parallel and self.topology.nodes_per_replica > 1:
            raise ValueError("multi-node expert parallel is not supported in v1")
        return self


class ArrayTasksConfig(AuthoredConfig):
    count: PositiveInt = 1
    max_concurrent: PositiveInt = 1

    @model_validator(mode="after")
    def validate_concurrency(self) -> ArrayTasksConfig:
        if self.max_concurrent > self.count:
            raise ValueError("array task concurrency must not exceed task count")
        return self


class SubmissionConfig(AuthoredConfig):
    account: Identifier | None = None
    partition: Identifier | None = None
    job_name: Identifier = "data-designer"
    time_limit: Annotated[str, StringConstraints(pattern=r"^(?:[0-9]+-)?[0-9]{2}:[0-9]{2}:[0-9]{2}$")] = "03:55:00"
    comment: Annotated[str, StringConstraints(max_length=256)] | None = None

    @field_validator("time_limit")
    @classmethod
    def validate_time_limit(cls, value: str) -> str:
        clock = value.rsplit("-", maxsplit=1)[-1]
        _, minutes, seconds = (int(part) for part in clock.split(":"))
        if minutes >= 60 or seconds >= 60:
            raise ValueError("time_limit minutes and seconds must be below 60")
        return value

    @field_validator("comment")
    @classmethod
    def validate_comment(cls, value: str | None) -> str | None:
        return None if value is None else validate_plain_text(value, field_name="submission comment")


class OutputConfig(AuthoredConfig):
    root: str | None = None
    format: Literal["parquet", "jsonl", "csv"] = "parquet"
    partitions: PositiveInt = 1
    require_exact_record_count: bool = False

    @field_validator("root")
    @classmethod
    def validate_root(cls, value: str | None) -> str | None:
        return None if value is None else validate_absolute_path(value)


class DataDesignerSlurmConfig(AuthoredConfig):
    """Complete portable intent for one Data Designer Slurm run."""

    schema_version: SchemaVersion
    name: Identifier
    builder: BuilderInput
    invocation: InvocationConfig
    client: ClientConfig
    deployments: list[ServerDeploymentConfig] = Field(min_length=1)
    array_tasks: ArrayTasksConfig = Field(default_factory=ArrayTasksConfig)
    submission: SubmissionConfig = Field(default_factory=SubmissionConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def validate_run(self) -> DataDesignerSlurmConfig:
        if self.array_tasks.count > self.invocation.num_records:
            raise ValueError("array task count must not exceed requested records")
        aliases = [deployment.model_alias for deployment in self.deployments]
        if len(aliases) != len(set(aliases)):
            raise ValueError("deployment model aliases must be unique")
        unknown_concurrency = set(self.invocation.model_concurrency).difference(aliases)
        if unknown_concurrency:
            raise ValueError(
                f"model concurrency references undeclared aliases: {', '.join(sorted(unknown_concurrency))}"
            )
        return self
