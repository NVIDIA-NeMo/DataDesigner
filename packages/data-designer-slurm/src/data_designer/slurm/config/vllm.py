# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authored vLLM configuration and launch-boundary validation."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Literal, TypeVar

from pydantic import Field, NonNegativeInt, PositiveInt, field_validator, model_validator

from data_designer.slurm.config.environment import (
    EnvironmentBinding,
    contains_secret_key,
    is_secret_name,
    validate_environment_bindings,
)
from data_designer.slurm.config.images import ImageRef
from data_designer.slurm.contracts import (
    AuthoredConfig,
    convert_duration_to_seconds,
    extract_option_flag,
    validate_plain_text,
)
from data_designer.slurm.types import Duration, EnvironmentName, NonNegativeDuration

__all__ = [
    "QueueBackpressureConfig",
    "VllmServerConfig",
    "validate_vllm_environment_bindings",
    "validate_vllm_extra_args",
    "validate_vllm_readiness_path",
]


class QueueBackpressureConfig(AuthoredConfig):
    max_waiting_requests: NonNegativeInt = 128
    retry_after_seconds: PositiveInt | None = 1


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

    @field_validator("readiness_path")
    @classmethod
    def validate_readiness_path(cls, value: str) -> str:
        return validate_vllm_readiness_path(value)

    @field_validator("extra_args")
    @classmethod
    def validate_extra_args(cls, values: list[str]) -> list[str]:
        return validate_vllm_extra_args(values)

    @field_validator("environment")
    @classmethod
    def validate_environment(
        cls, values: dict[EnvironmentName, EnvironmentBinding]
    ) -> dict[EnvironmentName, EnvironmentBinding]:
        return validate_vllm_environment_bindings(values)

    @model_validator(mode="after")
    def validate_timeouts(self) -> VllmServerConfig:
        if convert_duration_to_seconds(self.distributed_init_timeout) > convert_duration_to_seconds(
            self.startup_timeout
        ):
            raise ValueError("distributed_init_timeout must not exceed startup_timeout")
        return self


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
        flag = _canonicalize_vllm_option_flag(value)
        json_payload = _parse_vllm_json_argument(value)
        if flag == "--":
            raise ValueError("vLLM argument option terminators are not supported")
        if _is_owned_vllm_flag(flag):
            raise ValueError(f"vLLM argument {flag!r} is owned by the compiler or runtime")
        if _contains_owned_vllm_json_key(json_payload):
            raise ValueError("vLLM argument contains a field owned by the compiler or runtime")
        if contains_secret_key(json_payload):
            raise ValueError("secret-shaped vLLM JSON fields must use an environment secret reference")
        if is_secret_name(flag.lstrip("-")):
            raise ValueError("secret-shaped vLLM arguments must use an environment secret reference")
        if any(character.isspace() for character in value):
            raise ValueError("each vLLM argument must be one token")
        if re.match(r"^--?[A-Za-z]", flag) is not None:
            canonical_flag = f"--{flag.removeprefix('--no-')}" if flag.startswith("--no-") else flag
            if canonical_flag in seen_flags:
                raise ValueError(f"duplicate or conflicting vLLM argument {flag!r}")
            seen_flags.add(canonical_flag)
    return values


_Arguments = TypeVar("_Arguments", list[str], tuple[str, ...])
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
        "CACHE_ROOT",
        "CUDA_VISIBLE_DEVICES",
        "DP_GLOBAL",
        "DP_LOCAL",
        "ENABLE_EP",
        "GEN_WORKFLOW",
        "GLOBAL_REPLICA_COUNT",
        "GROUP_RANK",
        "HEAD_IP",
        "HEAD_NODE_NAME",
        "HF_HOME",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "LOG_DIR",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NODE_RANK",
        "NODE_GROUP_COUNT",
        "NVIDIA_VISIBLE_DEVICES",
        "PIPELINE_PARALLEL_SIZE",
        "PYTHON_EXEC",
        "RANK",
        "REPLICAS_PER_NODE_GROUP",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "SCRATCH_ROOT",
        "SERVER_HTTP_PORT",
        "SERVICE_DP_RPC_PORT",
        "SERVICE_GPU_IDS",
        "SERVICE_HTTP_PORT_BASE",
        "SERVICE_MASTER_PORT_BASE",
        "TENSOR_PARALLEL_SIZE",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING",
        "VLLM_API_KEY",
        "VLLM_CACHE_ROOT",
        "VLLM_DISTRIBUTED_TIMEOUT_SECONDS",
        "VLLM_HOST_IP",
        "VLLM_LEAD_BOOT_STANDOFF_SECONDS",
        "VLLM_MASTER_PORT_BASE",
        "VLLM_MODEL_REDIRECT_PATH",
        "VLLM_PLUGINS",
        "VLLM_PORT",
        "VLLM_RANK_LAUNCH_STAGGER_SECONDS",
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


def _contains_owned_vllm_json_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).replace("_", "-") in _OWNED_VLLM_JSON_KEYS or _contains_owned_vllm_json_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return any(_contains_owned_vllm_json_key(item) for item in value)
    return False


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
    return flag.startswith("-") and any(
        owned_flag.startswith(flag)
        for owned_flag in _OWNED_VLLM_FLAGS
        if owned_flag.startswith("--") == flag.startswith("--")
    )


def _canonicalize_vllm_option_flag(value: str) -> str:
    flag = extract_option_flag(value)
    return flag.replace("_", "-") if flag.startswith("--") else flag


def _parse_vllm_json_argument(value: str) -> object | None:
    candidate = value.partition("=")[2] if value.startswith("-") and "=" in value else value
    if not candidate.startswith(("{", "[")):
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ValueError("vLLM JSON arguments must contain valid JSON") from error
