# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation and normalization owned by the vLLM configuration boundary."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from data_designer.slurm.config.environment import (
    EnvironmentBinding,
    is_secret_bearing_name,
    validate_environment_bindings,
)
from data_designer.slurm.contracts import validate_plain_text
from data_designer.slurm.types import EnvironmentName


def validate_vllm_environment_bindings(values: Mapping[EnvironmentName, EnvironmentBinding]) -> None:
    """Reject environment names derived by the serving compiler or runtime."""
    validate_environment_bindings(values)
    conflicts = sorted(
        name
        for name in values
        if name in _RESERVED_VLLM_ENVIRONMENT_NAMES or name.startswith(_RESERVED_VLLM_ENVIRONMENT_PREFIXES)
    )
    if conflicts:
        raise ValueError(f"vLLM environment names are owned by the compiler or runtime: {', '.join(conflicts)}")


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


def validate_vllm_extra_args(values: Sequence[str]) -> None:
    """Validate shell-free vLLM argv without interpreting plugin-owned values."""
    seen_flags: set[str] = set()
    for value in values:
        validate_plain_text(value, field_name="vLLM argument")
        flag = value.partition("=")[0].replace("_", "-")
        candidate_flag = flag.split(maxsplit=1)[0]
        if flag == "--":
            raise ValueError("vLLM argument option terminators are not supported")
        if _is_reserved_vllm_flag(candidate_flag):
            raise ValueError(f"vLLM argument {candidate_flag!r} is owned by the compiler or runtime")
        if candidate_flag.startswith("--") and is_secret_bearing_name(candidate_flag.removeprefix("--")):
            raise ValueError("secret-shaped vLLM arguments must use an environment secret reference")
        if flag.startswith("-") and re.fullmatch(r"--[A-Za-z][A-Za-z0-9.-]*", flag) is None:
            raise ValueError(
                "short or combined vLLM options are owned by the compiler or runtime; "
                "each extra option must use one complete long flag in one token"
            )
        if flag.startswith("--"):
            canonical_flag = f"--{flag.removeprefix('--no-')}" if flag.startswith("--no-") else flag
            if canonical_flag in seen_flags:
                raise ValueError(f"duplicate or conflicting vLLM argument {flag!r}")
            seen_flags.add(canonical_flag)


_RESERVED_VLLM_FLAGS = frozenset(
    {
        "--api-key",
        "--api-server-count",
        "--config",
        "--cpu-distributed-timeout-seconds",
        "--cp-kv-cache-interleave-size",
        "--cpunodebind",
        "--dcp-comm-backend",
        "--dcp-kv-cache-interleave-size",
        "--default-mm-loras",
        "--distributed-executor-backend",
        "--distributed-init-address",
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
        "--lora-modules",
        "--master-addr",
        "--master-port",
        "--max-cpu-loras",
        "--max-loras",
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
        "--physcpubind",
        "--pipeline-parallel-size",
        "--port",
        "--reasoning-parser-plugin",
        "--root-path",
        "--served-model-name",
        "--spec-model",
        "--speculative-config",
        "--tensor-parallel-size",
        "--tool-parser-plugin",
        "--uds",
        "--weight-transfer-config",
        "--worker-cls",
        "--worker-extension-cls",
    }
)
_VLLM_EXACT_PASSTHROUGH_FLAGS = frozenset({"--reasoning-parser"})
_RESERVED_VLLM_FLAG_PREFIXES = (
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
_RESERVED_VLLM_ENVIRONMENT_NAMES = frozenset(
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
_RESERVED_VLLM_ENVIRONMENT_PREFIXES = (
    "SLURM_",
    "TORCHELASTIC_",
    "VLLM_DP_",
    "VLLM_LORA_",
    "VLLM_MOONCAKE_",
    "VLLM_NIXL_",
    "VLLM_RAY_",
)


def _is_reserved_vllm_flag(flag: str) -> bool:
    if flag in _VLLM_EXACT_PASSTHROUGH_FLAGS:
        return False
    return (
        flag in _RESERVED_VLLM_FLAGS
        or flag.startswith(_RESERVED_VLLM_FLAG_PREFIXES)
        or (flag.startswith("--") and any(reserved_flag.startswith(flag) for reserved_flag in _RESERVED_VLLM_FLAGS))
    )
