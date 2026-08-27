# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pydantic import ValidationError

from data_designer.slurm.config import QueueBackpressureConfig
from data_designer.slurm.serving.vllm import ResolvedVllmLaunchPolicy


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"readiness_path": "health"}, "absolute URL path"),
        ({"readiness_path": "//other-host/health"}, "absolute URL path"),
        ({"readiness_path": "/health check"}, "without whitespace"),
        ({"startup_timeout_seconds": 10, "distributed_init_timeout_seconds": 11}, "must not exceed"),
    ],
)
def test_launch_policy_rejects_invalid_readiness_or_deadlines(update: dict[str, object], message: str) -> None:
    payload = {
        "startup_timeout_seconds": 900,
        "distributed_init_timeout_seconds": 600,
        "lead_boot_standoff_seconds": 60,
        "rank_launch_stagger_seconds": 5,
        "readiness_path": "/health",
        "enable_expert_parallel": False,
        "queue_backpressure": {"max_waiting_requests": 128, "retry_after_seconds": 1},
        **update,
    }
    with pytest.raises(ValidationError, match=message):
        ResolvedVllmLaunchPolicy.model_validate(payload)


@pytest.mark.parametrize(
    "argument",
    [
        "-a",
        "-as",
        "-dc",
        "-dcp=2",
        "-e",
        "-n2",
        "-n+2",
        "-pc",
        "-pcp=2",
        "-r1",
        "-r0",
        "-t",
        "--kv-transfer-config={}",
        "--numa-bind",
        "--reasoning-parser-plugin=/tmp/reasoning.py",
        "--worker-cls=custom.Worker",
    ],
)
def test_launch_policy_rejects_runtime_owned_arguments(argument: str) -> None:
    with pytest.raises(ValidationError, match="owned by the compiler or runtime"):
        ResolvedVllmLaunchPolicy(
            startup_timeout_seconds=900,
            distributed_init_timeout_seconds=600,
            lead_boot_standoff_seconds=60,
            rank_launch_stagger_seconds=5,
            readiness_path="/health",
            enable_expert_parallel=False,
            queue_backpressure=QueueBackpressureConfig(),
            extra_args=(argument,),
        )


@pytest.mark.parametrize(
    "environment_name",
    ["CUDA_VISIBLE_DEVICES", "VLLM_API_KEY", "VLLM_DP_RANK", "VLLM_HOST_IP"],
)
def test_launch_policy_rejects_runtime_owned_environment_names(environment_name: str) -> None:
    with pytest.raises(ValidationError, match="owned by the compiler or runtime"):
        ResolvedVllmLaunchPolicy(
            startup_timeout_seconds=900,
            distributed_init_timeout_seconds=600,
            lead_boot_standoff_seconds=60,
            rank_launch_stagger_seconds=5,
            readiness_path="/health",
            enable_expert_parallel=False,
            queue_backpressure=QueueBackpressureConfig(),
            environment={environment_name: {"type": "secret", "environment": "EXTERNAL_VALUE"}},
        )
