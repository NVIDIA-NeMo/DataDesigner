# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authored vLLM server configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, NonNegativeInt, PositiveInt, field_validator, model_validator

from data_designer.slurm.config.environment import EnvironmentBinding
from data_designer.slurm.config.images import ImageRef
from data_designer.slurm.config.utils import convert_duration_to_seconds
from data_designer.slurm.config.vllm_validation import (
    validate_vllm_environment_bindings,
    validate_vllm_extra_args,
    validate_vllm_readiness_path,
)
from data_designer.slurm.contracts import AuthoredConfig
from data_designer.slurm.types import Duration, EnvironmentName, NonNegativeDuration

__all__ = [
    "QueueBackpressureConfig",
    "VllmServerConfig",
]


class QueueBackpressureConfig(AuthoredConfig):
    """Admission policy applied by the logical endpoint under queue pressure.

    Attributes:
        max_waiting_requests: Maximum queued requests before admission is rejected.
        retry_after_seconds: Retry delay returned to callers, or ``None`` to omit it.
    """

    max_waiting_requests: NonNegativeInt = 128
    retry_after_seconds: PositiveInt | None = 1


class VllmServerConfig(AuthoredConfig):
    """Authored vLLM image, lifecycle, and safe launch customization.

    Duration fields use Slurm duration strings; resolution converts them to integer
    seconds. ``extra_args`` contains individual argv items, never shell fragments.

    Attributes:
        type: Serving-backend discriminator.
        image: Authored image reference to inspect and resolve.
        startup_timeout: Total deadline for backend readiness.
        distributed_init_timeout: Deadline for distributed initialization.
        lead_boot_standoff: Delay before non-leading replicas launch.
        rank_launch_stagger: Additional delay per deployment-wide replica index.
        readiness_path: Absolute HTTP path used for readiness probes.
        enable_expert_parallel: Whether vLLM expert parallelism is requested.
        queue_backpressure: Logical-endpoint queue admission policy.
        extra_args: Additional shell-free vLLM argv items not reserved by orchestration.
        environment: Explicit literal or secret-reference environment bindings.
    """

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
        validate_vllm_extra_args(values)
        return values

    @field_validator("environment")
    @classmethod
    def validate_environment(
        cls, values: dict[EnvironmentName, EnvironmentBinding]
    ) -> dict[EnvironmentName, EnvironmentBinding]:
        validate_vllm_environment_bindings(values)
        return values

    @model_validator(mode="after")
    def validate_timeouts(self) -> VllmServerConfig:
        if convert_duration_to_seconds(self.distributed_init_timeout) > convert_duration_to_seconds(
            self.startup_timeout
        ):
            raise ValueError("distributed_init_timeout must not exceed startup_timeout")
        return self
