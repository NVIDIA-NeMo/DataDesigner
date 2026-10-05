# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import Field

from data_designer.config.base import ConfigBase


class RequestAdmissionTuningConfig(ConfigBase):
    """Advanced request-admission AIMD tuning for model API calls.

    Most workloads should tune model capacity with ``max_parallel_requests`` on
    inference parameters. These fields adjust the adaptive recovery behavior
    below that cap and are intended for provider/runtime support cases.
    """

    multiplicative_decrease_factor: float = Field(
        default=0.75,
        gt=0.0,
        lt=1.0,
        description="Factor applied to the adaptive concurrency limit after a provider rate-limit signal.",
    )
    additive_increase_step: int = Field(
        default=1,
        ge=1,
        description="Slots added to the adaptive concurrency limit after each successful recovery window.",
    )
    successes_until_increase: int = Field(
        default=25,
        ge=1,
        description="Successful releases required before additive recovery increases the adaptive limit.",
    )
    cooldown_seconds: float = Field(
        default=2.0,
        gt=0.0,
        description="Fallback cooldown after a rate-limit signal when the provider omits Retry-After.",
    )
    startup_ramp_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Startup ramp duration. When greater than zero, each request resource starts at one "
            "concurrent request and linearly ramps to its configured cap unless a rate-limit aborts the ramp."
        ),
    )
