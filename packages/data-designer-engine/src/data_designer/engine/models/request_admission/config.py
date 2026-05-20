# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from data_designer.engine.models.request_admission.resources import RequestResourceKey


@dataclass(frozen=True)
class RequestAdmissionConfig:
    initial_limits: Mapping[RequestResourceKey, int] = field(default_factory=dict)
    max_limit_clamps: Mapping[RequestResourceKey, int | None] = field(default_factory=dict)
    cooldown_seconds: float = 2.0
    multiplicative_decrease_factor: float = 0.75
    additive_increase_step: int = 1
    increase_after_successes: int = 25
    startup_ramp_seconds: float = 0.0
    default_queue_wait_timeout_seconds: float | None = None
