# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from data_designer.config.base import ConfigBase

if TYPE_CHECKING:
    from data_designer.config.run_config import RequestAdmissionTuningConfig

_THROTTLE_DEPRECATION_MESSAGE = (
    "RunConfig.throttle and ThrottleConfig are deprecated. Use RunConfig.request_admission with "
    "RequestAdmissionTuningConfig for supported advanced request-admission tuning."
)


class ThrottleConfig(ConfigBase):
    """Deprecated compatibility DTO for request-admission tuning.

    Use ``RequestAdmissionTuningConfig`` via ``RunConfig.request_admission``
    instead. ``ceiling_overshoot`` is accepted for compatibility but is not
    forwarded because request admission no longer exposes an overshoot knob.
    """

    reduce_factor: float = Field(
        default=0.75,
        gt=0.0,
        lt=1.0,
        description="Deprecated alias for RequestAdmissionTuningConfig.multiplicative_decrease_factor.",
    )
    additive_increase: int = Field(
        default=1,
        ge=1,
        description="Deprecated alias for RequestAdmissionTuningConfig.additive_increase_step.",
    )
    success_window: int = Field(
        default=25,
        ge=1,
        description="Deprecated alias for RequestAdmissionTuningConfig.successes_until_increase.",
    )
    cooldown_seconds: float = Field(
        default=2.0,
        gt=0.0,
        description="Deprecated alias for RequestAdmissionTuningConfig.cooldown_seconds.",
    )
    ceiling_overshoot: float = Field(
        default=0.10,
        ge=0.0,
        description="Deprecated compatibility field; not forwarded to request admission.",
    )
    rampup_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description=("Deprecated alias for RequestAdmissionTuningConfig.startup_ramp_seconds."),
    )

    def to_request_admission_tuning(self) -> RequestAdmissionTuningConfig:
        """Translate legacy throttle tuning into the request-admission DTO."""
        # Runtime import: module-level would cause circular import
        from data_designer.config.run_config import RequestAdmissionTuningConfig

        return RequestAdmissionTuningConfig(
            multiplicative_decrease_factor=self.reduce_factor,
            additive_increase_step=self.additive_increase,
            successes_until_increase=self.success_window,
            cooldown_seconds=self.cooldown_seconds,
            startup_ramp_seconds=self.rampup_seconds,
        )
