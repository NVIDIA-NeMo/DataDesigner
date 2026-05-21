# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from typing import Any

from pydantic import Field, model_validator
from typing_extensions import Self

from data_designer.config.base import ConfigBase
from data_designer.config.utils.type_helpers import StrEnum


class JinjaRenderingEngine(StrEnum):
    """Template renderer used by the engine for user-supplied Jinja templates."""

    NATIVE = "native"
    SECURE = "secure"


_THROTTLE_DEPRECATION_MESSAGE = (
    "RunConfig.throttle and ThrottleConfig are deprecated. Use RunConfig.request_admission with "
    "RequestAdmissionTuningConfig for supported advanced request-admission tuning."
)


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
        return RequestAdmissionTuningConfig(
            multiplicative_decrease_factor=self.reduce_factor,
            additive_increase_step=self.additive_increase,
            successes_until_increase=self.success_window,
            cooldown_seconds=self.cooldown_seconds,
            startup_ramp_seconds=self.rampup_seconds,
        )


class RunConfig(ConfigBase):
    """Runtime configuration for dataset generation.

    Groups configuration options that control generation behavior but aren't
    part of the dataset configuration itself.

    Attributes:
        disable_early_shutdown: If True, disables the executor's early-shutdown behavior entirely.
            Generation will continue regardless of error rate, and the early-shutdown exception
            will never be raised. Error counts and summaries are still collected. Default is False.
        shutdown_error_rate: Error rate threshold (0.0-1.0) that triggers early shutdown when
            early shutdown is enabled. Default is 0.5.
        shutdown_error_window: Minimum number of completed tasks before error rate
            monitoring begins. Must be >= 1. Default is 10.
        buffer_size: Number of records to process in each batch during dataset generation.
            A batch is processed end-to-end (column generation, post-batch processors, and writing the batch
            to artifact storage) before moving on to the next batch. Must be > 0. Default is 1000.
        non_inference_max_parallel_workers: Maximum number of worker threads used for non-inference
            cell-by-cell generators. Must be >= 1. Default is 4.
        max_conversation_restarts: Maximum number of full conversation restarts permitted when
            generation tasks call `ModelFacade.generate(...)`. Must be >= 0. Default is 5.
        max_conversation_correction_steps: Maximum number of correction rounds permitted within a
            single conversation when generation tasks call `ModelFacade.generate(...)`. Must be >= 0.
            Default is 0.
        async_trace: If True, collect per-task tracing data when using the async engine
            (DATA_DESIGNER_ASYNC_ENGINE=1). Has no effect on the sync path. Default is False.
        progress_bar: If True, display a sticky ANSI throughput chart panel instead of periodic
            log lines during generation. Requires a TTY; falls back to log lines in non-TTY
            environments. Default is True.
        progress_interval: How often (in seconds) the async progress reporter emits a
            consolidated log block. Must be > 0. Default is 5.0.
        jinja_rendering_engine: Template renderer used for engine-side Jinja evaluation.
            ``native`` uses Jinja2's built-in sandbox with the standard filter set and
            fewer Data Designer-specific restrictions. ``secure`` uses Data Designer's
            hardened sandbox with additional AST, filter, and output guards.
            Default is ``secure``.
        request_admission: Advanced AIMD request-admission tuning for provider/model calls.
            Most users should leave this unset and tune ``max_parallel_requests`` instead.

    Notes:
        Request-admission controller internals remain engine-owned. This field
        exposes only the supported tuning DTO and does not expose controller
        mutation APIs, leases, queues, or pressure snapshots.
    """

    disable_early_shutdown: bool = False
    shutdown_error_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    shutdown_error_window: int = Field(default=10, ge=1)
    buffer_size: int = Field(default=1000, gt=0)
    non_inference_max_parallel_workers: int = Field(default=4, ge=1)
    max_conversation_restarts: int = Field(default=5, ge=0)
    max_conversation_correction_steps: int = Field(default=0, ge=0)
    async_trace: bool = False
    progress_bar: bool = True
    progress_interval: float = Field(default=5.0, gt=0.0)
    jinja_rendering_engine: JinjaRenderingEngine = Field(
        default=JinjaRenderingEngine.SECURE,
        description=(
            "Template renderer used for engine-side Jinja evaluation. "
            "`native` uses Jinja2's built-in sandbox; `secure` uses Data Designer's hardened sandbox."
        ),
    )
    request_admission: RequestAdmissionTuningConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def translate_deprecated_throttle_config(cls, data: Any) -> Any:
        if isinstance(data, dict) and "throttle" in data:
            normalized = dict(data)
            throttle = normalized.pop("throttle")
            if normalized.get("request_admission") is not None:
                raise ValueError(
                    "Specify either RunConfig.throttle or RunConfig.request_admission, not both. "
                    "RunConfig.throttle is deprecated."
                )
            if throttle is not None:
                throttle_config = (
                    throttle if isinstance(throttle, ThrottleConfig) else ThrottleConfig.model_validate(throttle)
                )
                normalized["request_admission"] = throttle_config.to_request_admission_tuning()
            warnings.warn(
                _THROTTLE_DEPRECATION_MESSAGE,
                DeprecationWarning,
                stacklevel=2,
            )
            return normalized
        return data

    @model_validator(mode="after")
    def normalize_shutdown_settings(self) -> Self:
        """Normalize shutdown settings for compatibility."""
        if self.disable_early_shutdown:
            self.shutdown_error_rate = 1.0
        return self
