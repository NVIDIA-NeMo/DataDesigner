# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

from pydantic import Field, model_validator
from typing_extensions import Self

from data_designer.config.base import ConfigBase
from data_designer.config.run_config_deprecated import _THROTTLE_DEPRECATION_MESSAGE, ThrottleConfig
from data_designer.config.utils.type_helpers import StrEnum
from data_designer.config.utils.warning_helpers import warn_at_caller


class JinjaRenderingEngine(StrEnum):
    """Template renderer used by the engine for user-supplied Jinja templates."""

    NATIVE = "native"
    SECURE = "secure"


class ResumeMode(StrEnum):
    NEVER = "never"
    ALWAYS = "always"
    IF_POSSIBLE = "if_possible"


_PROGRESS_BAR_DEPRECATION_MESSAGE = "RunConfig.progress_bar is deprecated. Use RunConfig.display_tui instead."


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


class RunConfig(ConfigBase):
    """Runtime configuration for dataset generation.

    Groups configuration options that control generation behavior but aren't
    part of the dataset configuration itself.

    Notes:
        Request-admission controller internals remain engine-owned. ``request_admission``
        exposes only the supported tuning DTO and does not expose controller
        mutation APIs, leases, queues, or pressure snapshots.
    """

    # Early shutdown
    disable_early_shutdown: bool = Field(
        default=False,
        description=(
            "If True, disables the executor's early-shutdown behavior entirely. Generation will continue "
            "regardless of error rate, and the early-shutdown exception will never be raised. Error counts "
            "and summaries are still collected."
        ),
    )
    shutdown_error_rate: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Error rate threshold (0.0-1.0) that triggers early shutdown when early shutdown is enabled.",
    )
    shutdown_error_window: int = Field(
        default=10,
        ge=1,
        description="Minimum number of completed tasks before error rate monitoring begins.",
    )

    # Scheduling and concurrency
    buffer_size: int = Field(
        default=1000,
        gt=0,
        description="Number of records in each row group during dataset generation.",
    )
    max_concurrent_row_groups: int = Field(
        default=3,
        ge=1,
        description="Maximum number of row groups the async scheduler may keep active at once.",
    )
    max_in_flight_tasks: int = Field(
        default=1024,
        ge=1,
        description=(
            "Maximum number of async scheduler tasks that may hold task leases at once. "
            "Model API request concurrency is controlled separately by max_parallel_requests."
        ),
    )
    non_inference_max_parallel_workers: int = Field(
        default=4,
        ge=1,
        description="Maximum number of worker threads used for non-inference cell-by-cell generators.",
    )

    # Conversation recovery
    max_conversation_restarts: int = Field(
        default=5,
        ge=0,
        description=(
            "Maximum number of full conversation restarts permitted when generation tasks call "
            "`ModelFacade.generate(...)`."
        ),
    )
    max_conversation_correction_steps: int = Field(
        default=0,
        ge=0,
        description=(
            "Maximum number of correction rounds permitted within a single conversation when generation "
            "tasks call `ModelFacade.generate(...)`."
        ),
    )

    # Observability
    async_trace: bool = Field(default=False, description="If True, collect per-task tracing data.")
    write_scheduler_events: bool = Field(
        default=False,
        description=(
            "If True, create runs write structured scheduler diagnostics to ``scheduler_events.jsonl`` in the "
            "dataset directory. The file is JSONL, not direct Perfetto input, and may contain sensitive column, "
            "provider, model, task, and resource labels. Each event is flushed to disk, so enabling this option "
            "adds file I/O overhead. Preview runs do not write it."
        ),
    )
    display_tui: bool = Field(
        default=False,
        description=(
            "If True, display the terminal throughput TUI instead of periodic log lines during generation. "
            "Requires a TTY; falls back to log lines in non-TTY environments."
        ),
    )
    progress_interval: float = Field(
        default=5.0,
        gt=0.0,
        description="How often (in seconds) the async progress reporter emits a consolidated log block.",
    )
    otel_metrics_port: int | None = Field(
        default=9464,
        ge=1,
        le=65535,
        description=(
            "Loopback port for the pull-based OpenTelemetry metrics endpoint. "
            "None disables instrumentation for this create invocation; an existing process endpoint may remain "
            "available for prior metrics. Raw log records are not exposed."
        ),
    )

    # Output
    preserve_dropped_columns: bool = Field(
        default=True,
        description=(
            "Whether columns removed by drop processors are preserved in separate dropped-column parquet files."
        ),
    )

    # Templating
    jinja_rendering_engine: JinjaRenderingEngine = Field(
        default=JinjaRenderingEngine.SECURE,
        description=(
            "Template renderer used for engine-side Jinja evaluation. "
            "`native` uses Jinja2's built-in sandbox; `secure` uses Data Designer's hardened sandbox."
        ),
    )

    # Request admission
    request_admission: RequestAdmissionTuningConfig | None = Field(
        default=None,
        description=(
            "Advanced AIMD request-admission tuning for provider/model calls. "
            "Most users should leave this unset and tune ``max_parallel_requests`` instead."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def translate_deprecated_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        normalized = dict(data)

        if "progress_bar" in normalized:
            progress_bar = normalized.pop("progress_bar")
            normalized.setdefault("display_tui", progress_bar)
            warn_at_caller(
                _PROGRESS_BAR_DEPRECATION_MESSAGE,
                DeprecationWarning,
            )

        if "throttle" in normalized:
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
            warn_at_caller(
                _THROTTLE_DEPRECATION_MESSAGE,
                DeprecationWarning,
            )
            return normalized
        return normalized

    @property
    def progress_bar(self) -> bool:
        warnings.warn(
            _PROGRESS_BAR_DEPRECATION_MESSAGE,
            DeprecationWarning,
            stacklevel=2,
        )
        return self.display_tui

    @progress_bar.setter
    def progress_bar(self, value: bool) -> None:
        warnings.warn(
            _PROGRESS_BAR_DEPRECATION_MESSAGE,
            DeprecationWarning,
            stacklevel=2,
        )
        self.display_tui = value

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        if update is not None and "progress_bar" in update:
            normalized_update = dict(update)
            progress_bar = normalized_update.pop("progress_bar")
            normalized_update.setdefault("display_tui", progress_bar)
            warnings.warn(
                _PROGRESS_BAR_DEPRECATION_MESSAGE,
                DeprecationWarning,
                stacklevel=2,
            )
            update = normalized_update
        return super().model_copy(update=update, deep=deep)

    @property
    def effective_shutdown_error_rate(self) -> float:
        """Error rate handed to early-shutdown checks: 1.0 when early shutdown is disabled."""
        return 1.0 if self.disable_early_shutdown else self.shutdown_error_rate
