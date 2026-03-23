# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import ClassVar

from pydantic import Field, model_validator
from typing_extensions import Self

from data_designer.config.base import ConfigBase


class ThrottleConfig(ConfigBase):
    """AIMD throttle tuning parameters for adaptive concurrency control.

    These knobs configure the ``ThrottleManager`` that wraps every outbound
    model HTTP request.  The defaults are conservative and suitable for most
    workloads; override only when you understand the trade-offs.

    Attributes:
        reduce_factor: Multiplicative decrease factor applied to the per-domain
            concurrency limit on a 429 / rate-limit signal.  Must be in (0, 1).
            Default is 0.75 (reduce by 25% on rate-limit).
        additive_increase: Additive increase step applied after every
            ``success_window`` consecutive successes.  Default is 1.
        success_window: Number of consecutive successful releases before
            the additive increase is applied.  Default is 25.
        cooldown_seconds: Default cooldown duration (seconds) applied after a
            rate-limit when the provider does not include a ``Retry-After``
            header.  Default is 2.0.
        ceiling_overshoot: Fraction above the observed rate-limit ceiling
            that additive increase is allowed to probe before capping.
            Default is 0.10 (10% overshoot).
    """

    DEFAULT_REDUCE_FACTOR: ClassVar[float] = 0.75
    DEFAULT_ADDITIVE_INCREASE: ClassVar[int] = 1
    DEFAULT_SUCCESS_WINDOW: ClassVar[int] = 25
    DEFAULT_COOLDOWN_SECONDS: ClassVar[float] = 2.0
    DEFAULT_CEILING_OVERSHOOT: ClassVar[float] = 0.10

    reduce_factor: float = Field(
        default=DEFAULT_REDUCE_FACTOR,
        gt=0.0,
        lt=1.0,
        description="Multiplicative decrease factor applied to the per-domain concurrency limit on a 429 signal.",
    )
    additive_increase: int = Field(
        default=DEFAULT_ADDITIVE_INCREASE,
        ge=1,
        description="Additive increase step applied after every `success_window` consecutive successes.",
    )
    success_window: int = Field(
        default=DEFAULT_SUCCESS_WINDOW,
        ge=1,
        description="Number of consecutive successful releases before the additive increase is applied.",
    )
    cooldown_seconds: float = Field(
        default=DEFAULT_COOLDOWN_SECONDS,
        gt=0.0,
        description="Default cooldown duration (seconds) after a rate-limit when no Retry-After header is present.",
    )
    ceiling_overshoot: float = Field(
        default=DEFAULT_CEILING_OVERSHOOT,
        ge=0.0,
        description="Fraction above the rate-limit ceiling that additive increase is allowed to probe.",
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
        throttle: AIMD throttle tuning parameters.  See ``ThrottleConfig`` for details.
    """

    disable_early_shutdown: bool = False
    shutdown_error_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    shutdown_error_window: int = Field(default=10, ge=1)
    buffer_size: int = Field(default=1000, gt=0)
    non_inference_max_parallel_workers: int = Field(default=4, ge=1)
    max_conversation_restarts: int = Field(default=5, ge=0)
    max_conversation_correction_steps: int = Field(default=0, ge=0)
    async_trace: bool = False
    throttle: ThrottleConfig = Field(default_factory=ThrottleConfig)

    @model_validator(mode="after")
    def normalize_shutdown_settings(self) -> Self:
        """Normalize shutdown settings for compatibility."""
        if self.disable_early_shutdown:
            self.shutdown_error_rate = 1.0
        return self
