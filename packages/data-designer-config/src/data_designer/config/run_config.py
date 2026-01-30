# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import Field, model_validator
from typing_extensions import Self

from data_designer.config.base import ConfigBase


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
            monitoring begins. Must be >= 0. Default is 10.
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
        debug_override_save_all_column_traces: If True, overrides per-column `with_trace` settings
            and includes `__trace` columns for ALL LLM generations, containing the full ordered
            message history (system/user/assistant) for the final generation attempt.
            Useful for debugging. Default is False.
    """

    disable_early_shutdown: bool = False
    shutdown_error_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    shutdown_error_window: int = Field(default=10, ge=0)
    buffer_size: int = Field(default=1000, gt=0)
    non_inference_max_parallel_workers: int = Field(default=4, ge=1)
    max_conversation_restarts: int = Field(default=5, ge=0)
    max_conversation_correction_steps: int = Field(default=0, ge=0)
    debug_override_save_all_column_traces: bool = False

    @model_validator(mode="after")
    def normalize_shutdown_settings(self) -> Self:
        """Normalize shutdown settings for compatibility."""
        if self.disable_early_shutdown:
            self.shutdown_error_rate = 1.0
        return self
