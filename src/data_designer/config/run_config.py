# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field, model_validator
from typing_extensions import Self

from data_designer.config.base import ConfigBase


class RunConfig(ConfigBase):
    """Runtime configuration for dataset generation.

    Groups configuration options that control generation behavior but aren't
    part of the dataset configuration itself.

    Attributes:
        disable_early_shutdown: If True, disables early shutdown entirely. Generation
            will continue regardless of error rate. Default is False.
        shutdown_error_rate: Error rate threshold (0.0-1.0) that triggers early shutdown.
            When early shutdown is disabled, this value is normalized to 1.0. Default is 0.5.
        shutdown_error_window: Minimum number of completed tasks before error rate
            monitoring begins. Must be >= 0. Default is 10.
    """

    disable_early_shutdown: bool = False
    shutdown_error_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    shutdown_error_window: int = Field(default=10, ge=0)

    @model_validator(mode="after")
    def normalize_shutdown_settings(self) -> Self:
        """Set shutdown_error_rate to 1.0 when early shutdown is disabled."""
        if self.disable_early_shutdown:
            self.shutdown_error_rate = 1.0
        return self
