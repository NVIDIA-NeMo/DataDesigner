# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from data_designer.config.utils.type_helpers import StrEnum


class SamplerRetryMode(StrEnum):
    """Control whether sampler values remain fixed across record attempts."""

    PRESERVE = "preserve"
    RESAMPLE = "resample"


class RetryExhaustion(StrEnum):
    """Control how record retry reports an exhausted retry bound."""

    RAISE = "raise"
    RETURN_PARTIAL = "return_partial"


class RetryUntil(BaseModel):
    """Bounded workflow-stage policy for retrying rejected logical records."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    predicate_column: str = Field(strict=True, min_length=1, pattern=r".*\S.*")
    max_attempts: int | None = Field(default=None, strict=True, gt=0)
    max_candidate_records: int | None = Field(default=None, strict=True, gt=0)
    sampler_retry_mode: SamplerRetryMode = SamplerRetryMode.PRESERVE
    on_exhausted: RetryExhaustion = RetryExhaustion.RAISE

    @model_validator(mode="after")
    def validate_bounds(self) -> RetryUntil:
        """Require at least one finite retry bound."""
        if self.max_attempts is None and self.max_candidate_records is None:
            raise ValueError("retry_until requires at least one bound: max_attempts or max_candidate_records")
        return self
