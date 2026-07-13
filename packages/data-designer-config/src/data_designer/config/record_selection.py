# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import Field, field_validator

from data_designer.config.base import ConfigBase
from data_designer.config.utils.type_helpers import StrEnum


class RecordSelectionExhaustion(StrEnum):
    """Behavior when the candidate budget is exhausted before reaching the target."""

    RAISE = "raise"
    RETURN_PARTIAL = "return_partial"


class RecordSelectionConfig(ConfigBase):
    """Select records until the requested accepted-row count is reached.

    Attributes:
        predicate_column: Non-blank name of the boolean column used to accept or reject generated records.
        max_candidate_records: Strict positive integer limiting how many candidate records the engine may generate.
        on_exhausted: Behavior when the candidate budget is exhausted before reaching the target.
    """

    predicate_column: str = Field(min_length=1)
    max_candidate_records: int = Field(gt=0, strict=True)
    on_exhausted: RecordSelectionExhaustion = RecordSelectionExhaustion.RAISE

    @field_validator("predicate_column")
    @classmethod
    def _validate_predicate_column_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("predicate_column must contain a non-whitespace character")
        return value
