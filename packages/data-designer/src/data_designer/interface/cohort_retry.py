# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

from data_designer.config.utils.type_helpers import StrEnum
from data_designer.interface.errors import DataDesignerWorkflowError

_StrEnumT = TypeVar("_StrEnumT", bound=StrEnum)


class SamplerRetryMode(StrEnum):
    """Control whether sampler values remain fixed across cohort attempts."""

    PRESERVE = "preserve"
    RESAMPLE = "resample"


class RetryExhaustion(StrEnum):
    """Control how cohort retry reports an exhausted retry bound."""

    RAISE = "raise"
    RETURN_PARTIAL = "return_partial"


@dataclass(frozen=True)
class RetryUntil:
    """Bounded workflow-stage policy for retrying rejected cohort rows."""

    predicate_column: str
    max_attempts: int | None = None
    max_candidate_records: int | None = None
    sampler_retry_mode: SamplerRetryMode | str = SamplerRetryMode.PRESERVE
    on_exhausted: RetryExhaustion | str = RetryExhaustion.RAISE

    def __post_init__(self) -> None:
        if not isinstance(self.predicate_column, str) or not self.predicate_column.strip():
            raise DataDesignerWorkflowError("retry_until.predicate_column must be a non-empty string.")
        _validate_optional_positive_int(self.max_attempts, "max_attempts")
        _validate_optional_positive_int(self.max_candidate_records, "max_candidate_records")
        if self.max_attempts is None and self.max_candidate_records is None:
            raise DataDesignerWorkflowError(
                "retry_until requires at least one bound: max_attempts or max_candidate_records."
            )

        object.__setattr__(
            self,
            "sampler_retry_mode",
            _coerce_enum(self.sampler_retry_mode, SamplerRetryMode, "sampler_retry_mode"),
        )
        object.__setattr__(
            self,
            "on_exhausted",
            _coerce_enum(self.on_exhausted, RetryExhaustion, "on_exhausted"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-compatible payload used by metadata and fingerprints."""
        return {
            "predicate_column": self.predicate_column,
            "max_attempts": self.max_attempts,
            "max_candidate_records": self.max_candidate_records,
            "sampler_retry_mode": self.sampler_retry_mode.value,
            "on_exhausted": self.on_exhausted.value,
        }


def _validate_optional_positive_int(value: int | None, field_name: str) -> None:
    if value is None:
        return
    if type(value) is not int or value <= 0:
        raise DataDesignerWorkflowError(f"retry_until.{field_name} must be a strict positive integer.")


def _coerce_enum(value: _StrEnumT | str, enum_type: type[_StrEnumT], field_name: str) -> _StrEnumT:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(repr(item.value) for item in enum_type)
        raise DataDesignerWorkflowError(f"retry_until.{field_name} must be one of: {choices}.") from exc
