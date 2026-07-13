# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.record_selection import RecordSelectionConfig, RecordSelectionExhaustion
from data_designer.engine.dataset_builders.errors import DatasetGenerationError

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True, slots=True)
class CandidateBatch:
    """An immutable unit of candidate generation work."""

    candidate_batch_id: int
    row_group_id: int
    start_offset: int
    size: int


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    """Selection result for one completed candidate batch."""

    accepted_indices: tuple[int, ...]
    candidate_records: int
    accepted_records: int
    rejected_records: int
    null_predicate_records: int
    failed_generation_records: int
    trimmed_accepted_records: int

    def __post_init__(self) -> None:
        accounted = (
            self.accepted_records
            + self.rejected_records
            + self.null_predicate_records
            + self.failed_generation_records
            + self.trimmed_accepted_records
        )
        if accounted != self.candidate_records:
            raise ValueError(f"Selection decision accounts for {accounted} records, expected {self.candidate_records}.")


@dataclass(frozen=True, slots=True)
class SelectionBatchMarker:
    """Durable commit record for a completed candidate batch."""

    candidate_batch_id: int
    row_group_id: int
    candidate_start_offset: int
    candidate_records: int
    accepted_records: int
    rejected_records: int
    null_predicate_records: int
    failed_generation_records: int
    trimmed_accepted_records: int
    accepted_partition: str | None
    schema_materialized: bool = False
    non_retryable_error_type: str | None = None
    non_retryable_error_message: str | None = None
    terminal_error_kind: str | None = None
    terminal_error_message: str | None = None

    def __post_init__(self) -> None:
        nonnegative_fields = (
            self.candidate_batch_id,
            self.row_group_id,
            self.candidate_start_offset,
            self.candidate_records,
            self.accepted_records,
            self.rejected_records,
            self.null_predicate_records,
            self.failed_generation_records,
            self.trimmed_accepted_records,
        )
        if any(value < 0 for value in nonnegative_fields):
            raise ValueError("Selection marker counters must be non-negative.")
        accounted = (
            self.accepted_records
            + self.rejected_records
            + self.null_predicate_records
            + self.failed_generation_records
            + self.trimmed_accepted_records
        )
        if accounted != self.candidate_records:
            raise ValueError(f"Selection marker accounts for {accounted} records, expected {self.candidate_records}.")
        if (self.accepted_partition is None) != (self.accepted_records == 0):
            raise ValueError("accepted_partition must be present exactly when accepted_records is non-zero.")
        if (self.non_retryable_error_type is None) != (self.non_retryable_error_message is None):
            raise ValueError("non_retryable_error_type and non_retryable_error_message must be present together.")
        if (self.terminal_error_kind is None) != (self.terminal_error_message is None):
            raise ValueError("terminal_error_kind and terminal_error_message must be present together.")

    def to_dict(self) -> dict[str, int | bool | str | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SelectionBatchMarker:
        try:
            return cls(
                candidate_batch_id=_strict_int(value, "candidate_batch_id"),
                row_group_id=_strict_int(value, "row_group_id"),
                candidate_start_offset=_strict_int(value, "candidate_start_offset"),
                candidate_records=_strict_int(value, "candidate_records"),
                accepted_records=_strict_int(value, "accepted_records"),
                rejected_records=_strict_int(value, "rejected_records"),
                null_predicate_records=_strict_int(value, "null_predicate_records"),
                failed_generation_records=_strict_int(value, "failed_generation_records"),
                trimmed_accepted_records=_strict_int(value, "trimmed_accepted_records"),
                accepted_partition=_optional_string(value, "accepted_partition"),
                schema_materialized=_optional_bool(value, "schema_materialized", required=False),
                non_retryable_error_type=_optional_string(value, "non_retryable_error_type", required=False),
                non_retryable_error_message=_optional_string(value, "non_retryable_error_message", required=False),
                terminal_error_kind=_optional_string(value, "terminal_error_kind", required=False),
                terminal_error_message=_optional_string(value, "terminal_error_message", required=False),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid selection batch marker: {exc}") from exc


class AcceptanceController:
    """Track candidate attempts and accepted output records for a selection run."""

    def __init__(
        self,
        *,
        config: RecordSelectionConfig,
        target_records: int,
        buffer_size: int,
        markers: tuple[SelectionBatchMarker, ...] = (),
    ) -> None:
        if target_records <= 0:
            raise ValueError("target_records must be positive.")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")
        if config.max_candidate_records < target_records:
            raise DatasetGenerationError(
                "🛑 record_selection.max_candidate_records must be greater than or equal to "
                f"num_records ({config.max_candidate_records} < {target_records})."
            )
        self.config = config
        self.target_records = target_records
        self.buffer_size = buffer_size
        self._markers = list(markers)
        self._candidate_records = 0
        self._accepted_records = 0
        self._rejected_records = 0
        self._null_predicate_records = 0
        self._failed_generation_records = 0
        self._trimmed_accepted_records = 0
        self._accepted_partitions = 0
        self._first_non_retryable_error: dict[str, str] | None = None
        self._validate_marker_sequence()

    @property
    def markers(self) -> tuple[SelectionBatchMarker, ...]:
        return tuple(self._markers)

    @property
    def candidate_records(self) -> int:
        return self._candidate_records

    @property
    def accepted_records(self) -> int:
        return self._accepted_records

    @property
    def rejected_records(self) -> int:
        return self._rejected_records

    @property
    def null_predicate_records(self) -> int:
        return self._null_predicate_records

    @property
    def failed_generation_records(self) -> int:
        return self._failed_generation_records

    @property
    def trimmed_accepted_records(self) -> int:
        return self._trimmed_accepted_records

    @property
    def accepted_partitions(self) -> int:
        return self._accepted_partitions

    @property
    def first_non_retryable_error(self) -> dict[str, str] | None:
        if self._first_non_retryable_error is None:
            return None
        return dict(self._first_non_retryable_error)

    @property
    def terminal_error(self) -> dict[str, str] | None:
        if not self._markers:
            return None
        marker = self._markers[-1]
        kind = marker.terminal_error_kind
        message = marker.terminal_error_message
        if kind is None:
            return None
        if message is None:
            raise RuntimeError("Selection marker terminal-error fields are inconsistent.")
        return {"kind": kind, "message": message}

    @property
    def candidate_batches_completed(self) -> int:
        return len(self._markers)

    @property
    def has_reached_target(self) -> bool:
        return self.accepted_records >= self.target_records

    @property
    def has_candidate_budget(self) -> bool:
        return self.candidate_records < self.config.max_candidate_records

    @property
    def is_exhausted(self) -> bool:
        return not self.has_reached_target and not self.has_candidate_budget

    def next_candidate_batch(self) -> CandidateBatch:
        if self.has_reached_target:
            raise RuntimeError("The accepted-record target has already been reached.")
        remaining_budget = self.config.max_candidate_records - self.candidate_records
        if remaining_budget <= 0:
            raise RuntimeError("The candidate budget has already been exhausted.")
        batch_id = len(self._markers)
        return CandidateBatch(
            candidate_batch_id=batch_id,
            row_group_id=batch_id,
            start_offset=self.candidate_records,
            size=min(self.buffer_size, self.target_records, remaining_budget),
        )

    def select(
        self,
        dataframe: pd.DataFrame,
        *,
        batch: CandidateBatch,
        failed_generation_records: int,
    ) -> SelectionDecision:
        predicate_column = self.config.predicate_column
        if failed_generation_records < 0 or failed_generation_records + len(dataframe) != batch.size:
            raise DatasetGenerationError(
                f"🛑 Candidate batch {batch.candidate_batch_id} has inconsistent generated-row accounting."
            )
        if len(dataframe) == 0 and failed_generation_records == batch.size:
            return SelectionDecision(
                accepted_indices=(),
                candidate_records=batch.size,
                accepted_records=0,
                rejected_records=0,
                null_predicate_records=0,
                failed_generation_records=failed_generation_records,
                trimmed_accepted_records=0,
            )
        if predicate_column not in dataframe.columns:
            raise DatasetGenerationError(
                f"🛑 Record-selection predicate column {predicate_column!r} was not materialized for "
                f"candidate batch {batch.candidate_batch_id}."
            )

        true_indices: list[int] = []
        rejected_records = 0
        null_predicate_records = 0
        invalid: list[str] = []
        for index, value in enumerate(dataframe[predicate_column].tolist()):
            if isinstance(value, (bool, lazy.np.bool_)):
                if value:
                    true_indices.append(index)
                else:
                    rejected_records += 1
                continue
            is_null = lazy.pd.isna(value)
            if isinstance(is_null, (bool, lazy.np.bool_)) and is_null:
                null_predicate_records += 1
                continue
            if len(invalid) < 5:
                invalid.append(f"{value!r} ({type(value).__name__})")

        if invalid:
            samples = ", ".join(invalid)
            raise DatasetGenerationError(
                f"🛑 Record-selection predicate column {predicate_column!r} contains non-boolean values in "
                f"candidate batch {batch.candidate_batch_id}: {samples}. Expected only boolean or null values."
            )

        remaining = self.target_records - self.accepted_records
        accepted_indices = tuple(true_indices[:remaining])
        trimmed = len(true_indices) - len(accepted_indices)
        return SelectionDecision(
            accepted_indices=accepted_indices,
            candidate_records=batch.size,
            accepted_records=len(accepted_indices),
            rejected_records=rejected_records,
            null_predicate_records=null_predicate_records,
            failed_generation_records=failed_generation_records,
            trimmed_accepted_records=trimmed,
        )

    def record_checkpoint(
        self,
        *,
        batch: CandidateBatch,
        decision: SelectionDecision,
        accepted_partition: str | None,
        schema_materialized: bool = False,
        non_retryable_error: dict[str, str] | None = None,
        terminal_error: dict[str, str] | None = None,
    ) -> SelectionBatchMarker:
        expected = self.next_candidate_batch()
        if batch != expected:
            raise ValueError(f"Cannot checkpoint candidate batch {batch}; expected {expected}.")
        marker = SelectionBatchMarker(
            candidate_batch_id=batch.candidate_batch_id,
            row_group_id=batch.row_group_id,
            candidate_start_offset=batch.start_offset,
            candidate_records=decision.candidate_records,
            accepted_records=decision.accepted_records,
            rejected_records=decision.rejected_records,
            null_predicate_records=decision.null_predicate_records,
            failed_generation_records=decision.failed_generation_records,
            trimmed_accepted_records=decision.trimmed_accepted_records,
            accepted_partition=accepted_partition,
            schema_materialized=schema_materialized,
            non_retryable_error_type=(non_retryable_error["type"] if non_retryable_error is not None else None),
            non_retryable_error_message=(non_retryable_error["message"] if non_retryable_error is not None else None),
            terminal_error_kind=terminal_error["kind"] if terminal_error is not None else None,
            terminal_error_message=terminal_error["message"] if terminal_error is not None else None,
        )
        self._markers.append(marker)
        self._accumulate_marker(marker)
        return marker

    def replace_last_marker_terminal_error(self, terminal_error: dict[str, str]) -> SelectionBatchMarker:
        if not self._markers:
            raise RuntimeError("Cannot attach a terminal error without a completed candidate batch.")
        marker = replace(
            self._markers[-1],
            terminal_error_kind=terminal_error["kind"],
            terminal_error_message=terminal_error["message"],
        )
        self._markers[-1] = marker
        return marker

    def summary(self) -> dict[str, int | float | bool | str]:
        candidate_records = self.candidate_records
        return {
            "predicate_column": self.config.predicate_column,
            "max_candidate_records": self.config.max_candidate_records,
            "on_exhausted": RecordSelectionExhaustion(self.config.on_exhausted).value,
            "run_buffer_size": self.buffer_size,
            "candidate_records_generated": candidate_records,
            "candidate_batches_completed": self.candidate_batches_completed,
            "accepted_records": self.accepted_records,
            "rejected_records": self.rejected_records,
            "null_predicate_records": self.null_predicate_records,
            "failed_generation_records": self.failed_generation_records,
            "trimmed_accepted_records": self.trimmed_accepted_records,
            "acceptance_rate": self.accepted_records / candidate_records if candidate_records else 0.0,
            "selection_satisfied": self.has_reached_target,
            "selection_exhausted": self.is_exhausted,
            "next_candidate_batch_id": self.candidate_batches_completed,
            "next_candidate_offset": candidate_records,
        }

    def _validate_marker_sequence(self) -> None:
        expected_offset = 0
        for expected_id, marker in enumerate(self._markers):
            if marker.candidate_batch_id != expected_id or marker.row_group_id != expected_id:
                raise ValueError("Selection batch markers must have contiguous batch and row-group IDs.")
            if marker.candidate_start_offset != expected_offset:
                raise ValueError("Selection batch marker candidate offsets must be contiguous.")
            remaining_budget = self.config.max_candidate_records - expected_offset
            if remaining_budget <= 0:
                raise ValueError("Selection batch markers continue after the candidate budget was exhausted.")
            expected_size = min(self.buffer_size, self.target_records, remaining_budget)
            if marker.candidate_records != expected_size:
                raise ValueError(
                    f"Selection batch {expected_id} has size {marker.candidate_records}, expected {expected_size}."
                )
            self._accumulate_marker(marker)
            expected_offset = self.candidate_records
            if self.accepted_records > self.target_records:
                raise ValueError("Selection batch markers exceed the accepted-record target.")
            if self.accepted_records == self.target_records and expected_id != len(self._markers) - 1:
                raise ValueError("Selection batch markers continue after the accepted-record target was reached.")

    def _accumulate_marker(self, marker: SelectionBatchMarker) -> None:
        self._candidate_records += marker.candidate_records
        self._accepted_records += marker.accepted_records
        self._rejected_records += marker.rejected_records
        self._null_predicate_records += marker.null_predicate_records
        self._failed_generation_records += marker.failed_generation_records
        self._trimmed_accepted_records += marker.trimmed_accepted_records
        self._accepted_partitions += marker.accepted_partition is not None
        if self._first_non_retryable_error is None and marker.non_retryable_error_type is not None:
            message = marker.non_retryable_error_message
            if message is None:
                raise RuntimeError("Selection marker non-retryable-error fields are inconsistent.")
            self._first_non_retryable_error = {
                "type": marker.non_retryable_error_type,
                "message": message,
            }


def _strict_int(value: dict[str, Any], key: str) -> int:
    field = value[key]
    if isinstance(field, bool) or not isinstance(field, int):
        raise TypeError(f"{key} must be an integer")
    return field


def _optional_string(value: dict[str, Any], key: str, *, required: bool = True) -> str | None:
    field = value[key] if required else value.get(key)
    if field is not None and not isinstance(field, str):
        raise TypeError(f"{key} must be a string or null")
    return field


def _optional_bool(value: dict[str, Any], key: str, *, required: bool = True) -> bool:
    field = value[key] if required else value.get(key, False)
    if not isinstance(field, bool):
        raise TypeError(f"{key} must be a boolean")
    return field
