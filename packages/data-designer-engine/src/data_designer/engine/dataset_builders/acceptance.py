# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class SelectionBatchMarker(BaseModel):
    """Durable commit record for a completed candidate batch."""

    model_config = ConfigDict(frozen=True, strict=True)

    candidate_batch_id: int = Field(ge=0)
    row_group_id: int = Field(ge=0)
    candidate_start_offset: int = Field(ge=0)
    candidate_records: int = Field(ge=0)
    accepted_records: int = Field(ge=0)
    rejected_records: int = Field(ge=0)
    null_predicate_records: int = Field(ge=0)
    failed_generation_records: int = Field(ge=0)
    trimmed_accepted_records: int = Field(ge=0)
    accepted_partition: str | None
    schema_materialized: bool
    non_retryable_error: str | None
    stopped_early: bool

    @model_validator(mode="after")
    def validate_accounting(self) -> SelectionBatchMarker:
        """Validate persisted candidate accounting and partition consistency."""
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
        return self


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
        self._first_non_retryable_error: str | None = None
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
    def first_non_retryable_error(self) -> str | None:
        return self._first_non_retryable_error

    @property
    def last_batch_stopped_early(self) -> bool:
        return bool(self._markers and self._markers[-1].stopped_early)

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
        non_retryable_error: str | None = None,
        stopped_early: bool = False,
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
            non_retryable_error=non_retryable_error,
            stopped_early=stopped_early,
        )
        self._markers.append(marker)
        self._accumulate_marker(marker)
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
        if self._first_non_retryable_error is None and marker.non_retryable_error is not None:
            self._first_non_retryable_error = marker.non_retryable_error
