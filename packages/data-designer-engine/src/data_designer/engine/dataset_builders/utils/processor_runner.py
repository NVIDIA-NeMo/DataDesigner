# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import shutil
from enum import Enum
from typing import TYPE_CHECKING

from data_designer.engine.dataset_builders.errors import DatasetProcessingError
from data_designer.engine.storage.artifact_storage import BatchStage

if TYPE_CHECKING:
    import pandas as pd

    from data_designer.engine.processing.processors.base import Processor
    from data_designer.engine.storage.artifact_storage import ArtifactStorage

logger = logging.getLogger(__name__)


class ProcessorStage(str, Enum):
    """Valid processor callback stages."""

    PRE_BATCH = "process_before_batch"
    POST_BATCH = "process_after_batch"
    AFTER_GENERATION = "process_after_generation"


class ProcessorRunner:
    """Runs processor callbacks at various stages of dataset generation."""

    def __init__(
        self,
        processors: list[Processor],
        artifact_storage: ArtifactStorage,
    ):
        self._processors = processors
        self._artifact_storage = artifact_storage

    @property
    def processors(self) -> tuple[Processor, ...]:
        return tuple(self._processors)

    def has_processors_for(self, stage: ProcessorStage) -> bool:
        """Check if any processor implements the given stage."""
        return any(p.implements(stage.value) for p in self._processors)

    def _run_stage(self, df: pd.DataFrame, stage: ProcessorStage, **kwargs) -> pd.DataFrame:
        """Run a processor callback on all processors that implement it."""
        original_len = len(df)
        for processor in self._processors:
            if not processor.implements(stage.value):
                continue
            try:
                df = getattr(processor, stage.value)(df, **kwargs)
            except Exception as e:
                raise DatasetProcessingError(f"🛑 Failed in {stage.value} for {processor.name}: {e}") from e
        if len(df) != original_len:
            if stage == ProcessorStage.PRE_BATCH:
                self._raise_if_pre_batch_resized(original_len, len(df))
            delta = len(df) - original_len
            logger.info(f"ℹ️ {stage.name} processors changed the record count by {delta:+d} records.")
        return df

    @staticmethod
    def _raise_if_pre_batch_resized(original_len: int, new_len: int) -> None:
        if new_len != original_len:
            raise DatasetProcessingError(
                f"Pre-batch processor changed row count from {original_len} to {new_len}. "
                "Row-count changes in pre-batch processors are not supported; use workflow chaining instead."
            )

    def run_pre_batch_on_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run PRE_BATCH processors on a DataFrame and return the result.

        Args:
            df: Input DataFrame.
        """
        return self._run_stage(df, ProcessorStage.PRE_BATCH)

    def run_post_batch(
        self, df: pd.DataFrame, current_batch_number: int | None, *, strict_row_count: bool = False
    ) -> pd.DataFrame:
        """Run process_after_batch() on processors that implement it.

        Args:
            df: Input DataFrame.
            current_batch_number: Batch index passed to processors.
            strict_row_count: If True, raise ``DatasetProcessingError`` when a
                processor changes the row count.
        """
        original_len = len(df)
        df = self._run_stage(df, ProcessorStage.POST_BATCH, current_batch_number=current_batch_number)
        if strict_row_count and len(df) != original_len:
            raise DatasetProcessingError(
                f"Post-batch processor changed row count from {original_len} to {len(df)}. "
                "Row-count changes in post-batch processors are not supported; use workflow chaining instead."
            )
        return df

    def run_after_generation_on_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run process_after_generation() on a DataFrame (for preview mode)."""
        return self._run_stage(df, ProcessorStage.AFTER_GENERATION)

    def run_after_generation(self, batch_size: int, *, selection_publication: bool = False) -> None:
        """Load final dataset, run process_after_generation(), rewrite in chunks.

        Re-chunks the processed dataset using the given batch_size so that output
        files stay consistently sized regardless of how many rows the processor
        adds or removes. Record-selection publications opt into names whose width
        depends on the final partition count rather than the candidate-batch budget.
        """
        if not self.has_processors_for(ProcessorStage.AFTER_GENERATION):
            return

        logger.info("⏳ Running process_after_generation on final dataset...")
        df = self._artifact_storage.load_dataset()
        df = self._run_stage(df, ProcessorStage.AFTER_GENERATION)

        shutil.rmtree(self._artifact_storage.final_dataset_path)
        num_partitions = len(range(0, max(len(df), 1), batch_size))
        for i in range(0, max(len(df), 1), batch_size):
            batch_number = i // batch_size
            batch = df.iloc[i : i + batch_size]
            if selection_publication:
                self._artifact_storage.write_selection_publication_batch(
                    batch_number,
                    batch,
                    num_partitions=num_partitions,
                )
            else:
                self._artifact_storage.write_batch_to_parquet_file(
                    batch_number=batch_number,
                    dataframe=batch,
                    batch_stage=BatchStage.FINAL_RESULT,
                )
        logger.info(f"✅ process_after_generation complete. Final dataset has {len(df)} rows.")
