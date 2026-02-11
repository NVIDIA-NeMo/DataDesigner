# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import shutil
from enum import Enum
from typing import TYPE_CHECKING

from data_designer.engine.dataset_builders.artifact_storage import BatchStage
from data_designer.engine.dataset_builders.errors import DatasetProcessingError

if TYPE_CHECKING:
    import pandas as pd

    from data_designer.engine.dataset_builders.artifact_storage import ArtifactStorage
    from data_designer.engine.dataset_builders.utils.dataset_batch_manager import DatasetBatchManager
    from data_designer.engine.processing.processors.base import Processor

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

    def has_processors_for(self, stage: ProcessorStage) -> bool:
        """Check if any processor implements the given stage."""
        return any(p.implements(stage.value) for p in self._processors)

    def _run_stage(self, df: pd.DataFrame, stage: ProcessorStage, **kwargs) -> pd.DataFrame:
        """Run a processor callback on all processors that implement it."""
        for processor in self._processors:
            if not processor.implements(stage.value):
                continue
            try:
                df = getattr(processor, stage.value)(df, **kwargs)
            except Exception as e:
                raise DatasetProcessingError(f"🛑 Failed in {stage.value} for {processor.name}: {e}") from e
        return df

    def run_pre_batch(self, batch_manager: DatasetBatchManager) -> None:
        """Run process_before_batch() on current batch."""
        if not self.has_processors_for(ProcessorStage.PRE_BATCH):
            return

        df = batch_manager.get_current_batch(as_dataframe=True)
        original_len = len(df)
        df = self._run_stage(df, ProcessorStage.PRE_BATCH)
        if len(df) != original_len:
            delta = len(df) - original_len
            logger.info(f"ℹ️ PRE_BATCH processors changed the record count by {delta:+d} records.")
        batch_manager.replace_records(df.to_dict(orient="records"))

    def run_post_batch(self, df: pd.DataFrame, current_batch_number: int | None) -> pd.DataFrame:
        """Run process_after_batch() on processors that implement it."""
        return self._run_stage(df, ProcessorStage.POST_BATCH, current_batch_number=current_batch_number)

    def run_after_generation_on_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run process_after_generation() on a DataFrame (for preview mode)."""
        return self._run_stage(df, ProcessorStage.AFTER_GENERATION)

    def run_after_generation(self, batch_size: int) -> None:
        """Load final dataset, run process_after_generation(), rewrite in chunks.

        Re-chunks the processed dataset using the given batch_size so that output
        files stay consistently sized regardless of how many rows the processor
        adds or removes.
        """
        if not self.has_processors_for(ProcessorStage.AFTER_GENERATION):
            return

        logger.info("⏳ Running process_after_generation on final dataset...")
        df = self._artifact_storage.load_dataset()
        df = self._run_stage(df, ProcessorStage.AFTER_GENERATION)

        shutil.rmtree(self._artifact_storage.final_dataset_path)
        for i in range(0, max(len(df), 1), batch_size):
            self._artifact_storage.write_batch_to_parquet_file(
                batch_number=i // batch_size,
                dataframe=df.iloc[i : i + batch_size],
                batch_stage=BatchStage.FINAL_RESULT,
            )
        logger.info(f"✅ process_after_generation complete. Final dataset has {len(df)} rows.")
