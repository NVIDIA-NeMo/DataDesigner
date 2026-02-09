# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import shutil
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from data_designer.engine.dataset_builders.artifact_storage import BatchStage
from data_designer.engine.dataset_builders.errors import DatasetProcessingError

if TYPE_CHECKING:
    import pandas as pd

    from data_designer.engine.dataset_builders.artifact_storage import ArtifactStorage
    from data_designer.engine.dataset_builders.utils.dataset_batch_manager import DatasetBatchManager
    from data_designer.engine.processing.processors.base import Processor
    from data_designer.engine.resources.resource_provider import ResourceProvider

logger = logging.getLogger(__name__)


class ProcessorStage(str, Enum):
    """Valid processor callback stages."""

    PREPROCESS = "preprocess"
    PRE_BATCH = "process_before_batch"
    POST_BATCH = "process_after_batch"
    POSTPROCESS = "postprocess"


class ProcessorRunner:
    """Runs processor callbacks at various stages of dataset generation."""

    def __init__(
        self,
        processors: list[Processor],
        resource_provider: ResourceProvider,
        artifact_storage: ArtifactStorage,
    ):
        self._processors = processors
        self._resource_provider = resource_provider
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

    def run_preprocess(self) -> None:
        """Load seed data, run preprocess(), save preprocessed seed."""
        if not self.has_processors_for(ProcessorStage.PREPROCESS):
            return
        if self._resource_provider.seed_reader is None:
            return

        logger.info("⏳ Running preprocess on seed data...")
        seed_reader = self._resource_provider.seed_reader
        conn = seed_reader.create_duckdb_connection()
        df = conn.execute(f"SELECT * FROM '{seed_reader.get_dataset_uri()}'").fetchdf()
        original_len = len(df)

        df = self._run_stage(df, ProcessorStage.PREPROCESS)

        preprocessed_path = self._artifact_storage.base_dataset_path / "preprocessed_seed.parquet"
        self._artifact_storage.mkdir_if_needed(self._artifact_storage.base_dataset_path)
        df.to_parquet(preprocessed_path, index=False)
        self._resource_provider.preprocessed_seed_uri = str(preprocessed_path)
        logger.info(f"✅ Preprocess complete. Seed data has {len(df)} rows (was {original_len}).")

    def cleanup_preprocessed_seed(self) -> None:
        """Remove preprocessed seed file and reset URI."""
        if self._resource_provider.preprocessed_seed_uri is not None:
            preprocessed_path = Path(self._resource_provider.preprocessed_seed_uri)
            if preprocessed_path.exists():
                preprocessed_path.unlink()
            self._resource_provider.preprocessed_seed_uri = None

    def run_pre_batch(self, batch_manager: DatasetBatchManager) -> None:
        """Run process_before_batch() on current batch."""
        if not self.has_processors_for(ProcessorStage.PRE_BATCH):
            return

        df = batch_manager.get_current_batch(as_dataframe=True)
        original_len = len(df)
        df = self._run_stage(df, ProcessorStage.PRE_BATCH)
        if len(df) != original_len:
            logger.warning(
                f"⚠️ PRE_BATCH processors changed row count from {original_len} to {len(df)}. "
                "This may cause unexpected behavior in downstream generators."
            )
        batch_manager.update_records(df.to_dict(orient="records"))

    def run_post_batch(self, df: pd.DataFrame, current_batch_number: int | None) -> pd.DataFrame:
        """Run process_after_batch() on processors that implement it."""
        return self._run_stage(df, ProcessorStage.POST_BATCH, current_batch_number=current_batch_number)

    def run_postprocess_on_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run postprocess() on a DataFrame (for preview mode)."""
        return self._run_stage(df, ProcessorStage.POSTPROCESS)

    def run_postprocess(self) -> None:
        """Load final dataset, run postprocess(), rewrite dataset."""
        if not self.has_processors_for(ProcessorStage.POSTPROCESS):
            return

        logger.info("⏳ Running postprocess on final dataset...")
        df = self._artifact_storage.load_dataset()
        df = self._run_stage(df, ProcessorStage.POSTPROCESS)

        if self._artifact_storage.final_dataset_path.exists():
            shutil.rmtree(self._artifact_storage.final_dataset_path)
        self._artifact_storage.write_batch_to_parquet_file(
            batch_number=0,
            dataframe=df,
            batch_stage=BatchStage.FINAL_RESULT,
        )
        logger.info(f"✅ Postprocess complete. Final dataset has {len(df)} rows.")
