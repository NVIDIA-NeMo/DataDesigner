# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from data_designer.config.processors import DropColumnsProcessorConfig
from data_designer.engine.dataset_builders.artifact_storage import BatchStage
from data_designer.engine.processing.processors.base import Processor
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class DropColumnsProcessor(Processor[DropColumnsProcessorConfig]):
    """Drops specified columns from the dataset after each batch."""

    def process_after_batch(self, data: pd.DataFrame, *, batch_number: int) -> pd.DataFrame:
        logger.info(f"🙈 Dropping columns: {self.config.column_names}")
        self._save_dropped_columns(data, batch_number)
        return self._drop_columns(data)

    def _drop_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        for column in self.config.column_names:
            if column in data.columns:
                data.drop(columns=[column], inplace=True)
            else:
                logger.warning(f"⚠️ Cannot drop column: `{column}` not found in the dataset.")
        return data

    def _save_dropped_columns(self, data: pd.DataFrame, batch_number: int) -> None:
        # Only save columns that actually exist
        existing_columns = [col for col in self.config.column_names if col in data.columns]
        if not existing_columns:
            return

        logger.debug("📦 Saving dropped columns to dropped-columns directory")
        dropped_column_parquet_file_name = self.artifact_storage.create_batch_file_path(
            batch_number=batch_number,
            batch_stage=BatchStage.DROPPED_COLUMNS,
        ).name
        self.artifact_storage.write_parquet_file(
            parquet_file_name=dropped_column_parquet_file_name,
            dataframe=data[existing_columns],
            batch_stage=BatchStage.DROPPED_COLUMNS,
        )
