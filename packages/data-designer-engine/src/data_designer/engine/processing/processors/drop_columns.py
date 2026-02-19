# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from fnmatch import fnmatch
from typing import TYPE_CHECKING

from data_designer.config.processors import DropColumnsProcessorConfig
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.storage.artifact_storage import BatchStage

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class DropColumnsProcessor(Processor[DropColumnsProcessorConfig]):
    """Drops specified columns from the dataset after each batch."""

    def _resolve_columns(self, available: pd.Index) -> list[str]:
        """Expand column_names entries (including glob patterns) against available columns."""
        resolved: dict[str, None] = {}
        for name in self.config.column_names:
            if "*" in name:
                resolved.update({col: None for col in available if fnmatch(col, name)})
            elif name in available:
                resolved[name] = None
            else:
                logger.warning(f"⚠️ Cannot drop column: `{name}` not found in the dataset.")
        return list(resolved)

    def process_after_batch(self, data: pd.DataFrame, *, current_batch_number: int | None) -> pd.DataFrame:
        logger.info(f"🙈 Dropping columns: {self.config.column_names}")
        resolved = self._resolve_columns(data.columns)
        if current_batch_number is not None:
            self._save_dropped_columns(data, resolved, current_batch_number)
        if resolved:
            data.drop(columns=resolved, inplace=True)
        return data

    def _save_dropped_columns(self, data: pd.DataFrame, resolved: list[str], current_batch_number: int) -> None:
        if not resolved:
            return

        logger.debug("📦 Saving dropped columns to dropped-columns directory")
        dropped_column_parquet_file_name = self.artifact_storage.create_batch_file_path(
            batch_number=current_batch_number,
            batch_stage=BatchStage.DROPPED_COLUMNS,
        ).name
        self.artifact_storage.write_parquet_file(
            parquet_file_name=dropped_column_parquet_file_name,
            dataframe=data[resolved],
            batch_stage=BatchStage.DROPPED_COLUMNS,
        )
