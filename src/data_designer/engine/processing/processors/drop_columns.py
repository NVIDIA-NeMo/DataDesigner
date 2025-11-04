# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pandas as pd

from data_designer.config.processors import DropColumnsProcessorConfig
from data_designer.engine.configurable_task import ConfigurableTaskMetadata
from data_designer.engine.dataset_builders.artifact_storage import BatchStage
from data_designer.engine.processing.processors.base import Processor

logger = logging.getLogger(__name__)


class DropColumnsProcessor(Processor[DropColumnsProcessorConfig]):
    @staticmethod
    def metadata() -> ConfigurableTaskMetadata:
        return ConfigurableTaskMetadata(
            name="drop_columns",
            description="Drop columns from the input dataset.",
            required_resources=None,
        )

    def process(self, data: pd.DataFrame, *, current_batch_number: int | None = None) -> pd.DataFrame:
        logger.info(f"🙈 Dropping columns: {self.config.column_names}")
        if current_batch_number is not None:  # not in preview mode
            self._save_dropped_columns_if_needed(data, current_batch_number)
        for column in self.config.column_names:
            if column in data.columns:
                data.drop(columns=[column], inplace=True)
            else:
                logger.warning(f"⚠️ Cannot drop column: `{column}` not found in the dataset.")
        return data

    def _save_dropped_columns_if_needed(self, data: pd.DataFrame, current_batch_number: int) -> None:
        logger.debug("📦 Saving dropped columns to dropped-columns directory")
        dropped_column_parquet_file_name = self.artifact_storage.create_batch_file_path(
            batch_number=current_batch_number,
            batch_stage=BatchStage.DROPPED_COLUMNS,
        ).name
        self.artifact_storage.write_parquet_file(
            parquet_file_name=dropped_column_parquet_file_name,
            dataframe=data[self.config.column_names],
            batch_stage=BatchStage.DROPPED_COLUMNS,
        )
