# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING

from data_designer.config.seed import IndexRange, PartitionBlock, SamplingStrategy
from data_designer.engine.column_generators.generators.base import FromScratchColumnGenerator, GenerationStrategy
from data_designer.engine.column_generators.utils.errors import SeedDatasetError
from data_designer.engine.dataset_builders.multi_column_configs import SeedDatasetMultiColumnConfig
from data_designer.engine.processing.utils import concat_datasets
from data_designer.lazy_heavy_imports import duckdb, pd

if TYPE_CHECKING:
    import duckdb
    import pandas as pd

MAX_ZERO_RECORD_RESPONSE_FACTOR = 2

logger = logging.getLogger(__name__)


class SeedDatasetColumnGenerator(FromScratchColumnGenerator[SeedDatasetMultiColumnConfig]):
    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.FULL_COLUMN

    @property
    def num_records_sampled(self) -> int:
        return self._num_records_sampled

    @functools.cached_property
    def duckdb_conn(self) -> duckdb.DuckDBPyConnection:
        return self.resource_provider.seed_reader.create_duckdb_connection()

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        return concat_datasets([self.generate_from_scratch(len(data)), data])

    def generate_from_scratch(self, num_records: int) -> pd.DataFrame:
        if num_records <= 0:
            raise ValueError("🛑 `num_records` must be positive.")

        if self._batch_reader is None:
            self._reset_batch_reader(num_records)

        return self._sample_records(num_records)

    def _initialize(self) -> None:
        self._num_records_sampled = 0
        self._batch_reader = None
        self._df_remaining = None
        self._dataset_uri = self.resource_provider.seed_reader.get_dataset_uri()
        self._seed_dataset_size = self.duckdb_conn.execute(f"SELECT COUNT(*) FROM '{self._dataset_uri}'").fetchone()[0]
        self._index_range = self._resolve_index_range()

    def _validate_selection_strategy(self) -> None:
        err_msg = None
        if self.config.selection_strategy is not None:
            if (
                isinstance(self.config.selection_strategy, IndexRange)
                and self.config.selection_strategy.end >= self._seed_dataset_size
            ):
                err_msg = f"Selection strategy 'end' index {self.config.selection_strategy.end} is out of bounds for dataset size {self._seed_dataset_size}"
            elif (
                isinstance(self.config.selection_strategy, PartitionBlock)
                and self.config.selection_strategy.num_partitions > self._seed_dataset_size
            ):
                err_msg = f"Selection strategy 'num_partitions' {self.config.selection_strategy.num_partitions} is out of bounds for dataset size {self._seed_dataset_size}"
            if err_msg is not None:
                raise SeedDatasetError(err_msg)

    def _resolve_index_range(self) -> IndexRange | None:
        self._validate_selection_strategy()
        index_range = None
        if self.config.selection_strategy is not None:
            if isinstance(self.config.selection_strategy, IndexRange):
                index_range = self.config.selection_strategy
            elif isinstance(self.config.selection_strategy, PartitionBlock):
                index_range = self.config.selection_strategy.to_index_range(self._seed_dataset_size)
        return index_range

    def _reset_batch_reader(self, num_records: int) -> None:
        shuffle = self.config.sampling_strategy == SamplingStrategy.SHUFFLE
        shuffle_query = " ORDER BY RANDOM()" if shuffle else ""

        if self._index_range is not None:
            # Use LIMIT and OFFSET for efficient index range filtering
            # IndexRange uses 0-based indexing [start, end] inclusive
            # OFFSET skips the first 'start' rows (0-based)
            # LIMIT takes 'end - start + 1' rows to include both start and end (inclusive)
            offset_value = self._index_range.start
            limit_value = self._index_range.end - self._index_range.start + 1
            read_query = f"""
                SELECT * FROM '{self._dataset_uri}'
                LIMIT {limit_value} OFFSET {offset_value}
            """

            read_query = f"SELECT * FROM ({read_query}){shuffle_query}"
        else:
            read_query = f"SELECT * FROM '{self._dataset_uri}'{shuffle_query}"
        self._batch_reader = self.duckdb_conn.query(read_query).record_batch(batch_size=num_records)

    def _sample_records(self, num_records: int) -> pd.DataFrame:
        logger.info(f"🌱 Sampling {num_records} records from seed dataset")
        logger.info(f"  |-- seed dataset size: {self._seed_dataset_size} records")
        logger.info(f"  |-- sampling strategy: {self.config.sampling_strategy}")
        if self._index_range is not None:
            if isinstance(self.config.selection_strategy, IndexRange):
                logger.info(f"  |-- selection: rows [{self._index_range.start} to {self._index_range.end}] inclusive")
            else:
                logger.info(
                    f"  |-- selection: partition {self.config.selection_strategy.index + 1} of {self.config.selection_strategy.num_partitions}"
                )
            logger.info(f"  |-- seed dataset size after selection: {self._index_range.size} records")
        df_batch = pd.DataFrame()
        df_sample = pd.DataFrame() if self._df_remaining is None else self._df_remaining
        num_zero_record_responses = 0

        while len(df_sample) < num_records:
            try:
                df_batch = self._batch_reader.read_next_batch().to_pandas()
                df_sample = pd.concat([df_sample, df_batch], ignore_index=True)
            except StopIteration:
                self._reset_batch_reader(num_records)

            if len(df_batch) == 0:
                num_zero_record_responses += 1
                if num_zero_record_responses > MAX_ZERO_RECORD_RESPONSE_FACTOR * num_records:
                    raise RuntimeError(
                        "🛑 Something went wrong while reading from the datastore. "
                        "Please check your connection and try again. "
                        "If the issue persists, please contact support."
                    )

        self._df_remaining = None
        if len(df_sample) > num_records:
            self._df_remaining = df_sample.iloc[num_records:].reset_index(drop=True)
        df_sample = df_sample.iloc[:num_records]
        self._num_records_sampled += len(df_sample)

        return df_sample
