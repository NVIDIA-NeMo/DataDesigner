# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Container, Iterator

from data_designer.engine.dataset_builders.artifact_storage import ArtifactStorage, BatchStage
from data_designer.engine.dataset_builders.utils.errors import DatasetBatchManagementError
from data_designer.lazy_heavy_imports import pd, pq

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class DatasetBatchManager:
    def __init__(self, artifact_storage: ArtifactStorage):
        self._buffer: list[dict] = []
        self._current_batch_number = 0
        self._num_records_list: list[int] | None = None
        self._buffer_size: int | None = None
        self.artifact_storage = artifact_storage

    @property
    def num_batches(self) -> int:
        if self._num_records_list is None:
            return 0
        return len(self._num_records_list)

    @property
    def num_records_batch(self) -> int:
        if self._num_records_list is None or self._current_batch_number >= len(self._num_records_list):
            raise DatasetBatchManagementError("🛑 Invalid batch number or num_records_list not set.")
        return self._num_records_list[self._current_batch_number]

    @property
    def num_records_list(self) -> list[int]:
        if self._num_records_list is None:
            raise DatasetBatchManagementError("🛑 `num_records_list` is not set. Call start() first.")
        return self._num_records_list

    @property
    def num_records_in_buffer(self) -> int:
        return len(self._buffer)

    @property
    def buffer_is_empty(self) -> bool:
        return len(self._buffer) == 0

    @property
    def buffer_size(self) -> int:
        if self._buffer_size is None:
            raise DatasetBatchManagementError("🛑 `buffer_size` is not set. Call start() first.")
        return self._buffer_size

    def add_record(self, record: dict) -> None:
        self.add_records([record])

    def add_records(self, records: list[dict]) -> None:
        self._buffer.extend(records)
        if len(self._buffer) > self.buffer_size:
            raise DatasetBatchManagementError(
                f"🛑 Buffer size exceeded. Current: {len(self._buffer)}, Max: {self.buffer_size}. "
                "Flush the batch before adding more records."
            )

    def drop_records(self, index: Container[int]) -> None:
        self._buffer = [record for i, record in enumerate(self._buffer) if i not in index]

    def finish_batch(self, on_complete: Callable[[Path], None] | None = None) -> Path | None:
        """Finish the batch by moving the results from the partial results path to the final parquet folder.

        Returns:
            The path to the written parquet file.
        """
        if self._current_batch_number >= self.num_batches:
            raise DatasetBatchManagementError("🛑 All batches have been processed.")

        if self.write() is not None:
            final_file_path = self.artifact_storage.move_partial_result_to_final_file_path(self._current_batch_number)

            self.artifact_storage.write_metadata(
                {
                    "target_num_records": sum(self.num_records_list),
                    "total_num_batches": self.num_batches,
                    "buffer_size": self._buffer_size,
                    "schema": {field.name: str(field.type) for field in pq.read_schema(final_file_path)},
                    "file_paths": self.artifact_storage.get_file_paths(),
                    "num_completed_batches": self._current_batch_number + 1,
                    "dataset_name": self.artifact_storage.dataset_name,
                }
            )

            if on_complete:
                on_complete(final_file_path)
        else:
            final_file_path = None

            logger.warning(
                f"⚠️ Batch {self._current_batch_number + 1} finished without any results to write. "
                "A partial dataset containing the currently available columns has been written to the partial results "
                f"directory: {self.artifact_storage.partial_results_path}"
            )

        self._current_batch_number += 1
        self._buffer: list[dict] = []

        return final_file_path

    def finish(self) -> None:
        """Finish the dataset writing process by deleting the partial results path if it exists and is empty."""

        # If the partial results path is empty, delete it.
        if not any(self.artifact_storage.partial_results_path.iterdir()):
            self.artifact_storage.partial_results_path.rmdir()

        # Otherwise, log a warning, since existing partial results means the dataset is not complete.
        else:
            logger.warning("⚠️ Dataset writing finished with partial results.")

        self.reset()

    def get_current_batch_number(self) -> int:
        return self._current_batch_number

    def get_current_batch(self, *, as_dataframe: bool = False) -> pd.DataFrame | list[dict]:
        if as_dataframe:
            return pd.DataFrame(self._buffer)
        return self._buffer

    def iter_current_batch(self) -> Iterator[tuple[int, dict]]:
        for i, record in enumerate(self._buffer):
            yield i, record

    def reset(self, delete_files: bool = False) -> None:
        self._current_batch_number = 0
        self._buffer: list[dict] = []
        if delete_files:
            for dir_path in [
                self.artifact_storage.final_dataset_path,
                self.artifact_storage.partial_results_path,
                self.artifact_storage.dropped_columns_dataset_path,
                self.artifact_storage.base_dataset_path,
            ]:
                if dir_path.exists():
                    try:
                        shutil.rmtree(dir_path)
                    except OSError as e:
                        raise DatasetBatchManagementError(f"🛑 Failed to delete directory {dir_path}: {e}")

    def start(self, *, num_records: int, buffer_size: int) -> None:
        if num_records <= 0:
            raise DatasetBatchManagementError("🛑 num_records must be positive.")
        if buffer_size <= 0:
            raise DatasetBatchManagementError("🛑 buffer_size must be positive.")

        self._buffer_size = buffer_size
        self._num_records_list = [buffer_size] * (num_records // buffer_size)
        if remaining_records := num_records % buffer_size:
            self._num_records_list.append(remaining_records)
        self.reset()

    def write(self) -> Path | None:
        """Write the current batch to a parquet file.

        This method always writes results to the partial results path.

        Returns:
            The path to the written parquet file. If the buffer is empty, returns None.
        """
        if len(self._buffer) == 0:
            return None
        try:
            file_path = self.artifact_storage.write_batch_to_parquet_file(
                batch_number=self._current_batch_number,
                dataframe=pd.DataFrame(self._buffer),
                batch_stage=BatchStage.PARTIAL_RESULT,
            )
            return file_path
        except Exception as e:
            raise DatasetBatchManagementError(f"🛑 Failed to write batch {self._current_batch_number}: {e}")

    def update_record(self, index: int, record: dict) -> None:
        if index < 0 or index >= len(self._buffer):
            raise IndexError(f"🛑 Index {index} is out of bounds for buffer of size {len(self._buffer)}.")
        self._buffer[index] = record

    def update_records(self, records: list[dict]) -> None:
        if len(records) != len(self._buffer):
            raise DatasetBatchManagementError(
                f"🛑 Number of records to update ({len(records)}) must match "
                f"the number of records in the buffer ({len(self._buffer)})."
            )
        self._buffer = records
