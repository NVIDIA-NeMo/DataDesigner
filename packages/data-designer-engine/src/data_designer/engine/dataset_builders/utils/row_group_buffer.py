# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

import data_designer.lazy_heavy_imports as lazy

if TYPE_CHECKING:
    import pandas as pd

    from data_designer.engine.storage.artifact_storage import ArtifactStorage

logger = logging.getLogger(__name__)


class RowGroupBufferManager:
    """Per-row-group buffer manager for the async dataset builder.

    Each active row group gets its own ``list[dict]`` buffer. Cell-level
    writes (``update_cell``) are the only write path — whole-record replacement
    is unsafe under parallel column execution.

    The existing ``DatasetBatchManager`` is untouched; this class is used
    exclusively by the async scheduler.
    """

    def __init__(self, artifact_storage: ArtifactStorage) -> None:
        self._buffers: dict[int, list[dict]] = {}
        self._row_group_sizes: dict[int, int] = {}
        self._dropped: dict[int, set[int]] = {}
        self._artifact_storage = artifact_storage
        self._actual_num_records: int = 0
        self._total_num_batches: int = 0

    def init_row_group(self, row_group: int, size: int) -> None:
        """Allocate a buffer for *row_group* with *size* empty rows."""
        self._buffers[row_group] = [{} for _ in range(size)]
        self._row_group_sizes[row_group] = size
        self._dropped.setdefault(row_group, set())

    def update_cell(self, row_group: int, row_index: int, column: str, value: Any) -> None:
        """Write a single cell value. Thread-safe within the asyncio event loop."""
        self._buffers[row_group][row_index][column] = value

    def update_cells(self, row_group: int, row_index: int, values: dict[str, Any]) -> None:
        """Write multiple cell values for a single row."""
        self._buffers[row_group][row_index].update(values)

    def update_batch(self, row_group: int, column: str, values: list[Any]) -> None:
        """Write a full column for all rows in a row group."""
        buf = self._buffers[row_group]
        for ri, val in enumerate(values):
            buf[ri][column] = val

    def get_row(self, row_group: int, row_index: int) -> dict:
        return self._buffers[row_group][row_index]

    def get_dataframe(self, row_group: int) -> pd.DataFrame:
        """Return the row group as a DataFrame (excluding dropped rows)."""
        dropped = self._dropped.get(row_group, set())
        rows = [row for i, row in enumerate(self._buffers[row_group]) if i not in dropped]
        return lazy.pd.DataFrame(rows)

    def drop_row(self, row_group: int, row_index: int) -> None:
        self._dropped.setdefault(row_group, set()).add(row_index)

    def is_dropped(self, row_group: int, row_index: int) -> bool:
        return row_index in self._dropped.get(row_group, set())

    def checkpoint_row_group(
        self,
        row_group: int,
        on_complete: Callable | None = None,
    ) -> None:
        """Write the row group to parquet and free memory."""
        df = self.get_dataframe(row_group)
        if len(df) > 0:
            from data_designer.engine.storage.artifact_storage import BatchStage

            self._artifact_storage.write_batch_to_parquet_file(
                batch_number=row_group,
                dataframe=df,
                batch_stage=BatchStage.PARTIAL_RESULT,
            )
            final_path = self._artifact_storage.move_partial_result_to_final_file_path(row_group)
            self._actual_num_records += len(df)
            self._total_num_batches += 1

            if on_complete:
                on_complete(final_path)
        else:
            logger.warning(f"Row group {row_group} has no records to write after drops.")

        # Free memory
        del self._buffers[row_group]
        self._dropped.pop(row_group, None)

    def write_metadata(self, target_num_records: int, buffer_size: int) -> None:
        """Write final metadata after all row groups are checkpointed."""
        self._artifact_storage.write_metadata(
            {
                "target_num_records": target_num_records,
                "actual_num_records": self._actual_num_records,
                "total_num_batches": self._total_num_batches,
                "buffer_size": buffer_size,
                "dataset_name": self._artifact_storage.dataset_name,
                "file_paths": self._artifact_storage.get_file_paths(),
                "num_completed_batches": self._total_num_batches,
            }
        )

    @property
    def actual_num_records(self) -> int:
        return self._actual_num_records
