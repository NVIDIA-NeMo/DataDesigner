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
        if len(values) != len(buf):
            raise ValueError(
                f"update_batch received {len(values)} values but row group {row_group} has {len(buf)} rows."
            )
        for ri, val in enumerate(values):
            buf[ri][column] = val

    def get_row(self, row_group: int, row_index: int) -> dict[str, Any]:
        return self._buffers[row_group][row_index]

    def get_dataframe(self, row_group: int) -> pd.DataFrame:
        """Return the row group as a DataFrame (excluding dropped rows)."""
        dropped = self._dropped.get(row_group, set())
        rows = [row for i, row in enumerate(self._buffers[row_group]) if i not in dropped]
        return lazy.pd.DataFrame(rows)

    def replace_dataframe(self, row_group: int, df: pd.DataFrame) -> None:
        """Replace the buffer for a row group from a DataFrame (non-dropped rows only).

        If *df* has fewer rows than active slots, trailing slots are marked as dropped.
        """
        dropped = self._dropped.get(row_group, set())
        records = df.to_dict(orient="records")
        buf_idx = 0
        for ri in range(self._row_group_sizes[row_group]):
            if ri in dropped:
                continue
            if buf_idx < len(records):
                self._buffers[row_group][ri] = records[buf_idx]
            else:
                self._dropped.setdefault(row_group, set()).add(ri)
            buf_idx += 1

    def drop_row(self, row_group: int, row_index: int) -> None:
        self._dropped.setdefault(row_group, set()).add(row_index)

    def is_dropped(self, row_group: int, row_index: int) -> bool:
        return row_index in self._dropped.get(row_group, set())

    def checkpoint_row_group(
        self,
        row_group: int,
        on_complete: Callable[[str | None], None] | None = None,
    ) -> None:
        """Write the row group to parquet and free memory."""
        df = self.get_dataframe(row_group)
        final_path = None
        if len(df) > 0:
            # Runtime import: needed at call site; module-level would cause circular import
            from data_designer.engine.storage.artifact_storage import BatchStage

            self._artifact_storage.write_batch_to_parquet_file(
                batch_number=row_group,
                dataframe=df,
                batch_stage=BatchStage.PARTIAL_RESULT,
            )
            final_path = self._artifact_storage.move_partial_result_to_final_file_path(row_group)
            self._actual_num_records += len(df)
            self._total_num_batches += 1
        else:
            logger.warning(f"Row group {row_group} has no records to write after drops.")

        if on_complete:
            on_complete(final_path)

        # Free memory
        del self._buffers[row_group]
        self._dropped.pop(row_group, None)
        self._row_group_sizes.pop(row_group, None)

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
