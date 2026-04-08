# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record-inline skip tracking for conditional column generation.

All reads, writes, and DataFrame-stripping of the ``__internal_skipped_columns`` key go
through this module so sync, async, and buffer code do not diverge.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

SKIPPED_COLUMNS_RECORD_KEY: Final[str] = "__internal_skipped_columns"


def get_skipped_column_names(record: dict) -> set[str]:
    """Return a *copy* of skipped producer column names for this row (empty if unset)."""
    return set(record.get(SKIPPED_COLUMNS_RECORD_KEY, set()))


def apply_skip_to_record(
    record: dict,
    *,
    column_name: str,
    cell_value: bool | int | float | str | None,
    side_effect_columns: Sequence[str],
) -> None:
    """Mutate *record* in place: skip marker, primary cell value, side effects cleared.

    Side-effect columns (e.g. ``__trace``, ``__reasoning_content``) are set to
    ``None`` because the generator never ran — without this, records would have
    inconsistent keys, breaking DataFrame construction and leaving stale or
    missing values visible to downstream columns.
    """
    skipped: set[str] = record.setdefault(SKIPPED_COLUMNS_RECORD_KEY, set())
    skipped.add(column_name)
    record[column_name] = cell_value
    for se_col in side_effect_columns:
        record[se_col] = None


def strip_skip_metadata_for_dataframe_row(record: dict) -> dict:
    """Shallow copy of *record* without skip metadata — safe for ``pd.DataFrame(rows)``."""
    return {k: v for k, v in record.items() if k != SKIPPED_COLUMNS_RECORD_KEY}


def strip_skip_metadata_from_records(records: Sequence[dict]) -> list[dict]:
    """Map :func:`strip_skip_metadata_for_dataframe_row` over *records*."""
    return [strip_skip_metadata_for_dataframe_row(r) for r in records]


def restore_skip_metadata(old_records: Sequence[dict], new_records: Sequence[dict]) -> None:
    """Copy ``SKIPPED_COLUMNS_RECORD_KEY`` from *old_records* into *new_records* in-place.

    ``pd.DataFrame`` construction drops non-column keys, so skip metadata is
    lost when records round-trip through a DataFrame.  Call this after
    ``df.to_dict(orient="records")`` to re-attach the metadata before passing
    the records to ``replace_buffer``.  When lengths differ (e.g.
    ``allow_resize``), only positionally matched rows are restored.
    """
    for i in range(min(len(old_records), len(new_records))):
        meta = old_records[i].get(SKIPPED_COLUMNS_RECORD_KEY)
        if meta is not None:
            new_records[i][SKIPPED_COLUMNS_RECORD_KEY] = meta
