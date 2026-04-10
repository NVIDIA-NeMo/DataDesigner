# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record-inline skip tracking for conditional column generation.

All reads, writes, and DataFrame-stripping of the ``__internal_skipped_columns`` key go
through this module so sync, async, and buffer code do not diverge.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

SKIPPED_COLUMNS_RECORD_KEY: Final[str] = "__internal_skipped_columns"
SKIP_METADATA_RESTORE_ID_COLUMN_PREFIX: Final[str] = "__internal_skip_restore_id"


@dataclass(frozen=True, slots=True)
class SkipMetadataRestoreContext:
    """Metadata needed to restore skip provenance after a DataFrame round-trip."""

    restore_id_column: str
    source_ids: set[str]
    skipped_columns_by_source_id: dict[str, set[str]]


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


def prepare_records_for_skip_metadata_round_trip(
    records: Sequence[dict],
) -> tuple[list[dict], SkipMetadataRestoreContext | None]:
    """Prepare records for a DataFrame round-trip while preserving skip metadata.

    Returns stripped records ready for ``pd.DataFrame(...)``. If any record has
    skip metadata, injects a hidden restore-ID column and returns a context that
    can later be passed to :func:`restore_skip_metadata`.
    """
    if not any(SKIPPED_COLUMNS_RECORD_KEY in record for record in records):
        return strip_skip_metadata_from_records(records), None

    restore_id_column = _choose_restore_id_column(records)
    prepared_records: list[dict] = []
    source_ids: set[str] = set()
    skipped_columns_by_source_id: dict[str, set[str]] = {}

    for index, record in enumerate(records):
        source_id = str(index)
        source_ids.add(source_id)
        prepared_record = strip_skip_metadata_for_dataframe_row(record)
        prepared_record[restore_id_column] = source_id
        prepared_records.append(prepared_record)

        meta = record.get(SKIPPED_COLUMNS_RECORD_KEY)
        if meta is not None:
            skipped_columns_by_source_id[source_id] = set(meta)

    return prepared_records, SkipMetadataRestoreContext(
        restore_id_column=restore_id_column,
        source_ids=source_ids,
        skipped_columns_by_source_id=skipped_columns_by_source_id,
    )


def restore_skip_metadata(
    records: Sequence[dict],
    *,
    context: SkipMetadataRestoreContext,
    allow_resize: bool,
) -> None:
    """Restore skip provenance using hidden restore IDs instead of row position."""
    restored_source_ids: list[str] = []
    for record in records:
        if context.restore_id_column not in record:
            raise ValueError(
                f"Records returned from the DataFrame round-trip must preserve "
                f"the internal column {context.restore_id_column!r} so skip "
                "provenance can be restored."
            )

        source_id = str(record.pop(context.restore_id_column))
        if source_id not in context.source_ids:
            raise ValueError(
                f"Record returned unknown restore ID {source_id!r}. Skip provenance "
                "can only be restored for rows derived from the original input."
            )

        restored_source_ids.append(source_id)
        meta = context.skipped_columns_by_source_id.get(source_id)
        if meta is not None:
            record[SKIPPED_COLUMNS_RECORD_KEY] = set(meta)

    if not allow_resize:
        if len(restored_source_ids) != len(context.source_ids) or set(restored_source_ids) != context.source_ids:
            raise ValueError(
                "Full-column generation changed the row identity mapping while "
                "allow_resize=False. Returned rows must preserve a 1:1 mapping "
                "to the original input so skip provenance can be restored."
            )


def _choose_restore_id_column(records: Sequence[dict]) -> str:
    candidate = SKIP_METADATA_RESTORE_ID_COLUMN_PREFIX
    suffix = 0
    while any(candidate in record for record in records):
        suffix += 1
        candidate = f"{SKIP_METADATA_RESTORE_ID_COLUMN_PREFIX}_{suffix}"
    return candidate
