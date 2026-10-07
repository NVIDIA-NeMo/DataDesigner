# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
import threading
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.dataset_builders.errors import ArtifactStorageError

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa

BATCH_FILE_PATTERN = re.compile(r"^batch_(\d+)\.parquet$")

schema_lock = threading.RLock()


def dataframe_to_table(dataframe: pd.DataFrame) -> pa.Table:
    return lazy.pa.Table.from_pandas(dataframe, preserve_index=False).replace_schema_metadata(None)


def is_batch_file(path: Path) -> bool:
    return BATCH_FILE_PATTERN.match(path.name) is not None


def write_table_atomically(table: pa.Table, file_path: Path) -> None:
    tmp_path = file_path.with_name(f"{file_path.name}.tmp.{os.getpid()}")
    try:
        lazy.pq.write_table(table, tmp_path)
        os.replace(tmp_path, file_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def read_batch_schema(file_path: Path) -> pa.Schema:
    return lazy.pq.read_schema(file_path).remove_metadata()


def resolve_target_schema(file_path: Path, incoming: pa.Schema) -> pa.Schema:
    """Return the schema ``file_path`` must be written with, widening sibling batch files if needed.

    Siblings are the other ``batch_*.parquet`` files in the same directory. They all share one
    schema, so a single sibling is the reference; widening rewrites every sibling that differs.
    """
    siblings = (p for p in file_path.parent.glob("batch_*.parquet") if p != file_path and is_batch_file(p))
    first_sibling = next(siblings, None)
    if first_sibling is None:
        return incoming
    reference = read_batch_schema(first_sibling)
    if reference.equals(incoming):
        return reference
    unified = _unify(reference, incoming, file_path)
    if unified.equals(reference):
        return reference
    batch_files = [p for p in file_path.parent.glob("batch_*.parquet") if p != file_path and is_batch_file(p)]
    widen_files(sorted(batch_files), unified, trigger=file_path)
    return unified


def widen_files(batch_files: list[Path], schema: pa.Schema, *, trigger: Path) -> None:
    rewritten: list[tuple[Path, Path]] = []
    try:
        for batch_file in batch_files:
            if read_batch_schema(batch_file).equals(schema):
                continue
            table = cast_table(lazy.pq.read_table(batch_file), schema, batch_file, widened_for=trigger)
            tmp_path = batch_file.with_name(f"{batch_file.name}.widen.{os.getpid()}")
            rewritten.append((batch_file, tmp_path))
            lazy.pq.write_table(table, tmp_path)
        for batch_file, tmp_path in rewritten:
            os.replace(tmp_path, batch_file)
    finally:
        for _, tmp_path in rewritten:
            with suppress(FileNotFoundError):
                tmp_path.unlink()


def cast_table(table: pa.Table, schema: pa.Schema, file_path: Path, *, widened_for: Path | None = None) -> pa.Table:
    trigger = f" (widening required by {_batch_label(widened_for)})" if widened_for else ""
    columns = []
    for field in schema:
        if field.name not in table.column_names:
            columns.append(lazy.pa.nulls(table.num_rows, type=field.type))
            continue
        column = table.column(field.name)
        if column.type != field.type:
            try:
                column = column.cast(field.type, safe=True)
            except (lazy.pa.ArrowInvalid, lazy.pa.ArrowNotImplementedError, lazy.pa.ArrowTypeError) as e:
                raise ArtifactStorageError(
                    f"🛑 Cannot store column {field.name!r} of {_batch_label(file_path)} with the dataset schema{trigger} "
                    f"without losing data: batch type {column.type} -> dataset type {field.type} ({e})."
                ) from e
        columns.append(column)
    return lazy.pa.Table.from_arrays(columns, schema=schema)


def _unify(existing: pa.Schema, incoming: pa.Schema, file_path: Path) -> pa.Schema:
    try:
        return lazy.pa.unify_schemas([existing, incoming], promote_options="permissive")
    except (lazy.pa.ArrowInvalid, lazy.pa.ArrowTypeError) as e:
        conflicts = [
            f"column {name!r}: {existing.field(name).type} (earlier batches) vs {incoming.field(name).type} "
            f"({_batch_label(file_path)})"
            for name in incoming.names
            if name in existing.names and not _compatible(existing.field(name), incoming.field(name))
        ]
        detail = "; ".join(conflicts) or str(e)
        raise ArtifactStorageError(f"🛑 Batches of one dataset have incompatible column types — {detail}.") from e


def _compatible(left: pa.Field, right: pa.Field) -> bool:
    try:
        lazy.pa.unify_schemas([lazy.pa.schema([left]), lazy.pa.schema([right])], promote_options="permissive")
    except (lazy.pa.ArrowInvalid, lazy.pa.ArrowTypeError):
        return False
    return True


def _batch_label(file_path: Path) -> str:
    match = BATCH_FILE_PATTERN.match(file_path.name)
    return f"batch {int(match.group(1))}" if match else f"file {file_path.name!r}"
