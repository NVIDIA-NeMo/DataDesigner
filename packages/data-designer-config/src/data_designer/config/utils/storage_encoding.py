# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.errors import InvalidFileFormatError
from data_designer.config.utils.storage_schema import JSON_STORAGE_METADATA_KEY, JSON_STORAGE_RUN_KEY, JsonStoragePlan

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa


def read_storage_plan(schema: pa.Schema) -> JsonStoragePlan:
    """Read and validate the versioned contract embedded in a Parquet schema."""
    payload = (schema.metadata or {}).get(JSON_STORAGE_METADATA_KEY)
    if payload is None:
        return JsonStoragePlan()
    try:
        data = json.loads(payload)
        if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1:
            raise ValueError("Unsupported JSON storage contract version")
        plan = JsonStoragePlan.model_validate(data)
        if len(schema.names) != len(set(schema.names)):
            raise ValueError("JSON storage metadata requires unique column names")
        if set(plan.columns) - set(schema.names):
            raise ValueError("JSON storage metadata references missing columns")
        for name, paths in plan.columns.items():
            for path in paths:
                _validate_storage_path(schema.field(name).type, path)
        return plan
    except (ValueError, TypeError, ValidationError, RecursionError) as exc:
        raise InvalidFileFormatError("Invalid or unsupported DD JSON storage metadata") from exc


def read_dataframe_storage_plan(dataframe: pd.DataFrame) -> JsonStoragePlan:
    """Validate a logical DataFrame's carried contract and keep its present columns.

    Raises:
        InvalidFileFormatError: If the carried JSON storage contract is malformed.
    """
    try:
        plan = JsonStoragePlan.model_validate(dataframe.attrs.get(JSON_STORAGE_RUN_KEY, {}))
    except (ValueError, TypeError) as exc:
        raise InvalidFileFormatError("Invalid or unsupported DD JSON storage metadata in DataFrame") from exc
    return plan.for_columns(list(dataframe.columns))


def validate_storage_plans(schemas: list[pa.Schema]) -> JsonStoragePlan:
    """Require matching encoding contracts before combining files."""
    expected: JsonStoragePlan | None = None
    for schema in schemas:
        plan = read_storage_plan(schema)
        if expected is not None and plan != expected:
            raise InvalidFileFormatError("Incompatible JSON storage contracts across Parquet files")
        expected = plan
    return expected or JsonStoragePlan()


def encode_storage_table(dataframe: pd.DataFrame, plan: JsonStoragePlan) -> pa.Table:
    """Encode a storage copy, fixing leaf types even for empty/null-only batches."""
    active = plan.for_columns(list(dataframe.columns))
    frame = dataframe.copy(deep=False)
    for name, paths in active.columns.items():
        values = []
        for row, value in enumerate(dataframe[name]):
            try:
                for path in paths:
                    value = _transform_value(value, path, encode=True)
                values.append(value)
            except (ValueError, TypeError, RecursionError) as exc:
                raise InvalidFileFormatError(f"Cannot encode JSON storage column {name!r}, row {row}: {exc}") from exc
        frame[name] = lazy.pd.Series(values, index=frame.index, dtype=object)
    table = lazy.pa.Table.from_pandas(frame, preserve_index=False)
    for name, paths in active.columns.items():
        index = table.schema.get_field_index(name)
        field = table.schema.field(index)
        dtype = _storage_type(field.type, paths)
        table = table.set_column(
            index, field.with_type(dtype), lazy.pa.array(frame[name].tolist(), type=dtype, from_pandas=True)
        )
    if active.columns:
        table = table.replace_schema_metadata(
            {
                **(table.schema.metadata or {}),
                JSON_STORAGE_METADATA_KEY: active.model_dump_json().encode(),
            }
        )
    return table


def decode_storage_table(table: pa.Table) -> pd.DataFrame:
    """Restore logical JSON values without guessing from ordinary string contents."""
    plan = read_storage_plan(table.schema)
    frame = table.to_pandas(types_mapper=lazy.pd.ArrowDtype)
    for name, paths in plan.columns.items():
        values = []
        for row, value in enumerate(table[name].to_pylist()):
            try:
                for path in paths:
                    value = _transform_value(value, path, encode=False)
                values.append(value)
            except (ValueError, TypeError, RecursionError) as exc:
                raise InvalidFileFormatError(f"Cannot decode JSON storage column {name!r}, row {row}: {exc}") from exc
        frame[name] = lazy.pd.Series(values, index=frame.index, dtype=object)
    if plan.columns:
        frame.attrs[JSON_STORAGE_RUN_KEY] = plan.model_dump()
    return frame


def decode_storage_dataframe(dataframe: pd.DataFrame, plan: JsonStoragePlan) -> pd.DataFrame:
    """Decode query-engine batches after their file metadata has been discovered."""
    frame = dataframe.copy(deep=False)
    for name, paths in plan.for_columns(list(frame.columns)).columns.items():
        values = []
        for value in frame[name]:
            for path in paths:
                value = _transform_value(value, path, encode=False)
            values.append(value)
        frame[name] = lazy.pd.Series(values, index=frame.index, dtype=object)
    if plan.columns:
        frame.attrs[JSON_STORAGE_RUN_KEY] = plan.model_dump()
    return frame


def read_encoded_parquet(path: Path, schemas: list[pa.Schema]) -> pd.DataFrame:
    """Read annotated files individually to retain and validate each file's metadata."""
    plan = validate_storage_plans(schemas)
    dataset = lazy.pq.ParquetDataset(path)
    frames = []
    for fragment in sorted(dataset.fragments, key=lambda item: item.path):
        schema = fragment.physical_schema
        for field in dataset.schema:
            if field.name not in schema.names:
                schema = schema.append(field)
        frames.append(decode_storage_table(fragment.to_table(schema=schema)))
    frame = lazy.pd.concat(frames, ignore_index=True)
    frame.attrs[JSON_STORAGE_RUN_KEY] = plan.model_dump()
    return frame


def _transform_value(value: Any, path: list[str | None], *, encode: bool) -> Any:
    if value is None or value is lazy.pd.NA or (not encode and isinstance(value, float) and math.isnan(value)):
        return None
    if not path:
        if encode:
            _validate_json_value(value)
            return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if not isinstance(value, str):
            raise TypeError("Expected JSON text at an encoded leaf")
        decoded = json.loads(value)
        _validate_json_value(decoded)
        return decoded
    head, *tail = path
    if head is None:
        if not isinstance(value, list):
            # DuckDB/pandas can expose list columns as NumPy arrays.
            if not encode and isinstance(value, lazy.np.ndarray):
                value = value.tolist()
            else:
                raise TypeError("Expected a list on a JSON storage traversal path")
        return [_transform_value(item, tail, encode=encode) for item in value]
    if not isinstance(value, dict):
        raise TypeError("Expected an object on a JSON storage traversal path")
    result = deepcopy(value)
    if head in result:
        result[head] = _transform_value(result[head], tail, encode=encode)
    return result


def _validate_storage_path(dtype: pa.DataType, path: list[str | None]) -> None:
    if not path:
        if not (lazy.pa.types.is_string(dtype) or lazy.pa.types.is_large_string(dtype)):
            raise ValueError("JSON storage leaves must have an Arrow UTF-8 string type")
        return
    head, *tail = path
    if head is None:
        if not (lazy.pa.types.is_list(dtype) or lazy.pa.types.is_large_list(dtype)):
            raise ValueError("JSON storage traversal requires an Arrow list")
        _validate_storage_path(dtype.value_type, tail)
        return
    if not lazy.pa.types.is_struct(dtype):
        raise ValueError("JSON storage traversal requires an Arrow struct")
    indices = [index for index, field in enumerate(dtype) if field.name == head]
    if len(indices) != 1:
        raise ValueError("JSON storage traversal references a missing or duplicate struct field")
    _validate_storage_path(dtype.field(indices[0]).type, tail)


def _validate_json_value(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _validate_json_value(item)
        return
    raise TypeError("JSON storage requires string-keyed objects, lists, and finite JSON scalar values")


def _storage_type(dtype: pa.DataType, paths: list[list[str | None]]) -> pa.DataType:
    if [] in paths:
        return lazy.pa.string()
    if any(path[0] is None for path in paths):
        if not all(path[0] is None for path in paths):
            raise InvalidFileFormatError("Conflicting list/object storage paths")
        if lazy.pa.types.is_null(dtype):
            child = lazy.pa.null()
        elif lazy.pa.types.is_list(dtype) or lazy.pa.types.is_large_list(dtype):
            child = dtype.value_type
        else:
            raise InvalidFileFormatError("JSON storage path expects an Arrow list")
        return lazy.pa.list_(_storage_type(child, [path[1:] for path in paths]))
    if lazy.pa.types.is_null(dtype):
        fields: dict[str, Any] = {}
    elif lazy.pa.types.is_struct(dtype):
        fields = {field.name: field for field in dtype}
    else:
        raise InvalidFileFormatError("JSON storage path expects an Arrow struct")
    for name in dict.fromkeys(path[0] for path in paths):
        field = fields.get(name, lazy.pa.field(name, lazy.pa.null()))
        fields[name] = field.with_type(_storage_type(field.type, [path[1:] for path in paths if path[0] == name]))
    return lazy.pa.struct(list(fields.values()))
