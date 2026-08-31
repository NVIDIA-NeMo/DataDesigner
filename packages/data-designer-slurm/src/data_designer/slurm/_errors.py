# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe formatting for normalized Slurm boundary errors."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import yaml
from pydantic import BaseModel, ValidationError

_ERROR_DESCRIPTIONS = {
    "extra_forbidden": "field is not permitted",
    "greater_than": "must be greater than the allowed minimum",
    "greater_than_equal": "must be at least the allowed minimum",
    "less_than": "must be less than the allowed maximum",
    "less_than_equal": "must not exceed the allowed maximum",
    "literal_error": "must use an allowed value",
    "missing": "field is required",
    "string_pattern_mismatch": "does not match the required pattern",
    "string_too_long": "is longer than allowed",
    "string_too_short": "is shorter than allowed",
    "value_error": "value is invalid",
}
_SAFE_VALUE_ERROR_MESSAGES = {
    "Value error, builder content digest does not match the resolved input": (
        "resolved builder digest does not match its input"
    ),
    "Value error, image inspection digest does not match the resolved SQSH": (
        "resolved image digest does not match its inspection record"
    ),
    "Value error, mem_per_gpu requires GRES GPU request mode": "mem_per_gpu requires GRES GPU request mode",
    "Value error, resolved model aliases do not match the inline builder": (
        "resolved model aliases do not match the inline builder"
    ),
    "Value error, resolved deployment aliases must exactly cover Data Designer model aliases": (
        "resolved deployment aliases must exactly cover Data Designer model aliases"
    ),
}


def format_validation_error(
    error: ValidationError,
    *,
    subject: str,
    models: type[BaseModel] | tuple[type[BaseModel], ...],
) -> str:
    """Summarize validation without rendering user-controlled values."""
    candidates = models if isinstance(models, tuple) else (models,)
    model_type = next((model for model in candidates if model.__name__ == error.title), None)
    details = error.errors(include_url=False, include_context=False, include_input=False)
    summaries = sorted({_format_error_detail(detail, model_type=model_type) for detail in details})
    count = error.error_count()
    noun = "error" if count == 1 else "errors"
    summary = f": {'; '.join(summaries)}" if summaries else ""
    return f"{subject} failed validation ({count} {noun}{summary})"


def _format_error_detail(detail: dict[str, Any], *, model_type: type[BaseModel] | None) -> str:
    error_type = str(detail["type"])
    message = _SAFE_VALUE_ERROR_MESSAGES.get(str(detail.get("msg")))
    description = message or _ERROR_DESCRIPTIONS.get(error_type, error_type.replace("_", " "))
    location = _format_location(detail.get("loc", ()), model_type=model_type)
    return f"{location}: {description}" if location else description


def _format_location(location: Iterable[object], *, model_type: type[BaseModel] | None) -> str:
    if model_type is None:
        return ""
    parts: list[str] = []
    schemas: tuple[object, ...] = (model_type,)
    for segment in location:
        schemas = tuple(candidate for schema in schemas for candidate in _expand_schema(schema))
        if any(_is_mapping_schema(schema) for schema in schemas):
            break
        if isinstance(segment, int):
            item_schemas = tuple(item for schema in schemas for item in _sequence_item_schemas(schema, index=segment))
            if not parts or not item_schemas:
                break
            parts[-1] = f"{parts[-1]}[{segment}]"
            schemas = item_schemas
            continue
        if not isinstance(segment, str):
            break
        field_schemas = tuple(
            field.annotation
            for schema in schemas
            if isinstance(schema, type)
            and issubclass(schema, BaseModel)
            and (field := schema.model_fields.get(segment)) is not None
        )
        if not field_schemas:
            branch_schemas = _tagged_union_schemas(schemas, tag=segment)
            if not branch_schemas:
                break
            schemas = branch_schemas
            continue
        parts.append(segment)
        schemas = field_schemas
    return ".".join(parts)


def _expand_schema(schema: object) -> tuple[object, ...]:
    origin = get_origin(schema)
    if origin is Annotated:
        return _expand_schema(get_args(schema)[0])
    if origin in (Union, UnionType):
        return tuple(candidate for item in get_args(schema) for candidate in _expand_schema(item))
    return (schema,)


def _is_mapping_schema(schema: object) -> bool:
    origin = get_origin(schema)
    candidate = origin or schema
    return isinstance(candidate, type) and issubclass(candidate, Mapping)


def _tagged_union_schemas(schemas: tuple[object, ...], *, tag: str) -> tuple[object, ...]:
    if len(schemas) < 2:
        return ()
    return tuple(
        schema
        for schema in schemas
        if isinstance(schema, type)
        and issubclass(schema, BaseModel)
        and (
            schema.__name__ == tag
            or any(_annotation_contains_literal(field.annotation, value=tag) for field in schema.model_fields.values())
        )
    )


def _annotation_contains_literal(annotation: object, *, value: str) -> bool:
    return any(
        get_origin(candidate) is Literal and value in get_args(candidate) for candidate in _expand_schema(annotation)
    )


def _sequence_item_schemas(schema: object, *, index: int) -> tuple[object, ...]:
    origin = get_origin(schema)
    candidate = origin or schema
    if not isinstance(candidate, type) or not issubclass(candidate, Sequence):
        return ()
    arguments = get_args(schema)
    if not arguments:
        return ()
    if origin is tuple and arguments[-1] is not Ellipsis:
        return (arguments[index],) if index < len(arguments) else ()
    return (arguments[0],)


def format_parse_error(error: json.JSONDecodeError | yaml.YAMLError) -> str:
    """Summarize a parse failure without rendering source text."""
    if isinstance(error, json.JSONDecodeError):
        return f"invalid JSON at line {error.lineno}, column {error.colno}"
    mark = getattr(error, "problem_mark", None)
    if mark is None:
        return "invalid YAML"
    return f"invalid YAML at line {mark.line + 1}, column {mark.column + 1}"
