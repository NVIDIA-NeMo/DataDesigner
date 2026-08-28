# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe formatting for normalized Slurm boundary errors."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

import yaml
from pydantic import ValidationError

_LOCATION_SEGMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SENSITIVE_LOCATION = re.compile(r"(?:api_?key|credential|password|secret|token)", re.IGNORECASE)
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
    "Value error, resolved model aliases do not match the inline builder": (
        "resolved model aliases do not match the inline builder"
    ),
    "Value error, resolved deployment aliases must exactly cover Data Designer model aliases": (
        "resolved deployment aliases must exactly cover Data Designer model aliases"
    ),
}


def format_validation_error(error: ValidationError, *, subject: str) -> str:
    """Summarize validation without rendering user-controlled values."""
    details = error.errors(include_url=False, include_context=False, include_input=False)
    summaries = sorted({_format_error_detail(detail) for detail in details})
    count = error.error_count()
    noun = "error" if count == 1 else "errors"
    summary = f": {'; '.join(summaries)}" if summaries else ""
    return f"{subject} failed validation ({count} {noun}{summary})"


def _format_error_detail(detail: dict[str, Any]) -> str:
    error_type = str(detail["type"])
    message = _SAFE_VALUE_ERROR_MESSAGES.get(str(detail.get("msg")))
    description = message or _ERROR_DESCRIPTIONS.get(error_type, error_type.replace("_", " "))
    location = _format_location(detail.get("loc", ()))
    return f"{location}: {description}" if location else description


def _format_location(location: Iterable[object]) -> str:
    parts: list[str] = []
    for segment in location:
        if isinstance(segment, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{segment}]"
            continue
        if (
            not isinstance(segment, str)
            or _LOCATION_SEGMENT.fullmatch(segment) is None
            or _SENSITIVE_LOCATION.search(segment) is not None
        ):
            break
        parts.append(segment)
    return ".".join(parts)


def format_parse_error(error: json.JSONDecodeError | yaml.YAMLError) -> str:
    """Summarize a parse failure without rendering source text."""
    if isinstance(error, json.JSONDecodeError):
        return f"invalid JSON at line {error.lineno}, column {error.colno}"
    mark = getattr(error, "problem_mark", None)
    if mark is None:
        return "invalid YAML"
    return f"invalid YAML at line {mark.line + 1}, column {mark.column + 1}"
