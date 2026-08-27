# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe formatting for normalized Slurm boundary errors."""

from __future__ import annotations

import json

import yaml
from pydantic import ValidationError


def format_validation_error(error: ValidationError, *, subject: str) -> str:
    """Summarize validation without rendering user-controlled values."""
    error_types = sorted(
        {str(detail["type"]) for detail in error.errors(include_url=False, include_context=False, include_input=False)}
    )
    count = error.error_count()
    noun = "error" if count == 1 else "errors"
    kinds = f": {', '.join(error_types)}" if error_types else ""
    return f"{subject} failed validation ({count} {noun}{kinds})"


def format_parse_error(error: json.JSONDecodeError | yaml.YAMLError) -> str:
    """Summarize a parse failure without rendering source text."""
    if isinstance(error, json.JSONDecodeError):
        return f"invalid JSON at line {error.lineno}, column {error.colno}"
    mark = getattr(error, "problem_mark", None)
    if mark is None:
        return "invalid YAML"
    return f"invalid YAML at line {mark.line + 1}, column {mark.column + 1}"
