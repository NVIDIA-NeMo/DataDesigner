# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security helpers shared by Slurm process boundaries."""

from __future__ import annotations

import re

from data_designer.slurm.config.environment import is_secret_bearing_name

_ASSIGNMENT_START_PATTERN = re.compile(
    r"(?P<prefix>(?P<quote>[\"']?)(?P<name>-{0,2}[A-Za-z][A-Za-z0-9_.-]*)(?P=quote)\s*[:=]\s*)"
)
_OPTION_START_PATTERN = re.compile(r"(?P<prefix>(?P<name>--[A-Za-z][A-Za-z0-9.-]*)\s+)")
_AUTHORIZATION_PATTERN = re.compile(r"(?i)(?P<prefix>\bauthorization\s*[:=]\s*(?:basic|bearer)\s+)(?P<value>[^\s]+)")
_URL_USERINFO_PATTERN = re.compile(r"(?i)(?P<scheme>\bhttps?://)[^/@\s]+@")
_TOKEN_PATTERNS = (
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bnvapi-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
)
_REDACTION = "<redacted>"


def redact_sensitive_text(value: str) -> str:
    """Redact recognizable credentials without echoing their values."""
    redacted = _AUTHORIZATION_PATTERN.sub(lambda match: f"{match.group('prefix')}{_REDACTION}", value)
    redacted = _URL_USERINFO_PATTERN.sub(lambda match: f"{match.group('scheme')}{_REDACTION}@", redacted)
    redacted = _redact_named_values(redacted, _ASSIGNMENT_START_PATTERN)
    redacted = _redact_named_values(redacted, _OPTION_START_PATTERN)
    for pattern in _TOKEN_PATTERNS:
        redacted = pattern.sub(_REDACTION, redacted)
    return redacted


def _redact_named_values(value: str, start_pattern: re.Pattern[str]) -> str:
    """Redact secret-bearing named values without letting earlier matches overlap them."""
    parts: list[str] = []
    output_cursor = 0
    search_cursor = 0
    while match := start_pattern.search(value, search_cursor):
        if not is_secret_bearing_name(match.group("name").lstrip("-")):
            search_cursor = match.end()
            continue
        value_end = _find_named_value_end(value, match.end())
        parts.append(value[output_cursor : match.start()])
        parts.append(f"{match.group('prefix')}{_REDACTION}")
        output_cursor = value_end
        search_cursor = value_end
    parts.append(value[output_cursor:])
    return "".join(parts)


def _find_named_value_end(value: str, start: int) -> int:
    """Find the first unambiguous boundary for a secret-bearing value."""
    if start < len(value) and value[start] in {'"', "'"}:
        return _find_quoted_value_end(value, start)
    index = start
    while index < len(value) and not value[index].isspace():
        index += 1
    return index


def _find_quoted_value_end(value: str, start: int) -> int:
    quote = value[start]
    index = start + 1
    while index < len(value):
        if value[index] == "\\":
            index += 2
        elif value[index] == quote:
            return index + 1
        else:
            index += 1
    return len(value)


__all__ = ["redact_sensitive_text"]
