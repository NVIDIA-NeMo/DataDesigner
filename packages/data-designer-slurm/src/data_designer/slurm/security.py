# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security helpers shared by Slurm process boundaries."""

from __future__ import annotations

import re

from data_designer.slurm.config.environment import is_secret_bearing_name

_ASSIGNMENT_PATTERN = re.compile(
    r"(?P<prefix>(?P<quote>[\"']?)(?P<name>-{0,2}[A-Za-z][A-Za-z0-9_.-]*)"
    r"(?P=quote)\s*[:=]\s*)(?P<value>\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_OPTION_PATTERN = re.compile(r"(?P<prefix>(?P<name>--[A-Za-z][A-Za-z0-9.-]*)\s+)(?P<value>\"[^\"]*\"|'[^']*'|[^\s]+)")
_AUTHORIZATION_PATTERN = re.compile(r"(?i)(?P<prefix>\bauthorization\s*[:=]\s*(?:basic|bearer)\s+)(?P<value>[^\s,;]+)")
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
    redacted = _ASSIGNMENT_PATTERN.sub(_redact_named_value, redacted)
    redacted = _OPTION_PATTERN.sub(_redact_named_value, redacted)
    for pattern in _TOKEN_PATTERNS:
        redacted = pattern.sub(_REDACTION, redacted)
    return redacted


def _redact_named_value(match: re.Match[str]) -> str:
    name = match.group("name").lstrip("-")
    if not is_secret_bearing_name(name):
        return match.group(0)
    return f"{match.group('prefix')}{_REDACTION}"


__all__ = ["redact_sensitive_text"]
