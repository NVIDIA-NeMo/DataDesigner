# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skip expression evaluation for conditional column generation."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from jinja2 import StrictUndefined
from jinja2.exceptions import SecurityError, TemplateSyntaxError, UndefinedError
from jinja2.nativetypes import NativeEnvironment
from jinja2.sandbox import SandboxedEnvironment

from data_designer.engine.dataset_builders.utils.skip_tracker import SKIPPED_COLUMNS_RECORD_KEY
from data_designer.engine.processing.utils import deserialize_json_values

if TYPE_CHECKING:
    from jinja2 import Template

logger = logging.getLogger(__name__)


class NativeSandboxedEnvironment(SandboxedEnvironment, NativeEnvironment):
    """Sandboxed environment that returns native Python types instead of strings.

    Uses ``StrictUndefined`` so that references to missing variables raise
    ``UndefinedError`` instead of silently returning a truthy ``Undefined``
    object (which would cause every row to be skipped on a typo).
    """


_env = NativeSandboxedEnvironment(undefined=StrictUndefined)


def evaluate_skip_when(expression: str, record: dict) -> bool:
    """Render *expression* against *record*; return ``True`` if result is truthy.

    The caller is responsible for passing a raw record dict — deserialization
    of JSON string values is handled here so both sync and async engines get
    identical behavior.  On expected evaluation failures (``UndefinedError``,
    ``SecurityError``, ``TemplateSyntaxError``, ``TypeError``, ``ValueError``)
    a warning is logged and ``True`` is returned (fail-safe: skip the row
    rather than making an expensive LLM call on a row with unknown filter
    status).  Unexpected exceptions propagate to the caller.
    """
    try:
        template = _compile_skip_template(expression)
        deserialized = deserialize_json_values(record)
        result = template.render(deserialized)
        return bool(result)
    except (UndefinedError, SecurityError, TemplateSyntaxError, TypeError, ValueError):
        logger.warning(
            "skip.when evaluation failed for expression %r; treating as truthy (cell will be skipped)",
            expression,
            exc_info=True,
        )
        return True


def get_skipped_column_names(record: dict) -> set[str]:
    """Return a *copy* of skipped producer column names for this row (empty if unset)."""
    return set(record.get(SKIPPED_COLUMNS_RECORD_KEY, set()))


def should_skip_by_propagation(
    required_columns: list[str],
    skipped_columns_for_row: set[str],
) -> bool:
    """Return ``True`` if any required column was skipped.

    The caller is responsible for checking ``propagate_skip`` on the column
    config *before* calling this function (see ``ExecutionGraph.should_propagate_skip``).
    """
    return not skipped_columns_for_row.isdisjoint(required_columns)


def should_skip_column_for_record(
    record: dict,
    *,
    propagate_skip: bool,
    required_columns: list[str],
    skip_config_when: str | None,
) -> bool:
    """Unified skip decision for a single cell/record.

    Shared by both the sync and async engines so the logic stays in sync.
    Checks propagation first (cheaper), then the expression gate.

    Args:
        record: Current row dict (may contain ``__internal_skipped_columns``).
        propagate_skip: Whether this column auto-skips on upstream skips.
        required_columns: Config-level data dependencies for *column*.
        skip_config_when: The ``skip.when`` Jinja2 expression, or ``None``.
    """
    skipped_cols = get_skipped_column_names(record)
    if propagate_skip and should_skip_by_propagation(required_columns, skipped_cols):
        return True
    if skip_config_when is not None:
        return evaluate_skip_when(skip_config_when, record)
    return False


@lru_cache(maxsize=64)
def _compile_skip_template(expression: str) -> Template:
    return _env.from_string(expression)
