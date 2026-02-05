# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-facing utilities for custom column generation."""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# Expected parameter names by position (first param validated as row/df at runtime based on strategy)
EXPECTED_PARAMS = ({"row", "df"}, {"generator_params"}, {"models"})


def validate_generator_signature(fn: Callable[..., Any]) -> list[inspect.Parameter]:
    """Validate generator function signature. Returns positional params if valid."""
    params = [
        p
        for p in inspect.signature(fn).parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    n = len(params)
    if n == 0 or n > 3:
        raise TypeError(f"Generator '{fn.__name__}' must have 1-3 parameters, got {n}.")
    for i, param in enumerate(params):
        if param.name not in EXPECTED_PARAMS[i]:
            expected = " or ".join(f"'{p}'" for p in sorted(EXPECTED_PARAMS[i]))
            raise TypeError(f"Generator '{fn.__name__}' param {i + 1} must be {expected}, got '{param.name}'.")
    return params


def custom_column_generator(
    required_columns: list[str] | None = None,
    side_effect_columns: list[str] | None = None,
    model_aliases: list[str] | None = None,
) -> Callable[[F], F]:
    """Decorator to define metadata for a custom column generator function.

    Args:
        required_columns: Columns that must exist before this column runs (DAG ordering).
        side_effect_columns: Additional columns the function will create.
        model_aliases: Model aliases to include in the `models` dict (required for LLM access).
    """

    def decorator(fn: F) -> F:
        validate_generator_signature(fn)
        fn.custom_column_metadata = {  # type: ignore[attr-defined]
            "required_columns": required_columns or [],
            "side_effect_columns": side_effect_columns or [],
            "model_aliases": model_aliases or [],
        }

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        # Copy metadata to wrapper
        wrapper.custom_column_metadata = fn.custom_column_metadata  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator
