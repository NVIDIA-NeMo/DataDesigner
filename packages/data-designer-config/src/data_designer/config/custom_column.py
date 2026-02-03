# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-facing utilities for custom column generation."""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def custom_column_generator(
    required_columns: list[str] | None = None,
    side_effect_columns: list[str] | None = None,
    model_aliases: list[str] | None = None,
) -> Callable[[F], F]:
    """Decorator to define metadata for a custom column generator function.

    Args:
        required_columns: Columns that must exist before this column runs (DAG ordering).
        side_effect_columns: Additional columns the function will create.
        model_aliases: Model aliases used (enables health checks).
    """

    def decorator(fn: F) -> F:
        fn._custom_column_metadata = {  # type: ignore[attr-defined]
            "required_columns": required_columns or [],
            "side_effect_columns": side_effect_columns or [],
            "model_aliases": model_aliases or [],
        }

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        # Copy metadata to wrapper
        wrapper._custom_column_metadata = fn._custom_column_metadata  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator
