# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for emitting warnings that attribute correctly to user code.

Pydantic v2 dispatches ``@model_validator`` / ``@field_validator`` callbacks
through several internal frames. ``warnings.warn(stacklevel=N)`` from inside a
validator therefore tends to land inside pydantic's machinery rather than at
the user's ``ModelConfig(...)`` / ``ModelProviderRegistry(...)`` call site.

That breaks two things:

1. Attribution — the displayed source location is unhelpful.
2. Deduplication — Python's default once-per-location dedup key is
   ``(category, module, lineno)``. When every call resolves to the same
   pydantic-internal line, every warning after the first is silently
   suppressed regardless of which user file triggered it.

``warn_at_caller`` walks the stack to the first frame outside pydantic (and
outside this helper / the calling validator) and uses
``warnings.warn_explicit`` to attribute the warning there.
"""

from __future__ import annotations

import sys
import warnings


def warn_at_caller(message: str, category: type[Warning]) -> None:
    """Emit ``message`` attributed to the first non-pydantic frame above the caller.

    Intended to be invoked from inside a pydantic validator. The walk skips this
    helper's own frame and the calling validator's frame, then walks past any
    pydantic-internal frames until it finds the user's call site.

    The user frame's ``__warningregistry__`` is passed to
    ``warnings.warn_explicit`` so Python's built-in once-per-location dedup keys
    on the *user's* (filename, lineno) rather than an internal pydantic frame.
    That matches how ``warnings.warn`` would dedup if ``stacklevel`` could
    correctly point at the user.

    We deliberately do *not* pass ``module_globals`` — it's only used for
    ``linecache`` source-line display, and for scripts run with ``python -c``
    (where the user frame's ``__loader__`` is ``BuiltinImporter`` for
    ``__main__``) the lookup raises ``ImportError("'__main__' is not a built-in
    module")``. Skipping ``module_globals`` keeps the warning path robust at
    the cost of an empty source line in the formatted output.
    """
    # Skip frame 0 (warn_at_caller) and frame 1 (the validator that called us).
    frame = sys._getframe(2) if hasattr(sys, "_getframe") else None
    while frame is not None:
        module_name = frame.f_globals.get("__name__", "")
        if not module_name.startswith("pydantic"):
            warnings.warn_explicit(
                message,
                category,
                frame.f_code.co_filename,
                frame.f_lineno,
                module=module_name,
                registry=frame.f_globals.setdefault("__warningregistry__", {}),
            )
            return
        frame = frame.f_back

    # Fallback: never escaped pydantic frames (or no frame access). Use stacklevel.
    warnings.warn(message, category, stacklevel=3)
