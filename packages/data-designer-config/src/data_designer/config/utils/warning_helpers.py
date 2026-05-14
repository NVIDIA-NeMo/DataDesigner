# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for emitting warnings that attribute correctly to user code.

Library-internal warnings (typically emitted from a pydantic ``@model_validator``
or from a helper function) need to be attributed to the *user's* call site, not
to the library frame that happened to fire them. Two reasons:

1. Attribution — a source location pointing at library internals isn't
   actionable.
2. Visibility under default filters — Python's default ``DeprecationWarning``
   filter is::

       default::DeprecationWarning:__main__
       ignore::DeprecationWarning

   Library-attributed ``DeprecationWarning`` entries fall under the second
   filter and are silenced. Attributing to user code is what gets the warning
   actually shown.

3. Deduplication — Python's once-per-location dedup key is
   ``(category, module, lineno)``. When every call resolves to the same
   library-internal line, every warning after the first is silently suppressed
   regardless of which user file triggered it.

``warn_at_caller`` walks the stack past frames whose module belongs to a known
internal package (pydantic, data_designer) and uses ``warnings.warn_explicit``
to attribute the warning at the first user frame.
"""

from __future__ import annotations

import sys
import warnings

DEFAULT_INTERNAL_PREFIXES: tuple[str, ...] = ("pydantic", "pydantic_core", "data_designer")
"""Modules whose frames are skipped when walking up to the user's call site.

Matching is exact-or-dotted-prefix (see ``_module_in_prefixes``), so
``pydantic_helpers`` is *not* treated as part of ``pydantic``."""


def _module_in_prefixes(module_name: str, prefixes: tuple[str, ...]) -> bool:
    """Return True if ``module_name`` belongs to one of the prefix-rooted packages.

    Uses exact-equality plus dotted-prefix matching so that, e.g.,
    ``pydantic_helpers`` is NOT treated as part of the ``pydantic`` package
    while ``pydantic.fields`` is. Same for ``data_designer`` vs. a hypothetical
    ``data_designer_other``.
    """
    return any(module_name == prefix or module_name.startswith(prefix + ".") for prefix in prefixes)


def warn_at_caller(
    message: str,
    category: type[Warning],
    *,
    skip_prefixes: tuple[str, ...] = DEFAULT_INTERNAL_PREFIXES,
) -> None:
    """Emit ``message`` attributed to the first frame outside ``skip_prefixes``.

    Intended for warnings whose root cause is the user's call site but whose
    emission point is library code (a pydantic validator, an internal helper,
    etc.). The walk starts above this helper's own frame and skips every frame
    whose module belongs to a package in ``skip_prefixes`` until it reaches a
    user frame.

    The default skip set covers:

    * ``pydantic`` / ``pydantic_core`` — so warnings emitted from
      ``@model_validator`` callbacks escape pydantic's dispatch frames.
    * ``data_designer`` — so warnings emitted from a registry / model-config
      built deep inside a DataDesigner helper still attribute to the outermost
      user call. Without this, attribution lands on a library file and Python's
      default ``DeprecationWarning`` filter silences the warning entirely.

    The user frame's ``__warningregistry__`` is passed to
    ``warnings.warn_explicit`` so Python's built-in once-per-location dedup keys
    on the *user's* (filename, lineno) rather than an internal frame.

    We deliberately do *not* pass ``module_globals`` — it's only used for
    ``linecache`` source-line display, and for scripts run with ``python -c``
    (where the user frame's ``__loader__`` is ``BuiltinImporter`` for
    ``__main__``) the lookup raises ``ImportError("'__main__' is not a built-in
    module")``. Skipping ``module_globals`` keeps the warning path robust at
    the cost of an empty source line in the formatted output.
    """
    frame = sys._getframe(1) if hasattr(sys, "_getframe") else None
    while frame is not None:
        module_name = frame.f_globals.get("__name__", "")
        if not _module_in_prefixes(module_name, skip_prefixes):
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

    # Fallback: never escaped library frames (or no frame access). Use stacklevel.
    warnings.warn(message, category, stacklevel=3)
