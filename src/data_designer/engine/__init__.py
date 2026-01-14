# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Engine module with fully automatic lazy loading.

This module automatically discovers ALL engine modules and their public classes/functions,
providing a facade that lazily imports components only when accessed. This significantly
improves import performance while requiring ZERO maintenance - just add a module and it's
automatically exported.

Note: Private modules (starting with _) are excluded from auto-discovery.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path


def _discover_all_engine_exports() -> dict[str, tuple[str, str]]:
    """Automatically discover all public classes/functions in the engine package.

    Scans the engine directory recursively for all Python files, parses them
    with AST (without importing), and builds a mapping of all public exports.

    Returns:
        Dictionary mapping public names to (module_path, attribute_name) tuples.
    """
    lazy_imports = {}
    engine_dir = Path(__file__).parent

    # Find all Python files in engine directory recursively
    for py_file in engine_dir.rglob("*.py"):
        # Skip __init__.py files and private modules (starting with _)
        if py_file.name.startswith("_"):
            continue

        # Convert file path to module path
        # e.g., dataset_builders/column_wise_builder.py -> data_designer.engine.dataset_builders.column_wise_builder
        rel_path = py_file.relative_to(engine_dir.parent)
        module_parts = list(rel_path.parts[:-1]) + [rel_path.stem]
        module_path = ".".join(["data_designer"] + module_parts)

        try:
            # Parse the Python file with AST (doesn't import it - fast!)
            with open(py_file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(py_file))

            # Find all top-level public classes and functions
            for node in tree.body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = node.name
                    # Only export public items (no leading underscore)
                    if not name.startswith("_"):
                        # Avoid name collisions - first one wins
                        if name not in lazy_imports:
                            lazy_imports[name] = (module_path, name)
        except Exception:
            # If AST parsing fails, skip this module silently
            pass

    return lazy_imports


# Cache for lazy imports - built on first access
_LAZY_IMPORTS_CACHE: dict[str, tuple[str, str]] | None = None


def __getattr__(name: str) -> object:
    """Lazily import engine components when accessed.

    On first access, automatically discovers all public classes/functions in the
    engine package. Subsequent accesses use the cached mapping for fast lookups.

    Args:
        name: The name of the attribute to import.

    Returns:
        The imported class, function, or object.

    Raises:
        AttributeError: If the attribute is not found in any engine module.
    """
    global _LAZY_IMPORTS_CACHE

    # Build cache on first access
    if _LAZY_IMPORTS_CACHE is None:
        _LAZY_IMPORTS_CACHE = _discover_all_engine_exports()

    if name in _LAZY_IMPORTS_CACHE:
        module_path, attr_name = _LAZY_IMPORTS_CACHE[name]
        # Dynamically import the module
        module = importlib.import_module(module_path)
        # Get the attribute from the module
        return getattr(module, attr_name)

    raise AttributeError(f"module 'data_designer.engine' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return list of all available lazy imports for introspection."""
    global _LAZY_IMPORTS_CACHE

    # Build cache if not already built
    if _LAZY_IMPORTS_CACHE is None:
        _LAZY_IMPORTS_CACHE = _discover_all_engine_exports()

    return list(_LAZY_IMPORTS_CACHE.keys())
