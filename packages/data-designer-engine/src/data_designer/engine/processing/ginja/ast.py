# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from typing import Any

from jinja2 import nodes as j_nodes

# An access chain is a (root_name, accessors) pair, where accessors is the
# ordered list of attribute names / subscript keys applied to the root.
# E.g. ``{{ person.address["street"] }}`` -> ``("person", ["address", "street"])``.
AccessChain = tuple[str, list[str | int]]


def ast_max_depth(node: j_nodes.Node) -> int:
    """Calculate the depth of a Jinja AST from a given node.

    Args:
        node (jinja2.nodes.Node): The starting Jinja2 AST node

    Returns:
        int: The maximum depth of the tree
    """
    # Each entry is (node, depth)
    queue = deque([(node, 1)])
    max_depth = 0

    while queue:
        current_node, current_depth = queue.popleft()

        # Update maximum depth seen so far
        max_depth = max(max_depth, current_depth)

        # Add all children with incremented depth
        for child in current_node.iter_child_nodes():
            queue.append((child, current_depth + 1))

    return max_depth


def ast_descendant_count(ast: j_nodes.Node, only_type: type[j_nodes.Node] | None = None) -> int:
    """Count the number of nodes which descend from the given node.

    Args:
        ast (jinja2.nodes.Node): The starting Jinja2 AST node
        only_type (Type[jinja2.nodes.Node]): If specified, then only
            nodes of this type will be counted.

    Returns:
        int: The number of nodes descended from the given node.
    """
    if only_type is None:
        only_type = j_nodes.Node

    return len(list(ast.find_all(only_type)))


def ast_count_name_references(ast: j_nodes.Node, name: str) -> int:
    """Count the number of nodes descended from the current that refer to name.

    Args:
        ast (jinja2.nodes.Node): The starting Jinja2 AST node

    Returns:
        int: The number of nodes descended from the provided node whose
            name field matches the given name.
    """
    referenced_names = [node.name for node in ast.find_all(j_nodes.Name) if node.name == name]
    return len(referenced_names)


def ast_extract_access_chains(root: j_nodes.Node) -> list[AccessChain]:
    """Extract every top-level access chain rooted at a named variable.

    Each output tuple is ``(root_name, accessors)`` where ``accessors``
    is the ordered list of attribute/subscript keys applied to the root.
    Top-level means the chain is not contained inside a longer chain,
    so ``{{ a.b.c }}`` yields ``("a", ["b", "c"])`` rather than three
    overlapping entries. Dynamic subscripts inside a chain (``a[b].c``)
    cause the chain to be skipped; the inner ``b`` is still extracted
    as its own chain.

    Args:
        root: A parsed Jinja2 AST node (typically the ``Template``).

    Returns:
        Every extractable access chain, in source-order. Duplicates are
        preserved so the caller can decide how to dedupe.
    """
    chains: list[AccessChain] = []

    def visit(node: j_nodes.Node, in_chain: bool) -> None:
        if isinstance(node, (j_nodes.Getattr, j_nodes.Getitem)):
            if not in_chain:
                chain = _build_access_chain(node)
                if chain is not None:
                    chains.append(chain)
            # Descend through ``.node`` as "in chain" so we don't re-emit
            # the prefixes ``a`` and ``a.b`` for ``{{ a.b.c }}``.
            visit(node.node, in_chain=True)
            if isinstance(node, j_nodes.Getitem):
                # The subscript expression is a separate scope and may
                # contain its own variable references.
                visit(node.arg, in_chain=False)
            return
        if isinstance(node, j_nodes.Name):
            if not in_chain:
                chains.append((node.name, []))
            return
        for child in node.iter_child_nodes():
            visit(child, in_chain=False)

    visit(root, in_chain=False)
    return chains


def resolve_access_chain(record: dict, name: str, accessors: list[str | int]) -> tuple[bool, Any, list[str | int]]:
    """Walk an access chain against a record dict.

    Args:
        record: The sanitized record dict that would be used as template
            context. Values are expected to be JSON-compatible types.
        name: Root variable name.
        accessors: Ordered attribute names / subscript keys to apply.

    Returns:
        A tuple ``(resolved, value, prefix)``:
            - ``resolved`` is ``True`` iff every accessor matched.
            - ``value`` is the final value when ``resolved`` is True,
              otherwise ``None``.
            - ``prefix`` is the longest accessor list that did match.
              When ``resolved`` is False, the next accessor (``accessors
              [len(prefix)]``) is the one that broke the chain.
    """
    if name not in record:
        return (False, None, [])
    current: Any = record[name]
    prefix: list[str | int] = []
    for acc in accessors:
        if isinstance(current, dict):
            if not isinstance(acc, str) or acc not in current:
                return (False, None, prefix)
            current = current[acc]
        elif isinstance(current, list):
            if not isinstance(acc, int):
                return (False, None, prefix)
            if acc >= len(current) or acc < -len(current):
                return (False, None, prefix)
            current = current[acc]
        else:
            # The chain wants to go deeper but the value is a scalar.
            return (False, None, prefix)
        prefix.append(acc)
    return (True, current, prefix)


def _build_access_chain(node: j_nodes.Node) -> AccessChain | None:
    """Reduce a Getattr/Getitem/Name node to a top-level access chain.

    Walks down the ``.node`` field of nested ``Getattr``/``Getitem`` nodes
    until reaching the root ``Name``. Returns ``None`` if the chain is
    rooted in something other than a ``Name`` (e.g. a function call) or
    if a ``Getitem`` uses a non-constant subscript expression (e.g.
    ``a[b]`` where ``b`` is itself a variable).
    """
    accessors: list[str | int] = []
    current: j_nodes.Node = node
    while True:
        if isinstance(current, j_nodes.Getattr):
            accessors.append(current.attr)
            current = current.node
        elif isinstance(current, j_nodes.Getitem):
            arg = current.arg
            if isinstance(arg, j_nodes.Const) and isinstance(arg.value, (str, int)):
                accessors.append(arg.value)
                current = current.node
            else:
                # Dynamic subscript like ``a[b]`` -- not a fixed access chain.
                return None
        elif isinstance(current, j_nodes.Name):
            return (current.name, list(reversed(accessors)))
        else:
            return None
