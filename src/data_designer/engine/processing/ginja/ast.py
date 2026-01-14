# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque

from jinja2 import nodes as j_nodes


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
