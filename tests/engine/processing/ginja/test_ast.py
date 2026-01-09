# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from jinja2 import nodes as j_nodes

from data_designer.engine.processing.ginja.ast import (
    ast_count_name_references,
    ast_descendant_count,
    ast_max_depth,
)


@pytest.fixture
def stub_node():
    return Mock(spec=j_nodes.Node)


@pytest.fixture
def stub_name_node():
    return Mock(spec=j_nodes.Name)


@pytest.mark.parametrize(
    "test_case,children_structure,expected_depth",
    [
        ("single_node", [], 1),
        ("two_levels", [[Mock(spec=j_nodes.Node)]], 2),
        ("three_levels", [[Mock(spec=j_nodes.Node)], [Mock(spec=j_nodes.Node)]], 3),
        ("unbalanced_tree", [[Mock(spec=j_nodes.Node)]], 3),
        ("empty_tree", [], 1),
    ],
)
def test_ast_max_depth(stub_node, test_case, children_structure, expected_depth):
    if test_case == "three_levels":
        root = Mock(spec=j_nodes.Node)
        child1 = Mock(spec=j_nodes.Node)
        child2 = Mock(spec=j_nodes.Node)
        grandchild = Mock(spec=j_nodes.Node)

        grandchild.iter_child_nodes.return_value = []
        child1.iter_child_nodes.return_value = [grandchild]
        child2.iter_child_nodes.return_value = []
        root.iter_child_nodes.return_value = [child1, child2]

        result = ast_max_depth(root)
        assert result == expected_depth
    elif test_case == "unbalanced_tree":
        root = Mock(spec=j_nodes.Node)
        child1 = Mock(spec=j_nodes.Node)
        child2 = Mock(spec=j_nodes.Node)
        grandchild = Mock(spec=j_nodes.Node)

        grandchild.iter_child_nodes.return_value = []
        child1.iter_child_nodes.return_value = [grandchild]
        child2.iter_child_nodes.return_value = []
        root.iter_child_nodes.return_value = [child1, child2]

        result = ast_max_depth(root)
        assert result == expected_depth
    else:
        if test_case == "two_levels":
            child = Mock(spec=j_nodes.Node)
            child.iter_child_nodes.return_value = []
            stub_node.iter_child_nodes.return_value = [child]
        else:
            stub_node.iter_child_nodes.return_value = children_structure

        result = ast_max_depth(stub_node)
        assert result == expected_depth


@pytest.mark.parametrize(
    "test_case,find_all_return,only_type,expected_count,expected_call",
    [
        ("single_node", [Mock()], None, 1, j_nodes.Node),
        ("multiple_nodes", [Mock(), Mock(), Mock()], None, 3, j_nodes.Node),
        ("with_type_filter", [Mock(), Mock()], j_nodes.Name, 2, j_nodes.Name),
        ("with_none_type_filter", [Mock(), Mock(), Mock()], None, 3, j_nodes.Node),
        ("empty_tree", [], None, 0, j_nodes.Node),
    ],
)
def test_ast_descendant_count(stub_node, test_case, find_all_return, only_type, expected_count, expected_call):
    stub_node.find_all.return_value = find_all_return

    if only_type is None:
        result = ast_descendant_count(stub_node)
    else:
        result = ast_descendant_count(stub_node, only_type=only_type)

    assert result == expected_count
    stub_node.find_all.assert_called_once_with(expected_call)


@pytest.mark.parametrize(
    "test_case,name_nodes,search_name,expected_count",
    [
        ("single_reference", ["test_name"], "test_name", 1),
        ("multiple_references", ["test_name", "test_name", "other_name"], "test_name", 2),
        ("no_references", ["other_name"], "test_name", 0),
        ("empty_tree", [], "test_name", 0),
        ("case_sensitive", ["Test_Name", "test_name"], "test_name", 1),
        ("with_non_name_nodes", ["test_name"], "test_name", 1),
        ("empty_name", [""], "", 1),
    ],
)
def test_ast_count_name_references(stub_node, stub_name_node, test_case, name_nodes, search_name, expected_count):
    def mock_find_all(node_type):
        if node_type == j_nodes.Name:
            mock_nodes = []
            for name in name_nodes:
                mock_name_node = Mock(spec=j_nodes.Name)
                mock_name_node.name = name
                mock_nodes.append(mock_name_node)
            return mock_nodes
        return []

    stub_node.find_all.side_effect = mock_find_all

    result = ast_count_name_references(stub_node, search_name)

    assert result == expected_count
