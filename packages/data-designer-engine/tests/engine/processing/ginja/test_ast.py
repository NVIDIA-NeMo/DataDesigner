# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from jinja2 import Environment
from jinja2 import nodes as j_nodes

from data_designer.engine.processing.ginja.ast import (
    ast_count_name_references,
    ast_descendant_count,
    ast_extract_access_chains,
    ast_max_depth,
    resolve_access_chain,
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


# Use a real Jinja Environment for chain-extraction tests; mocking the AST
# directly would just re-implement the helper.
@pytest.fixture
def jinja_env():
    return Environment()


@pytest.mark.parametrize(
    "template,expected_chains",
    [
        ("plain text, no vars", []),
        ("{{ x }}", [("x", [])]),
        ("{{ person.first_name }}", [("person", ["first_name"])]),
        ("{{ person.address.street }}", [("person", ["address", "street"])]),
        ("{{ y['key']['sub'] }}", [("y", ["key", "sub"])]),
        ("{{ y[0] }}", [("y", [0])]),
        (
            "{{ a.b }} and {{ c.d.e }} and {{ z }}",
            [("a", ["b"]), ("c", ["d", "e"]), ("z", [])],
        ),
        # `a.b` repeated -> two entries, caller dedupes if needed
        ("{{ a.b }}{{ a.b }}", [("a", ["b"]), ("a", ["b"])]),
        # Mixed attr + subscript
        ("{{ data['rows'][0].name }}", [("data", ["rows", 0, "name"])]),
        # Dynamic subscripts skip the chain; the inner name is still extracted
        ("{{ a[b] }}", [("b", [])]),
        ("{{ a[b].c }}", [("b", [])]),
        # Loops + conditions still extract chains
        ("{% if person.active %}{{ person.name }}{% endif %}", [("person", ["active"]), ("person", ["name"])]),
    ],
)
def test_ast_extract_access_chains(jinja_env, template, expected_chains):
    ast = jinja_env.parse(template)
    assert ast_extract_access_chains(ast) == expected_chains


@pytest.mark.parametrize(
    "record,name,accessors,expected",
    [
        # Fully resolved cases
        ({"x": 42}, "x", [], (True, 42, [])),
        ({"x": None}, "x", [], (True, None, [])),
        ({"x": ""}, "x", [], (True, "", [])),
        ({"person": {"name": "John"}}, "person", ["name"], (True, "John", ["name"])),
        (
            {"person": {"address": {"street": "123 Main"}}},
            "person",
            ["address", "street"],
            (True, "123 Main", ["address", "street"]),
        ),
        ({"items": ["a", "b", "c"]}, "items", [1], (True, "b", [1])),
        # Failed resolution cases
        ({}, "missing", [], (False, None, [])),
        ({"x": {}}, "x", ["y"], (False, None, [])),
        ({"x": {"y": {}}}, "x", ["y", "z"], (False, None, ["y"])),
        # Type mismatches
        ({"x": "scalar"}, "x", ["y"], (False, None, [])),
        ({"x": [1, 2]}, "x", ["key"], (False, None, [])),
        ({"x": [1, 2]}, "x", [5], (False, None, [])),
        ({"x": {"y": 1}}, "x", [0], (False, None, [])),
    ],
)
def test_resolve_access_chain(record, name, accessors, expected):
    assert resolve_access_chain(record, name, accessors) == expected
