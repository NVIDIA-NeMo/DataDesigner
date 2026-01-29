# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for execution graph."""

import pytest

from data_designer.engine.execution_graph.completion import CompletionTracker
from data_designer.engine.execution_graph.graph import ExecutionGraph
from data_designer.engine.execution_graph.node_id import BarrierNodeId, CellNodeId
from data_designer.engine.execution_graph.traits import ExecutionTraits

from .conftest import create_descriptor  # noqa: TID252

# --- Init tests ---


def test_init_with_valid_params() -> None:
    desc = create_descriptor("col", ExecutionTraits.START)
    graph = ExecutionGraph(num_records=10, column_descriptors=[desc])

    assert graph.num_records == 10
    assert graph.num_columns == 1


def test_init_rejects_zero_records() -> None:
    desc = create_descriptor("col", ExecutionTraits.START)
    with pytest.raises(ValueError, match="num_records must be positive"):
        ExecutionGraph(num_records=0, column_descriptors=[desc])


def test_init_rejects_negative_records() -> None:
    desc = create_descriptor("col", ExecutionTraits.START)
    with pytest.raises(ValueError, match="num_records must be positive"):
        ExecutionGraph(num_records=-5, column_descriptors=[desc])


def test_init_rejects_empty_descriptors() -> None:
    with pytest.raises(ValueError, match="column_descriptors cannot be empty"):
        ExecutionGraph(num_records=10, column_descriptors=[])


def test_init_validates_dependencies_by_default() -> None:
    desc = create_descriptor("col", ExecutionTraits.NONE, dependencies=["nonexistent"])
    with pytest.raises(ValueError, match="depends on unknown column"):
        ExecutionGraph(num_records=10, column_descriptors=[desc])


def test_init_skips_validation_when_strict_false() -> None:
    desc = create_descriptor("col", ExecutionTraits.START, dependencies=["nonexistent"])
    # Should not raise
    graph = ExecutionGraph(num_records=10, column_descriptors=[desc], strict=False)
    assert graph.num_columns == 1


# --- Properties tests ---


def test_num_nodes_without_barriers() -> None:
    descriptors = [
        create_descriptor("a", ExecutionTraits.START | ExecutionTraits.ROW_STREAMABLE),
        create_descriptor("b", ExecutionTraits.CELL_BY_CELL | ExecutionTraits.ROW_STREAMABLE, ["a"]),
    ]
    graph = ExecutionGraph(num_records=100, column_descriptors=descriptors)
    # 2 columns × 100 records = 200 nodes
    assert graph.num_nodes == 200


def test_num_nodes_with_barriers() -> None:
    descriptors = [
        create_descriptor("a", ExecutionTraits.START | ExecutionTraits.ROW_STREAMABLE),
        create_descriptor("b", ExecutionTraits.BARRIER, ["a"]),
    ]
    graph = ExecutionGraph(num_records=100, column_descriptors=descriptors)
    # 2 columns × 100 records + 1 barrier = 201 nodes
    assert graph.num_nodes == 201


def test_start_columns() -> None:
    descriptors = [
        create_descriptor("starter", ExecutionTraits.START),
        create_descriptor("dependent", ExecutionTraits.CELL_BY_CELL, ["starter"]),
    ]
    graph = ExecutionGraph(num_records=10, column_descriptors=descriptors)
    assert graph.start_columns == ["starter"]


def test_column_names() -> None:
    descriptors = [
        create_descriptor("a", ExecutionTraits.START),
        create_descriptor("b", ExecutionTraits.CELL_BY_CELL, ["a"]),
        create_descriptor("c", ExecutionTraits.CELL_BY_CELL, ["b"]),
    ]
    graph = ExecutionGraph(num_records=10, column_descriptors=descriptors)
    assert graph.column_names == ["a", "b", "c"]


# --- Dependencies tests ---


def test_start_column_no_dependencies() -> None:
    desc = create_descriptor("start", ExecutionTraits.START)
    graph = ExecutionGraph(num_records=10, column_descriptors=[desc])

    deps = graph.get_dependencies(CellNodeId(0, "start"))
    assert deps == []


def test_cell_by_cell_row_local_dependencies() -> None:
    descriptors = [
        create_descriptor("a", ExecutionTraits.START | ExecutionTraits.ROW_STREAMABLE),
        create_descriptor(
            "b",
            ExecutionTraits.CELL_BY_CELL | ExecutionTraits.ROW_STREAMABLE,
            ["a"],
        ),
    ]
    graph = ExecutionGraph(num_records=10, column_descriptors=descriptors)

    # Row 5 of column b depends on row 5 of column a
    deps = graph.get_dependencies(CellNodeId(5, "b"))
    assert deps == [CellNodeId(5, "a")]


def test_barrier_node_depends_on_all_cells() -> None:
    descriptors = [
        create_descriptor("a", ExecutionTraits.START | ExecutionTraits.ROW_STREAMABLE),
        create_descriptor("b", ExecutionTraits.BARRIER, ["a"]),
    ]
    graph = ExecutionGraph(num_records=3, column_descriptors=descriptors)

    # Barrier depends on all cells of dependency column
    barrier_deps = graph.get_dependencies(BarrierNodeId("b"))
    expected = [CellNodeId(0, "a"), CellNodeId(1, "a"), CellNodeId(2, "a")]
    assert barrier_deps == expected


def test_barrier_cell_depends_on_barrier() -> None:
    descriptors = [
        create_descriptor("a", ExecutionTraits.START | ExecutionTraits.ROW_STREAMABLE),
        create_descriptor("b", ExecutionTraits.BARRIER, ["a"]),
    ]
    graph = ExecutionGraph(num_records=10, column_descriptors=descriptors)

    # Cell nodes of barrier column depend on the barrier node
    deps = graph.get_dependencies(CellNodeId(5, "b"))
    assert deps == [BarrierNodeId("b")]


def test_dependency_on_side_effect_column() -> None:
    descriptors = [
        create_descriptor(
            "person_name",
            ExecutionTraits.START | ExecutionTraits.ROW_STREAMABLE,
            side_effects=["person_email"],
        ),
        create_descriptor(
            "greeting",
            ExecutionTraits.CELL_BY_CELL | ExecutionTraits.ROW_STREAMABLE,
            ["person_email"],  # Depends on side effect, not primary
        ),
    ]
    graph = ExecutionGraph(num_records=10, column_descriptors=descriptors)

    # Should resolve to the parent column
    deps = graph.get_dependencies(CellNodeId(5, "greeting"))
    assert deps == [CellNodeId(5, "person_name")]


# --- Dependents tests ---


def test_get_dependents_of_start_cell() -> None:
    descriptors = [
        create_descriptor("a", ExecutionTraits.START | ExecutionTraits.ROW_STREAMABLE),
        create_descriptor(
            "b",
            ExecutionTraits.CELL_BY_CELL | ExecutionTraits.ROW_STREAMABLE,
            ["a"],
        ),
    ]
    graph = ExecutionGraph(num_records=10, column_descriptors=descriptors)

    # Row 5 of column a should have row 5 of column b as dependent
    dependents = graph.get_dependents(CellNodeId(5, "a"))
    assert CellNodeId(5, "b") in dependents


def test_get_dependents_of_barrier() -> None:
    descriptors = [
        create_descriptor("a", ExecutionTraits.START | ExecutionTraits.ROW_STREAMABLE),
        create_descriptor("b", ExecutionTraits.BARRIER, ["a"]),
    ]
    graph = ExecutionGraph(num_records=3, column_descriptors=descriptors)

    # Barrier completion triggers all cells of that column
    dependents = graph.get_dependents(BarrierNodeId("b"))
    expected = [CellNodeId(0, "b"), CellNodeId(1, "b"), CellNodeId(2, "b")]
    assert dependents == expected


# --- Iteration tests ---


def test_iter_nodes() -> None:
    descriptors = [
        create_descriptor("a", ExecutionTraits.START | ExecutionTraits.ROW_STREAMABLE),
        create_descriptor(
            "b",
            ExecutionTraits.CELL_BY_CELL | ExecutionTraits.ROW_STREAMABLE,
            ["a"],
        ),
    ]
    graph = ExecutionGraph(num_records=3, column_descriptors=descriptors)

    nodes = list(graph.iter_nodes())
    # 2 columns × 3 records = 6 nodes
    assert len(nodes) == 6

    # First column's cells come first
    assert nodes[0] == CellNodeId(0, "a")
    assert nodes[1] == CellNodeId(1, "a")
    assert nodes[2] == CellNodeId(2, "a")

    # Second column's cells come next
    assert nodes[3] == CellNodeId(0, "b")
    assert nodes[4] == CellNodeId(1, "b")
    assert nodes[5] == CellNodeId(2, "b")


def test_iter_nodes_with_barrier() -> None:
    descriptors = [
        create_descriptor("a", ExecutionTraits.START | ExecutionTraits.ROW_STREAMABLE),
        create_descriptor("b", ExecutionTraits.BARRIER, ["a"]),
    ]
    graph = ExecutionGraph(num_records=2, column_descriptors=descriptors)

    nodes = list(graph.iter_nodes())
    # Column a: 2 cells
    # Column b: 1 barrier + 2 cells
    assert len(nodes) == 5

    # Barrier comes before cells of barrier column
    assert nodes[2] == BarrierNodeId("b")
    assert nodes[3] == CellNodeId(0, "b")
    assert nodes[4] == CellNodeId(1, "b")


def test_iter_start_nodes() -> None:
    descriptors = [
        create_descriptor("starter", ExecutionTraits.START | ExecutionTraits.ROW_STREAMABLE),
        create_descriptor(
            "dependent",
            ExecutionTraits.CELL_BY_CELL | ExecutionTraits.ROW_STREAMABLE,
            ["starter"],
        ),
    ]
    graph = ExecutionGraph(num_records=3, column_descriptors=descriptors)

    start_nodes = list(graph.iter_start_nodes())
    assert len(start_nodes) == 3
    assert all(node.column == "starter" for node in start_nodes)


def test_iter_ready_nodes_initial() -> None:
    descriptors = [
        create_descriptor("a", ExecutionTraits.START | ExecutionTraits.ROW_STREAMABLE),
        create_descriptor(
            "b",
            ExecutionTraits.CELL_BY_CELL | ExecutionTraits.ROW_STREAMABLE,
            ["a"],
        ),
    ]
    graph = ExecutionGraph(num_records=3, column_descriptors=descriptors)
    tracker = CompletionTracker(3)

    # Initially, only start column cells are ready
    ready = list(graph.iter_ready_nodes(tracker))
    assert len(ready) == 3
    assert all(node.column == "a" for node in ready)


def test_iter_ready_nodes_after_completion() -> None:
    descriptors = [
        create_descriptor("a", ExecutionTraits.START | ExecutionTraits.ROW_STREAMABLE),
        create_descriptor(
            "b",
            ExecutionTraits.CELL_BY_CELL | ExecutionTraits.ROW_STREAMABLE,
            ["a"],
        ),
    ]
    graph = ExecutionGraph(num_records=3, column_descriptors=descriptors)
    tracker = CompletionTracker(3)

    # Complete row 0 of column a
    tracker.mark_complete(CellNodeId(0, "a"))

    ready = list(graph.iter_ready_nodes(tracker))
    # Remaining rows 1, 2 of column a are still ready
    # Row 0 of column b is now also ready
    assert CellNodeId(1, "a") in ready
    assert CellNodeId(2, "a") in ready
    assert CellNodeId(0, "b") in ready
    assert CellNodeId(0, "a") not in ready  # Already completed


def test_iter_ready_nodes_for_column() -> None:
    descriptors = [
        create_descriptor("a", ExecutionTraits.START | ExecutionTraits.ROW_STREAMABLE),
        create_descriptor(
            "b",
            ExecutionTraits.CELL_BY_CELL | ExecutionTraits.ROW_STREAMABLE,
            ["a"],
        ),
    ]
    graph = ExecutionGraph(num_records=3, column_descriptors=descriptors)
    tracker = CompletionTracker(3)

    # Complete row 0 of column a
    tracker.mark_complete(CellNodeId(0, "a"))

    # Check only column b
    ready_b = list(graph.iter_ready_nodes_for_column("b", tracker))
    assert ready_b == [CellNodeId(0, "b")]


# --- Completion tests ---


def test_is_complete_empty_tracker() -> None:
    desc = create_descriptor("a", ExecutionTraits.START | ExecutionTraits.ROW_STREAMABLE)
    graph = ExecutionGraph(num_records=10, column_descriptors=[desc])
    tracker = CompletionTracker(10)

    assert not graph.is_complete(tracker)


def test_is_complete_all_done() -> None:
    desc = create_descriptor("a", ExecutionTraits.START | ExecutionTraits.ROW_STREAMABLE)
    graph = ExecutionGraph(num_records=3, column_descriptors=[desc])
    tracker = CompletionTracker(3)

    for r in range(3):
        tracker.mark_complete(CellNodeId(r, "a"))

    assert graph.is_complete(tracker)


# --- Getter tests ---


def test_get_column_descriptor() -> None:
    desc = create_descriptor("my_col", ExecutionTraits.START)
    graph = ExecutionGraph(num_records=10, column_descriptors=[desc])

    retrieved = graph.get_column_descriptor("my_col")
    assert retrieved.name == "my_col"


def test_get_column_descriptor_not_found() -> None:
    desc = create_descriptor("my_col", ExecutionTraits.START)
    graph = ExecutionGraph(num_records=10, column_descriptors=[desc])

    with pytest.raises(KeyError):
        graph.get_column_descriptor("nonexistent")


def test_get_generator_and_config() -> None:
    desc = create_descriptor("my_col", ExecutionTraits.START)
    graph = ExecutionGraph(num_records=10, column_descriptors=[desc])

    gen_cls, config = graph.get_generator_and_config(CellNodeId(0, "my_col"))
    assert gen_cls is desc.generator_cls
    assert config is desc.config
