# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for node identification types."""

from data_designer.engine.execution_graph.node_id import BarrierNodeId, CellNodeId, NodeId

# --- CellNodeId tests ---


def test_cell_node_id_creation() -> None:
    node = CellNodeId(row=5, column="test_col")
    assert node.row == 5
    assert node.column == "test_col"


def test_cell_node_id_frozen() -> None:
    node = CellNodeId(row=0, column="col")
    try:
        node.row = 1  # type: ignore[misc]
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass  # Expected


def test_cell_node_id_equality() -> None:
    node1 = CellNodeId(row=0, column="col")
    node2 = CellNodeId(row=0, column="col")
    node3 = CellNodeId(row=1, column="col")
    node4 = CellNodeId(row=0, column="other")

    assert node1 == node2
    assert node1 != node3
    assert node1 != node4


def test_cell_node_id_hashable() -> None:
    node1 = CellNodeId(row=0, column="col")
    node2 = CellNodeId(row=0, column="col")
    node_set = {node1, node2}
    assert len(node_set) == 1


def test_cell_node_id_repr() -> None:
    node = CellNodeId(row=5, column="test_col")
    assert repr(node) == "Cell(5, 'test_col')"


# --- BarrierNodeId tests ---


def test_barrier_node_id_creation() -> None:
    node = BarrierNodeId(column="barrier_col")
    assert node.column == "barrier_col"


def test_barrier_node_id_frozen() -> None:
    node = BarrierNodeId(column="col")
    try:
        node.column = "other"  # type: ignore[misc]
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass  # Expected


def test_barrier_node_id_equality() -> None:
    node1 = BarrierNodeId(column="col")
    node2 = BarrierNodeId(column="col")
    node3 = BarrierNodeId(column="other")

    assert node1 == node2
    assert node1 != node3


def test_barrier_node_id_hashable() -> None:
    node1 = BarrierNodeId(column="col")
    node2 = BarrierNodeId(column="col")
    node_set = {node1, node2}
    assert len(node_set) == 1


def test_barrier_node_id_repr() -> None:
    node = BarrierNodeId(column="barrier_col")
    assert repr(node) == "Barrier('barrier_col')"


# --- NodeId type alias tests ---


def test_node_id_can_be_cell_or_barrier() -> None:
    cell: NodeId = CellNodeId(row=0, column="col")
    barrier: NodeId = BarrierNodeId(column="col")

    assert isinstance(cell, CellNodeId)
    assert isinstance(barrier, BarrierNodeId)


def test_cell_and_barrier_not_equal() -> None:
    cell = CellNodeId(row=0, column="col")
    barrier = BarrierNodeId(column="col")
    assert cell != barrier
