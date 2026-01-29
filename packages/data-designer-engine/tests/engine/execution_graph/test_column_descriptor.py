# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for column descriptor."""

from data_designer.engine.execution_graph.traits import ExecutionTraits

from .conftest import create_descriptor  # noqa: TID252


def test_is_start_column() -> None:
    start = create_descriptor("col", ExecutionTraits.START)
    non_start = create_descriptor("col", ExecutionTraits.CELL_BY_CELL)

    assert start.is_start_column
    assert not non_start.is_start_column


def test_is_cell_by_cell() -> None:
    cell = create_descriptor("col", ExecutionTraits.CELL_BY_CELL)
    non_cell = create_descriptor("col", ExecutionTraits.BARRIER)

    assert cell.is_cell_by_cell
    assert not non_cell.is_cell_by_cell


def test_is_row_streamable() -> None:
    streamable = create_descriptor("col", ExecutionTraits.ROW_STREAMABLE)
    non_streamable = create_descriptor("col", ExecutionTraits.BARRIER)

    assert streamable.is_row_streamable
    assert not non_streamable.is_row_streamable


def test_is_barrier() -> None:
    barrier = create_descriptor("col", ExecutionTraits.BARRIER)
    non_barrier = create_descriptor("col", ExecutionTraits.CELL_BY_CELL)

    assert barrier.is_barrier
    assert not non_barrier.is_barrier


def test_has_dependencies() -> None:
    with_deps = create_descriptor("col", ExecutionTraits.NONE, dependencies=["dep1"])
    without_deps = create_descriptor("col", ExecutionTraits.NONE)

    assert with_deps.has_dependencies
    assert not without_deps.has_dependencies


def test_has_side_effects() -> None:
    with_effects = create_descriptor("col", ExecutionTraits.NONE, side_effects=["effect1"])
    without_effects = create_descriptor("col", ExecutionTraits.NONE)

    assert with_effects.has_side_effects
    assert not without_effects.has_side_effects


def test_all_produced_columns() -> None:
    without_effects = create_descriptor("primary", ExecutionTraits.NONE)
    with_effects = create_descriptor(
        "primary",
        ExecutionTraits.NONE,
        side_effects=["secondary", "tertiary"],
    )

    assert without_effects.all_produced_columns == ["primary"]
    assert with_effects.all_produced_columns == ["primary", "secondary", "tertiary"]


def test_combined_traits() -> None:
    combined = create_descriptor(
        "col",
        ExecutionTraits.START | ExecutionTraits.ROW_STREAMABLE,
    )

    assert combined.is_start_column
    assert combined.is_row_streamable
    assert not combined.is_barrier
    assert not combined.is_cell_by_cell
