# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for execution traits."""

from data_designer.engine.execution_graph.traits import ExecutionTraits


def test_none_trait() -> None:
    traits = ExecutionTraits.NONE
    assert traits == ExecutionTraits.NONE
    assert not bool(traits)


def test_individual_traits() -> None:
    assert ExecutionTraits.START != ExecutionTraits.NONE
    assert ExecutionTraits.CELL_BY_CELL != ExecutionTraits.NONE
    assert ExecutionTraits.ROW_STREAMABLE != ExecutionTraits.NONE
    assert ExecutionTraits.BARRIER != ExecutionTraits.NONE


def test_traits_are_distinct() -> None:
    traits = [
        ExecutionTraits.START,
        ExecutionTraits.CELL_BY_CELL,
        ExecutionTraits.ROW_STREAMABLE,
        ExecutionTraits.BARRIER,
    ]
    # Each trait should be distinct
    for i, t1 in enumerate(traits):
        for j, t2 in enumerate(traits):
            if i != j:
                assert t1 != t2


def test_trait_combination() -> None:
    combined = ExecutionTraits.START | ExecutionTraits.ROW_STREAMABLE
    assert combined & ExecutionTraits.START
    assert combined & ExecutionTraits.ROW_STREAMABLE
    assert not (combined & ExecutionTraits.BARRIER)
    assert not (combined & ExecutionTraits.CELL_BY_CELL)


def test_trait_check_with_bool() -> None:
    traits = ExecutionTraits.START | ExecutionTraits.CELL_BY_CELL

    assert bool(traits & ExecutionTraits.START)
    assert bool(traits & ExecutionTraits.CELL_BY_CELL)
    assert not bool(traits & ExecutionTraits.BARRIER)


def test_cell_by_cell_implies_row_streamable() -> None:
    # This is a common pattern: cell-by-cell generators are always row-streamable
    traits = ExecutionTraits.CELL_BY_CELL | ExecutionTraits.ROW_STREAMABLE
    assert bool(traits & ExecutionTraits.ROW_STREAMABLE)


def test_start_with_barrier() -> None:
    # A generator can start from scratch but still be a barrier
    traits = ExecutionTraits.START | ExecutionTraits.BARRIER
    assert bool(traits & ExecutionTraits.START)
    assert bool(traits & ExecutionTraits.BARRIER)
