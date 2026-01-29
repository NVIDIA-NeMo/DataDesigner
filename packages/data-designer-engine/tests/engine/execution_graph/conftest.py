# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for execution graph tests."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from data_designer.engine.execution_graph.column_descriptor import ColumnDescriptor
from data_designer.engine.execution_graph.traits import ExecutionTraits

if TYPE_CHECKING:
    from data_designer.config.base import ConfigBase
    from data_designer.engine.column_generators.generators.base import ColumnGenerator


def create_mock_config(name: str) -> ConfigBase:
    """Create a mock config with a name attribute."""
    mock = Mock()
    mock.name = name
    mock.required_columns = []
    mock.side_effect_columns = []
    return mock


def create_mock_generator() -> type[ColumnGenerator]:
    """Create a mock generator class."""
    return Mock()


def create_descriptor(
    name: str,
    traits: ExecutionTraits,
    dependencies: list[str] | None = None,
    side_effects: list[str] | None = None,
) -> ColumnDescriptor:
    """Helper to create column descriptors for testing."""
    return ColumnDescriptor(
        name=name,
        config=create_mock_config(name),
        generator_cls=create_mock_generator(),
        traits=traits,
        dependencies=dependencies or [],
        side_effects=side_effects or [],
    )


@pytest.fixture
def start_descriptor() -> ColumnDescriptor:
    """A descriptor for a start column (can generate from scratch)."""
    return create_descriptor(
        name="category",
        traits=ExecutionTraits.START | ExecutionTraits.ROW_STREAMABLE,
    )


@pytest.fixture
def cell_by_cell_descriptor() -> ColumnDescriptor:
    """A descriptor for a cell-by-cell column with a dependency."""
    return create_descriptor(
        name="question",
        traits=ExecutionTraits.CELL_BY_CELL | ExecutionTraits.ROW_STREAMABLE,
        dependencies=["category"],
    )


@pytest.fixture
def barrier_descriptor() -> ColumnDescriptor:
    """A descriptor for a barrier column."""
    return create_descriptor(
        name="validation",
        traits=ExecutionTraits.BARRIER,
        dependencies=["question"],
    )


@pytest.fixture
def multi_column_descriptor() -> ColumnDescriptor:
    """A descriptor for a multi-column generator with side effects."""
    return create_descriptor(
        name="person_name",
        traits=ExecutionTraits.START | ExecutionTraits.ROW_STREAMABLE,
        side_effects=["person_email", "person_phone"],
    )
