# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Column descriptor for the execution graph.

This module defines the ColumnDescriptor dataclass that stores metadata about
a column in the execution graph, including its configuration, generator class,
execution traits, and dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from data_designer.engine.execution_graph.traits import ExecutionTraits

if TYPE_CHECKING:
    from data_designer.config.base import ConfigBase
    from data_designer.engine.column_generators.generators.base import ColumnGenerator


@dataclass(slots=True)
class ColumnDescriptor:
    """Metadata describing a column in the execution graph.

    Attributes:
        name: The primary column name (for multi-column configs, this is the first column).
        config: The column configuration object.
        generator_cls: The generator class to use for this column.
        traits: Execution traits inferred from the generator.
        dependencies: List of column names this column depends on (from required_columns).
        side_effects: List of additional column names this generator produces.
    """

    name: str
    config: ConfigBase
    generator_cls: type[ColumnGenerator]
    traits: ExecutionTraits
    dependencies: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)

    @property
    def is_start_column(self) -> bool:
        """Whether this column can generate data from scratch."""
        return bool(self.traits & ExecutionTraits.START)

    @property
    def is_cell_by_cell(self) -> bool:
        """Whether this column processes individual cells independently."""
        return bool(self.traits & ExecutionTraits.CELL_BY_CELL)

    @property
    def is_row_streamable(self) -> bool:
        """Whether this column can emit results as rows complete."""
        return bool(self.traits & ExecutionTraits.ROW_STREAMABLE)

    @property
    def is_barrier(self) -> bool:
        """Whether this column requires all inputs before producing any output."""
        return bool(self.traits & ExecutionTraits.BARRIER)

    @property
    def has_dependencies(self) -> bool:
        """Whether this column has any dependencies."""
        return len(self.dependencies) > 0

    @property
    def has_side_effects(self) -> bool:
        """Whether this column produces additional columns as side effects."""
        return len(self.side_effects) > 0

    @property
    def all_produced_columns(self) -> list[str]:
        """All column names produced by this generator (primary + side effects)."""
        return [self.name, *self.side_effects]
