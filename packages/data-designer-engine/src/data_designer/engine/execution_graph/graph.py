# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution graph for async dataset generation.

This module provides the ExecutionGraph class, which implements a hybrid
representation where column structure is stored explicitly while cell-level
nodes are virtual/computed on-demand to handle millions of records efficiently.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from data_designer.engine.execution_graph.column_descriptor import ColumnDescriptor
from data_designer.engine.execution_graph.node_id import BarrierNodeId, CellNodeId, NodeId

if TYPE_CHECKING:
    from data_designer.config.base import ConfigBase
    from data_designer.engine.column_generators.generators.base import ColumnGenerator


@runtime_checkable
class CompletionTrackerProtocol(Protocol):
    """Protocol for completion tracking - enables duck typing with any tracker implementation."""

    def is_complete(self, node: NodeId) -> bool:
        """Check if a node is completed."""
        ...

    def __contains__(self, node: NodeId) -> bool:
        """Support `node in tracker` syntax."""
        ...


class ExecutionGraph:
    """Execution graph for async dataset generation with hybrid representation.

    The ExecutionGraph models cell-level dependencies for dataset generation while
    maintaining memory efficiency through a hybrid representation:

    - **Explicit**: Column structure (ColumnDescriptors) is stored in memory
    - **Virtual**: Cell-level nodes are computed on-demand, not stored

    This allows handling datasets with millions of records without creating
    millions of explicit nodes and edges.

    The graph supports different execution traits:
    - START: Columns that can generate data from scratch (no dependencies)
    - CELL_BY_CELL: Columns that process individual cells independently
    - ROW_STREAMABLE: Columns that can emit results as rows complete
    - BARRIER: Columns that require all input rows before producing any output

    Attributes:
        num_records: The number of records (rows) in the dataset.
        num_columns: The number of columns in the dataset.
        num_nodes: The total number of virtual nodes (cells + barriers).

    Examples:
        >>> graph = ExecutionGraph(num_records=1000, column_descriptors=[...])
        >>> for node in graph.iter_start_nodes():
        ...     print(node)  # Cell(0, 'category'), Cell(1, 'category'), ...
        >>> deps = graph.get_dependencies(CellNodeId(5, 'question'))
        >>> # Returns [CellNodeId(5, 'context')] for a cell-by-cell column
    """

    def __init__(
        self,
        num_records: int,
        column_descriptors: list[ColumnDescriptor],
        *,
        strict: bool = True,
    ) -> None:
        """Initialize the execution graph.

        Args:
            num_records: The number of records (rows) to generate.
            column_descriptors: List of column descriptors in topological order.
            strict: If True, validate all dependencies exist during construction.

        Raises:
            ValueError: If num_records is not positive, column_descriptors is empty,
                or strict=True and dependencies are invalid.
        """
        if num_records <= 0:
            raise ValueError(f"num_records must be positive, got {num_records}")
        if not column_descriptors:
            raise ValueError("column_descriptors cannot be empty")

        self._num_records = num_records
        self._columns: dict[str, ColumnDescriptor] = {desc.name: desc for desc in column_descriptors}
        self._topo_order: list[str] = [desc.name for desc in column_descriptors]

        # Build reverse mapping for side effects
        self._side_effect_to_parent: dict[str, str] = {}
        for desc in column_descriptors:
            for side_effect in desc.side_effects:
                self._side_effect_to_parent[side_effect] = desc.name

        # Cache start columns
        self._start_columns: list[str] = [name for name, desc in self._columns.items() if desc.is_start_column]

        # Cache barrier columns for efficient lookup
        self._barrier_columns: set[str] = {name for name, desc in self._columns.items() if desc.is_barrier}

        if strict:
            self._validate_dependencies()

    def _validate_dependencies(self) -> None:
        """Validate all dependencies can be resolved.

        Raises:
            ValueError: If any column has an unresolvable dependency.
        """
        all_columns = set(self._columns.keys())
        all_side_effects = set(self._side_effect_to_parent.keys())
        known_names = all_columns | all_side_effects

        errors: list[str] = []
        for col_name, desc in self._columns.items():
            for dep in desc.dependencies:
                if dep not in known_names:
                    errors.append(f"Column '{col_name}' depends on unknown column '{dep}'")

        if errors:
            msg = "Invalid dependencies in execution graph:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(msg)

    @property
    def num_records(self) -> int:
        """The number of records (rows) in the dataset."""
        return self._num_records

    @property
    def num_columns(self) -> int:
        """The number of columns in the dataset."""
        return len(self._columns)

    @property
    def num_nodes(self) -> int:
        """The total number of virtual nodes in the graph.

        This is computed as:
        - For each non-barrier column: num_records cell nodes
        - For each barrier column: 1 barrier node + num_records cell nodes
        """
        barrier_count = len(self._barrier_columns)
        return (self.num_columns * self._num_records) + barrier_count

    @property
    def start_columns(self) -> list[str]:
        """List of column names that can generate data from scratch."""
        return self._start_columns.copy()

    @property
    def column_names(self) -> list[str]:
        """List of column names in topological order."""
        return self._topo_order.copy()

    def get_column_descriptor(self, column: str) -> ColumnDescriptor:
        """Get the descriptor for a column.

        Args:
            column: The column name.

        Returns:
            The ColumnDescriptor for the specified column.

        Raises:
            KeyError: If the column does not exist.
        """
        return self._columns[column]

    def get_generator_and_config(self, node: NodeId) -> tuple[type[ColumnGenerator], ConfigBase]:
        """Get the generator class and config for a node.

        For barrier nodes, returns the generator/config of the associated column.
        For cell nodes, returns the generator/config of that column.

        Args:
            node: The node identifier.

        Returns:
            A tuple of (generator_class, column_config).

        Raises:
            KeyError: If the column does not exist.
        """
        column = node.column
        desc = self._columns[column]
        return desc.generator_cls, desc.config

    def _resolve_column(self, name: str) -> str | None:
        """Resolve a column name, following side effect mappings if needed.

        Args:
            name: The column name to resolve.

        Returns:
            The resolved column name, or None if not found.
        """
        if name in self._columns:
            return name
        if name in self._side_effect_to_parent:
            return self._side_effect_to_parent[name]
        return None

    def get_dependencies(self, node: NodeId) -> list[NodeId]:
        """Get the dependencies for a node.

        Dependency resolution depends on the column's execution traits:

        - **START columns**: No dependencies (empty list)
        - **CELL_BY_CELL / ROW_STREAMABLE**: Row-local dependencies
          - Cell (row=r, col=C) depends on Cell (row=r, col=D) for each D in required_columns
        - **BARRIER columns**:
          - BarrierNodeId depends on ALL cells of ALL dependency columns
          - CellNodeId depends on the BarrierNodeId of its column

        Args:
            node: The node to get dependencies for.

        Returns:
            List of NodeIds that this node depends on.
        """
        column = node.column
        desc = self._columns[column]

        # START columns with no dependencies
        if desc.is_start_column and not desc.has_dependencies:
            return []

        # Handle barrier columns
        if desc.is_barrier:
            if isinstance(node, BarrierNodeId):
                # Barrier depends on ALL cells of ALL dependency columns
                deps: list[NodeId] = []
                for dep_name in desc.dependencies:
                    resolved = self._resolve_column(dep_name)
                    if resolved is None:
                        continue

                    dep_desc = self._columns[resolved]
                    # If dep is also a barrier, depend on its barrier node
                    if dep_desc.is_barrier:
                        deps.append(BarrierNodeId(resolved))
                    else:
                        # Depend on all cells of the dependency column
                        for r in range(self._num_records):
                            deps.append(CellNodeId(r, resolved))
                return deps
            elif isinstance(node, CellNodeId):
                # Output cells depend on the barrier
                return [BarrierNodeId(column)]

        # CELL_BY_CELL and ROW_STREAMABLE: row-local dependencies
        if isinstance(node, CellNodeId):
            row = node.row
            deps = []
            for dep_name in desc.dependencies:
                resolved = self._resolve_column(dep_name)
                if resolved is None:
                    continue

                dep_desc = self._columns[resolved]
                # If dep is a barrier, depend on the barrier node (not individual cells)
                if dep_desc.is_barrier:
                    deps.append(BarrierNodeId(resolved))
                else:
                    deps.append(CellNodeId(row, resolved))
            return deps

        return []

    def get_dependents(self, node: NodeId) -> list[NodeId]:
        """Get the nodes that depend on this node.

        This is the reverse of get_dependencies. Useful for scheduling
        dependent tasks when a node completes.

        Args:
            node: The node to get dependents for.

        Returns:
            List of NodeIds that depend on this node.
        """
        dependents: list[NodeId] = []
        column = node.column

        if isinstance(node, CellNodeId):
            row = node.row

            # Find columns that depend on this column
            for col_name, desc in self._columns.items():
                if column not in desc.dependencies:
                    # Check if this column is the parent of a side effect dependency
                    is_side_effect_parent = any(
                        self._side_effect_to_parent.get(dep) == column for dep in desc.dependencies
                    )
                    if not is_side_effect_parent:
                        continue

                if desc.is_barrier:
                    # Barrier depends on all cells, so add barrier node
                    dependents.append(BarrierNodeId(col_name))
                else:
                    # Row-local dependency
                    dependents.append(CellNodeId(row, col_name))

        elif isinstance(node, BarrierNodeId):
            # Barrier completion triggers all cells of that column
            for r in range(self._num_records):
                dependents.append(CellNodeId(r, column))

            # Also check if other barriers depend on this barrier
            for col_name, desc in self._columns.items():
                if desc.is_barrier and column in desc.dependencies:
                    dependents.append(BarrierNodeId(col_name))

        return dependents

    def iter_nodes(self) -> Iterator[NodeId]:
        """Iterate over all virtual nodes in the graph.

        Yields nodes in a consistent order:
        1. For each column in topological order:
           - If barrier column: yield BarrierNodeId, then all CellNodeIds
           - Otherwise: yield all CellNodeIds

        Yields:
            NodeId instances (CellNodeId or BarrierNodeId).
        """
        for col_name in self._topo_order:
            desc = self._columns[col_name]
            if desc.is_barrier:
                yield BarrierNodeId(col_name)
            for row in range(self._num_records):
                yield CellNodeId(row, col_name)

    def iter_start_nodes(self) -> Iterator[CellNodeId]:
        """Iterate over nodes that can start immediately (no dependencies).

        These are all cell nodes from START columns.

        Yields:
            CellNodeId instances for start columns.
        """
        for col_name in self._start_columns:
            for row in range(self._num_records):
                yield CellNodeId(row, col_name)

    def iter_ready_nodes(self, completed: CompletionTrackerProtocol) -> Iterator[NodeId]:
        """Iterate over nodes whose dependencies are all satisfied.

        A node is ready if all of its dependencies are in the completed set.
        This method is the primary interface for async execution engines.

        Args:
            completed: A completion tracker (or set) indicating completed nodes.

        Yields:
            NodeId instances that are ready for execution.

        Note:
            For large datasets, consider using iter_ready_nodes_for_column
            for more efficient targeted queries.
        """
        for col_name in self._topo_order:
            desc = self._columns[col_name]

            if desc.is_barrier:
                barrier_node = BarrierNodeId(col_name)
                if barrier_node not in completed:
                    deps = self.get_dependencies(barrier_node)
                    if all(dep in completed for dep in deps):
                        yield barrier_node

                # Check cell nodes only if barrier is complete
                if barrier_node in completed:
                    for row in range(self._num_records):
                        cell_node = CellNodeId(row, col_name)
                        if cell_node not in completed:
                            yield cell_node
            else:
                for row in range(self._num_records):
                    cell_node = CellNodeId(row, col_name)
                    if cell_node not in completed:
                        deps = self.get_dependencies(cell_node)
                        if all(dep in completed for dep in deps):
                            yield cell_node

    def iter_ready_nodes_for_column(self, column: str, completed: CompletionTrackerProtocol) -> Iterator[NodeId]:
        """Iterate over ready nodes for a specific column.

        More efficient than iter_ready_nodes when you know which column
        to check, as it avoids scanning all columns.

        Args:
            column: The column name to check.
            completed: A completion tracker (or set) indicating completed nodes.

        Yields:
            NodeId instances from the specified column that are ready.
        """
        desc = self._columns[column]

        if desc.is_barrier:
            barrier_node = BarrierNodeId(column)
            if barrier_node not in completed:
                deps = self.get_dependencies(barrier_node)
                if all(dep in completed for dep in deps):
                    yield barrier_node
                return  # Can't yield cells until barrier is complete

            # Barrier is complete, yield all incomplete cells
            for row in range(self._num_records):
                cell_node = CellNodeId(row, column)
                if cell_node not in completed:
                    yield cell_node
        else:
            for row in range(self._num_records):
                cell_node = CellNodeId(row, column)
                if cell_node not in completed:
                    deps = self.get_dependencies(cell_node)
                    if all(dep in completed for dep in deps):
                        yield cell_node

    def is_complete(self, completed: CompletionTrackerProtocol) -> bool:
        """Check if all nodes in the graph have been completed.

        Args:
            completed: A completion tracker (or set) indicating completed nodes.

        Returns:
            True if all nodes are in the completed set.
        """
        # Quick check: minimum required completions
        expected = self.num_nodes
        if hasattr(completed, "__len__") and len(completed) < expected:
            return False

        # Verify all expected nodes are present
        for node in self.iter_nodes():
            if node not in completed:
                return False
        return True

    def is_row_complete(self, row: int, completed: CompletionTrackerProtocol) -> bool:
        """Check if all cells for a row are complete.

        A row is considered complete when all cells across all columns for that
        row have been completed. For barrier columns, this also requires that
        the barrier itself is complete.

        Args:
            row: The row index to check.
            completed: A completion tracker (or set) indicating completed nodes.

        Returns:
            True if all cells for the row are complete, False otherwise.

        Raises:
            ValueError: If row is out of range.
        """
        if row < 0 or row >= self._num_records:
            raise ValueError(f"Row {row} is out of range [0, {self._num_records})")

        for col_name in self._topo_order:
            desc = self._columns[col_name]
            if desc.is_barrier:
                if BarrierNodeId(col_name) not in completed:
                    return False
            if CellNodeId(row, col_name) not in completed:
                return False
        return True

    def get_completed_row_count(self, completed: CompletionTrackerProtocol) -> int:
        """Get the count of contiguous complete rows starting from row 0.

        Returns the highest N where rows 0..N-1 are all complete. This is useful
        for checkpoint/restart scenarios where you want to know how many rows
        can be safely saved as a usable partial dataset.

        Args:
            completed: A completion tracker (or set) indicating completed nodes.

        Returns:
            The count of contiguous complete rows starting from row 0.
        """
        count = 0
        for row in range(self._num_records):
            if self.is_row_complete(row, completed):
                count += 1
            else:
                break
        return count
