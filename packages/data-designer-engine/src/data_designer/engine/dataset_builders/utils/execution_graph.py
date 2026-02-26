# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

from data_designer.config.column_configs import GenerationStrategy
from data_designer.config.column_types import ColumnConfigT
from data_designer.engine.dataset_builders.multi_column_configs import MultiColumnConfig
from data_designer.engine.dataset_builders.utils.errors import DAGCircularDependencyError

DatasetBuilderColumnConfigT = ColumnConfigT | MultiColumnConfig


@dataclass
class ExecutionGraph:
    """Column-level static execution graph built from column configs.

    Nodes are columns (O(C)); edges are dependency relationships (O(C²) worst-case).
    The graph is fixed for the lifetime of a run — runtime readiness is tracked
    separately by ``CompletionTracker``.
    """

    _upstream: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    _downstream: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    _strategies: dict[str, GenerationStrategy] = field(default_factory=dict)
    _side_effect_map: dict[str, str] = field(default_factory=dict)
    _columns: list[str] = field(default_factory=list)

    def upstream(self, column: str) -> set[str]:
        """Direct dependencies of *column*."""
        return self._upstream.get(column, set())

    def downstream(self, column: str) -> set[str]:
        """Columns that depend on *column*."""
        return self._downstream.get(column, set())

    def strategy(self, column: str) -> GenerationStrategy:
        return self._strategies[column]

    @property
    def columns(self) -> list[str]:
        """All column names in insertion order."""
        return list(self._columns)

    def topological_order(self) -> list[str]:
        """Return a valid topological ordering of columns (Kahn's algorithm)."""
        in_degree: dict[str, int] = {col: 0 for col in self._columns}
        for col, deps in self._upstream.items():
            if col in in_degree:
                in_degree[col] = len(deps)

        queue = deque(col for col, deg in in_degree.items() if deg == 0)
        order: list[str] = []
        while queue:
            col = queue.popleft()
            order.append(col)
            for child in self._downstream.get(col, set()):
                if child in in_degree:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

        if len(order) != len(self._columns):
            raise DAGCircularDependencyError(
                f"The execution graph contains cyclic dependencies. Resolved {len(order)}/{len(self._columns)} columns."
            )
        return order

    def critical_path(self) -> list[str]:
        """Longest dependency chain (by number of columns)."""
        order = self.topological_order()
        dist: dict[str, int] = {col: 0 for col in order}
        pred: dict[str, str | None] = {col: None for col in order}

        for col in order:
            for child in self._downstream.get(col, set()):
                if dist[col] + 1 > dist[child]:
                    dist[child] = dist[col] + 1
                    pred[child] = col

        end = max(order, key=lambda c: dist[c])
        path: list[str] = []
        cur: str | None = end
        while cur is not None:
            path.append(cur)
            cur = pred[cur]
        path.reverse()
        return path

    def task_count(self, num_records: int, buffer_size: int) -> dict[str, int]:
        """Exact task count per column before the run starts.

        Cell-by-cell columns produce ``num_records`` tasks.
        Full-column columns (including from-scratch) produce ``ceil(num_records / buffer_size)`` tasks.
        """
        num_row_groups = math.ceil(num_records / buffer_size)
        counts: dict[str, int] = {}
        for col in self._columns:
            strat = self._strategies[col]
            if strat == GenerationStrategy.CELL_BY_CELL:
                counts[col] = num_records
            else:
                counts[col] = num_row_groups
        return counts

    def cell_dependencies(
        self,
        column: str,
        row_group: int,
        row_index: int | None,
        row_group_size: int,
    ) -> list[tuple[str, int, int | None]]:
        """Derive cell-level deps on demand from column-level DAG + strategy.

        Returns a list of ``(upstream_column, row_group, row_index)`` tuples
        that must be complete before this task can run.
        """
        deps: list[tuple[str, int, int | None]] = []
        for up_col in self.upstream(column):
            up_strategy = self._strategies[up_col]
            if up_strategy == GenerationStrategy.CELL_BY_CELL:
                if row_index is not None:
                    deps.append((up_col, row_group, row_index))
                else:
                    for ri in range(row_group_size):
                        deps.append((up_col, row_group, ri))
            else:
                deps.append((up_col, row_group, None))
        return deps

    def to_mermaid(self) -> str:
        """Mermaid diagram string with strategy annotations."""
        lines = ["graph TD"]
        for col in self._columns:
            strat = self._strategies[col]
            label = f"{col} [{strat.value}]"
            lines.append(f'    {col}["{label}"]')
        for col in self._columns:
            for dep in sorted(self._upstream.get(col, set())):
                lines.append(f"    {dep} --> {col}")
        return "\n".join(lines)


def build_execution_graph(
    column_configs: list[DatasetBuilderColumnConfigT],
    strategies: dict[str, GenerationStrategy],
) -> ExecutionGraph:
    """Build an ``ExecutionGraph`` from column configs and pre-computed strategies.

    Args:
        column_configs: Ordered list of ``ColumnConfigT`` or ``MultiColumnConfig``.
        strategies: Map of column name → ``GenerationStrategy``, obtained from
            each generator's ``get_generation_strategy()``.
    """
    graph = ExecutionGraph()

    # First pass: register all columns, strategies, and side-effect mappings
    for config in column_configs:
        if isinstance(config, MultiColumnConfig):
            sub_configs = config.columns
        else:
            sub_configs = [config]

        for sub in sub_configs:
            name = sub.name
            graph._columns.append(name)
            graph._strategies[name] = strategies[name]

            for se_col in sub.side_effect_columns:
                graph._side_effect_map[se_col] = name

    known_columns = set(graph._columns)

    # Second pass: build edges
    for config in column_configs:
        if isinstance(config, MultiColumnConfig):
            sub_configs = config.columns
        else:
            sub_configs = [config]

        for sub in sub_configs:
            name = sub.name
            for req in sub.required_columns:
                resolved = graph._side_effect_map.get(req, req)
                if resolved not in known_columns:
                    raise ValueError(
                        f"Column '{name}' requires '{req}' (resolved to '{resolved}') which is not a known producer."
                    )
                if resolved == name:
                    continue  # skip self-dependency
                graph._upstream[name].add(resolved)
                graph._downstream[resolved].add(name)

    # Validate acyclicity
    graph.topological_order()

    return graph
