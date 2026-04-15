# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
from collections import deque
from typing import TYPE_CHECKING

from data_designer.config.column_configs import GenerationStrategy
from data_designer.config.column_types import ColumnConfigT
from data_designer.engine.column_generators.utils.generator_classification import column_type_used_in_execution_dag
from data_designer.engine.dataset_builders.multi_column_configs import (
    DatasetBuilderColumnConfigT,
    MultiColumnConfig,
)
from data_designer.engine.dataset_builders.utils.errors import ConfigCompilationError, DAGCircularDependencyError
from data_designer.engine.dataset_builders.utils.task_model import SliceRef
from data_designer.logging import LOG_INDENT

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from data_designer.config.base import SkipConfig


class ExecutionGraph:
    """Column-level static execution graph built from column configs.

    Nodes are columns (O(C)); edges are dependency relationships (O(C²) worst-case).
    The graph is fixed for the lifetime of a run — runtime readiness is tracked
    separately by ``CompletionTracker``.
    """

    def __init__(self) -> None:
        self._upstream: dict[str, set[str]] = {}
        self._downstream: dict[str, set[str]] = {}
        self._strategies: dict[str, GenerationStrategy] = {}
        self._side_effect_map: dict[str, str] = {}
        self._columns: list[str] = []
        self._topological_order_cache: list[str] | None = None
        self._upstream_by_strategy_cache: dict[str, tuple[list[str], list[str]]] = {}
        self._required_columns: dict[str, list[str]] = {}
        self._skip_configs: dict[str, SkipConfig] = {}
        self._propagate_skip: dict[str, bool] = {}
        self._producer_to_side_effect_map: dict[str, list[str]] = {}

    @property
    def columns(self) -> list[str]:
        """All column names in insertion order."""
        return list(self._columns)

    @classmethod
    def create(
        cls,
        column_configs: list[DatasetBuilderColumnConfigT],
        strategies: dict[str, GenerationStrategy],
    ) -> ExecutionGraph:
        """Build an ``ExecutionGraph`` from column configs and pre-computed strategies.

        Args:
            column_configs: Ordered list of ``ColumnConfigT`` or ``MultiColumnConfig``.
            strategies: Map of column name → ``GenerationStrategy``, obtained from
                each generator's ``get_generation_strategy()``.
        """
        graph = cls()

        # First pass: register all columns, strategies, side-effect mappings, and skip metadata
        for config in column_configs:
            if isinstance(config, MultiColumnConfig):
                sub_configs = config.columns
            else:
                sub_configs = [config]

            for sub in sub_configs:
                name = sub.name
                graph.add_column(name, strategies[name])

                for se_col in sub.side_effect_columns:
                    graph.set_side_effect(se_col, name)

                graph.set_propagate_skip(name, sub.propagate_skip)
                if sub.skip is not None:
                    graph.set_skip_config(name, sub.skip)

        known_columns = set(graph.columns)

        # Second pass: build edges (required_columns + skip.columns)
        for config in column_configs:
            if isinstance(config, MultiColumnConfig):
                sub_configs = config.columns
            else:
                sub_configs = [config]

            for sub in sub_configs:
                name = sub.name
                resolved_required: list[str] = []
                for req in sub.required_columns:
                    resolved = graph.resolve_side_effect(req)
                    if resolved not in known_columns:
                        raise ValueError(
                            f"Column '{name}' requires '{req}' (resolved to '{resolved}') which is not a known producer."
                        )
                    if resolved == name:
                        continue
                    if resolved not in resolved_required:
                        resolved_required.append(resolved)
                    graph.add_edge(upstream=resolved, downstream=name)
                graph.set_required_columns(name, resolved_required)

                if sub.skip is not None:
                    for skip_col in sub.skip.columns:
                        resolved = graph.resolve_side_effect(skip_col)
                        if resolved not in known_columns:
                            raise ValueError(
                                f"Column '{name}' skip.when references '{skip_col}' "
                                f"(resolved to '{resolved}') which is not a known producer."
                            )
                        if resolved == name:
                            continue
                        graph.add_edge(upstream=resolved, downstream=name)

        # Validate acyclicity
        graph.get_topological_order()

        return graph

    def add_column(self, name: str, strategy: GenerationStrategy) -> None:
        """Register a column and its generation strategy."""
        if name in self._strategies:
            raise ValueError(f"Column '{name}' is already registered.")
        self._columns.append(name)
        self._strategies[name] = strategy

    def add_edge(self, upstream: str, downstream: str) -> None:
        """Add a dependency edge: *downstream* depends on *upstream*."""
        self._upstream.setdefault(downstream, set()).add(upstream)
        self._downstream.setdefault(upstream, set()).add(downstream)

    def set_side_effect(self, side_effect_col: str, producer: str) -> None:
        """Map a side-effect column name to its producing column.

        Each side-effect column must have exactly one producer.  Duplicate
        registrations from a different producer are a configuration error -
        use distinct column names for each pipeline stage instead.
        """
        existing = self._side_effect_map.get(side_effect_col)
        if existing is not None and existing != producer:
            raise ConfigCompilationError(
                f"Side-effect column {side_effect_col!r} is already produced by {existing!r}; "
                f"cannot register a second producer {producer!r}. "
                f"Use distinct side-effect column names for each pipeline stage."
            )
        self._side_effect_map[side_effect_col] = producer
        self._producer_to_side_effect_map.setdefault(producer, []).append(side_effect_col)

    def set_required_columns(self, column: str, required: list[str]) -> None:
        """Store producer-resolved ``required_columns`` for skip propagation."""
        self._required_columns[column] = required

    def set_propagate_skip(self, column: str, propagate: bool) -> None:
        """Store whether *column* should auto-skip when an upstream was skipped."""
        self._propagate_skip[column] = propagate

    def set_skip_config(self, column: str, skip_config: SkipConfig) -> None:
        """Attach a ``SkipConfig`` to *column*."""
        self._skip_configs[column] = skip_config

    def resolve_side_effect(self, column: str) -> str:
        """Resolve a column name through the side-effect map.

        If a real column exists with the same name as a side-effect alias,
        the real column wins.
        """
        if column in self._strategies:
            return column
        return self._side_effect_map.get(column, column)

    def get_upstream_columns(self, column: str) -> set[str]:
        """Direct dependencies of *column*."""
        return set(self._upstream.get(column, set()))

    def get_downstream_columns(self, column: str) -> set[str]:
        """Columns that depend on *column*."""
        return set(self._downstream.get(column, set()))

    def get_required_columns(self, column: str) -> list[str]:
        """Producer-resolved ``required_columns`` for *column* (data dependencies only)."""
        return list(self._required_columns.get(column, []))

    def get_skip_config(self, column: str) -> SkipConfig | None:
        """Return the ``SkipConfig`` for *column*, or ``None`` if not configured."""
        return self._skip_configs.get(column)

    def should_propagate_skip(self, column: str) -> bool:
        """Whether *column* auto-skips when an upstream was skipped."""
        return self._propagate_skip.get(column, True)

    def get_side_effect_columns(self, column: str) -> list[str]:
        """Return side-effect column names produced by *column*."""
        return list(self._producer_to_side_effect_map.get(column, []))

    def get_strategy(self, column: str) -> GenerationStrategy:
        return self._strategies[column]

    def get_root_columns(self) -> list[str]:
        """Columns with no upstream dependencies, in topological order."""
        return [col for col in self.get_topological_order() if not self._upstream.get(col)]

    def split_upstream_by_strategy(self, column: str) -> tuple[list[str], list[str]]:
        """Split upstream columns into (batch, cell_by_cell) by strategy. Cached."""
        cached = self._upstream_by_strategy_cache.get(column)
        if cached is not None:
            return cached
        batch: list[str] = []
        cell: list[str] = []
        for up_col in self.get_upstream_columns(column):
            if self._strategies[up_col] == GenerationStrategy.CELL_BY_CELL:
                cell.append(up_col)
            else:
                batch.append(up_col)
        result = (batch, cell)
        self._upstream_by_strategy_cache[column] = result
        return result

    def get_topological_order(self) -> list[str]:
        """Return a valid topological ordering of columns (Kahn's algorithm).

        Result is cached after first successful computation since the graph is
        immutable after construction.
        """
        if self._topological_order_cache is not None:
            return list(self._topological_order_cache)

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

        self._topological_order_cache = order
        return list(order)

    def get_longest_dependency_chain(self) -> list[str]:
        """Longest dependency chain (by number of columns)."""
        order = self.get_topological_order()
        if not order:
            return []
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

    def compute_task_count(self, num_records: int, buffer_size: int) -> dict[str, int]:
        """Exact task count per column before the run starts.

        Cell-by-cell columns produce ``num_records`` tasks.
        Full-column columns (including from-scratch) produce ``ceil(num_records / buffer_size)`` tasks.
        """
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be a positive integer, got {buffer_size}")
        num_row_groups = math.ceil(num_records / buffer_size)
        counts: dict[str, int] = {}
        for col in self._columns:
            strat = self._strategies[col]
            if strat == GenerationStrategy.CELL_BY_CELL:
                counts[col] = num_records
            else:
                counts[col] = num_row_groups
        return counts

    def compute_cell_dependencies(
        self,
        column: str,
        row_group: int,
        row_index: int | None,
        row_group_size: int,
    ) -> list[SliceRef]:
        """Derive cell-level deps on demand from column-level DAG + strategy.

        Returns a list of ``SliceRef`` that must be complete before this task can run.
        """
        deps: list[SliceRef] = []
        for up_col in self.get_upstream_columns(column):
            up_strategy = self._strategies[up_col]
            if up_strategy == GenerationStrategy.CELL_BY_CELL:
                if row_index is not None:
                    deps.append(SliceRef(up_col, row_group, row_index))
                else:
                    for ri in range(row_group_size):
                        deps.append(SliceRef(up_col, row_group, ri))
            else:
                deps.append(SliceRef(up_col, row_group, None))
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


def topologically_sort_column_configs(column_configs: list[ColumnConfigT]) -> list[ColumnConfigT]:
    non_dag_cols = [col for col in column_configs if not column_type_used_in_execution_dag(col.column_type)]
    dag_col_dict = {col.name: col for col in column_configs if column_type_used_in_execution_dag(col.column_type)}

    if not dag_col_dict:
        return non_dag_cols

    # side_effect_col_name -> producing column name
    side_effect_map: dict[str, str] = {}
    for name, col in dag_col_dict.items():
        for se_col in col.side_effect_columns:
            existing = side_effect_map.get(se_col)
            if existing is not None and existing != name:
                raise ConfigCompilationError(
                    f"Side-effect column {se_col!r} is already produced by {existing!r}; "
                    f"cannot register a second producer {name!r}. "
                    f"Use distinct side-effect column names for each pipeline stage."
                )
            side_effect_map[se_col] = name

    def resolve(col_name: str) -> str | None:
        if col_name in dag_col_dict:
            return col_name
        return side_effect_map.get(col_name)

    upstream: dict[str, set[str]] = {name: set() for name in dag_col_dict}
    downstream: dict[str, set[str]] = {name: set() for name in dag_col_dict}

    # Only required_columns edges are added here. skip.columns edges are intentionally
    # omitted: ExecutionGraph.create handles them in its own two-pass build, which is
    # the authoritative execution-order graph. This function only needs to produce a
    # valid topological ordering of ColumnConfigT objects for config compilation.
    logger.info("⛓️ Sorting column configs into a Directed Acyclic Graph")
    for name, col in dag_col_dict.items():
        for req in col.required_columns:
            resolved = resolve(req)
            if resolved is None or resolved == name:
                continue
            logger.debug(f"{LOG_INDENT}🔗 `{name}` depends on `{resolved}`")
            upstream[name].add(resolved)
            downstream[resolved].add(name)

    in_degree = {name: len(ups) for name, ups in upstream.items()}
    queue: deque[str] = deque(name for name, deg in in_degree.items() if deg == 0)
    order: list[str] = []
    while queue:
        name = queue.popleft()
        order.append(name)
        for child in downstream.get(name, set()):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(order) != len(dag_col_dict):
        raise DAGCircularDependencyError(
            "🛑 The Data Designer column configurations contain cyclic dependencies. Please "
            "inspect the column configurations and ensure they can be sorted without "
            "circular references."
        )

    return non_dag_cols + [dag_col_dict[n] for n in order]
