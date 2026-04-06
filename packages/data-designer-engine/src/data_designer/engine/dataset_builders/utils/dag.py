# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_types import ColumnConfigT
from data_designer.engine.column_generators.utils.generator_classification import column_type_used_in_execution_dag
from data_designer.engine.dataset_builders.utils.errors import DAGCircularDependencyError
from data_designer.logging import LOG_INDENT

logger = logging.getLogger(__name__)


def _add_dependency_edges(
    dag: lazy.nx.DiGraph,
    name: str,
    dep_names: list[str],
    dag_column_config_dict: dict[str, ColumnConfigT],
    side_effect_dict: dict[str, list[str]],
    label: str,
) -> None:
    """Add DAG edges from *dep_names* to *name*, resolving through side-effect parents."""
    for dep in dep_names:
        if dep in dag_column_config_dict:
            logger.debug(f"{LOG_INDENT}🔗 `{name}` {label} depends on `{dep}`")
            dag.add_edge(dep, name)
        elif dep in sum(side_effect_dict.values(), []):
            for parent, cols in side_effect_dict.items():
                if dep in cols:
                    logger.debug(f"{LOG_INDENT}🔗 `{name}` {label} depends on `{parent}` via `{dep}`")
                    dag.add_edge(parent, name)
                    break


def topologically_sort_column_configs(column_configs: list[ColumnConfigT]) -> list[ColumnConfigT]:
    dag = lazy.nx.DiGraph()

    non_dag_column_config_list = [
        col for col in column_configs if not column_type_used_in_execution_dag(col.column_type)
    ]
    dag_column_config_dict = {
        col.name: col for col in column_configs if column_type_used_in_execution_dag(col.column_type)
    }

    if len(dag_column_config_dict) == 0:
        return non_dag_column_config_list

    side_effect_dict = {n: list(c.side_effect_columns) for n, c in dag_column_config_dict.items()}

    logger.info("⛓️ Sorting column configs into a Directed Acyclic Graph")
    for name, col in dag_column_config_dict.items():
        dag.add_node(name)
        _add_dependency_edges(dag, name, list(col.required_columns), dag_column_config_dict, side_effect_dict, "")
        if col.skip is not None:
            _add_dependency_edges(dag, name, col.skip.columns, dag_column_config_dict, side_effect_dict, "skip.when")

    if not lazy.nx.is_directed_acyclic_graph(dag):
        raise DAGCircularDependencyError(
            "🛑 The Data Designer column configurations contain cyclic dependencies. Please "
            "inspect the column configurations and ensure they can be sorted without "
            "circular references."
        )

    sorted_columns = non_dag_column_config_list
    sorted_columns.extend([dag_column_config_dict[n] for n in list(lazy.nx.topological_sort(dag))])

    return sorted_columns
