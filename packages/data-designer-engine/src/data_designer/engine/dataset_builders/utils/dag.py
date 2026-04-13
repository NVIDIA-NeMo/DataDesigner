# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_types import ColumnConfigT
from data_designer.engine.column_generators.utils.generator_classification import column_type_used_in_execution_dag
from data_designer.engine.dataset_builders.utils.errors import ConfigCompilationError, DAGCircularDependencyError
from data_designer.logging import LOG_INDENT

logger = logging.getLogger(__name__)


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

    side_effect_to_producer: dict[str, str] = {}
    for producer, cols in side_effect_dict.items():
        for col in cols:
            existing = side_effect_to_producer.get(col)
            if existing is not None and existing != producer:
                raise ConfigCompilationError(
                    f"Side-effect column {col!r} is already produced by {existing!r}; "
                    f"cannot register a second producer {producer!r}. "
                    f"Use distinct side-effect column names for each pipeline stage."
                )
            side_effect_to_producer[col] = producer

    logger.info("⛓️ Sorting column configs into a Directed Acyclic Graph")
    for name, col in dag_column_config_dict.items():
        dag.add_node(name)
        for req_col_name in col.required_columns:
            if req_col_name in list(dag_column_config_dict.keys()):
                logger.debug(f"{LOG_INDENT}🔗 `{name}` depends on `{req_col_name}`")
                dag.add_edge(req_col_name, name)

            # If the required column is a side effect of another column,
            # add an edge from the parent column to the current column.
            elif req_col_name in sum(side_effect_dict.values(), []):
                for parent, cols in side_effect_dict.items():
                    if req_col_name in cols:
                        logger.debug(f"{LOG_INDENT}🔗 `{name}` depends on `{parent}` via `{req_col_name}`")
                        dag.add_edge(parent, name)
                        break

    if not lazy.nx.is_directed_acyclic_graph(dag):
        raise DAGCircularDependencyError(
            "🛑 The Data Designer column configurations contain cyclic dependencies. Please "
            "inspect the column configurations and ensure they can be sorted without "
            "circular references."
        )

    sorted_columns = non_dag_column_config_list
    sorted_columns.extend([dag_column_config_dict[n] for n in list(lazy.nx.topological_sort(dag))])

    return sorted_columns
