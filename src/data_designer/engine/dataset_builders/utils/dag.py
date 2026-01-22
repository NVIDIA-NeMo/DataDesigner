# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from data_designer.config.column_types import ColumnConfigT
from data_designer.engine.column_generators.utils.generator_classification import column_type_used_in_execution_dag
from data_designer.engine.dataset_builders.utils.errors import DAGCircularDependencyError
from data_designer.lazy_heavy_imports import nx

if TYPE_CHECKING:
    import networkx as nx

logger = logging.getLogger(__name__)


def topologically_sort_column_configs(column_configs: list[ColumnConfigT]) -> list[ColumnConfigT]:
    dag = nx.DiGraph()

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
        for req_col_name in col.required_columns:
            if req_col_name in list(dag_column_config_dict.keys()):
                logger.debug(f"  |-- 🔗 `{name}` depends on `{req_col_name}`")
                dag.add_edge(req_col_name, name)

            # If the required column is a side effect of another column,
            # add an edge from the parent column to the current column.
            elif req_col_name in sum(side_effect_dict.values(), []):
                for parent, cols in side_effect_dict.items():
                    if req_col_name in cols:
                        logger.debug(f"  |-- 🔗 `{name}` depends on `{parent}` via `{req_col_name}`")
                        dag.add_edge(parent, name)
                        break

    if not nx.is_directed_acyclic_graph(dag):
        raise DAGCircularDependencyError(
            "🛑 The Data Designer column configurations contain cyclic dependencies. Please "
            "inspect the column configurations and ensure they can be sorted without "
            "circular references."
        )

    sorted_columns = non_dag_column_config_list
    sorted_columns.extend([dag_column_config_dict[n] for n in list(nx.topological_sort(dag))])

    return sorted_columns
