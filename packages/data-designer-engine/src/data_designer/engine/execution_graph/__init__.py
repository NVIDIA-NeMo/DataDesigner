# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution graph for async dataset generation.

This package provides a memory-efficient execution graph for modeling cell-level
dependencies in dataset generation. The graph supports different generator
execution traits (start, cell-by-cell, row-streamable, barrier) with a hybrid
representation that can handle millions of records efficiently.

Example:
    >>> from data_designer.engine.execution_graph import (
    ...     GraphBuilder,
    ...     ExecutionGraph,
    ...     CompletionTracker,
    ... )
    >>>
    >>> # Build graph from config
    >>> builder = GraphBuilder(column_generator_registry)
    >>> graph = builder.build(config, num_records=1_000_000)
    >>>
    >>> # Execute with completion tracking
    >>> tracker = CompletionTracker(graph.num_records)
    >>> for node in graph.iter_ready_nodes(tracker):
    ...     gen_cls, config = graph.get_generator_and_config(node)
    ...     # Execute node...
    ...     tracker.mark_complete(node)
"""

from data_designer.engine.execution_graph.builder import GraphBuilder
from data_designer.engine.execution_graph.column_descriptor import ColumnDescriptor
from data_designer.engine.execution_graph.completion import (
    CompletionTracker,
    ThreadSafeCompletionTracker,
)
from data_designer.engine.execution_graph.graph import (
    CompletionTrackerProtocol,
    ExecutionGraph,
)
from data_designer.engine.execution_graph.node_id import (
    BarrierNodeId,
    CellNodeId,
    NodeId,
)
from data_designer.engine.execution_graph.traits import ExecutionTraits

__all__ = [
    # Node identification
    "CellNodeId",
    "BarrierNodeId",
    "NodeId",
    # Traits
    "ExecutionTraits",
    # Column descriptor
    "ColumnDescriptor",
    # Graph
    "ExecutionGraph",
    "CompletionTrackerProtocol",
    # Builder
    "GraphBuilder",
    # Completion tracking
    "CompletionTracker",
    "ThreadSafeCompletionTracker",
]
