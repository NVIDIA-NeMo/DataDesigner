# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Node identification types for the execution graph.

This module defines the node ID types used to identify individual units of work
in the execution graph. There are two types of nodes:

- CellNodeId: Identifies a single cell (row, column) in the dataset
- BarrierNodeId: Identifies a barrier synchronization point for a column

Using frozen dataclasses with slots=True for memory efficiency when handling
millions of nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class CellNodeId:
    """Identifies a single cell in the dataset.

    Attributes:
        row: The row index (0-based).
        column: The column name.
    """

    row: int
    column: str

    def __repr__(self) -> str:
        return f"Cell({self.row}, {self.column!r})"


@dataclass(frozen=True, slots=True)
class BarrierNodeId:
    """Identifies a barrier synchronization point for a column.

    Barrier nodes represent a synchronization point where all input cells
    must complete before any output cells can begin. This is used for
    full-column generators that cannot process rows independently.

    Attributes:
        column: The column name this barrier is for.
    """

    column: str

    def __repr__(self) -> str:
        return f"Barrier({self.column!r})"


NodeId: TypeAlias = CellNodeId | BarrierNodeId
"""Type alias for any node identifier in the execution graph."""
