# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution traits for column generators.

This module defines the ExecutionTraits flag enum that describes the execution
characteristics of column generators. Traits are inferred from generator properties
(not hardcoded class names) to support plugin generators.
"""

from __future__ import annotations

from enum import Flag, auto


class ExecutionTraits(Flag):
    """Flags describing execution characteristics of a column generator.

    Traits can be combined using bitwise operators:
        traits = ExecutionTraits.START | ExecutionTraits.ROW_STREAMABLE

    Attributes:
        NONE: No special traits (default).
        START: Generator can produce data from scratch without dependencies.
            Inferred from `can_generate_from_scratch = True`.
        CELL_BY_CELL: Generator processes individual cells independently.
            Inferred from `get_generation_strategy() == CELL_BY_CELL`.
        ROW_STREAMABLE: Generator can emit results as rows complete.
            Inferred from `is_row_streamable = True`.
        BARRIER: Generator requires all input rows before producing any output.
            Inferred from full-column strategy with `is_row_streamable = False`.
    """

    NONE = 0
    START = auto()
    CELL_BY_CELL = auto()
    ROW_STREAMABLE = auto()
    BARRIER = auto()
