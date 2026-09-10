# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True, slots=True)
class TerminalTaskFailure:
    """Terminal column failure for an omitted seed row.

    ``seed_row_index`` is the zero-based position in the requested generation
    sequence. It is not necessarily the raw source index for shuffled, selected,
    or cycled seed datasets.
    """

    seed_row_index: int
    column: str
