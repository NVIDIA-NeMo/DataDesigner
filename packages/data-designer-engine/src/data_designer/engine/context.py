# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextvars import ContextVar

# Set by the async scheduler before executing each task.
# Value: (current_rg_index, total_rg_count) or None.
current_row_group: ContextVar[tuple[int, int] | None] = ContextVar("current_row_group", default=None)


def format_row_group_tag() -> str:
    """Return a '(x/X) ' prefix if a row group context is active, else ''."""
    rg = current_row_group.get()
    if rg is None:
        return ""
    current, total = rg[0] + 1, rg[1]
    width = len(str(total))
    return f"({current:0{width}d}/{total}) "
