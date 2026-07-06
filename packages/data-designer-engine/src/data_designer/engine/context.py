# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextvars import ContextVar
from threading import Event

# Set per row group by the async scheduler before each task executes.
# Value: (current_rg_index, total_rg_count) or None.
current_row_group: ContextVar[tuple[int, int] | None] = ContextVar("current_row_group", default=None)

# Shared cancellation signal for sync generator work running in thread-pool
# workers. Context variables copy the Event object into worker threads, and the
# scheduler flips the Event on cancellation.
current_run_cancel_event: ContextVar[Event | None] = ContextVar("current_run_cancel_event", default=None)


def is_run_cancellation_requested() -> bool:
    cancel_event = current_run_cancel_event.get()
    return cancel_event.is_set() if cancel_event is not None else False


# Set while generating a row group. The value is the row group's planned start
# offset in the full dataset, including row groups skipped during resume.
current_row_group_start_offset: ContextVar[int | None] = ContextVar("current_row_group_start_offset", default=None)


def format_row_group_tag() -> str:
    """Return a '(x/X) ' prefix if a row group context is active, else ''."""
    rg = current_row_group.get()
    if rg is None:
        return ""
    current, total = rg[0] + 1, rg[1]
    width = len(str(total))
    return f"({current:0{width}d}/{total}) "
