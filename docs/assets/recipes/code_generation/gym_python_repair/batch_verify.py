# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent checks for the CSV repair experiment."""

from __future__ import annotations

import json
import runpy
from collections.abc import Callable
from copy import deepcopy


def order(order_id: str, day: int, status: str) -> dict[str, str]:
    return {"order_id": order_id, "updated_at": f"2026-11-{day:02}", "status": status}


def check(
    dedupe: Callable[[list[dict[str, str]]], list[dict[str, str]]], rows: list[dict[str, str]], winners: list[int]
) -> bool:
    original = deepcopy(rows)
    return dedupe(rows) == [original[index] for index in winners] and rows == original


dedupe = runpy.run_path("/workspace/clean.py")["dedupe"]
cases = {
    "older_then_newer": ([order("B9", 1, "old"), order("B9", 4, "new")], [1]),
    "newer_then_older": ([order("X1", 4, "new"), order("X1", 2, "old")], [0]),
    "equal_dates_first_wins": (
        [order("Q7", 5, "first"), order("Q7", 5, "second")],
        [0],
    ),
    "interleaved_ids_and_ties": (
        [
            order("Z2", 2, "old"),
            order("A1", 9, "new"),
            order("R3", 1, "only"),
            order("Z2", 6, "new"),
            order("A1", 3, "old"),
            order("Z2", 6, "duplicate"),
        ],
        [3, 1, 2],
    ),
    "empty_input": ([], []),
}
report = {}
for name, (rows, winners) in cases.items():
    try:
        report[name] = {"passed": check(dedupe, rows, winners)}
    except Exception as error:  # noqa: BLE001 - Record exceptions from the candidate.
        report[name] = {"passed": False, "error": type(error).__name__}
print(json.dumps(report))
raise SystemExit(0 if all(result["passed"] for result in report.values()) else 1)
