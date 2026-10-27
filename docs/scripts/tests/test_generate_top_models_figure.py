# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import runpy
from pathlib import Path

_MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "generate_top_models_figure.py"))
rank_rows = _MODULE["rank_rows"]


def test_rank_by_combined_input_and_output_tokens() -> None:
    rows = [("input-heavy", 100, 1), ("output-heavy", 2, 200), ("balanced", 75, 75)]

    assert rank_rows(rows) == [("output-heavy", 2, 200), ("balanced", 75, 75), ("input-heavy", 100, 1)]


def test_keep_top_ten_and_preserve_both_totals_in_other() -> None:
    rows = [(f"model-{i}", i * 100, i * 10) for i in range(1, 13)]
    rows += [("Other", 50, 5), ("other", 20, 2)]
    original = list(rows)

    ranked = rank_rows(rows)

    assert [row[0] for row in ranked[:-1]] == [f"model-{i}" for i in range(12, 2, -1)]
    assert ranked[-1] == ("Other", 370, 37)
    assert sum(row[1] for row in ranked) == sum(row[1] for row in rows)
    assert sum(row[2] for row in ranked) == sum(row[2] for row in rows)
    assert rows == original


def test_fold_remainder_into_other_when_source_has_no_aggregate() -> None:
    rows = [(f"model-{i}", i, 0) for i in range(11)]

    ranked = rank_rows(rows)

    assert len(ranked) == 11
    assert ranked[-1] == ("Other", 0, 0)
    assert ranked[0] == ("model-10", 10, 0)
    assert sum(row[1] for row in ranked) == 55
