# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from data_designer.engine.dataset_builders.row_group_plan import ExplicitRowGroupPlan, normalize_row_group_plan


def test_explicit_plan_base_offset_does_not_change_scheduled_total() -> None:
    plan = ExplicitRowGroupPlan(((7, 3), (8, 2)), base_offset=10)
    assert plan.row_group_start_offset(7) == 10
    assert plan.row_group_start_offset(8) == 13
    assert plan.scheduled_total_rows == 5
    assert normalize_row_group_plan(plan) is plan


def test_explicit_plan_rejects_negative_base_offset() -> None:
    with pytest.raises(ValueError, match="base_offset"):
        ExplicitRowGroupPlan(((0, 1),), base_offset=-1)


def test_explicit_plan_rejects_duplicate_ids_and_nonpositive_sizes() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        ExplicitRowGroupPlan(((0, 1), (0, 2)))
    with pytest.raises(ValueError, match="positive"):
        ExplicitRowGroupPlan(((0, 0),))
