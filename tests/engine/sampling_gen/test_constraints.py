# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from data_designer.config.sampler_constraints import (
    ColumnInequalityConstraint,
    ConstraintType,
    ScalarInequalityConstraint,
)
from data_designer.engine.sampling_gen.constraints import get_constraint_checker
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd


@pytest.mark.parametrize(
    "test_case,constraint_type,constraint_class,target_column,operator,rhs,expected_columns,test_data_success,test_data_failure",
    [
        (
            "scalar_inequality",
            ConstraintType.SCALAR_INEQUALITY,
            ScalarInequalityConstraint,
            "balance",
            "gt",
            0,
            ("balance",),
            {"balance": [1, 2, 3]},
            {"balance": [-1, -2, -3]},
        ),
        (
            "column_inequality",
            ConstraintType.COLUMN_INEQUALITY,
            ColumnInequalityConstraint,
            "balance",
            "gt",
            "credit",
            ("balance", "credit"),
            {"balance": [1, 2, 3], "credit": [0, 1, 2]},
            {"balance": [1, 0, -1], "credit": [2, 1, 0]},
        ),
    ],
)
def test_constraint_scenarios(
    test_case,
    constraint_type,
    constraint_class,
    target_column,
    operator,
    rhs,
    expected_columns,
    test_data_success,
    test_data_failure,
):
    constraint = get_constraint_checker(constraint_type)(
        constraint=constraint_class(target_column=target_column, operator=operator, rhs=rhs)
    )

    assert constraint.constraint.target_column == target_column
    assert constraint.constraint.operator == operator
    assert constraint.constraint.rhs == rhs

    if isinstance(expected_columns, tuple):
        assert constraint.get_required_column_names() == expected_columns
    else:
        assert set(constraint.get_required_column_names()) == expected_columns

    success_df = pd.DataFrame(test_data_success)
    assert constraint.check(success_df).all()

    failure_df = pd.DataFrame(test_data_failure)
    assert not constraint.check(failure_df).any()


@pytest.mark.parametrize(
    "test_case,constraint_class,target_column,operator,rhs,expected_dump",
    [
        (
            "scalar_inequality_serialization",
            ScalarInequalityConstraint,
            "balance",
            "gt",
            0,
            {"target_column": "balance", "operator": "gt", "rhs": 0},
        ),
        (
            "column_inequality_serialization",
            ColumnInequalityConstraint,
            "balance",
            "gt",
            "credit",
            {"target_column": "balance", "operator": "gt", "rhs": "credit"},
        ),
    ],
)
def test_constraint_serialization(test_case, constraint_class, target_column, operator, rhs, expected_dump):
    constraint = constraint_class(
        target_column=target_column,
        operator=operator,
        rhs=rhs,
    )
    assert constraint.model_dump() == expected_dump
