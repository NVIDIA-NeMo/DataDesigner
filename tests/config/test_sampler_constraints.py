# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from data_designer.config.sampler_constraints import (
    ColumnInequalityConstraint,
    ConstraintType,
    InequalityOperator,
    ScalarInequalityConstraint,
)


def test_scalar_inequality_constraint():
    constraint = ScalarInequalityConstraint(target_column="test", rhs=1, operator=InequalityOperator.LT)
    assert constraint.target_column == "test"
    assert constraint.rhs == 1
    assert constraint.operator == InequalityOperator.LT
    assert constraint.constraint_type == ConstraintType.SCALAR_INEQUALITY


def test_column_inequality_constraint():
    constraint = ColumnInequalityConstraint(target_column="test", rhs="test2", operator=InequalityOperator.LT)
    assert constraint.target_column == "test"
    assert constraint.rhs == "test2"
    assert constraint.operator == InequalityOperator.LT
    assert constraint.constraint_type == ConstraintType.COLUMN_INEQUALITY
