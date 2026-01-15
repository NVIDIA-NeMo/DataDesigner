# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from numpy.typing import NDArray

from data_designer.config.base import ConfigBase
from data_designer.config.sampler_constraints import (
    ColumnInequalityConstraint,
    Constraint,
    ConstraintType,
    InequalityOperator,
    ScalarInequalityConstraint,
)
from data_designer.lazy_heavy_imports import np, pd

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class ConstraintChecker(ConfigBase, ABC):
    constraint: Constraint

    def get_required_column_names(self) -> tuple[str, ...]:
        return (self.constraint.target_column,)

    @abstractmethod
    def check(self, dataframe: pd.DataFrame) -> NDArray[np.bool_]: ...


class WithCompareMixin:
    @property
    def lhs(self) -> str:
        return self.constraint.target_column

    def compare(self, lhs: float | int | NDArray, rhs: float | int | NDArray) -> bool | NDArray[np.bool_]:
        operator = {
            InequalityOperator.LT: np.less,
            InequalityOperator.LE: np.less_equal,
            InequalityOperator.GT: np.greater,
            InequalityOperator.GE: np.greater_equal,
        }[InequalityOperator(self.constraint.operator)]
        return operator(lhs, rhs)


class ScalarInequalityChecker(ConstraintChecker, WithCompareMixin):
    """Compare a column to a scalar value.

    Args:
        column_name: Name of the constrained column. Will be
            used as the left-hand side (lhs) of the comparison.
        operator: Comparison operator.
        rhs: Scalar value to compare against.
    """

    constraint: ScalarInequalityConstraint

    def check(self, dataframe: pd.DataFrame) -> NDArray[np.bool_]:
        return self.compare(dataframe[self.lhs].values, self.constraint.rhs)


class ColumnInequalityChecker(ConstraintChecker, WithCompareMixin):
    """Compare the values of two columns.

    Args:
        column_name: Name of the constrained column. Will be
            used as the left-hand side (lhs) of the comparison.
        operator: Comparison operator.
        rhs: Name of the column to compare against.
    """

    constraint: ColumnInequalityConstraint

    def get_required_column_names(self) -> tuple[str, ...]:
        """Return the names of columns required for the constraint.

        Note that order matters. Edges in the DAG are created as column_names[1], column_names[0].
        """
        return (self.lhs, self.constraint.rhs)

    def check(self, dataframe: pd.DataFrame) -> NDArray[np.bool_]:
        return self.compare(
            dataframe[self.lhs].values,
            dataframe[self.constraint.rhs].values,
        )


CONSTRAINT_TYPE_TO_CHECKER = {
    ConstraintType.SCALAR_INEQUALITY: ScalarInequalityChecker,
    ConstraintType.COLUMN_INEQUALITY: ColumnInequalityChecker,
}


def get_constraint_checker(constraint_type: ConstraintType) -> type[ConstraintChecker]:
    return CONSTRAINT_TYPE_TO_CHECKER[ConstraintType(constraint_type)]
