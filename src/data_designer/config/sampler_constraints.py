# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from enum import Enum
from typing import Union

from typing_extensions import TypeAlias

from data_designer.config.base import ConfigBase


class ConstraintType(str, Enum):
    SCALAR_INEQUALITY = "scalar_inequality"
    COLUMN_INEQUALITY = "column_inequality"


class InequalityOperator(str, Enum):
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"


class Constraint(ConfigBase, ABC):
    target_column: str

    @property
    @abstractmethod
    def constraint_type(self) -> ConstraintType: ...


class ScalarInequalityConstraint(Constraint):
    rhs: float
    operator: InequalityOperator

    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.SCALAR_INEQUALITY


class ColumnInequalityConstraint(Constraint):
    rhs: str
    operator: InequalityOperator

    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.COLUMN_INEQUALITY


ColumnConstraintT: TypeAlias = Union[ScalarInequalityConstraint, ColumnInequalityConstraint]
