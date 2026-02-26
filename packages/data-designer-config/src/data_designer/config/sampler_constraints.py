# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from pydantic import Field
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
    """Base class for sampler column constraints."""

    target_column: str = Field(description="Name of the sampler column this constraint applies to")

    @property
    @abstractmethod
    def constraint_type(self) -> ConstraintType: ...


class ScalarInequalityConstraint(Constraint):
    """Sampler constraint that compares a sampler column's generated values against a scalar threshold."""

    rhs: float = Field(description="Scalar value to compare against")
    operator: InequalityOperator = Field(description="Comparison operator (lt, le, gt, ge)")

    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.SCALAR_INEQUALITY


class ColumnInequalityConstraint(Constraint):
    """Sampler constraint that compares a sampler column's generated values against another sampler column's values."""

    rhs: str = Field(description="Name of the other column to compare against")
    operator: InequalityOperator = Field(description="Comparison operator (lt, le, gt, ge)")

    @property
    def constraint_type(self) -> ConstraintType:
        return ConstraintType.COLUMN_INEQUALITY


ColumnConstraintT: TypeAlias = ScalarInequalityConstraint | ColumnInequalityConstraint
