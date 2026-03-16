# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from abc import ABC
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import Discriminator, Field, Tag
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
    """Base class for sampler constraints. Use a concrete subclass, not this class directly."""

    target_column: str = Field(description="Name of the sampler column this constraint applies to")
    constraint_type: ConstraintType = Field(description="Constraint type discriminator")


class ScalarInequalityConstraint(Constraint):
    """Constrain a sampler column to be less/greater than a scalar value.

    Only applies to sampler columns.
    """

    rhs: float = Field(description="Scalar value to compare against")
    operator: InequalityOperator = Field(description="Comparison operator")
    constraint_type: Literal[ConstraintType.SCALAR_INEQUALITY] = Field(
        default=ConstraintType.SCALAR_INEQUALITY,
        description="Constraint type discriminator, always 'scalar_inequality' for this constraint",
    )


class ColumnInequalityConstraint(Constraint):
    """Constrain a sampler column to be less/greater than another sampler column.

    Only applies to sampler columns.
    """

    rhs: str = Field(description="Name of the other sampler column to compare against")
    operator: InequalityOperator = Field(description="Comparison operator")
    constraint_type: Literal[ConstraintType.COLUMN_INEQUALITY] = Field(
        default=ConstraintType.COLUMN_INEQUALITY,
        description="Constraint type discriminator, always 'column_inequality' for this constraint",
    )


# Plain union for engine-layer type hints on already-validated constraint instances.
ColumnConstraintT: TypeAlias = ScalarInequalityConstraint | ColumnInequalityConstraint


def resolve_constraint_input_type(value: Any) -> ConstraintType | str | None:
    """Resolve the constraint type for both tagged and legacy config shapes."""
    if isinstance(value, dict):
        if (constraint_type := value.get("constraint_type")) is not None:
            return constraint_type

        # rhs is required on both concrete types, so when it's missing we default to
        # SCALAR_INEQUALITY — Pydantic will surface a clear "rhs field required" error.
        if (rhs := value.get("rhs")) is None:
            return ConstraintType.SCALAR_INEQUALITY
        if isinstance(rhs, str):
            return ConstraintType.SCALAR_INEQUALITY if _can_coerce_to_float(rhs) else ConstraintType.COLUMN_INEQUALITY
        return ConstraintType.SCALAR_INEQUALITY

    return getattr(value, "constraint_type", None)


def _can_coerce_to_float(value: str) -> bool:
    try:
        result = float(value)
    except ValueError:
        return False
    return math.isfinite(result)


# Discriminated union for deserialization — supports both tagged and legacy config shapes.
ColumnConstraintInputT: TypeAlias = Annotated[
    Annotated[ScalarInequalityConstraint, Tag(ConstraintType.SCALAR_INEQUALITY)]
    | Annotated[ColumnInequalityConstraint, Tag(ConstraintType.COLUMN_INEQUALITY)],
    Discriminator(resolve_constraint_input_type),
]
