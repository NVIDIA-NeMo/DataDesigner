# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numbers
from numbers import Number
from typing import Any, Type

from data_designer.config.utils.constants import REPORTING_PRECISION


def is_int(val: Any) -> bool:
    return isinstance(val, numbers.Integral)


def is_float(val: Any) -> bool:
    return isinstance(val, numbers.Real) and not isinstance(val, numbers.Integral)


def prepare_number_for_reporting(
    value: Number,
    target_type: Type[Number],
    precision: int = REPORTING_PRECISION,
) -> Number:
    """Ensure native python types and round to `precision` decimal digits."""
    value = target_type(value)
    if is_float(value):
        return round(value, precision)
    return value
