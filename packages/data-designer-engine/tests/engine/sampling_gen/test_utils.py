# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from data_designer.engine.sampling_gen.utils import check_random_state
from data_designer.lazy_heavy_imports import np

if TYPE_CHECKING:
    import numpy as np


@pytest.mark.parametrize(
    "test_case,input_value,expected_type,expected_seed",
    [
        ("none_input", None, "np.random.mtrand._rand", None),
        ("np_random_input", np.random, "np.random.mtrand._rand", None),
        ("integer_input", 42, "np.random.RandomState", 42),
        ("random_state_input", np.random.RandomState(123), "np.random.RandomState", 123),
    ],
)
def test_check_random_state_scenarios(test_case, input_value, expected_type, expected_seed):
    if test_case == "random_state_input":
        result = check_random_state(input_value)
        assert result is input_value
    else:
        result = check_random_state(input_value)

        if expected_type == "np.random.mtrand._rand":
            assert result is np.random.mtrand._rand
        elif expected_type == "np.random.RandomState":
            assert isinstance(result, np.random.RandomState)
            if expected_seed is not None:
                assert result.get_state()[1][0] == expected_seed


def test_check_random_state_invalid():
    with pytest.raises(ValueError, match="'invalid' cannot be used to seed a numpy.random.RandomState instance"):
        check_random_state("invalid")
