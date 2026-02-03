# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import date
from unittest.mock import patch

from data_designer.engine.sampling_gen.entities.national_id_utils import generate_ssn


def test_generate_ssn_format():
    ssn = generate_ssn("NY", date(1990, 1, 1))
    assert isinstance(ssn, str)
    assert len(ssn) == 11
    assert ssn.count("-") == 2
    assert ssn[3] == "-" and ssn[6] == "-"


def test_generate_ssn_digits_only():
    ssn = generate_ssn("NY", date(1990, 1, 1))
    digits_only = ssn.replace("-", "")
    assert digits_only.isdigit()
    assert len(digits_only) == 9


def test_generate_ssn_no_666_area():
    for _ in range(10):
        ssn = generate_ssn("NY", date(1990, 1, 1))
        area_code = ssn[:3]
        assert area_code != "666"


def test_generate_ssn_pre_2011_state_based():
    birth_date = date(2010, 1, 1)

    with patch("random.random", return_value=0.1):
        ssn = generate_ssn("NY", birth_date)
        area_code = int(ssn[:3])

        assert 50 <= area_code <= 134


def test_generate_ssn_post_2011_random():
    birth_date = date(2015, 1, 1)

    ssn = generate_ssn("NY", birth_date)
    area_code = int(ssn[:3])

    assert 1 <= area_code <= 899
    assert area_code != 666


def test_generate_ssn_different_states():
    birth_date = date(1990, 1, 1)

    test_cases = [("CA", (545, 573)), ("TX", (449, 467)), ("FL", (261, 267)), ("NY", (50, 134))]

    for state, expected_range in test_cases:
        with patch("random.random", return_value=0.1):
            ssn = generate_ssn(state, birth_date)
            area_code = int(ssn[:3])
            assert expected_range[0] <= area_code <= expected_range[1]
