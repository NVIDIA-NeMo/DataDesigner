# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import date
from unittest.mock import patch

import pytest

from data_designer.engine.sampling_gen.entities.email_address_utils import (
    get_email_address,
    get_email_basename_by_name,
    get_email_domain_by_age,
    get_email_suffix_by_birth_date,
)


def test_get_email_address_basic():
    email = get_email_address(
        first_name="John", middle_name="Michael", last_name="Doe", age=30, birth_date=date(1990, 1, 1)
    )

    assert isinstance(email, str)
    assert "@" in email
    assert "." in email.split("@")[1]


def test_get_email_address_without_middle_name():
    email = get_email_address(
        first_name="Jane", middle_name=None, last_name="Smith", age=25, birth_date=date(1995, 5, 15)
    )

    assert isinstance(email, str)
    assert "@" in email


def test_get_email_address_components():
    with patch("data_designer.engine.sampling_gen.entities.email_address_utils.get_email_domain_by_age") as mock_domain:
        with patch(
            "data_designer.engine.sampling_gen.entities.email_address_utils.get_email_basename_by_name"
        ) as mock_basename:
            with patch(
                "data_designer.engine.sampling_gen.entities.email_address_utils.get_email_suffix_by_birth_date"
            ) as mock_suffix:
                mock_domain.return_value = "gmail.com"
                mock_basename.return_value = "johndoe"
                mock_suffix.return_value = "90"

                email = get_email_address(
                    first_name="John", middle_name=None, last_name="Doe", age=30, birth_date=date(1990, 1, 1)
                )

                assert email == "johndoe90@gmail.com"
                mock_domain.assert_called_once_with(30)
                mock_basename.assert_called_once_with("John", None, "Doe")
                mock_suffix.assert_called_once_with(date(1990, 1, 1))


def test_get_email_domain_age_groups():
    with patch("random.choices") as mock_choices:
        mock_choices.return_value = ["test.com"]

        domain_25 = get_email_domain_by_age(25)
        domain_35 = get_email_domain_by_age(35)
        domain_55 = get_email_domain_by_age(55)

        assert domain_25 == "test.com"
        assert domain_35 == "test.com"
        assert domain_55 == "test.com"


def test_get_email_basename_basic():
    basename = get_email_basename_by_name("John", None, "Doe")
    assert isinstance(basename, str)
    assert len(basename) > 0


def test_get_email_basename_with_middle_name():
    basename = get_email_basename_by_name("John", "Michael", "Doe")
    assert isinstance(basename, str)
    assert len(basename) > 0


def test_get_email_basename_special_characters():
    basename = get_email_basename_by_name("José", None, "O'Connor")
    assert isinstance(basename, str)
    assert len(basename) > 0
    assert basename.isalnum() or "." in basename or "_" in basename


def test_get_email_basename_empty_names():
    with pytest.raises(AssertionError, match="Both first and last name must be non-empty"):
        get_email_basename_by_name("", None, "Doe")


def test_get_email_suffix_basic():
    suffix = get_email_suffix_by_birth_date(date(1990, 1, 1))
    assert isinstance(suffix, str)


def test_get_email_suffix_different_dates():
    test_dates = [date(1980, 5, 15), date(2000, 12, 31), date(1995, 6, 1)]

    for test_date in test_dates:
        suffix = get_email_suffix_by_birth_date(test_date)
        assert isinstance(suffix, str)
