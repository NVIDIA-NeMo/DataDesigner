# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from data_designer.engine.sampling_gen.entities.phone_number import PhoneNumber, get_area_code


@pytest.fixture
def stub_sample_phone_number():
    return PhoneNumber(country_code="1", area_code="555", prefix="123", line_number="4567")


def test_phone_number_creation(stub_sample_phone_number):
    assert stub_sample_phone_number.country_code == "1"
    assert stub_sample_phone_number.area_code == "555"
    assert stub_sample_phone_number.prefix == "123"
    assert stub_sample_phone_number.line_number == "4567"


def test_phone_number_default_country_code():
    phone = PhoneNumber(area_code="555", prefix="123", line_number="4567")
    assert phone.country_code == "1"


def test_validate_digits_invalid():
    with pytest.raises(ValueError, match="Must contain only digits"):
        PhoneNumber(country_code="1a", area_code="555", prefix="123", line_number="4567")


def test_validate_country_code_length_invalid():
    with pytest.raises(ValueError, match="Country code 1234 is longer than 3 digits"):
        PhoneNumber(country_code="1234", area_code="555", prefix="123", line_number="4567")


def test_format_styles(stub_sample_phone_number):
    assert stub_sample_phone_number.format("dash") == "555-123-4567"
    assert stub_sample_phone_number.format("parentheses") == "(555) 123-4567"
    assert stub_sample_phone_number.format("dot") == "555.123.4567"
    assert stub_sample_phone_number.format("space") == "555 123 4567"
    assert stub_sample_phone_number.format("no_separation") == "5551234567"
    assert stub_sample_phone_number.format("international_plus") == "+1 555 123 4567"
    assert stub_sample_phone_number.format("international") == "001 555 123 4567"


def test_format_invalid_style(stub_sample_phone_number):
    with pytest.raises(ValueError, match="Unsupported format style"):
        stub_sample_phone_number.format("invalid_style")


def test_from_area_code():
    with patch("random.randint") as mock_randint:
        mock_randint.side_effect = [500, 1234]

        phone = PhoneNumber.from_area_code("555")

        assert phone.area_code == "555"
        assert phone.prefix == "500"
        assert phone.line_number == "1234"


def test_str_and_repr_representations(stub_sample_phone_number):
    assert str(stub_sample_phone_number) == "555-123-4567"
    assert repr(stub_sample_phone_number) == "PhoneNumber(555-123-4567)"


def test_get_area_code_exact_zipcode():
    with patch("data_designer.engine.sampling_gen.entities.phone_number.ZIPCODE_AREA_CODE_MAP", {"12345": "555"}):
        result = get_area_code("12345")
        assert result == "555"


def test_get_area_code_exact_zipcode_not_found():
    with patch("data_designer.engine.sampling_gen.entities.phone_number.ZIPCODE_AREA_CODE_MAP", {}):
        with pytest.raises(ValueError, match="ZIP code 12345 not found"):
            get_area_code("12345")


def test_get_area_code_prefix_match():
    with patch(
        "data_designer.engine.sampling_gen.entities.phone_number.ZIPCODE_POPULATION_MAP",
        {"12345": 100, "12346": 200},
    ):
        with patch(
            "data_designer.engine.sampling_gen.entities.phone_number.ZIPCODE_AREA_CODE_MAP",
            {"12345": "555", "12346": "666"},
        ):
            with patch("random.choices") as mock_choices:
                mock_choices.return_value = ["12345"]

                result = get_area_code("123")
                assert result == "555"
