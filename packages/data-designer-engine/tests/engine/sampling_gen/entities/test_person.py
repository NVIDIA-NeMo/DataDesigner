# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import date

import pytest

from data_designer.engine.sampling_gen.entities.dataset_based_person_fields import PERSONA_FIELDS, PII_FIELDS
from data_designer.engine.sampling_gen.entities.errors import MissingPersonFieldsError
from data_designer.engine.sampling_gen.entities.person import (
    convert_age_to_birth_date,
    generate_and_insert_derived_fields,
    generate_email_address,
    generate_phone_number,
    get_national_id,
)


def test_convert_age_to_birth_date():
    """Test that convert_age_to_birth_date generates a valid birth date."""
    age = 30
    birth_date = convert_age_to_birth_date(age)

    assert isinstance(birth_date, date)
    today = date.today()
    expected_year = today.year - age
    assert birth_date.year in [expected_year, expected_year - 1]


def test_convert_age_to_birth_date_child():
    """Test birth date generation for a child."""
    age = 10
    birth_date = convert_age_to_birth_date(age)

    assert isinstance(birth_date, date)
    today = date.today()
    expected_year = today.year - age
    assert birth_date.year in [expected_year, expected_year - 1]


def test_generate_email_address_adult():
    """Test email generation for adults."""
    birth_date = date(1995, 6, 15)
    email = generate_email_address(
        first_name="John", middle_name="Michael", last_name="Doe", age=30, birth_date=birth_date
    )

    assert email is not None
    assert "@" in str(email)


def test_generate_email_address_adult_no_middle_name():
    """Test email generation for adults without middle name."""
    birth_date = date(2000, 3, 20)
    email = generate_email_address(
        first_name="Jane", middle_name=None, last_name="Smith", age=25, birth_date=birth_date
    )

    assert email is not None
    assert "@" in str(email)


def test_generate_email_address_child():
    """Test that children don't get email addresses."""
    birth_date = date(2015, 8, 10)
    email = generate_email_address(
        first_name="Child", middle_name=None, last_name="Person", age=10, birth_date=birth_date
    )

    assert email is None


def test_get_national_id_us_locale():
    """Test SSN generation for US locale."""
    birth_date = date(1995, 6, 15)
    national_id = get_national_id(locale="en_US", region="NY", birth_date=birth_date)

    assert national_id is not None
    assert isinstance(national_id, str)
    assert len(national_id) == 11
    assert national_id.count("-") == 2


def test_get_national_id_non_us_locale():
    """Test that non-US locales don't get national IDs."""
    birth_date = date(1995, 6, 15)
    national_id = get_national_id(locale="en_CA", region="Ontario", birth_date=birth_date)

    assert national_id is None


def test_get_national_id_missing_data():
    """Test that national ID is None when required data is missing."""
    birth_date = date(1995, 6, 15)
    national_id = get_national_id(locale="en_US", region=None, birth_date=birth_date)

    assert national_id is None


def test_generate_phone_number_adult():
    """Test phone number generation for adults."""
    phone = generate_phone_number(locale="en_US", age=30, postcode="10001")

    assert phone is not None
    assert hasattr(phone, "format")


def test_generate_phone_number_child():
    """Test that children don't get phone numbers."""
    phone = generate_phone_number(locale="en_US", age=10, postcode="10001")

    assert phone is None


def test_generate_phone_number_non_us_locale():
    """Test that non-US locales don't get phone numbers."""
    phone = generate_phone_number(locale="en_CA", age=30, postcode="M5H2N2")

    assert phone is None


def test_generate_phone_number_missing_data():
    """Test that phone number is None when required data is missing."""
    phone = generate_phone_number(locale="en_US", age=30, postcode=None)

    assert phone is None


def test_generate_and_insert_derived_fields():
    """Test that derived fields are inserted into a person record."""
    person_record = {
        "first_name": "John",
        "middle_name": "Michael",
        "last_name": "Doe",
        "age": 30,
        "locale": "en_US",
        "region": "NY",
        "postcode": "10001",
    }

    result = generate_and_insert_derived_fields(person_record)

    assert "birth_date" in result
    assert "phone_number" in result
    assert "email_address" in result
    assert "national_id" in result

    assert result["birth_date"] is not None
    assert isinstance(result["birth_date"], str)
    assert result["phone_number"] is not None
    assert result["email_address"] is not None
    assert result["national_id"] is not None


def test_generate_and_insert_derived_fields_child():
    """Test derived fields for a child record."""
    person_record = {
        "first_name": "Child",
        "middle_name": None,
        "last_name": "Person",
        "age": 10,
        "locale": "en_US",
        "region": "NY",
        "postcode": "10001",
    }

    result = generate_and_insert_derived_fields(person_record)

    assert "birth_date" in result
    assert "phone_number" in result
    assert "email_address" in result
    assert "national_id" in result

    # Children should have a birth date
    assert result["birth_date"] is not None
    assert isinstance(result["birth_date"], str)
    # Children should not have phone or email
    assert result["phone_number"] is None
    assert result["email_address"] is None
    # But they should still have an SSN
    assert result["national_id"] is not None


def test_generate_and_insert_derived_fields_non_us():
    """Test derived fields for non-US locale."""
    person_record = {
        "first_name": "Foreign",
        "middle_name": None,
        "last_name": "Person",
        "age": 30,
        "locale": "en_CA",
        "region": "Ontario",
        "postcode": "M5H2N2",
    }

    result = generate_and_insert_derived_fields(person_record)

    assert "birth_date" in result
    assert "phone_number" in result
    assert "email_address" in result
    assert "national_id" in result

    # Everyone should have a birth date
    assert result["birth_date"] is not None
    # Date should be serializable
    assert isinstance(result["birth_date"], str)
    # Non-US should not have phone or national ID
    assert result["phone_number"] is None
    assert result["national_id"] is None
    # But should have email
    assert result["email_address"] is not None


def test_pii_fields_loaded():
    """Test that PII_FIELDS is loaded correctly."""
    assert isinstance(PII_FIELDS, list)
    assert len(PII_FIELDS) > 0


def test_persona_fields_loaded():
    """Test that PERSONA_FIELDS is loaded correctly."""
    assert isinstance(PERSONA_FIELDS, list)
    assert len(PERSONA_FIELDS) > 0


def test_generate_and_insert_derived_fields_missing_required_field():
    """Test that MissingPersonFieldsError is raised when required fields are missing."""
    # Missing 'age' field
    person_record = {
        "first_name": "John",
        "last_name": "Doe",
        "locale": "en_US",
    }

    with pytest.raises(MissingPersonFieldsError) as exc_info:
        generate_and_insert_derived_fields(person_record)

    assert "age" in str(exc_info.value)


def test_generate_and_insert_derived_fields_missing_multiple_required_fields():
    """Test that MissingPersonFieldsError is raised when multiple required fields are missing."""
    # Missing 'first_name', 'last_name', and 'age' fields
    person_record = {
        "locale": "en_US",
    }

    with pytest.raises(MissingPersonFieldsError) as exc_info:
        generate_and_insert_derived_fields(person_record)

    error_message = str(exc_info.value)
    assert "first_name" in error_message
    assert "last_name" in error_message
    assert "age" in error_message


def test_generate_and_insert_derived_fields_missing_locale():
    """Test that MissingPersonFieldsError is raised when locale is missing."""
    person_record = {
        "first_name": "John",
        "last_name": "Doe",
        "age": 30,
    }

    with pytest.raises(MissingPersonFieldsError) as exc_info:
        generate_and_insert_derived_fields(person_record)

    assert "locale" in str(exc_info.value)


def test_generate_and_insert_derived_fields_with_all_required_fields():
    """Test that no error is raised when all required fields are present."""
    # Minimal record with only required fields
    person_record = {
        "first_name": "John",
        "last_name": "Doe",
        "age": 30,
        "locale": "en_US",
    }

    # Should not raise an error
    result = generate_and_insert_derived_fields(person_record)

    # Verify that derived fields were added
    assert "birth_date" in result
    assert result["birth_date"] is not None
