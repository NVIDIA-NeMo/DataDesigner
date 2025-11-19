# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import date, timedelta
import random
from typing import Any, Literal, TypeAlias

from data_designer.config.utils.constants import LOCALES_WITH_MANAGED_DATASETS
from data_designer.engine.resources.managed_dataset_generator import ManagedDatasetGenerator
from data_designer.engine.resources.managed_dataset_repository import load_managed_dataset_repository
from data_designer.engine.resources.managed_storage import ManagedBlobStorage
from data_designer.engine.sampling_gen.entities.dataset_based_person_fields import (
    PERSONA_FIELDS,
    PII_FIELDS,
    REQUIRED_FIELDS,
)
from data_designer.engine.sampling_gen.entities.email_address_utils import get_email_address
from data_designer.engine.sampling_gen.entities.errors import MissingPersonFieldsError
from data_designer.engine.sampling_gen.entities.national_id_utils import generate_ssn
from data_designer.engine.sampling_gen.entities.phone_number import PhoneNumber
from data_designer.engine.sampling_gen.errors import DatasetNotAvailableForLocaleError

SexT: TypeAlias = Literal["Male", "Female"]


def convert_age_to_birth_date(age: int) -> date:
    today = date.today()
    start_date = today.replace(year=today.year - age - 1)
    end_date = today.replace(year=today.year - age)
    days_between = (end_date - start_date).days
    random_days = random.randint(0, days_between)
    birthdate = start_date + timedelta(days=random_days)
    return birthdate


def generate_email_address(
    first_name: str,
    middle_name: str | None,
    last_name: str,
    age: int,
    birth_date: date,
) -> str | None:
    """
    Generate an email address based on the person's attributes.
    Email address is None for children. Uses common free email domains.
    """
    if age < 18:
        return None
    return get_email_address(
        first_name=first_name,
        middle_name=middle_name,
        last_name=last_name,
        age=age,
        birth_date=birth_date,
    )


def get_national_id(locale: str | None, region: str | None, birth_date: date) -> str | None:
    if locale != "en_US":
        return None
    if region is None:
        return None
    return generate_ssn(state=region, birth_date=birth_date)


def generate_phone_number(locale: str, age: int, postcode: str | None, style: str = "dash") -> str | None:
    """
    Generate a phone number correlated with location (postcode).
    Phone number is None for children.
    """
    if locale != "en_US":
        return None
    if age < 18:
        return None
    if postcode is None:
        return None
    locality_var = random.random()
    if locality_var < 0.6:
        # Exact match to postcode 60% of the time
        return PhoneNumber.from_zip_prefix(postcode).format(style=style)
    elif locality_var < 0.8:
        # Nearby postcodes 20% of the time
        return PhoneNumber.from_zip_prefix(postcode[:4]).format(style=style)
    elif locality_var < 0.9:
        # More distant postcodes 10% of the time
        return PhoneNumber.from_zip_prefix(postcode[:3]).format(style=style)
    # Random (population-weighted) area code 10% of the time
    return PhoneNumber.generate().format(style=style)


def generate_and_insert_derived_fields(person_record: dict[str, Any]) -> dict[str, str | None]:
    _verify_required_fields(person_record)
    birth_date = convert_age_to_birth_date(person_record.get("age"))
    person_record.update(
        {
            # Note: All data must be serializable to JSON.
            "birth_date": birth_date.isoformat(),
            "phone_number": generate_phone_number(
                locale=person_record.get("locale"),
                age=person_record.get("age"),
                postcode=person_record.get("postcode"),
            ),
            "email_address": generate_email_address(
                first_name=person_record.get("first_name"),
                middle_name=person_record.get("middle_name"),
                last_name=person_record.get("last_name"),
                age=person_record.get("age"),
                birth_date=birth_date,
            ),
            "national_id": get_national_id(
                locale=person_record.get("locale"),
                region=person_record.get("region"),
                birth_date=birth_date,
            ),
        }
    )
    if person_record.get("locale") == "en_US" and "region" in person_record and "state" not in person_record:
        state = person_record.pop("region")
        person_record.update({"state": state})

    return {
        **{k: v for k, v in person_record.items() if k in PII_FIELDS},
        **{k: v for k, v in person_record.items() if k in ["state", "phone_number", "email_address", "national_id"]},
        **{k: v for k, v in person_record.items() if k in PERSONA_FIELDS},
    }


def load_person_data_sampler(blob_storage: ManagedBlobStorage, locale: str) -> ManagedDatasetGenerator:
    if locale not in LOCALES_WITH_MANAGED_DATASETS:
        raise DatasetNotAvailableForLocaleError(f"Locale {locale} is not supported by the managed dataset generator.")

    return ManagedDatasetGenerator(
        managed_datasets=load_managed_dataset_repository(blob_storage, [locale]),
        dataset_name=locale,
    )


def _verify_required_fields(person_record: dict[str, Any]) -> None:
    """Verify that the person record contains all required fields."""
    missing_fields = REQUIRED_FIELDS - set(person_record.keys())
    if missing_fields:
        raise MissingPersonFieldsError(f"Person data is missing the following required fields: {missing_fields}")
