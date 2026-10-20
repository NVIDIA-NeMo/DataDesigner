# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from faker import Faker
from faker.config import AVAILABLE_LOCALES as FAKER_AVAILABLE_LOCALES

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.sampler_params import PersonSamplerParams
from data_designer.config.utils.constants import AVAILABLE_LOCALES
from data_designer.engine.resources.person_reader import LocalPersonReader
from data_designer.engine.sampling_gen.data_sources.sources import PersonSampler
from data_designer.engine.sampling_gen.entities.person import load_person_data_sampler
from data_designer.engine.sampling_gen.people_gen import PeopleGenFaker, PeopleGenFromDataset

NUM_PEOPLE = 100
FAKER_LOCALE = "en_GB"
PGM_LOCALE = "en_US"


def test_all_available_locales(stub_people_gen_resource):
    # Filter out deprecated locales to avoid warnings
    deprecated_locales = {"fr_QC"}  # fr_QC is deprecated, use fr_CA instead
    available_locales = [locale for locale in AVAILABLE_LOCALES if locale not in deprecated_locales]

    for locale in available_locales:
        if locale == PGM_LOCALE:
            pop = stub_people_gen_resource[locale].generate(NUM_PEOPLE)
        else:
            pop = PeopleGenFaker(Faker(locale), locale).generate(NUM_PEOPLE)
        assert {p["locale"] for p in pop} == {locale}
        assert len(pop) == NUM_PEOPLE


def test_available_locales_are_the_same_as_faker_available_locales():
    faker_locales = set(FAKER_AVAILABLE_LOCALES)
    faker_locales.remove("fr_QC")
    assert set(AVAILABLE_LOCALES) == set(faker_locales)


def test_people_fix_parameters_faker():
    pop = PeopleGenFaker(Faker(FAKER_LOCALE), FAKER_LOCALE).generate(NUM_PEOPLE, sex="Male", city="London")
    assert len(pop) == NUM_PEOPLE
    assert {p["sex"] for p in pop} == {"Male"}
    assert {p["city"] for p in pop} == {"London"}
    assert {p["locale"] for p in pop} == {FAKER_LOCALE}


def test_people_fix_parameters_pgm(stub_people_gen_pgm):
    pop = stub_people_gen_pgm.generate(NUM_PEOPLE, sex="Female", city="Brooklyn")
    assert len(pop) == NUM_PEOPLE
    assert {p["sex"] for p in pop} == {"Female"}
    assert {p["city"] for p in pop} == {"Brooklyn"}
    assert {p["locale"] for p in pop} == {PGM_LOCALE}


def test_people_with_personas_pgm(stub_people_gen_with_personas):
    pop = stub_people_gen_with_personas.generate(NUM_PEOPLE, with_synthetic_personas=True)
    assert len(pop) == NUM_PEOPLE
    assert {p["locale"] for p in pop} == {PGM_LOCALE}
    # Check that at least one person has the career_goals_and_ambitions field
    assert any("career_goals_and_ambitions" in p for p in pop)


@pytest.mark.parametrize(
    "locale,demographics",
    [
        (
            "en_GB",
            {
                "district": "Camden",
                "ethnicity": "White British",
                "country_of_birth": "United Kingdom",
                "disability": "Not disabled",
                "region": "London",
                "religion": "No religion",
                "industry": "Education",
            },
        ),
        (
            "de_DE",
            {
                "migration_background": "No migration background",
                "employment_status": "Employed",
                "job_position": "Employee",
                "district": "Berlin",
                "region": "Berlin",
                "occupation_kldb": "831",
                "last_occupation": "Teacher",
                "last_occupation_kldb": "841",
                "kreis_code": "11000",
                "nationality": "German",
                "household_type": "Single person",
                "housing_tenure": "Rented",
                "religion": "No religion",
                "industry": "Education",
            },
        ),
    ],
)
@pytest.mark.parametrize("with_synthetic_personas", [False, True])
def test_managed_uk_germany_sampling(
    tmp_path: Path, locale: str, demographics: dict[str, Any], with_synthetic_personas: bool
) -> None:
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    record = {
        "first_name": "Alex",
        "last_name": "Smith",
        "age": 35,
        "sex": "Weiblich" if locale == "de_DE" else "Female",
        "persona": "Enjoys learning languages.",
        "finance_persona": "Saves regularly.",
        "healthcare_persona": "Exercises daily.",
        "openness": '{"t_score": 60, "label": "High"}',
        "internal_metadata": "excluded",
        **demographics,
    }
    geography_field = "district"
    other_record = {**record, geography_field: "Elsewhere"}
    lazy.pd.DataFrame([record, other_record]).to_parquet(dataset_dir / f"{locale}.parquet")
    engine = load_person_data_sampler(LocalPersonReader(tmp_path), locale)
    generator = PeopleGenFromDataset(engine, locale)

    params = PersonSamplerParams(
        locale=locale,
        age_range=[30, 40],
        sex="Female",
        select_field_values={geography_field: [demographics[geography_field]]},
        with_synthetic_personas=with_synthetic_personas,
    )
    sampler = PersonSampler(params, people_gen_resource={params.people_gen_key: generator})
    people = sampler.sample(1)

    assert len(people) == 1
    person = people[0]
    assert person["locale"] == locale
    assert person["age"] == 35
    assert person["sex"] == "Female"
    assert person["birth_date"]
    assert person["email_address"]
    assert person["phone_number"] is None
    assert person["national_id"] is None
    assert "internal_metadata" not in person
    for field, value in demographics.items():
        assert person[field] == value
    for field in ["persona", "finance_persona", "healthcare_persona", "openness"]:
        assert (field in person) == with_synthetic_personas
        if with_synthetic_personas:
            assert person[field] == record[field]
