# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from faker import Faker
from faker.config import AVAILABLE_LOCALES as FAKER_AVAILABLE_LOCALES

from data_designer.config.utils.constants import AVAILABLE_LOCALES
from data_designer.engine.sampling_gen.people_gen import PeopleGenFaker

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
