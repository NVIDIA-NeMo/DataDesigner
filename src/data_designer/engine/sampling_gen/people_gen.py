# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
import random
from typing import TYPE_CHECKING, Any, Union
import uuid

from faker import Faker
import pandas as pd

from data_designer.config.utils.constants import AVAILABLE_LOCALES, DEFAULT_AGE_RANGE
from data_designer.engine.resources.managed_dataset_generator import ManagedDatasetGenerator
from data_designer.engine.sampling_gen.entities.dataset_based_person_fields import PERSONA_FIELDS, PII_FIELDS
from data_designer.engine.sampling_gen.entities.person import (
    convert_age_to_birth_date,
    generate_and_insert_derived_fields,
)
from data_designer.engine.sampling_gen.errors import ManagedDatasetGeneratorError
from data_designer.engine.sampling_gen.person_constants import faker_constants

if TYPE_CHECKING:
    from data_designer.engine.sampling_gen.schema import DataSchema


EngineT = Union[Faker, ManagedDatasetGenerator]


class PeopleGen(ABC):
    """Unified interface for generating people data."""

    def __init__(self, engine: EngineT, locale: str):
        if locale not in AVAILABLE_LOCALES:
            raise ValueError(
                f"Locale {locale} is not a supported locale.Supported locales: {', '.join(AVAILABLE_LOCALES)}"
            )
        self.locale = locale
        self._engine = engine

    def set_engine(self, engine: EngineT) -> None:
        self._engine = engine

    @abstractmethod
    def generate(self, n: int, **kwargs) -> list[dict[str, Any]]: ...


class PeopleGenFaker(PeopleGen):
    @property
    def _fake(self) -> Faker:
        return self._engine

    def try_fake_else_none(self, attr_name: str, none_fill: Any | None = None) -> type:
        try:
            return getattr(self._fake, attr_name)()
        except AttributeError:
            return none_fill

    def _generate_name_and_sex(self, **kwargs) -> dict[str, str]:
        options = faker_constants.sex
        if "sex" in kwargs and kwargs["sex"] in [*options, *[[o] for o in options]]:
            sex = random.choice(kwargs["sex"]) if isinstance(kwargs["sex"], list) else kwargs["sex"]
        else:
            sex = random.choice(options)

        return {
            "first_name": getattr(self._fake, f"first_name_{sex.lower()}")(),
            "last_name": getattr(self._fake, f"last_name_{sex.lower()}")(),
            "middle_name": None,
            "sex": sex,
        }

    def _generate_address_fields(self, **kwargs) -> dict[str, str]:
        address = {
            "street_number": self.try_fake_else_none(faker_constants.attr_map["street_number"]),
            "street_name": self.try_fake_else_none("street_name"),
        }

        # Location fields can be filtered using the fixed_kwargs.
        for attr in faker_constants.location:
            if attr in kwargs:
                address[attr] = random.choice(kwargs[attr]) if isinstance(kwargs[attr], list) else kwargs[attr]
            else:
                address[attr] = self.try_fake_else_none(attr)

        return address

    def _generate_age(self, **kwargs) -> int:
        return random.randint(*kwargs.get("age_range", DEFAULT_AGE_RANGE))

    def _generate_marital_status(self, **kwargs) -> str:
        return random.choice(faker_constants.marital_status)

    def _generate_bachelors_field(self, **kwargs) -> str:
        return random.choice(faker_constants.bachelors)

    def _generate_education_level(self, **kwargs) -> str:
        return random.choice(faker_constants.education_level)

    def make_person(self, **kwargs) -> dict[str, Any]:
        person = {"uuid": str(uuid.uuid4()), "locale": self.locale}
        person.update(self._generate_name_and_sex(**kwargs))
        person.update(self._generate_address_fields(**kwargs))
        person.update({"age": self._generate_age(**kwargs)})
        person.update({"birth_date": convert_age_to_birth_date(person["age"]).isoformat()})
        person.update({"country": self.try_fake_else_none("country")})
        person.update({"marital_status": self._generate_marital_status(**kwargs)})
        person.update({"education_level": self._generate_education_level(**kwargs)})
        person.update({"unit": ""})
        person.update({"occupation": self.try_fake_else_none(faker_constants.attr_map["occupation"])})
        person.update({"phone_number": (self.try_fake_else_none("phone_number") if person["age"] >= 18 else None)})
        if person["education_level"] in faker_constants.college_level:
            person.update({"bachelors_field": self._generate_bachelors_field(**kwargs)})
        else:
            person.update({"bachelors_field": "no_degree"})
        return person

    def generate(self, n: int, **kwargs) -> list[dict[str, Any]]:
        return [self.make_person(**kwargs) for _ in range(n)]


class PeopleGenFromDataset(PeopleGen):
    def _get_ages(self, age_range: tuple[int, int]) -> list[int]:
        return list(range(age_range[0], age_range[1] + 1))

    def _generate_from_dataset(self, n: int, **kwargs) -> pd.DataFrame:
        kw = deepcopy(kwargs)
        with_synthetic_personas = kw.pop("with_synthetic_personas", False)
        kw["age"] = self._get_ages(kw.pop("age_range", DEFAULT_AGE_RANGE))

        # Generate samples and drop columns where all rows are null.
        df = self._engine.generate_samples(size=n, evidence=kw).dropna(axis=1, how="all")

        # We need this for derived fields.
        df["locale"] = self.locale

        # Only keep columns that are listed in the schema.
        fields = [field for field in PII_FIELDS if field in df.columns]
        if with_synthetic_personas:
            fields.extend([field for field in PERSONA_FIELDS if field in df.columns])

        return df[fields]

    def generate(self, n: int, **kwargs) -> list[dict[str, Any]]:
        return [
            generate_and_insert_derived_fields(p)
            for p in self._generate_from_dataset(n, **kwargs).to_dict(orient="records")
        ]


def create_people_gen_resource(
    schema: DataSchema,
    person_generator_loader: Callable[[bool], ManagedDatasetGenerator] | None = None,
) -> dict[str, PeopleGen]:
    """Creates resource of unique people generators needed to generate the dataset.

    The resource is a dictionary of person generators, where the keys are the following:
        - {locale} for dataset-based person generators
        - {locale}_with_personas for dataset-based person generators with synthetic personas
        - {locale}_faker for faker-based person generators

    Args:
        schema: Schema of the dataset that we will generate.
        person_generator_loader: Function that loads a managed dataset generator.

    Returns:
        Dictionary of unique people generators needed to generate the dataset.
    """
    people_gen_resource = {}

    # ------------------------------------------------------------
    # Preload dataset-based person generators
    # ------------------------------------------------------------

    for column in schema.get_columns_by_sampler_type("person"):
        for params in [column.params, *list(column.conditional_params.values())]:
            if params.people_gen_key not in people_gen_resource:
                try:
                    engine = person_generator_loader(locale=params.locale)
                    people_gen_resource[params.people_gen_key] = PeopleGenFromDataset(
                        engine=engine, locale=params.locale
                    )
                except Exception as e:
                    raise ManagedDatasetGeneratorError(
                        f"🛑 Failed to load dataset-based person generator for locale {params.locale}. "
                        "Please check if you have access to person data for this locale. "
                    ) from e

    # ------------------------------------------------------------
    # Preload faker-based person generators
    # ------------------------------------------------------------

    for column in schema.get_columns_by_sampler_type("person_from_faker"):
        for params in [column.params, *list(column.conditional_params.values())]:
            if params.people_gen_key not in people_gen_resource:
                people_gen_resource[params.people_gen_key] = PeopleGenFaker(
                    engine=Faker(params.locale), locale=params.locale
                )

    return people_gen_resource
