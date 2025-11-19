# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from decimal import Decimal
from functools import partial

import pandas as pd
import pytest

from data_designer.config.sampler_constraints import ColumnInequalityConstraint, ScalarInequalityConstraint
from data_designer.config.sampler_params import SamplerType
from data_designer.engine.sampling_gen.errors import RejectionSamplingError
from data_designer.engine.sampling_gen.generator import DatasetGenerator

TEST_LOCALE_1 = "en_GB"
TEST_LOCALE_2 = "fr_FR"
PGM_LOCALE = "en_US"
NUM_SAMPLES = 100


def test_dataset_generator(stub_sampler_columns, stub_person_generator_loader):
    generator = DatasetGenerator(
        sampler_columns=stub_sampler_columns, person_generator_loader=stub_person_generator_loader
    )
    dataset = generator.generate(NUM_SAMPLES)
    assert dataset.shape == (NUM_SAMPLES, len(stub_sampler_columns.columns))


def test_float_column_stays_float(stub_schema_builder):
    stub_schema_builder.add_column(
        name="col_1",
        sampler_type=SamplerType.GAUSSIAN,
        params={"mean": 0, "stddev": 1},
    )
    generator = DatasetGenerator(sampler_columns=stub_schema_builder.to_sampler_columns())
    dataset = generator.generate(NUM_SAMPLES)
    assert dataset["col_1"].dtype == "float64"


def test_dataset_column_convert_to_int(stub_schema_builder):
    stub_schema_builder.add_column(
        name="col_1",
        sampler_type=SamplerType.GAUSSIAN,
        params={"mean": 0, "stddev": 1},
        convert_to="int",
    )
    generator = DatasetGenerator(sampler_columns=stub_schema_builder.to_sampler_columns())
    dataset = generator.generate(NUM_SAMPLES)
    assert dataset["col_1"].dtype == "int64"


def test_datetime_formats(stub_schema_builder):
    stub_schema_builder.add_column(
        name="year",
        sampler_type=SamplerType.DATETIME,
        params={"start": "2020-01-01", "end": "2025-01-01", "unit": "Y"},
    )

    stub_schema_builder.add_column(
        name="datetime",
        sampler_type=SamplerType.DATETIME,
        params={"start": "2020-01-01", "end": "2025-01-01", "unit": "s"},
    )

    generator = DatasetGenerator(sampler_columns=stub_schema_builder.to_sampler_columns())
    dataset = generator.generate(100)

    assert dataset["year"].str.match(r"\d{4}").all()
    assert dataset["datetime"].str.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}").all()


def test_timedelta(stub_schema_builder):
    stub_schema_builder.add_column(
        name="reference_date",
        sampler_type=SamplerType.DATETIME,
        params={
            "start": "2020-01-01",
            "end": "2025-01-01",
            "unit": "D",
        },
    )

    stub_schema_builder.add_column(
        name="new_date",
        sampler_type=SamplerType.TIMEDELTA,
        params={
            "dt_min": 5,
            "dt_max": 10,
            "reference_column_name": "reference_date",
            "unit": "D",
        },
    )

    generator = DatasetGenerator(sampler_columns=stub_schema_builder.to_sampler_columns())
    dataset = generator.generate(100)

    assert dataset["new_date"].str.match(r"\d{4}-\d{2}-\d{2}").all()

    dt = pd.to_datetime(dataset["new_date"]) - pd.to_datetime(dataset["reference_date"])
    assert (dt <= pd.Timedelta(days=10)).all()
    assert (dt >= pd.Timedelta(days=5)).all()


@pytest.mark.parametrize(
    ("sampler_type", "params"),
    [
        (SamplerType.POISSON, {"mean": 5}),
        (SamplerType.BINOMIAL, {"n": 10, "p": 0.5}),
    ],
)
def test_discrete_samplers_return_int_without_convert_to(stub_schema_builder, sampler_type, params):
    stub_schema_builder.add_column(
        name="col_1",
        sampler_type=sampler_type,
        params=params,
    )

    # Add constraint to ensure he hit rejection sampling.
    stub_schema_builder.add_constraint(ScalarInequalityConstraint(target_column="col_1", operator="gt", rhs=2))
    generator = DatasetGenerator(sampler_columns=stub_schema_builder.to_sampler_columns())
    dataset = generator.generate(NUM_SAMPLES)
    assert dataset["col_1"].dtype == "int64"


def test_dataset_column_convert_datetime_format(stub_schema_builder):
    stub_schema_builder.add_column(
        name="col_1",
        sampler_type=SamplerType.DATETIME,
        params={
            "start": "2020-01-01",
            "end": "2025-01-01",
            "unit": "D",
        },
        convert_to="%m/%d/%Y",
    )
    generator = DatasetGenerator(sampler_columns=stub_schema_builder.to_sampler_columns())
    dataset = generator.generate(NUM_SAMPLES)
    assert dataset["col_1"].dtype == "object"
    assert dataset["col_1"].str.contains(r"\d{2}/\d{2}/\d{4}").all()
    assert pd.to_datetime(dataset["col_1"], format="%m/%d/%Y").notna().all()


def test_dataset_with_conditionals(stub_schema_builder):
    stub_schema_builder.add_column(
        name="col_1",
        sampler_type=SamplerType.UNIFORM,
        params={"low": 0, "high": 1},
        conditional_params={
            "col_2 == 'high'": {"low": 2, "high": 5},
        },
    )
    stub_schema_builder.add_column(
        name="col_2",
        sampler_type=SamplerType.CATEGORY,
        params={
            "values": ["low", "high"],
        },
    )
    generator = DatasetGenerator(sampler_columns=stub_schema_builder.to_sampler_columns())
    dataset = generator.generate(NUM_SAMPLES)
    assert dataset.query("col_2 == 'low'")["col_1"].between(0, 1).all()
    assert dataset.query("col_2 == 'high'")["col_1"].between(2, 5).all()


def test_dataset_with_constraints(stub_schema_builder):
    stub_schema_builder.add_column(
        name="col_1",
        sampler_type=SamplerType.UNIFORM,
        params={"low": 0, "high": 1},
    )
    stub_schema_builder.add_column(
        name="col_2",
        sampler_type=SamplerType.UNIFORM,
        params={"low": 0.3, "high": 1},
    )
    stub_schema_builder.add_constraint(ColumnInequalityConstraint(target_column="col_1", operator="lt", rhs="col_2"))
    generator = DatasetGenerator(sampler_columns=stub_schema_builder.to_sampler_columns(max_rejections_factor=10))
    dataset = generator.generate(NUM_SAMPLES)
    assert (dataset["col_1"] < dataset["col_2"]).all()


def test_subcategory_generation(stub_schema_builder):
    subcategory_values = {
        "electronics": ["laptop", "smartphone", "tablet"],
        "clothing": ["shirt", "pants", "shoes", "hat"],
        "furniture": ["sofa", "table", "chair"],
        "appliances": ["refrigerator", "oven"],
    }

    stub_schema_builder.add_column(
        name="department",
        sampler_type="category",
        params={"values": list(subcategory_values.keys())},
    )

    stub_schema_builder.add_column(
        name="products",
        sampler_type="subcategory",
        params={
            "category": "department",
            "values": subcategory_values,
        },
    )

    generator = DatasetGenerator(sampler_columns=stub_schema_builder.to_sampler_columns())
    df = generator.generate(NUM_SAMPLES)

    assert df.shape == (NUM_SAMPLES, len(stub_schema_builder.to_sampler_columns().columns))
    for cat, vals in subcategory_values.items():
        assert df.query(f"department == '{cat}'")["products"].isin(vals).all()


def test_generation_with_people(stub_schema_builder, stub_person_generator_loader):
    stub_schema_builder.add_column(name="random_number", sampler_type="uniform", params={"low": 0, "high": 100})

    # Use person_from_faker for non-managed locales
    stub_schema_builder.add_column(
        name="some_dude",
        sampler_type="person_from_faker",
        params={"locale": TEST_LOCALE_1, "sex": "Male"},
        conditional_params={"random_number > 50": {"locale": TEST_LOCALE_2, "sex": "Male"}},
    )

    stub_schema_builder.add_column(
        name="some_lady",
        sampler_type="person_from_faker",
        params={"locale": TEST_LOCALE_1, "sex": "Female"},
        conditional_params={"random_number > 50": {"locale": TEST_LOCALE_2, "sex": "Female"}},
    )

    # Use person for managed locale (en_US)
    stub_schema_builder.add_column(
        name="american_dude",
        sampler_type="person",
        params={"locale": "en_US", "sex": "Male"},
    )

    stub_schema_builder.add_column(
        name="person_with_personas",
        sampler_type="person",
        params={"locale": "en_US", "with_synthetic_personas": True},
    )

    generator = DatasetGenerator(
        sampler_columns=stub_schema_builder.to_sampler_columns(),
        # TODO: Revamp how we mock person generation. We shouldn't need to write hacks like this partial.
        person_generator_loader=partial(stub_person_generator_loader, with_synthetic_personas=True),
    )

    df = generator.generate(NUM_SAMPLES)

    assert df["american_dude"].apply(lambda x: x["locale"] == "en_US").all()
    assert df["american_dude"].apply(lambda x: x["sex"] == "Male").all()
    assert df["person_with_personas"].apply(lambda x: "career_goals_and_ambitions" in x).all()

    for col, sex in zip(["some_dude", "some_lady"], ["Male", "Female"], strict=True):
        assert df.query("random_number < 50")[col].apply(lambda x: x["locale"] == TEST_LOCALE_1).all()
        assert df.query("random_number > 50")[col].apply(lambda x: x["locale"] == TEST_LOCALE_2).all()
        assert df[col].apply(lambda x: x["sex"] == sex).all()


def test_person_with_age_range(stub_schema_builder, stub_person_generator_loader):
    age_min = 18
    age_max = 30

    # Use person_from_faker for non-managed locale
    stub_schema_builder.add_column(
        name="some_dude",
        sampler_type="person_from_faker",
        params={
            "locale": TEST_LOCALE_1,
            "sex": "Male",
            "age_range": [age_min, age_max],
        },
    )

    # Use person for managed locale (en_US)
    stub_schema_builder.add_column(
        name="some_lady",
        sampler_type="person",
        params={
            "locale": PGM_LOCALE,
            "sex": "Female",
            "age_range": [age_min, age_max],
        },
    )

    generator = DatasetGenerator(
        sampler_columns=stub_schema_builder.to_sampler_columns(),
        person_generator_loader=stub_person_generator_loader,
    )
    df = generator.generate(NUM_SAMPLES)

    assert df["some_dude"].apply(lambda x: x["age"]).min() >= age_min
    assert df["some_dude"].apply(lambda x: x["age"]).max() <= age_max
    assert df["some_lady"].apply(lambda x: x["age"]).min() >= age_min
    assert df["some_lady"].apply(lambda x: x["age"]).max() <= age_max


def test_person_with_state(stub_schema_builder, stub_person_generator_loader):
    stub_schema_builder.add_column(
        name="some_dude",
        sampler_type="person",
        params={"locale": PGM_LOCALE, "sex": "Male", "select_field_values": {"state": ["CA"]}},
    )

    stub_schema_builder.add_column(
        name="some_lady",
        sampler_type="person",
        params={"locale": PGM_LOCALE, "sex": "Female", "select_field_values": {"state": ["NV", "NY"]}},
    )

    generator = DatasetGenerator(
        sampler_columns=stub_schema_builder.to_sampler_columns(),
        person_generator_loader=stub_person_generator_loader,
    )
    df = generator.generate(NUM_SAMPLES)

    assert df["some_dude"].apply(lambda x: x["state"]).isin(["CA"]).all()
    assert df["some_lady"].apply(lambda x: x["state"]).isin(["NV", "NY"]).all()


def test_error_person_with_state_and_non_en_us_locale(stub_schema_builder):
    # PersonSamplerParams now only supports locales with managed datasets
    with pytest.raises(ValueError):
        stub_schema_builder.add_column(
            name="some_dude",
            sampler_type="person",
            params={"locale": "en_GB", "sex": "Male", "select_field_values": {"state": ["CA", "NV", "DC"]}},
        )


def test_bernoulli_mixture(stub_schema_builder):
    stub_schema_builder.add_column(
        name="agi",
        sampler_type=SamplerType.CATEGORY,
        params={"values": [">50000", "<50000"]},
    )

    stub_schema_builder.add_column(
        name="bern_x",
        sampler_type=SamplerType.BERNOULLI_MIXTURE,
        params={
            "p": 0.0001,
            "dist_name": "expon",
            "dist_params": {"scale": 1},
        },
        conditional_params={
            "agi == '>50000'": {
                "p": 0.99,
                "dist_name": "expon",
                "dist_params": {"scale": 1000},
            },
        },
    )

    sampler_columns = stub_schema_builder.to_sampler_columns()
    generator = DatasetGenerator(sampler_columns=sampler_columns)
    df = generator.generate(NUM_SAMPLES)

    assert df.shape == (NUM_SAMPLES, len(sampler_columns.columns))
    assert df.query("agi == '<50000'")["bern_x"].sum() < df.query("agi == '>50000'")["bern_x"].sum()


def test_decimal_places(stub_schema_builder):
    stub_schema_builder.add_column(
        name="col_1",
        sampler_type=SamplerType.GAUSSIAN,
        params={"mean": 10, "stddev": 1, "decimal_places": 2},
    )
    stub_schema_builder.add_column(
        name="col_2",
        sampler_type=SamplerType.UNIFORM,
        params={"low": 0.1, "high": 1, "decimal_places": 3},
    )
    generator = DatasetGenerator(sampler_columns=stub_schema_builder.to_sampler_columns())
    df = generator.generate(NUM_SAMPLES)
    assert df["col_1"].apply(lambda x: abs(Decimal(str(x)).as_tuple().exponent) <= 2).all()
    assert df["col_2"].apply(lambda x: abs(Decimal(str(x)).as_tuple().exponent) <= 3).all()


def test_e2e_example(stub_schema_builder, stub_person_generator_loader):
    stub_schema_builder.add_column(
        name="employee_id",
        sampler_type="uuid",
        params={"prefix": "ZZZ-", "short_form": True, "uppercase": True},
        conditional_params={
            "department == 'electronics'": {
                "prefix": "EEE-",
                "short_form": True,
                "uppercase": True,
            },
            "department == 'clothing'": {
                "prefix": "CCC-",
                "short_form": True,
                "uppercase": True,
            },
            "department == 'furniture'": {
                "prefix": "FFF-",
                "short_form": True,
                "uppercase": True,
            },
            "department == 'appliances'": {
                "prefix": "AAA-",
                "short_form": True,
                "uppercase": True,
            },
        },
    )

    stub_schema_builder.add_column(
        name="department",
        sampler_type="category",
        params={
            "values": ["electronics", "clothing", "furniture", "appliances"],
            "weights": [0.4, 0.3, 0.2, 0.1],
        },
    )

    stub_schema_builder.add_column(
        name="products",
        sampler_type="subcategory",
        params={
            "category": "department",
            "values": {
                "electronics": ["laptop", "smartphone", "tablet"],
                "clothing": ["shirt", "pants", "shoes"],
                "furniture": ["sofa", "table", "chair"],
                "appliances": ["refrigerator", "microwave", "oven"],
            },
        },
    )

    stub_schema_builder.add_column(
        name="age",
        sampler_type="scipy",
        params={"dist_name": "norm", "dist_params": {"loc": 25, "scale": 8}},
        conditional_params={
            "department == 'electronics'": {
                "dist_name": "norm",
                "dist_params": {"loc": 25, "scale": 5},
            },
            "department == 'clothing'": {
                "dist_name": "norm",
                "dist_params": {"loc": 20, "scale": 5},
            },
            "department == 'furniture'": {
                "dist_name": "norm",
                "dist_params": {"loc": 40, "scale": 8},
            },
            "department == 'appliances'": {
                "dist_name": "norm",
                "dist_params": {"loc": 45, "scale": 9},
            },
        },
        convert_to="int",
    )

    stub_schema_builder.add_column(
        name="start_date",
        sampler_type="datetime",
        params={"start": "2020-01-01", "end": "2025-01-01", "unit": "D"},
        convert_to="%m/%d/%Y",
    )

    stub_schema_builder.add_column(
        name="end_date",
        sampler_type="datetime",
        params={"start": "2021-01-01", "end": "2026-01-01", "unit": "D"},
        convert_to="%m/%d/%Y",
    )

    stub_schema_builder.add_column(
        name="random_number",
        sampler_type="uniform",
        params={"low": 0, "high": 100},
    )

    stub_schema_builder.add_column(
        name="some_lady",
        sampler_type="person",
        params={"locale": "en_US", "sex": "Female", "city": "New York"},
    )

    stub_schema_builder.add_constraint(ScalarInequalityConstraint(target_column="age", operator="gt", rhs=0))

    stub_schema_builder.add_constraint(
        ColumnInequalityConstraint(target_column="start_date", operator="lt", rhs="end_date")
    )

    sampler_columns = stub_schema_builder.to_sampler_columns()
    generator = DatasetGenerator(sampler_columns=sampler_columns, person_generator_loader=stub_person_generator_loader)
    df = generator.generate(NUM_SAMPLES)

    assert df.shape == (NUM_SAMPLES, len(sampler_columns.columns))
    assert (pd.to_datetime(df["start_date"]) < pd.to_datetime(df["end_date"])).all()
    assert df.query("department == 'clothing'")["products"].isin(["shirt", "pants", "shoes"]).all()
    for d, p in zip(
        ["electronics", "clothing", "furniture", "appliances"],
        ["EEE", "CCC", "FFF", "AAA"],
        strict=True,
    ):
        assert df.query(f"department == '{d}'")["employee_id"].str.startswith(f"{p}").all()

    some_lady = {k: df["some_lady"].apply(lambda x: x[k]).to_numpy() for k in ["locale", "city", "sex", "age"]}

    assert (some_lady["locale"] == "en_US").all()
    assert (some_lady["city"] == "New York").all()
    assert (some_lady["sex"] == "Female").all()


def test_max_rejections_factor_error(stub_schema_builder):
    stub_schema_builder.add_column(
        name="col_1",
        sampler_type=SamplerType.UNIFORM,
        params={"low": 2, "high": 3},
    )
    stub_schema_builder.add_column(
        name="col_2",
        sampler_type=SamplerType.UNIFORM,
        params={"low": 0, "high": 1},
    )
    stub_schema_builder.add_constraint(ColumnInequalityConstraint(target_column="col_1", operator="lt", rhs="col_2"))
    generator = DatasetGenerator(sampler_columns=stub_schema_builder.to_sampler_columns(max_rejections_factor=1))

    with pytest.raises(
        RejectionSamplingError,
        match="Exceeded the maximum number of rejections",
    ):
        generator.generate(NUM_SAMPLES)
