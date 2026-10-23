# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import operator

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.sampler_constraints import ColumnInequalityConstraint, ScalarInequalityConstraint
from data_designer.config.sampler_params import GaussianSamplerParams, ScipySamplerParams, UniformSamplerParams
from data_designer.engine.sampling_gen.errors import RejectionSamplingError
from data_designer.engine.sampling_gen.generator import DatasetGenerator
from data_designer.engine.sampling_gen.schema_builder import SchemaBuilder


@pytest.mark.parametrize(
    "params",
    [
        UniformSamplerParams(low=0, high=1),
        GaussianSamplerParams(mean=0.5, stddev=0.2),
        ScipySamplerParams(dist_name="uniform", dist_params={"loc": 0, "scale": 1}),
    ],
    ids=["uniform", "gaussian", "scipy"],
)
@pytest.mark.parametrize(("comparison", "rhs"), [("gt", 0.25), ("ge", 0.6), ("lt", 0.75), ("le", 0.4)])
@pytest.mark.parametrize("decimal_places", [None, 0], ids=["unrounded", "rounded"])
def test_scalar_constraints_use_final_values(
    stub_schema_builder: SchemaBuilder,
    params: UniformSamplerParams | GaussianSamplerParams | ScipySamplerParams,
    comparison: str,
    rhs: float,
    decimal_places: int | None,
) -> None:
    params = params.model_copy(update={"decimal_places": decimal_places})
    stub_schema_builder.add_column(name="value", sampler_type=params.sampler_type, params=params.model_dump())
    stub_schema_builder.add_constraint(ScalarInequalityConstraint(target_column="value", operator=comparison, rhs=rhs))

    frame = DatasetGenerator(stub_schema_builder.to_sampler_columns(), random_state=42).generate(100)

    assert getattr(operator, comparison)(frame["value"], rhs).all()
    if decimal_places is not None:
        assert frame["value"].equals(frame["value"].round(decimal_places))


@pytest.mark.parametrize(("comparison", "low", "high"), [("lt", 0.6, 0.9), ("gt", 0.1, 0.4)])
@pytest.mark.parametrize("decimal_places", [None, 0], ids=["unrounded", "rounded"])
def test_column_constraints_use_final_values(
    stub_schema_builder: SchemaBuilder, comparison: str, low: float, high: float, decimal_places: int | None
) -> None:
    stub_schema_builder.add_column(
        name="value", sampler_type="uniform", params={"low": 0, "high": 1, "decimal_places": decimal_places}
    )
    stub_schema_builder.add_column(
        name="limit", sampler_type="uniform", params={"low": low, "high": high, "decimal_places": decimal_places}
    )
    stub_schema_builder.add_constraint(
        ColumnInequalityConstraint(target_column="value", operator=comparison, rhs="limit")
    )

    frame = DatasetGenerator(stub_schema_builder.to_sampler_columns(), random_state=42).generate(100)

    assert getattr(operator, comparison)(frame["value"], frame["limit"]).all()


@pytest.mark.parametrize("convert_to", [None, "float", "int", "str"])
@pytest.mark.parametrize("decimal_places", [None, 0], ids=["unrounded", "rounded"])
def test_conditional_sampling_uses_final_values(
    stub_schema_builder: SchemaBuilder, convert_to: str | None, decimal_places: int | None
) -> None:
    stub_schema_builder.add_column(
        name="value",
        sampler_type="uniform",
        params={"low": 0.41, "high": 0.49, "decimal_places": decimal_places},
        convert_to=convert_to,
    )
    stub_schema_builder.add_column(
        name="label",
        sampler_type="category",
        params={"values": ["low"]},
        conditional_params={"value | float > 0.4": {"values": ["high"]}},
    )

    frame = DatasetGenerator(stub_schema_builder.to_sampler_columns(), random_state=42).generate(16)

    expected = lazy.np.where(frame["value"].astype(float) > 0.4, "high", "low")
    assert (frame["label"] == expected).all()


@pytest.mark.parametrize("convert_to", [None, "str"])
@pytest.mark.parametrize("unrounded_precision", [{}, {"decimal_places": None}], ids=["omitted", "none"])
def test_conditional_precision_survives_rejection_sampling(
    stub_schema_builder: SchemaBuilder, convert_to: str | None, unrounded_precision: dict[str, None]
) -> None:
    stub_schema_builder.add_column(
        name="group", sampler_type="category", params={"values": ["whole", "tenths", "unrounded"]}
    )
    stub_schema_builder.add_column(
        name="value",
        sampler_type="uniform",
        params={"low": 0.41, "high": 0.49, "decimal_places": 0},
        conditional_params={
            "group == 'tenths'": {"low": 0.41, "high": 0.49, "decimal_places": 1},
            "group == 'unrounded'": {"low": 0.411, "high": 0.419, **unrounded_precision},
        },
        convert_to=convert_to,
    )
    if convert_to is None:
        stub_schema_builder.add_constraint(ScalarInequalityConstraint(target_column="value", operator="lt", rhs=0.45))

    frame = DatasetGenerator(stub_schema_builder.to_sampler_columns(), random_state=42).generate(100)
    values = frame["value"].astype(float)

    assert set(frame["group"]) == {"whole", "tenths", "unrounded"}
    assert (values[frame["group"] == "whole"] == 0).all()
    tenths = values[frame["group"] == "tenths"]
    assert tenths.isin([0.4] if convert_to is None else [0.4, 0.5]).all()
    assert values[frame["group"] == "unrounded"].between(0.411, 0.419).all()


@pytest.mark.parametrize("constraint_type", ["scalar", "column"])
@pytest.mark.parametrize("decimal_places", [None, 0], ids=["unrounded", "rounded"])
def test_constraints_made_impossible_by_rounding(
    stub_schema_builder: SchemaBuilder, constraint_type: str, decimal_places: int | None
) -> None:
    stub_schema_builder.add_column(
        name="value", sampler_type="uniform", params={"low": 0.41, "high": 0.44, "decimal_places": decimal_places}
    )
    if constraint_type == "scalar":
        stub_schema_builder.add_constraint(ScalarInequalityConstraint(target_column="value", operator="gt", rhs=0.4))
    else:
        stub_schema_builder.add_column(
            name="limit", sampler_type="uniform", params={"low": 0.46, "high": 0.49, "decimal_places": decimal_places}
        )
        stub_schema_builder.add_constraint(
            ColumnInequalityConstraint(target_column="value", operator="lt", rhs="limit")
        )

    generator = DatasetGenerator(stub_schema_builder.to_sampler_columns(max_rejections_factor=1), random_state=42)

    if decimal_places is None:
        assert len(generator.generate(4)) == 4
    else:
        with pytest.raises(RejectionSamplingError, match="Exceeded the maximum number of rejections"):
            generator.generate(4)


@pytest.mark.parametrize(
    ("convert_to", "decimal_places", "low", "high", "expected", "dtype"),
    [
        ("int", 1, 0.51, 0.54, 1, "int64"),
        ("int", -1, 14.6, 14.9, 20, "int64"),
        ("float", -1, 14.6, 14.9, 10.0, "float64"),
        ("str", -1, 14.6, 14.9, "10.0", "str"),
        ("str", 2, 0.411, 0.414, "0.41", "str"),
    ],
    ids=["int-before-precision", "int-negative-precision", "float-negative-precision", "str-negative-precision", "str"],
)
def test_rounding_preserves_type_conversion(
    stub_schema_builder: SchemaBuilder,
    convert_to: str,
    decimal_places: int,
    low: float,
    high: float,
    expected: int | float | str,
    dtype: str,
) -> None:
    stub_schema_builder.add_column(
        name="value",
        sampler_type="uniform",
        params={"low": low, "high": high, "decimal_places": decimal_places},
        convert_to=convert_to,
    )

    frame = DatasetGenerator(stub_schema_builder.to_sampler_columns(), random_state=42).generate(8)

    assert (frame["value"] == expected).all()
    if dtype == "str":
        assert lazy.pd.api.types.is_string_dtype(frame["value"].dtype)
    else:
        assert frame["value"].dtype == dtype


@pytest.mark.parametrize("convert_to", [None, "float", "int"])
def test_negative_precision_constraints(stub_schema_builder: SchemaBuilder, convert_to: str | None) -> None:
    stub_schema_builder.add_column(
        name="value",
        sampler_type="uniform",
        params={"low": 0, "high": 30, "decimal_places": -1},
        convert_to=convert_to,
    )
    stub_schema_builder.add_constraint(ScalarInequalityConstraint(target_column="value", operator="lt", rhs=18))

    frame = DatasetGenerator(stub_schema_builder.to_sampler_columns(), random_state=42).generate(100)

    assert frame["value"].isin([0, 10]).all()
