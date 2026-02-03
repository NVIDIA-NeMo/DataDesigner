# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from data_designer.config.sampler_constraints import ColumnInequalityConstraint, ScalarInequalityConstraint
from data_designer.config.sampler_params import SamplerType


def test_valid_dag(stub_schema_builder):
    stub_schema_builder.add_column(
        name="col_1",
        sampler_type=SamplerType.UNIFORM,
        params={"low": 0, "high": 1},
        conditional_params={
            "col_2 == 'not_this_value'": {"low": 0, "high": 0.5},
        },
    )

    stub_schema_builder.add_column(
        name="col_2",
        sampler_type=SamplerType.CATEGORY,
        params={
            "values": ["this_value", "not_this_value"],
        },
    )

    schema = stub_schema_builder.build()

    assert schema.dag.to_networkx().is_directed()
    assert schema.dag.nodes == {"col_1", "col_2"}
    assert schema.dag.edges == {("col_2", "col_1")}


def test_invalid_dag(stub_schema_builder):
    stub_schema_builder.add_column(
        name="col_1",
        sampler_type=SamplerType.UNIFORM,
        params={"low": 0, "high": 1},
        conditional_params={
            "col_2 == 'not_this_value'": {"low": 0, "high": 0.5},
        },
    )

    stub_schema_builder.add_column(
        name="col_2",
        sampler_type=SamplerType.CATEGORY,
        params={"values": ["this_value", "not_this_value"]},
        conditional_params={
            "col_1 == 0": {"values": ["this_value"]},
        },
    )

    with pytest.raises(ValidationError, match="There are circular dependencies"):
        stub_schema_builder.build()


def test_conditional_columns_and_constraints(stub_schema_builder):
    stub_schema_builder.add_column(
        name="department",
        sampler_type=SamplerType.CATEGORY,
        params={
            "values": ["electronics", "clothing", "furniture", "appliances"],
            "weights": [0.4, 0.3, 0.2, 0.1],
        },
    )

    stub_schema_builder.add_column(
        name="age",
        sampler_type=SamplerType.SCIPY,
        params={
            "dist_name": "norm",
            "dist_params": {"loc": 25, "scale": 8},
        },
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
        sampler_type=SamplerType.DATETIME,
        params={
            "start": "2020-01-01",
            "end": "2025-01-01",
            "unit": "D",
        },
        convert_to="%m-%d-%Y",
    )

    stub_schema_builder.add_column(
        name="end_date",
        sampler_type=SamplerType.DATETIME,
        params={
            "start": "2021-01-01",
            "end": "2026-01-01",
            "unit": "D",
        },
        convert_to="%m-%d-%Y",
    )

    stub_schema_builder.add_constraint(ScalarInequalityConstraint(target_column="age", operator="gt", rhs=0))

    stub_schema_builder.add_constraint(
        ColumnInequalityConstraint(target_column="start_date", operator="lt", rhs="end_date")
    )

    schema = stub_schema_builder.build()

    assert len(schema.get_column("age").conditions) == 5
    assert len(schema.get_constraint_checkers("age")) == 1
    assert len(schema.get_constraint_checkers("start_date")) == 1
    assert len(schema.constraints) == 2


def test_no_constraint_on_column_not_in_schema(stub_schema_builder):
    stub_schema_builder.add_column(
        name="col_1",
        sampler_type=SamplerType.UNIFORM,
        params={"low": 0, "high": 1},
    )

    stub_schema_builder.add_constraint(ScalarInequalityConstraint(target_column="col_2", operator="gt", rhs=0))

    with pytest.raises(
        ValidationError,
        match="These constrained columns are missing in the definitions of your sampler columns",
    ):
        stub_schema_builder.build()


def test_subcategory_with_no_category_column(stub_schema_builder):
    stub_schema_builder.add_column(
        name="col_1",
        sampler_type=SamplerType.UNIFORM,
        params={"low": 0, "high": 1},
    )

    stub_schema_builder.add_column(
        name="col_2",
        sampler_type=SamplerType.SUBCATEGORY,
        params={"category": "department", "values": {"electronics": ["laptop"]}},
    )

    with pytest.raises(ValidationError, match="Column 'department' not found in schema"):
        stub_schema_builder.build()


def test_subcategory_with_non_category_parent_column(stub_schema_builder):
    stub_schema_builder.add_column(
        name="col_1",
        sampler_type=SamplerType.UNIFORM,
        params={"low": 0, "high": 1},
    )

    stub_schema_builder.add_column(
        name="department",
        sampler_type=SamplerType.SUBCATEGORY,
        params={"category": "col_1", "values": {"electronics": ["laptop"]}},
    )

    with pytest.raises(ValidationError, match="The parent of subcategory column"):
        stub_schema_builder.build()


def test_subcategory_extra_category_value(stub_schema_builder):
    stub_schema_builder.add_column(
        name="department",
        sampler_type=SamplerType.CATEGORY,
        params={
            "values": ["electronics", "clothing", "furniture", "appliances"],
        },
    )

    stub_schema_builder.add_column(
        name="col_2",
        sampler_type=SamplerType.SUBCATEGORY,
        params={
            "category": "department",
            "values": {
                "electronics": ["laptop"],
                "clothing": ["shirt", "pants"],
                "furniture": ["sofa", "chair"],
                "appliances": ["refrigerator", "microwave"],
                "other": ["other"],
            },
        },
    )

    with pytest.raises(ValidationError, match="Subcategory column 'col_2' must have values"):
        stub_schema_builder.build()


@pytest.mark.parametrize(
    "sub_values",
    [
        {"electronics": ["laptop"], "clothing": ["shirt"], "furniture": ["chair"]},
        {"electronics": ["laptop"]},
    ],
)
def test_subcategory_missing_category_value(sub_values, stub_schema_builder):
    stub_schema_builder.add_column(
        name="department",
        sampler_type=SamplerType.CATEGORY,
        params={"values": ["electronics", "clothing"]},
    )

    stub_schema_builder.add_column(
        name="col_2",
        sampler_type=SamplerType.SUBCATEGORY,
        params={
            "category": "department",
            "values": sub_values,
        },
    )

    with pytest.raises(ValidationError, match="Subcategory column 'col_2' must have values"):
        stub_schema_builder.build()


def test_subcategory_empty_values(stub_schema_builder):
    stub_schema_builder.add_column(
        name="department",
        sampler_type=SamplerType.CATEGORY,
        params={"values": ["electronics", "clothing"]},
    )

    stub_schema_builder.add_column(
        name="col_2",
        sampler_type=SamplerType.SUBCATEGORY,
        params={
            "category": "department",
            "values": {"electronics": ["laptop"], "clothing": []},
        },
    )

    with pytest.raises(ValidationError, match="Subcategory column 'col_2' must have non-empty values"):
        stub_schema_builder.build()
