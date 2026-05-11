# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile

import pytest
import yaml

from data_designer.config.data_designer_config import DataDesignerConfig


def test_data_designer_config_to_dict(stub_data_designer_config):
    assert isinstance(stub_data_designer_config.to_dict(), dict)


def test_data_designer_config_to_yaml(stub_data_designer_config):
    assert isinstance(stub_data_designer_config.to_yaml(), str)
    with tempfile.NamedTemporaryFile(suffix=".yaml") as tmp_file:
        result = stub_data_designer_config.to_yaml(tmp_file.name)
        assert result is None
        with open(tmp_file.name, "r") as f:
            assert yaml.safe_load(f) == stub_data_designer_config.to_dict()


def test_data_designer_config_to_json(stub_data_designer_config):
    assert isinstance(stub_data_designer_config.to_json(), str)
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp_file:
        result = stub_data_designer_config.to_json(tmp_file.name)
        assert result is None
        with open(tmp_file.name, "r") as f:
            assert json.loads(f.read()) == stub_data_designer_config.to_dict()


def test_data_designer_config_parses_constraint_type_from_legacy_shape() -> None:
    config = DataDesignerConfig.model_validate(
        {
            "columns": [
                {
                    "name": "age",
                    "column_type": "sampler",
                    "sampler_type": "uniform",
                    "params": {"low": 18, "high": 99},
                }
            ],
            "constraints": [
                {"target_column": "age", "operator": "lt", "rhs": 65},
                {"target_column": "age", "operator": "le", "rhs": "65"},
                {"target_column": "age", "operator": "gt", "rhs": "minimum_age"},
            ],
        }
    )

    serialized_constraints = [constraint.model_dump(mode="json") for constraint in config.constraints]
    assert serialized_constraints == [
        {
            "target_column": "age",
            "operator": "lt",
            "rhs": 65.0,
            "constraint_type": "scalar_inequality",
        },
        {
            "target_column": "age",
            "operator": "le",
            "rhs": 65.0,
            "constraint_type": "scalar_inequality",
        },
        {
            "target_column": "age",
            "operator": "gt",
            "rhs": "minimum_age",
            "constraint_type": "column_inequality",
        },
    ]


def test_data_designer_config_parses_constraint_type_from_tagged_shape() -> None:
    config = DataDesignerConfig.model_validate(
        {
            "columns": [
                {
                    "name": "age",
                    "column_type": "sampler",
                    "sampler_type": "uniform",
                    "params": {"low": 18, "high": 99},
                }
            ],
            "constraints": [
                {"target_column": "age", "operator": "lt", "rhs": 65.0, "constraint_type": "scalar_inequality"},
                {
                    "target_column": "age",
                    "operator": "gt",
                    "rhs": "minimum_age",
                    "constraint_type": "column_inequality",
                },
            ],
        }
    )

    serialized_constraints = [constraint.model_dump(mode="json") for constraint in config.constraints]
    assert serialized_constraints == [
        {
            "target_column": "age",
            "operator": "lt",
            "rhs": 65.0,
            "constraint_type": "scalar_inequality",
        },
        {
            "target_column": "age",
            "operator": "gt",
            "rhs": "minimum_age",
            "constraint_type": "column_inequality",
        },
    ]


def test_data_designer_config_constraint_missing_rhs_raises_validation_error() -> None:
    with pytest.raises(Exception):
        DataDesignerConfig.model_validate(
            {
                "columns": [
                    {
                        "name": "age",
                        "column_type": "sampler",
                        "sampler_type": "uniform",
                        "params": {"low": 18, "high": 99},
                    }
                ],
                "constraints": [
                    {"target_column": "age", "operator": "lt"},
                ],
            }
        )


def test_subcategory_parent_must_be_a_sampler_column() -> None:
    with pytest.raises(ValueError, match=r"Subcategory column 'ski_category'.*'llm-text' column"):
        DataDesignerConfig.model_validate(
            {
                "columns": [
                    {
                        "name": "package_type",
                        "column_type": "llm-text",
                        "prompt": "describe a package",
                        "model_alias": "default",
                    },
                    {
                        "name": "ski_category",
                        "column_type": "sampler",
                        "sampler_type": "subcategory",
                        "params": {
                            "category": "package_type",
                            "values": {"basic": ["a"], "premium": ["b"]},
                        },
                    },
                ]
            }
        )


def test_subcategory_parent_as_category_sampler_is_valid() -> None:
    config = DataDesignerConfig.model_validate(
        {
            "columns": [
                {
                    "name": "package_type",
                    "column_type": "sampler",
                    "sampler_type": "category",
                    "params": {"values": ["basic", "premium"]},
                },
                {
                    "name": "ski_category",
                    "column_type": "sampler",
                    "sampler_type": "subcategory",
                    "params": {
                        "category": "package_type",
                        "values": {"basic": ["a"], "premium": ["b"]},
                    },
                },
            ]
        }
    )
    assert len(config.columns) == 2


def test_subcategory_parent_must_be_a_category_sampler() -> None:
    with pytest.raises(ValueError, match=r"sampler_type='uniform'.*sampler_type='category'"):
        DataDesignerConfig.model_validate(
            {
                "columns": [
                    {
                        "name": "package_type",
                        "column_type": "sampler",
                        "sampler_type": "uniform",
                        "params": {"low": 0, "high": 1},
                    },
                    {
                        "name": "ski_category",
                        "column_type": "sampler",
                        "sampler_type": "subcategory",
                        "params": {
                            "category": "package_type",
                            "values": {"basic": ["a"], "premium": ["b"]},
                        },
                    },
                ]
            }
        )


def test_subcategory_parent_missing_defers_to_schema_validator() -> None:
    config = DataDesignerConfig.model_validate(
        {
            "columns": [
                {
                    "name": "ski_category",
                    "column_type": "sampler",
                    "sampler_type": "subcategory",
                    "params": {
                        "category": "does_not_exist",
                        "values": {"basic": ["a"]},
                    },
                },
            ]
        }
    )
    assert len(config.columns) == 1
