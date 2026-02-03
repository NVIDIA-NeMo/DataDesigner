# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Self

from data_designer.config.base import ConfigBase
from data_designer.config.sampler_constraints import ColumnConstraintT
from data_designer.config.sampler_params import SamplerType
from data_designer.engine.sampling_gen.column import ConditionalDataColumn
from data_designer.engine.sampling_gen.constraints import ConstraintChecker, get_constraint_checker
from data_designer.lazy_heavy_imports import nx

if TYPE_CHECKING:
    import networkx as nx


class Dag(BaseModel):
    nodes: set[str]
    edges: set[tuple[str, str]]

    @model_validator(mode="after")
    def validate_is_dag(self) -> Self:
        if not nx.is_directed_acyclic_graph(self.to_networkx()):
            raise ValueError("There are circular dependencies in the definitions of your sampler columns.")
        return self

    def to_networkx(self) -> nx.DiGraph:
        dag = nx.DiGraph()
        for node in self.nodes:
            dag.add_node(node)
        for edge in self.edges:
            dag.add_edge(*edge)
        return dag


class DataSchema(ConfigBase):
    """Defines the data schema for synthetic data generation.

    A DataSchema represents a collection of columns and their relationships through
    conditional parameters and/or constraints. Upon initialization, the schema validates
    that column dependencies form a DAG and that all constraints reference valid columns.
    """

    columns: list[ConditionalDataColumn] = Field(..., min_length=1)
    constraints: list[ColumnConstraintT] = []

    @cached_property
    def constraint_checkers(self) -> list[ConstraintChecker]:
        return [get_constraint_checker(c.constraint_type)(constraint=c) for c in self.constraints]

    @property
    def column_names(self) -> list[str]:
        return [column.name for column in self.columns]

    @property
    def dag(self) -> Dag:
        nodes = set()
        edges = set()

        for column in self.columns:
            nodes.add(column.name)

            # Add edges for the conditional columns.
            for conditional_column in column.conditional_column_names:
                edges.add((conditional_column, column.name))

            # Add edges if the source has required columns.
            for condition in column.conditions:
                source = column.get_sampler(condition)
                for required_column in source.get_required_column_names():
                    edges.add((required_column, column.name))

        for checker in self.constraint_checkers:
            column_names = checker.get_required_column_names()
            if len(column_names) == 2:
                edges.add((column_names[1], column_names[0]))
        return Dag(nodes=nodes, edges=edges)

    @field_validator("columns", mode="after")
    def check_unique_column_names(cls, columns: list[ConditionalDataColumn]) -> list[ConditionalDataColumn]:
        column_names = [column.name for column in columns]
        if len(column_names) != len(set(column_names)):
            raise ValueError("Column names must be unique")
        return columns

    @model_validator(mode="after")
    def validate_constraints(self) -> Self:
        column_names = [column.name for column in self.columns]

        # Check if all columns required by constraints are present in the schema.
        for checker in self.constraint_checkers:
            constrained_column_names = checker.get_required_column_names()
            if not set(constrained_column_names).issubset(column_names):
                missing = set(constrained_column_names) - set(column_names)
                raise ValueError(
                    f"These constrained columns are missing in the definitions of your sampler columns: {missing}"
                )

        return self

    @model_validator(mode="after")
    def validate_dag(self) -> Self:
        self.dag
        return self

    @model_validator(mode="after")
    def validate_subcategory_columns_if_present(self) -> Self:
        for sub in self.get_columns_by_sampler_type(SamplerType.SUBCATEGORY):
            cat = self.get_column(sub.params.category)
            if cat.sampler_type != SamplerType.CATEGORY:
                raise ValueError(
                    f"The parent of subcategory column '{sub.name}' must be a category "
                    f"source type, but '{cat.name}' is of type '{cat.sampler_type}'."
                )
            cat_vals = set(cat.params.values)
            for params in cat.conditional_params.values():
                cat_vals.update(params.values)
            sub_vals = set(sub.params.values.keys())
            if cat_vals.symmetric_difference(sub_vals):
                raise ValueError(
                    f"Subcategory column '{sub.name}' must have values for each value of "
                    f"its parent category '{sub.params.category}'. The following "
                    f"values need attention: {cat_vals.symmetric_difference(sub_vals)}"
                )
            if not all(len(v) > 0 for v in sub.params.values.values()):
                raise ValueError(
                    f"Subcategory column '{sub.name}' must have non-empty values for "
                    f"each value of its parent category '{sub.params.category}'."
                )
        return self

    def get_column(self, column_name: str) -> ConditionalDataColumn:
        if column_name not in self.column_names:
            raise ValueError(f"Column '{column_name}' not found in schema")
        return next(column for column in self.columns if column.name == column_name)

    def get_columns_by_sampler_type(self, sampler_type: SamplerType) -> list[ConditionalDataColumn]:
        return [c for c in self.columns if c.sampler_type == sampler_type]

    def get_constraint_checkers(self, column_name: str) -> list[ConstraintChecker]:
        return [c for c in self.constraint_checkers if column_name == c.constraint.target_column]
