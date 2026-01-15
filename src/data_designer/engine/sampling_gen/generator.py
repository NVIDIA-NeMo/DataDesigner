# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from data_designer.engine.sampling_gen.data_sources.base import RadomStateT
from data_designer.engine.sampling_gen.errors import RejectionSamplingError
from data_designer.engine.sampling_gen.jinja_utils import JinjaDataFrame
from data_designer.engine.sampling_gen.people_gen import create_people_gen_resource
from data_designer.engine.sampling_gen.schema import DataSchema
from data_designer.engine.sampling_gen.utils import check_random_state
from data_designer.lazy_heavy_imports import np, nx, pd

if TYPE_CHECKING:
    import networkx as nx
    import numpy as np
    import pandas as pd

    from data_designer.engine.dataset_builders.multi_column_configs import SamplerMultiColumnConfig
    from data_designer.engine.resources.managed_dataset_generator import ManagedDatasetGenerator
    from data_designer.engine.sampling_gen.column import ConditionalDataColumn


class DatasetGenerator:
    """Generates synthetic datasets based on the given schema definition.

    This object generates synthetic data based on the schema using sampling-based
    methods (implemented as "data sources"), including handling conditional generation
    and enforcing constraints through rejection sampling.

    Args:
        sampler_columns: Sampler columns to generate.
        random_state: Random number generator or seed for reproducibility.
        person_generator_loader: A function that loads a person generator. If None,
            person generation will not be supported.

    Note:
        The generator leverages the schema's DAG to topologically sort the columns
        and uses rejection sampling to satisfy constraints. If constraints are too strict,
        generation may fail with a RejectionSamplingError.
    """

    def __init__(
        self,
        sampler_columns: SamplerMultiColumnConfig | None,
        random_state: RadomStateT | None = None,
        person_generator_loader: Callable[[bool], ManagedDatasetGenerator] | None = None,
        *,
        schema: DataSchema | None = None,
        max_rejections_factor: int = 5,
    ):
        # This is temporary while we need the legacy and refactored code to coexist.
        if schema is not None:
            self.schema = schema
            self.max_rejections_factor = max_rejections_factor
        else:
            self.schema = DataSchema(
                columns=[column.model_dump() for column in sampler_columns.columns],
                constraints=sampler_columns.constraints,
            )
            self.max_rejections_factor = sampler_columns.max_rejections_factor

        self.rng = check_random_state(random_state)
        self._dag = self.schema.dag.to_networkx()
        self._shared_sampler_kwargs = {
            "random_state": self.rng,
            "people_gen_resource": create_people_gen_resource(self.schema, person_generator_loader),
        }

    def _round_if_needed(self, column: ConditionalDataColumn, df: pd.DataFrame) -> pd.DataFrame:
        if hasattr(column.params, "decimal_places") and column.params.decimal_places is not None:
            df[column.name] = df[column.name].round(column.params.decimal_places)
        return df

    def _run_rejection_sampling(self, df: pd.DataFrame, column: ConditionalDataColumn) -> pd.DataFrame:
        name = column.name
        num_iterations = 0
        num_samples = len(df)
        needs_samples = np.ones(num_samples, dtype=bool)

        while needs_samples.any():
            for condition in column.conditions:
                index = JinjaDataFrame(condition).select_index(df[needs_samples])
                src = column.get_sampler(condition, **self._shared_sampler_kwargs)
                df = src.inject_data_column(df, name, index)

            df[name] = column.get_default_sampler(**self._shared_sampler_kwargs).preproc(df[name], column.convert_to)

            # Check all constraints on the column.
            batch_mask = np.ones(num_samples, dtype=bool)
            for constraint in self.schema.get_constraint_checkers(name):
                batch_mask &= constraint.check(df)
            needs_samples[batch_mask] = False
            num_iterations += 1

            if num_iterations > self.max_rejections_factor * num_samples:
                raise RejectionSamplingError(
                    "Exceeded the maximum number of rejections (max_rejections_factor * "
                    f"num_samples = {self.max_rejections_factor * num_samples}) while "
                    f"sampling `{column.name}`. Please consider adjusting the constraints "
                    "and/or column's generation configuration."
                )

        return df

    def generate(self, num_samples: int) -> pd.DataFrame:
        dataset = pd.DataFrame(index=range(num_samples))

        for column_name in nx.topological_sort(self._dag):
            column = self.schema.get_column(column_name)
            dataset = self._run_rejection_sampling(dataset, column)

        for column in self.schema.columns:
            dataset[column.name] = column.get_default_sampler(**self._shared_sampler_kwargs).postproc(
                dataset[column.name], column.convert_to
            )
            dataset = self._round_if_needed(column, dataset)

        return dataset[self.schema.column_names]
