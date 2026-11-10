# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_designer.config.analysis.column_statistics import ColumnDistributionType
from data_designer.config.column_configs import ExpressionColumnConfig, SamplerColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ModelProvider
from data_designer.config.run_config import RunConfig
from data_designer.config.sampler_params import SamplerType, UniformSamplerParams
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.interface.data_designer import DataDesigner


@pytest.fixture
def designer(tmp_path: Path, stub_model_providers: list[ModelProvider]) -> DataDesigner:
    managed = tmp_path / "managed"
    managed.mkdir()
    designer = DataDesigner(
        artifact_path=tmp_path / "artifacts",
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=managed,
    )
    designer.set_run_config(RunConfig(otel_metrics_port=None))
    return designer


def _statistics(analysis, column_name: str):
    return next(stat for stat in analysis.column_statistics if stat.column_name == column_name)


@pytest.mark.parametrize("run", ["preview", "create"])
@pytest.mark.parametrize(
    "values",
    [
        pytest.param(pd.Series([7, None], dtype="Int64"), id="nullable-int"),
        pytest.param(pd.Series(["a", None], dtype="string"), id="nullable-string"),
        pytest.param(pd.Series([True, None], dtype="boolean"), id="nullable-boolean"),
    ],
)
def test_profile_counts_a_missing_seed_value_as_null_not_as_a_distinct_value(
    designer: DataDesigner, values: pd.Series, run: str
) -> None:
    config = DataDesignerConfigBuilder().with_seed_dataset(DataFrameSeedSource(df=pd.DataFrame({"value": values})))
    config.add_column(
        ExpressionColumnConfig(name="rendered", expr="{{ 'MISSING' if value is none else 'PRESENT' }}", dtype="str")
    )

    if run == "preview":
        analysis = designer.preview(config, num_records=2).analysis
    else:
        analysis = designer.create(config, num_records=2, dataset_name="profile_nulls").load_analysis()

    statistics = _statistics(analysis, "value")
    assert statistics.num_null == 1
    assert statistics.num_unique == 1


@pytest.mark.parametrize("run", ["preview", "create"])
def test_profile_keeps_the_distribution_of_a_single_record_numerical_sampler(designer: DataDesigner, run: str) -> None:
    config = DataDesignerConfigBuilder()
    config.add_column(
        SamplerColumnConfig(name="value", sampler_type=SamplerType.UNIFORM, params=UniformSamplerParams(low=0, high=10))
    )

    if run == "preview":
        result = designer.preview(config, num_records=1)
        analysis, dataset = result.analysis, result.dataset
    else:
        created = designer.create(config, num_records=1, dataset_name="profile_single")
        analysis, dataset = created.load_analysis(), created.load_dataset()

    statistics = _statistics(analysis, "value")
    assert statistics.distribution_type == ColumnDistributionType.NUMERICAL
    sampled = float(dataset["value"].iloc[0])
    assert statistics.distribution.min == pytest.approx(sampled, abs=0.01)
    assert statistics.distribution.max == pytest.approx(sampled, abs=0.01)
    assert statistics.distribution.mean == pytest.approx(sampled, abs=0.01)
    assert statistics.distribution.median == pytest.approx(sampled, abs=0.01)
    assert statistics.distribution.stddev != statistics.distribution.stddev  # NaN: undefined for one record
